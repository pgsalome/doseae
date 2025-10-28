"""
Lung-specific preprocessor for dose autoencoder training.
Handles lung segmentation, dose normalization, and patch extraction.
"""

import os
import pickle
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from entities.base.base_preprocessor import BasePreprocessor

# Standardized lobe naming system
STANDARD_LOBE_NAMES = {
    'left_upper': 'left_upper_lobe',
    'left_lower': 'left_lower_lobe',
    'right_upper': 'right_upper_lobe',
    'right_middle': 'right_middle_lobe',
    'right_lower': 'right_lower_lobe'
}

LOBE_TO_SIDE = {
    'left_upper_lobe': 'left',
    'left_lower_lobe': 'left',
    'right_upper_lobe': 'right',
    'right_middle_lobe': 'right',
    'right_lower_lobe': 'right'
}


class LungPreprocessor(BasePreprocessor):
    """
    Lung-specific preprocessor for dose autoencoder training.
    This implementation focuses on robustness and resumability: each patient
    is cached individually so interrupted preprocessing runs can resume.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        
        # Lung-specific configuration
        self.lung_config = config.get('lung', {})
        self.patch_size = self.lung_config.get('patch_size', 50)
        self.overlap_percentage = self.lung_config.get('overlap_percentage', 20)
        self.dose_threshold = self.lung_config.get('dose_threshold', 0.1)
        self.lobe_acceptance = float(self.lung_config.get('mask_acceptance', 0.7))
        self.lobe_composite_threshold = float(self.lung_config.get('lobe_composite_threshold', 0.1))
        self.max_patches_per_lobe = int(self.lung_config.get('max_patches_per_lobe', 0) or 0)
        cpu_count = max(1, os.cpu_count() or 1)
        self.patch_workers = int(self.lung_config.get('patch_workers', min(8, cpu_count)))

        # Preprocessing configuration
        self.preproc_config = config.get('preprocessing', {})
        self.target_spacing = tuple(self.preproc_config.get('voxel_spacing', [0.5, 0.5, 0.5]))
        self.ct_preprocessing = self.preproc_config.get('ct_preprocessing', {})
        self.dose_preprocessing = self.preproc_config.get('dose_preprocessing', {})

        # Segmentation configuration
        self.segmentation_config = config.get('segmentation', {})
        self.segmentation_device_cfg = self.segmentation_config.get('device', 'auto')
        self.segmentation_devices = self._parse_segmentation_devices(self.segmentation_device_cfg)
        self._segmentation_device_lock = threading.Lock()
        self._next_segmentation_device = 0
        max_concurrent = int(self.segmentation_config.get('max_concurrent', 1))
        self._segmentation_semaphore = threading.Semaphore(max(1, max_concurrent))

        # Output organization
        self.output_root = self.output_dir  # Base root for all experiment outputs
        self.patch_root = self.output_root / 'processed_patches'
        self.image_root = self.output_root / 'processed_images'
    
    # ------------------------------------------------------------------
    # Directory helpers / caching utilities
    # ------------------------------------------------------------------
    def _get_split_dir(self, split_name: Optional[str], kind: str, ensure: bool = False) -> Path:
        if kind == 'patch':
            base = self.patch_root
        elif kind == 'image':
            base = self.image_root
        else:
            base = self.output_root
        path = base
        if split_name:
            path = path / split_name
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    def _get_image_patient_dir(self, split_name: Optional[str], patient_id: str,
                               ensure: bool = False) -> Path:
        path = self._get_split_dir(split_name, kind='image', ensure=ensure)
        patient_dir = path / patient_id
        if ensure:
            patient_dir.mkdir(parents=True, exist_ok=True)
        return patient_dir
    
    def _get_patch_cache_file(self, split_name: Optional[str], patient_id: str,
                              ensure_dir: bool = False) -> Path:
        split_dir = self._get_split_dir(split_name, kind='patch', ensure=ensure_dir)
        cache_dir = split_dir / 'patient_cache'
        if ensure_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{patient_id}.h5"

    def _is_patch_cache_valid(self, split_name: Optional[str], patient_id: str) -> bool:
        cache_path = self._get_patch_cache_file(split_name, patient_id, ensure_dir=False)
        if not cache_path.exists():
            return False
        try:
            import h5py
            import json
            with h5py.File(cache_path, 'r') as f:
                if 'metadata' not in f:
                    return False
                meta_ds = f['metadata']
                itemsize = getattr(getattr(meta_ds, 'dtype', None), 'itemsize', None)
                if itemsize and itemsize > 10_000_000:
                    return False
                metadata_raw = meta_ds[()]
                metadata = json.loads(self._decode_hdf5_json_blob(metadata_raw))
                if isinstance(metadata, list) and metadata:
                    sample = metadata[0]
                    if isinstance(sample, dict) and (
                        'ct_patch' in sample or 'dose_patch' in sample
                    ):
                        return False
            return True
        except Exception as exc:
            self.logger.debug(
                "Patch cache validation failed for %s: %s",
                cache_path,
                exc
            )
            return False
    
    def has_processed_patient(self, split_name: Optional[str], patient_id: str,
                              experiment_type: str) -> bool:
        experiment_type = experiment_type.lower()
        if experiment_type in ['patch', 'patches', 'patches_only']:
            return self._is_patch_cache_valid(split_name, patient_id)
        patient_dir = self._get_image_patient_dir(split_name, patient_id, ensure=False)
        return (patient_dir / f"{patient_id}_ct_processed.nrrd").exists() and \
               (patient_dir / f"{patient_id}_dose_processed.nrrd").exists()

    def _to_serializable(self, value):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.int32, np.int64, np.int16, np.uint8)):
            return int(value)
        if isinstance(value, (np.floating, np.float32, np.float64)):
            return float(value)
        if isinstance(value, dict):
            return {k: self._to_serializable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_serializable(v) for v in value]
        return value

    def _to_serializable_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: self._to_serializable(v) for k, v in data.items()}

    @staticmethod
    def _decode_hdf5_json_blob(blob: Any) -> str:
        if isinstance(blob, np.ndarray):
            blob = blob.tobytes()
        if isinstance(blob, (bytes, bytearray, np.bytes_)):
            return blob.decode('utf-8')
        if isinstance(blob, str):
            return blob
        return str(blob)
    
    # ------------------------------------------------------------------
    # Segmentation utilities
    # ------------------------------------------------------------------
    def segment_organs(self, ct_image: sitk.Image) -> Dict[str, sitk.Image]:
        """
        Segment lung lobes from CT image using TotalSegmentator.
        """
        with self._segmentation_semaphore:
            device, env = self._determine_segmentation_device()
            try:
                # Force garbage collection before segmentation
                import gc
                gc.collect()
                
                with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
                    sitk.WriteImage(ct_image, tmp_file.name)
                    temp_ct_path = tmp_file.name
                
                temp_output_dir = tempfile.mkdtemp()
                
                cmd = [
                    "TotalSegmentator",
                    "-i", temp_ct_path,
                    "-o", temp_output_dir,
                    "-ta", "total",
                    "-f",
                    "-d", device,
                    "-q"
                ]
                roi_subset = self.segmentation_config.get('roi_subset')
                if roi_subset:
                    if len(roi_subset) == 1:
                        cmd.extend(["--roi_subset", roi_subset[0]])
                    else:
                        cmd.append("--roi_subset")
                        cmd.extend(roi_subset)
                nr_thr_resamp = self.segmentation_config.get('nr_thr_resamp', 1)
                nr_thr_saving = self.segmentation_config.get('nr_thr_saving', 1)
                if nr_thr_resamp:
                    cmd.extend(["--nr_thr_resamp", str(nr_thr_resamp)])
                if nr_thr_saving:
                    cmd.extend(["--nr_thr_saving", str(nr_thr_saving)])
                
                # Enhanced environment variables for memory management
                env.setdefault('JOBLIB_MULTIPROCESSING', '0')
                env.setdefault('MP_NO_SEM', '1')
                env.setdefault('CUDA_LAUNCH_BLOCKING', '1')  # For better error reporting
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64')  # Further limit CUDA memory allocation
                env.setdefault('OMP_NUM_THREADS', '1')  # Limit OpenMP threads
                env.setdefault('MKL_NUM_THREADS', '1')  # Limit MKL threads
                env.setdefault('CUDA_MEMORY_FRACTION', '0.5')  # Limit CUDA memory usage
                env.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:64,roundup_power2_divisions:16')  # Better memory management
                
                self.logger.info(f"Running TotalSegmentator: {' '.join(cmd)} on device={device}")
                self.logger.info(f"Environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'all')}")
                
                # Add memory monitoring
                try:
                    import psutil
                    process = psutil.Process()
                    memory_before = process.memory_info().rss / 1024 / 1024  # MB
                    self.logger.info(f"Memory before segmentation: {memory_before:.1f} MB")
                except ImportError:
                    pass
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
                    if result.returncode != 0:
                        self.logger.error(f"TotalSegmentator failed with return code {result.returncode}")
                        self.logger.error(f"STDOUT: {result.stdout}")
                        self.logger.error(f"STDERR: {result.stderr}")
                        
                        # Try fallback to CPU if GPU failed
                        if device != 'cpu':
                            self.logger.warning("GPU segmentation failed, trying CPU fallback...")
                            cmd_cpu = cmd.copy()
                            cmd_cpu[cmd_cpu.index('-d') + 1] = 'cpu'
                            env_cpu = env.copy()
                            env_cpu.pop('CUDA_VISIBLE_DEVICES', None)  # Remove GPU restriction
                            env_cpu['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU
                            
                            result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=1800, env=env_cpu)
                            if result.returncode != 0:
                                self.logger.error(f"CPU fallback also failed: {result.stderr}")
                                raise RuntimeError(f"TotalSegmentator failed on both GPU and CPU: {result.stderr}")
                            else:
                                self.logger.info("CPU fallback succeeded")
                        else:
                            raise RuntimeError(f"TotalSegmentator failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    self.logger.error("TotalSegmentator timed out after 30 minutes")
                    raise RuntimeError("TotalSegmentator timed out")
                except Exception as e:
                    self.logger.error(f"Unexpected error during TotalSegmentator execution: {e}")
                    raise RuntimeError(f"TotalSegmentator execution failed: {e}")

                temp_output_path = Path(temp_output_dir)
                lung_masks = {}
                mapping = {
                    'lung_upper_lobe_left': 'left_upper_lobe',
                    'lung_lower_lobe_left': 'left_lower_lobe',
                    'lung_upper_lobe_right': 'right_upper_lobe',
                    'lung_middle_lobe_right': 'right_middle_lobe',
                    'lung_lower_lobe_right': 'right_lower_lobe',
                    'aorta': 'aorta',
                    'trachea': 'trachea'
                }
                for seg_name, organ_name in mapping.items():
                    seg_path = temp_output_path / f"{seg_name}.nii.gz"
                    if seg_path.exists():
                        mask = sitk.ReadImage(str(seg_path))
                        mask = self._resample_to_reference(mask, ct_image)
                        lung_masks[organ_name] = mask
                    else:
                        self.logger.warning(f"Organ segmentation missing: {seg_path}")
                
                if {'left_upper_lobe', 'left_lower_lobe'}.issubset(lung_masks):
                    combined = self._combine_lung_masks(lung_masks['left_upper_lobe'],
                                                        lung_masks['left_lower_lobe'])
                    lung_masks['left_lung'] = self._resample_to_reference(combined, ct_image)
                if {'right_upper_lobe', 'right_middle_lobe', 'right_lower_lobe'}.issubset(lung_masks):
                    combined = self._combine_lung_masks(lung_masks['right_upper_lobe'],
                                                        lung_masks['right_middle_lobe'],
                                                        lung_masks['right_lower_lobe'])
                    lung_masks['right_lung'] = self._resample_to_reference(combined, ct_image)

                required_masks = ['left_lung', 'right_lung']
                if not all(mask in lung_masks for mask in required_masks):
                    missing = [mask for mask in required_masks if mask not in lung_masks]
                    raise RuntimeError(f"Missing required lung masks after segmentation: {missing}")
                
                return lung_masks
            finally:
                # Clean up temporary files and directories
                try:
                    if 'temp_ct_path' in locals():
                        os.unlink(temp_ct_path)
                    if 'temp_output_dir' in locals():
                        import shutil
                        shutil.rmtree(temp_output_dir, ignore_errors=True)
                except:
                    pass
                
                # Force garbage collection and CUDA cache clearing
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except:
                    pass
    
    def _check_cuda_available(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _parse_segmentation_devices(self, cfg_value: Optional[str]) -> List[str]:
        if not cfg_value:
            return []
        cfg_value = str(cfg_value)
        cfg_lower = cfg_value.lower()
        devices: List[str] = []
        if cfg_lower.startswith(('gpu', 'cuda')):
            if ':' in cfg_value:
                _, device_str = cfg_value.split(':', 1)
                devices = [d.strip() for d in device_str.split(',') if d.strip()]
            else:
                suffix = cfg_lower.replace('gpu', '').replace('cuda', '').strip()
                if suffix:
                    devices = [suffix]
        elif ',' in cfg_value:
            devices = [d.strip() for d in cfg_value.split(',') if d.strip()]
        return devices

    def _determine_segmentation_device(self) -> Tuple[str, dict]:
        cfg = (self.segmentation_device_cfg or 'auto')
        cfg_lower = str(cfg).lower()
        env = os.environ.copy()
        if cfg_lower.startswith('cpu'):
            return 'cpu', env
        if cfg_lower.startswith(('gpu', 'cuda')):
            if self.segmentation_devices:
                with self._segmentation_device_lock:
                    device_id = self.segmentation_devices[self._next_segmentation_device % len(self.segmentation_devices)]
                    self._next_segmentation_device += 1
                env['CUDA_VISIBLE_DEVICES'] = device_id
            else:
                device_ids = []
                if ':' in cfg:
                    _, device_str = cfg.split(':', 1)
                    device_ids = [d.strip() for d in device_str.split(',') if d.strip()]
                elif cfg_lower not in ['gpu', 'cuda']:
                    suffix = cfg_lower.replace('gpu', '').replace('cuda', '').strip()
                    if suffix:
                        device_ids = [suffix]
                if device_ids:
                    env['CUDA_VISIBLE_DEVICES'] = ','.join(device_ids)
            return 'gpu', env
        if self.segmentation_devices:
            with self._segmentation_device_lock:
                device_id = self.segmentation_devices[self._next_segmentation_device % len(self.segmentation_devices)]
                self._next_segmentation_device += 1
            env['CUDA_VISIBLE_DEVICES'] = device_id
            return 'gpu', env
        if self._check_cuda_available():
            return 'gpu', env
        return 'cpu', env
    
    def _resample_to_reference(self, image: sitk.Image, reference: sitk.Image) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetSize(reference.GetSize())
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        return resampler.Execute(image)
    
    def _combine_lung_masks(self, *masks: sitk.Image) -> sitk.Image:
        arrays = [sitk.GetArrayFromImage(mask) for mask in masks]
        combined = np.zeros_like(arrays[0], dtype=np.uint8)
        for array in arrays:
            combined = np.logical_or(combined, array > 0).astype(np.uint8)
        mask_image = sitk.GetImageFromArray(combined)
        mask_image.CopyInformation(masks[0])
        return mask_image
    
    def _determine_ipsi_contra_lungs(self, left_lung: sitk.Image,
                                     right_lung: sitk.Image,
                                     dose_image: sitk.Image) -> Tuple[sitk.Image, sitk.Image]:
        left = sitk.GetArrayFromImage(left_lung).astype(bool)
        right = sitk.GetArrayFromImage(right_lung).astype(bool)
        dose = sitk.GetArrayFromImage(dose_image)
        left_dose = np.mean(dose[left]) if np.any(left) else 0.0
        right_dose = np.mean(dose[right]) if np.any(right) else 0.0
        if left_dose >= right_dose:
            self.logger.info(f"Left lung identified as ipsilateral (dose {left_dose:.3f})")
            return left_lung, right_lung
        self.logger.info(f"Right lung identified as ipsilateral (dose {right_dose:.3f})")
        return right_lung, left_lung
    
    # ------------------------------------------------------------------
    # Generic helpers inherited / extended
    # ------------------------------------------------------------------
    def resample_to_common_spacing(self, image: sitk.Image,
                                  target_spacing: Tuple[float, float, float]) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
                    for i in range(3)]
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(new_size)
        return resampler.Execute(image)
    
    def resize_image(self, image: sitk.Image, target_size: List[int]) -> sitk.Image:
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_spacing = [
            original_spacing[0] * original_size[0] / target_size[0],
            original_spacing[1] * original_size[1] / target_size[1],
            original_spacing[2] * original_size[2] / target_size[2],
        ]
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        return resampler.Execute(image)
    
    def apply_mask(self, image: sitk.Image, mask: sitk.Image) -> sitk.Image:
        image_array = sitk.GetArrayFromImage(image)
        mask_array = sitk.GetArrayFromImage(mask)
        masked_array = np.where(mask_array > 0, image_array, 0)
        masked_image = sitk.GetImageFromArray(masked_array)
        masked_image.CopyInformation(image)
        return masked_image
    
    # ------------------------------------------------------------------
    # CT / dose preprocessing
    # ------------------------------------------------------------------
    def apply_ct_preprocessing(self, ct_image: sitk.Image) -> sitk.Image:
        method = self.ct_preprocessing.get('method', 'lung_window').lower()
        if method in ['lung_window', 'lung']:
            return self._apply_lung_window_preprocessing(ct_image)
        return self._apply_cOOpD_style_preprocessing(ct_image)
    
    def _apply_cOOpD_style_preprocessing(self, ct_image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(ct_image)
        window_min = 40 - 400 / 2
        window_max = 40 + 400 / 2
        array = np.clip(array, window_min, window_max)
        array = (array - window_min) / (window_max - window_min)
        array = np.clip(array, 0.0, 1.0).astype(np.float32)
        processed_image = sitk.GetImageFromArray(array)
        processed_image.CopyInformation(ct_image)
        return processed_image
    
    def _apply_lung_window_preprocessing(self, ct_image: sitk.Image) -> sitk.Image:
        array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
        center = float(self.ct_preprocessing.get('window_center', -600))
        width = float(self.ct_preprocessing.get('window_width', 1500))
        clip_range = self.ct_preprocessing.get('clip_range')
        if clip_range and len(clip_range) == 2:
            array = np.clip(array, float(clip_range[0]), float(clip_range[1]))
        window_min = center - width / 2.0
        window_max = center + width / 2.0
        array = np.clip(array, window_min, window_max)
        array = (array - window_min) / max(width, 1e-6)
        array = np.clip(array, 0.0, 1.0).astype(np.float32)
        processed_image = sitk.GetImageFromArray(array)
        processed_image.CopyInformation(ct_image)
        return processed_image
    
    def normalize_dose(self, dose_image: sitk.Image,
                      prescribed_dose: Optional[float] = None) -> Tuple[sitk.Image, Dict[str, Any]]:
        array = sitk.GetArrayFromImage(dose_image)
        if prescribed_dose and prescribed_dose > 0:
            scale = float(prescribed_dose)
            normalized = array / scale
            scale_info = {'method': 'prescribed', 'scale': scale}
            self.logger.info(f"Normalized dose with prescribed dose {scale:.2f} Gy")
        else:
            p99 = np.percentile(array, 99)
            scale = float(p99 if p99 > 0 else 1.0)
            normalized = array / scale
            scale_info = {'method': 'p99', 'scale': scale}
            self.logger.info(f"Normalized dose to 99th percentile {scale:.2f} Gy")
        normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        norm_image = sitk.GetImageFromArray(normalized)
        norm_image.CopyInformation(dose_image)
        return norm_image, scale_info
    
    # ------------------------------------------------------------------
    # Patch extraction
    # ------------------------------------------------------------------
    def _extract_patches_from_organ(self, ct_image: sitk.Image, dose_image: sitk.Image,
                                    organ_mask: sitk.Image, organ_name: str):
        ct_array = sitk.GetArrayFromImage(ct_image)
        dose_array = sitk.GetArrayFromImage(dose_image)
        mask_array = sitk.GetArrayFromImage(organ_mask)
        
        coords = np.where(mask_array > 0)
        if len(coords[0]) == 0:
            return [], [], []
        
        min_coords = [np.min(coords[i]) for i in range(3)]
        max_coords = [np.min([np.max(coords[i]) + 1, ct_array.shape[i]]) for i in range(3)]
        
        patch_size = self.patch_size
        overlap = int(patch_size * self.overlap_percentage / 100)
        step = max(patch_size - overlap, 1)
        threshold = 0.1
        
        z_starts = list(range(min_coords[0], max_coords[0] - patch_size + 1, step))
        y_starts = list(range(min_coords[1], max_coords[1] - patch_size + 1, step))
        x_starts = list(range(min_coords[2], max_coords[2] - patch_size + 1, step))
        if not z_starts or not y_starts or not x_starts:
            return [], [], []
        
        def process_z(z_start: int):
            local_ct, local_dose, local_meta = [], [], []
            for y_start in y_starts:
                for x_start in x_starts:
                    ct_patch = ct_array[z_start:z_start+patch_size,
                                        y_start:y_start+patch_size,
                                        x_start:x_start+patch_size]
                    dose_patch = dose_array[z_start:z_start+patch_size,
                                            y_start:y_start+patch_size,
                                            x_start:x_start+patch_size]
                    patch_mask = mask_array[z_start:z_start+patch_size,
                                            y_start:y_start+patch_size,
                                            x_start:x_start+patch_size]
                    organ_ratio = np.mean(patch_mask > 0)
                    if organ_ratio < threshold:
                        continue
                    local_ct.append(ct_patch.copy())
                    local_dose.append(dose_patch.copy())
                    local_meta.append({
                        'coordinates': (z_start, y_start, x_start),
                        'organ_ratio': organ_ratio,
                        'organ_name': organ_name
                    })
            return local_ct, local_dose, local_meta
        
        ct_patches, dose_patches, metadata = [], [], []
        if len(z_starts) == 1 or self.patch_workers <= 1:
            local_ct, local_dose, local_meta = process_z(z_starts[0])
            ct_patches.extend(local_ct)
            dose_patches.extend(local_dose)
            metadata.extend(local_meta)
        else:
            max_workers = min(self.patch_workers, len(z_starts))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for local_ct, local_dose, local_meta in executor.map(process_z, z_starts):
                    if not local_ct:
                        continue
                    ct_patches.extend(local_ct)
                    dose_patches.extend(local_dose)
                    metadata.extend(local_meta)
        return ct_patches, dose_patches, metadata
    
    def extract_patches(self, ct_image: sitk.Image,
                       dose_image: sitk.Image,
                       organ_masks: Dict[str, sitk.Image],
                       patient_id: str = "unknown") -> Dict[str, Any]:
        patches_data: Dict[str, Any] = {
            'ipsilateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()},
            'contralateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()},
            'summary': {}
        }

        if 'left_lung' in organ_masks and 'right_lung' in organ_masks:
            ipsi_lung, contra_lung = self._determine_ipsi_contra_lungs(
                organ_masks['left_lung'], organ_masks['right_lung'], dose_image
            )
            ipsi_side = 'left' if ipsi_lung == organ_masks['left_lung'] else 'right'
        else:
            ipsi_side = 'unknown'

        ct_array = sitk.GetArrayFromImage(ct_image)
        dose_array = sitk.GetArrayFromImage(dose_image)

        left_array = sitk.GetArrayFromImage(organ_masks['left_lung']).astype(bool) if 'left_lung' in organ_masks else None
        right_array = sitk.GetArrayFromImage(organ_masks['right_lung']).astype(bool) if 'right_lung' in organ_masks else None

        if left_array is None and right_array is None:
            lung_arrays = []
            for lobe in STANDARD_LOBE_NAMES.values():
                if lobe in organ_masks:
                    lung_arrays.append(sitk.GetArrayFromImage(organ_masks[lobe]).astype(bool))
            if not lung_arrays:
                return patches_data
            combined_lung = np.any(lung_arrays, axis=0)
        else:
            combined_lung = np.zeros_like(ct_array, dtype=bool)
            if left_array is not None:
                combined_lung |= left_array
            if right_array is not None:
                combined_lung |= right_array

        if not np.any(combined_lung):
            return patches_data

        lobe_arrays: Dict[str, np.ndarray] = {}
        for lobe_name in ['left_upper_lobe', 'left_lower_lobe', 'right_upper_lobe', 'right_middle_lobe', 'right_lower_lobe']:
            if lobe_name in organ_masks:
                lobe_arrays[lobe_name] = sitk.GetArrayFromImage(organ_masks[lobe_name]).astype(bool)

        patch_size = self.patch_size
        if ct_array.shape[0] < patch_size or ct_array.shape[1] < patch_size or ct_array.shape[2] < patch_size:
            return patches_data

        overlap_fraction = max(0.0, min(0.95, self.overlap_percentage / 100.0))
        step = max(int(round(patch_size * (1.0 - overlap_fraction))), 1)

        z_limit = ct_array.shape[0] - patch_size + 1
        y_limit = ct_array.shape[1] - patch_size + 1
        x_limit = ct_array.shape[2] - patch_size + 1
        if z_limit <= 0 or y_limit <= 0 or x_limit <= 0:
            return patches_data

        total_ipsi = 0
        total_contra = 0
        lobe_summary: List[str] = []

        for z_start in range(0, z_limit, step):
            for y_start in range(0, y_limit, step):
                for x_start in range(0, x_limit, step):
                    lung_slice = combined_lung[
                        z_start:z_start + patch_size,
                        y_start:y_start + patch_size,
                        x_start:x_start + patch_size
                    ]
                    lung_voxels = int(np.count_nonzero(lung_slice))
                    if lung_voxels == 0:
                        continue
                    lung_ratio = float(lung_voxels) / lung_slice.size
                    if lung_ratio < self.lobe_acceptance:
                        continue

                    lobe_counts = {}
                    for lobe_name, mask_array in lobe_arrays.items():
                        lobe_counts[lobe_name] = int(np.count_nonzero(
                            mask_array[
                                z_start:z_start + patch_size,
                                y_start:y_start + patch_size,
                                x_start:x_start + patch_size
                            ]
                        ))

                    if not lobe_counts:
                        continue
                    best_lobe = max(lobe_counts, key=lobe_counts.get)
                    best_count = lobe_counts[best_lobe]
                    if best_count == 0:
                        continue

                    lobe_fractions = {
                        lobe_name: (count / float(lung_voxels)) if lung_voxels else 0.0
                        for lobe_name, count in lobe_counts.items()
                    }
                    patch_fractions = {
                        lobe_name: count / float(lung_slice.size)
                        for lobe_name, count in lobe_counts.items()
                    }
                    significant_lobes = [
                        lobe_name for lobe_name, fraction in lobe_fractions.items()
                        if fraction >= self.lobe_composite_threshold
                    ]
                    if not significant_lobes:
                        significant_lobes = [best_lobe]
                    composite_label = '+'.join(sorted(significant_lobes))

                    side_type = 'ipsilateral' if LOBE_TO_SIDE.get(best_lobe, 'unknown') == ipsi_side else 'contralateral'
                    bucket = patches_data[side_type][best_lobe]
                    if self.max_patches_per_lobe and len(bucket) >= self.max_patches_per_lobe:
                        continue

                    ct_patch = ct_array[
                        z_start:z_start + patch_size,
                        y_start:y_start + patch_size,
                        x_start:x_start + patch_size
                    ].copy()
                    dose_patch = dose_array[
                        z_start:z_start + patch_size,
                        y_start:y_start + patch_size,
                        x_start:x_start + patch_size
                    ].copy()

                    bucket.append({
                        'ct_patch': ct_patch,
                        'dose_patch': dose_patch,
                        'start_coords': [z_start, y_start, x_start],
                        'spatial_coord': [hash(patient_id) % 1_000_000, y_start, x_start, z_start],
                        'center_coords': [
                            x_start + patch_size // 2,
                            y_start + patch_size // 2,
                            z_start + patch_size // 2
                        ],
                        'organ_ratio': lung_ratio,
                        'lobe_name': best_lobe,
                        'side_type': side_type,
                        'anatomical_side': LOBE_TO_SIDE.get(best_lobe, 'unknown'),
                        'dose_mean': float(np.mean(dose_patch)),
                        'dose_max': float(np.max(dose_patch)),
                        'dose_std': float(np.std(dose_patch)),
                        'is_high_dose': bool(np.max(dose_patch) > 0.5),
                        'lobe_voxel_fraction': lobe_fractions.get(best_lobe, 0.0),
                        'lobe_patch_fraction': patch_fractions.get(best_lobe, 0.0),
                        'lobe_fractions': lobe_fractions,
                        'lobe_patch_fractions': patch_fractions,
                        'composite_lobes': significant_lobes,
                        'composite_label': composite_label
                    })
                    if side_type == 'ipsilateral':
                        total_ipsi += 1
                    else:
                        total_contra += 1
                    if best_lobe not in lobe_summary:
                        lobe_summary.append(best_lobe)

        patches_data['summary'] = {
            'ipsi_side': ipsi_side,
            'contra_side': 'right' if ipsi_side == 'left' else 'left',
            'total_ipsi_patches': total_ipsi,
            'total_contra_patches': total_contra,
            'lobes_with_patches': lobe_summary
        }
        return patches_data
    
    # ------------------------------------------------------------------
    # Visualization helpers (minimal)
    # ------------------------------------------------------------------
    def _central_slice(self, volume: np.ndarray) -> np.ndarray:
        """Return central axial slice from a 3D volume."""
        if volume.ndim == 3:
            z = volume.shape[0] // 2
            return volume[z]
        return volume

    def save_visualization_patches(self, patches_data: Dict[str, Any], patient_id: str,
                                   output_dir: Path, ct_image: sitk.Image = None) -> str:
        """
        Save representative CT/dose patches for each lobe as PNG images.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = patches_data.get('summary', {})
        summary_path = output_dir / f"{patient_id}_patch_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        if not patches_data:
            return str(output_dir)

        for side_key in ['ipsilateral', 'contralateral']:
            lobe_dict = patches_data.get(side_key, {})
            for lobe, patch_list in lobe_dict.items():
                if not patch_list:
                    continue
                best_patch = max(patch_list, key=lambda x: x.get('organ_ratio', 0))
                ct_patch = np.array(best_patch['ct_patch'], dtype=np.float32)
                dose_patch = np.array(best_patch['dose_patch'], dtype=np.float32)
                patch_dir = output_dir / f"{side_key}_{lobe}"
                patch_dir.mkdir(parents=True, exist_ok=True)

                ct_slice = self._central_slice(ct_patch)
                dose_slice = self._central_slice(dose_patch)

                fig, ax = plt.subplots(1, 2, figsize=(8, 4))
                ax[0].imshow(ct_slice, cmap='gray')
                ax[0].set_title(f"{lobe} CT\nratio={best_patch.get('organ_ratio', 0):.2f}")
                ax[0].axis('off')

                ax[1].imshow(ct_slice, cmap='gray')
                im = ax[1].imshow(dose_slice, cmap='hot', alpha=0.6)
                ax[1].set_title("Dose overlay")
                ax[1].axis('off')
                fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

                fig_path = patch_dir / f"{patient_id}_{lobe}_patch.png"
                plt.tight_layout()
                plt.savefig(fig_path, dpi=120)
                plt.close(fig)

        return str(output_dir)
    
    def create_visualization(self, ct_image: sitk.Image,
                             dose_image: sitk.Image,
                             organ_masks: Dict[str, sitk.Image],
                             patient_id: str,
                             patches_data: Dict[str, Any] = None,
                             output_dir: Optional[Path] = None,
                             dose_stats: Optional[Dict[str, Any]] = None) -> str:
        if output_dir is None:
            output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        ct_array = sitk.GetArrayFromImage(ct_image)
        dose_array = sitk.GetArrayFromImage(dose_image)
        middle_slice = ct_array.shape[0] // 2
        
        fig, ax = plt.subplots(figsize=(8, 8))
        base_slice = ct_array[middle_slice]
        ax.imshow(base_slice, cmap='gray', alpha=0.9)
        im_dose = ax.imshow(dose_array[middle_slice], cmap='hot', alpha=0.5)
        ax.axis('off')
        cbar = plt.colorbar(im_dose, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Dose (normalized)')

        mask_colors = {
            'left_lung': 'red',
            'right_lung': 'blue',
            'aorta': 'yellow',
            'trachea': 'green'
        }
        handles = []
        for organ, color in mask_colors.items():
            if organ in organ_masks:
                mask_slice = sitk.GetArrayFromImage(organ_masks[organ])[middle_slice]
                ax.contour(mask_slice, levels=[0.5], colors=[color], linewidths=1.0)
                handles.append(patches.Patch(color=color, label=organ.replace('_', ' ').title()))
        if handles:
            ax.legend(handles=handles, loc='upper right', fontsize=8)

        title_lines = [f"Lung Preprocessing - {patient_id}"]
        if dose_stats:
            raw_max = dose_stats.get('raw_max')
            norm_max = dose_stats.get('normalized_max')
            scale_info = dose_stats.get('scale_info', {})
            scale_val = scale_info.get('scale')
            method = scale_info.get('method')
            if raw_max is not None and norm_max is not None:
                title_lines.append(f"Dose max: {raw_max:.2f} Gy | Normalized max: {norm_max:.3f}")
            if scale_val is not None and method:
                title_lines.append(f"Normalization: {method} ({scale_val:.2f} Gy)")
        ax.set_title('\n'.join(title_lines))
        
        viz_path = output_dir / f"{patient_id}_lung_preprocessing.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=120)
        plt.close(fig)
        return str(viz_path)
    
    # ------------------------------------------------------------------
    # Patch caching / consolidation
    # ------------------------------------------------------------------
    def _collect_patch_data_for_patient(self, patient_result: Dict[str, Any]) -> Dict[str, Any]:
        patches_data = patient_result.get('patches_data', {})
        patient_id = patient_result.get('patient_id', 'unknown')
        ct_patches = []
        dose_patches = []
        spatial_coords = []
        metadata = []
        ipsi_indices = []
        contra_indices = []
        lobe_mapping = {
            'ipsilateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()},
            'contralateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()}
        }
        
        patch_idx = 0
        for side_type in ['ipsilateral', 'contralateral']:
            if side_type not in patches_data:
                continue
            for lobe_name, items in patches_data[side_type].items():
                for patch in items:
                    ct_patches.append(np.asarray(patch['ct_patch'], dtype=np.float32))
                    dose_patches.append(np.asarray(patch['dose_patch'], dtype=np.float32))
                    spatial_coords.append(np.array(patch['spatial_coord'], dtype=np.int64))
                    entry = {
                        k: self._to_serializable(v)
                        for k, v in patch.items()
                        if k not in {'ct_patch', 'dose_patch'}
                    }
                    entry['patch_id'] = patch_idx
                    entry['patient_id'] = patient_id
                    metadata.append(entry)
                    if side_type == 'ipsilateral':
                        ipsi_indices.append(patch_idx)
                    else:
                        contra_indices.append(patch_idx)
                    lobe_mapping[side_type][lobe_name].append(patch_idx)
                    patch_idx += 1
        if not ct_patches:
            return {}
        return {
            'ct_patches': np.stack(ct_patches, axis=0).astype(np.float32),
            'dose_patches': np.stack(dose_patches, axis=0).astype(np.float32),
            'spatial_coords': np.stack(spatial_coords, axis=0),
            'metadata': metadata,
            'ipsi_indices': ipsi_indices,
            'contra_indices': contra_indices,
            'lobe_mapping': lobe_mapping,
            'patient_summary': patches_data.get('summary', {}),
            'patch_count': len(ct_patches),
            'patient_id': patient_id
        }
    
    def _save_patch_patient_cache(self, patient_result: Dict[str, Any],
                                  split_name: Optional[str]):
        data = self._collect_patch_data_for_patient(patient_result)
        if not data:
            return
        cache_path = self._get_patch_cache_file(split_name, data['patient_id'], ensure_dir=True)
        import h5py
        import json
        with h5py.File(cache_path, 'w') as f:
            f.create_dataset('ct_patches', data=data['ct_patches'], compression='gzip', compression_opts=4)
            f.create_dataset('dose_patches', data=data['dose_patches'], compression='gzip', compression_opts=4)
            f.create_dataset('spatial_coords', data=data['spatial_coords'], compression='gzip', compression_opts=4)
            f.create_dataset('ipsi_indices', data=np.array(data['ipsi_indices'], dtype=np.int64))
            f.create_dataset('contra_indices', data=np.array(data['contra_indices'], dtype=np.int64))
            f.create_dataset('metadata', data=json.dumps(data['metadata'], indent=2).encode('utf-8'))
            f.create_dataset('lobe_mapping', data=json.dumps(data['lobe_mapping'], indent=2).encode('utf-8'))
            f.create_dataset('patient_summary', data=json.dumps(data['patient_summary'], indent=2).encode('utf-8'))
            f.attrs['patient_id'] = data['patient_id']
            f.attrs['patch_count'] = data['patch_count']
    
    def consolidate_patches_from_cache(self, split_name: Optional[str], patient_ids: List[str],
                                       output_path: str):
        import h5py
        import json
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.unlink(missing_ok=True)

        metadata_all: List[Dict[str, Any]] = []
        ipsi_indices: List[int] = []
        contra_indices: List[int] = []
        lobe_mapping = {
            'ipsilateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()},
            'contralateral': {lobe: [] for lobe in STANDARD_LOBE_NAMES.values()}
        }
        patient_summary: Dict[str, Any] = {}
        corrupted_patients = []

        ct_ds = dose_ds = coords_ds = None
        patch_shape = None
        coord_width = None
        offset = 0
        file_handle = None

        try:
            for pid in patient_ids:
                cache_path = self._get_patch_cache_file(split_name, pid, ensure_dir=False)
                if not cache_path.exists():
                    continue
                try:
                    with h5py.File(cache_path, 'r') as f_cache:
                        ct = f_cache['ct_patches'][()]
                        dose = f_cache['dose_patches'][()]
                        coords = f_cache['spatial_coords'][()]
                        ipsi_local = f_cache['ipsi_indices'][()].astype(np.int64).tolist()
                        contra_local = f_cache['contra_indices'][()].astype(np.int64).tolist()
                        metadata_raw = f_cache['metadata'][()]
                        metadata_local = json.loads(self._decode_hdf5_json_blob(metadata_raw))
                        lobe_local_raw = f_cache['lobe_mapping'][()]
                        lobe_local = json.loads(self._decode_hdf5_json_blob(lobe_local_raw))
                        summary_raw = f_cache['patient_summary'][()]
                        summary_local = json.loads(self._decode_hdf5_json_blob(summary_raw))
                except (OSError, KeyError, json.JSONDecodeError) as exc:
                    self.logger.error(
                        "Corrupted or incompatible patch cache detected for %s at %s: %s",
                        pid, cache_path, exc
                    )
                    corrupted_patients.append((pid, str(cache_path), str(exc)))
                    continue

                if ct.size == 0:
                    continue

                if file_handle is None:
                    file_handle = h5py.File(output_path, 'w')
                    patch_shape = ct.shape[1:]
                    coord_width = coords.shape[1]
                    chunk_len = max(1, min(64, ct.shape[0]))
                    ct_ds = file_handle.create_dataset(
                        'ct_patches',
                        shape=(0,) + patch_shape,
                        maxshape=(None,) + patch_shape,
                        chunks=(chunk_len,) + patch_shape,
                        dtype=ct.dtype,
                        compression='gzip',
                        compression_opts=4
                    )
                    dose_ds = file_handle.create_dataset(
                        'dose_patches',
                        shape=(0,) + patch_shape,
                        maxshape=(None,) + patch_shape,
                        chunks=(chunk_len,) + patch_shape,
                        dtype=dose.dtype,
                        compression='gzip',
                        compression_opts=4
                    )
                    coords_ds = file_handle.create_dataset(
                        'spatial_coords',
                        shape=(0, coord_width),
                        maxshape=(None, coord_width),
                        chunks=(max(1, min(2048, coords.shape[0])), coord_width),
                        dtype=coords.dtype,
                        compression='gzip',
                        compression_opts=4
                    )

                new_offset = offset + ct.shape[0]
                ct_ds.resize((new_offset,) + patch_shape)
                ct_ds[offset:new_offset] = ct
                dose_ds.resize((new_offset,) + patch_shape)
                dose_ds[offset:new_offset] = dose
                coords_ds.resize((new_offset, coord_width))
                coords_ds[offset:new_offset] = coords

                for meta in metadata_local:
                    meta['patch_id'] = meta['patch_id'] + offset
                    metadata_all.append(meta)
                ipsi_indices.extend([idx + offset for idx in ipsi_local])
                contra_indices.extend([idx + offset for idx in contra_local])
                for side in ['ipsilateral', 'contralateral']:
                    for lobe, indices in lobe_local.get(side, {}).items():
                        lobe_mapping[side][lobe].extend([idx + offset for idx in indices])
                patient_summary[pid] = summary_local
                offset = new_offset

            if file_handle is None:
                if corrupted_patients:
                    issues = "; ".join([f"{pid} ({path})" for pid, path, _ in corrupted_patients])
                    raise RuntimeError(
                        "Unable to consolidate cached patches because the following cache files "
                        f"are corrupted or outdated: {issues}. "
                        "Please delete these files and rerun preprocessing for those patients."
                    )
                self.logger.warning("No cached patch data available for consolidation")
                return

            file_handle.create_dataset('ipsi_indices', data=np.array(ipsi_indices, dtype=np.int64))
            file_handle.create_dataset('contra_indices', data=np.array(contra_indices, dtype=np.int64))
            file_handle.create_dataset(
                'patch_metadata',
                data=json.dumps(metadata_all, indent=2).encode('utf-8')
            )
            file_handle.create_dataset(
                'lobe_mapping',
                data=json.dumps(lobe_mapping, indent=2).encode('utf-8')
            )
            file_handle.create_dataset(
                'patient_summary',
                data=json.dumps(patient_summary, indent=2).encode('utf-8')
            )
            file_handle.attrs['total_patches'] = offset
            file_handle.attrs['total_ipsi_patches'] = len(ipsi_indices)
            file_handle.attrs['total_contra_patches'] = len(contra_indices)
            file_handle.attrs['n_patients'] = len(patient_summary)
        finally:
            if file_handle is not None:
                file_handle.close()
    
    def consolidate_patches_to_hdf5(self, patient_results: List[Dict[str, Any]],
                                    output_path: str, split_name: Optional[str] = None,
                                    all_patient_ids: Optional[List[str]] = None):
        ids: List[str] = list(all_patient_ids) if all_patient_ids else []
        for result in patient_results:
            if not result:
                continue
            if result.get('patches_data'):
                self._save_patch_patient_cache(result, split_name)
            pid = result.get('patient_id')
            if pid:
                ids.append(pid)
        unique_ids = list(dict.fromkeys(ids))
        if unique_ids:
            self.consolidate_patches_from_cache(split_name, unique_ids, output_path)
    
    def consolidate_images_to_hdf5(self, patient_results: List[Dict[str, Any]], output_path: str):
        import h5py
        from datetime import datetime
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'a') as f:
            if 'created_at' not in f.attrs:
                f.attrs['created_at'] = datetime.utcnow().isoformat()
            f.attrs['dataset_type'] = self.config.get('dataset', {}).get('dataset_type', 'full_images')
            f.attrs['voxel_spacing'] = self.target_spacing
            for result in patient_results:
                patient_id = result['patient_id']
                if patient_id in f:
                    del f[patient_id]
                group = f.create_group(patient_id)
                ct_array = sitk.GetArrayFromImage(result['ct_image']).astype(np.float32)
                dose_array = sitk.GetArrayFromImage(result['dose_image']).astype(np.float32)
                ct_ds = group.create_dataset('ct', data=ct_array, compression='gzip', compression_opts=4)
                ct_ds.attrs['spacing'] = result['ct_image'].GetSpacing()
                ct_ds.attrs['origin'] = result['ct_image'].GetOrigin()
                ct_ds.attrs['direction'] = result['ct_image'].GetDirection()
                dose_ds = group.create_dataset('dose', data=dose_array, compression='gzip', compression_opts=4)
                dose_ds.attrs['spacing'] = result['dose_image'].GetSpacing()
                dose_ds.attrs['origin'] = result['dose_image'].GetOrigin()
                dose_ds.attrs['direction'] = result['dose_image'].GetDirection()
    
    def consolidate_images_from_disk(self, split_name: Optional[str], patient_ids: List[str],
                                     output_path: str):
        import h5py
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, 'w') as f:
            count = 0
            for pid in patient_ids:
                patient_dir = self._get_image_patient_dir(split_name, pid, ensure=False)
                ct_path = patient_dir / f"{pid}_ct_processed.nrrd"
                dose_path = patient_dir / f"{pid}_dose_processed.nrrd"
                if not ct_path.exists() or not dose_path.exists():
                    continue
                ct_image = sitk.ReadImage(str(ct_path))
                dose_image = sitk.ReadImage(str(dose_path))
                ct_array = sitk.GetArrayFromImage(ct_image).astype(np.float32)
                dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
                grp = f.create_group(pid)
                grp.create_dataset('ct', data=ct_array, compression='gzip', compression_opts=4)
                grp.create_dataset('dose', data=dose_array, compression='gzip', compression_opts=4)
                grp['ct'].attrs['spacing'] = ct_image.GetSpacing()
                grp['ct'].attrs['origin'] = ct_image.GetOrigin()
                grp['ct'].attrs['direction'] = ct_image.GetDirection()
                grp['dose'].attrs['spacing'] = dose_image.GetSpacing()
                grp['dose'].attrs['origin'] = dose_image.GetOrigin()
                grp['dose'].attrs['direction'] = dose_image.GetDirection()
                count += 1
            f.attrs['n_patients'] = count
    
    # ------------------------------------------------------------------
    # Main patient processing entry point
    # ------------------------------------------------------------------
    def process_patient(self, patient_id: str,
                       ct_path: str,
                       dose_path: str,
                       prescribed_dose: Optional[float] = None,
                       split_name: Optional[str] = None,
                       experiment_type: Optional[str] = None) -> Dict[str, Any]:
        self.logger.info(f"Processing patient {patient_id}")
        
        # Force garbage collection at start
        import gc
        gc.collect()
        
        try:
            ct_image = sitk.ReadImage(ct_path)
            dose_image = sitk.ReadImage(dose_path)
            
            # Log image sizes for debugging
            ct_size = ct_image.GetSize()
            dose_size = dose_image.GetSize()
            self.logger.info(f"CT image size: {ct_size}, Dose image size: {dose_size}")
            
            dose_image = self.check_and_resample_dose_to_ct_space(ct_image, dose_image)
            
            # Force garbage collection before segmentation
            gc.collect()
            
            lung_masks = self.segment_organs(ct_image)
            
            # Force garbage collection after segmentation
            gc.collect()
            
            ct_image = self.resample_to_common_spacing(ct_image, self.target_spacing)
            dose_image = self.resample_to_common_spacing(dose_image, self.target_spacing)
            
            resampled_masks = {}
            for name, mask in lung_masks.items():
                resampled_masks[name] = self.resample_to_common_spacing(mask, self.target_spacing)
            lung_masks = resampled_masks
            
            ct_image = self.apply_ct_preprocessing(ct_image)
            raw_dose_array = sitk.GetArrayFromImage(dose_image).astype(np.float32)
            dose_image, scale_info = self.normalize_dose(dose_image, prescribed_dose)
            norm_dose_array = sitk.GetArrayFromImage(dose_image)
            dose_stats = {
                'raw_max': float(np.max(raw_dose_array)),
                'normalized_max': float(np.max(norm_dose_array)),
                'scale_info': scale_info
            }
            
            if 'left_lung' in lung_masks and 'right_lung' in lung_masks:
                ipsi_lung, contra_lung = self._determine_ipsi_contra_lungs(
                    lung_masks['left_lung'], lung_masks['right_lung'], dose_image
                )
                lung_masks['ipsi_lung'] = ipsi_lung
                lung_masks['contra_lung'] = contra_lung
            
            mode = (experiment_type or self.config.get('dataset', {}).get('dataset_type', 'patches')).lower()
            produce_patches = mode in ['patch', 'patches', 'patches_only']
            produce_images = mode in ['image', 'full_images']
            
            patches_data = self.extract_patches(ct_image, dose_image, lung_masks, patient_id) if produce_patches else {}
            ct_for_patch_visuals = ct_image
            
            if produce_images:
                resize_to = self.preproc_config.get('resize_to')
                if resize_to:
                    self.logger.info(f"🔄 Full image experiments: resizing to standardized size: {resize_to}")
                    ct_image = self.resize_image(ct_image, resize_to)
                    dose_image = self.resize_image(dose_image, resize_to)
                    resized_masks = {}
                    for name, mask in lung_masks.items():
                        resized_masks[name] = self.resize_image(mask, resize_to)
                    lung_masks = resized_masks
            
            image_split_dir = self._get_split_dir(split_name, kind='image', ensure=True)
            patch_split_dir = self._get_split_dir(split_name, kind='patch', ensure=produce_patches)
            
            viz_path = self.create_visualization(
                ct_image, dose_image, lung_masks,
                patient_id, patches_data if produce_patches else {},
                output_dir=image_split_dir,
                dose_stats=dose_stats
            )
            
            results: Dict[str, Any] = {
                'patient_id': patient_id,
                'ct_image': ct_image,
                'dose_image': dose_image,
                'lung_masks': lung_masks,
                'patches_data': patches_data if produce_patches else {},
                'visualization_path': viz_path,
                'prescribed_dose': prescribed_dose
            }
            
            if produce_patches:
                self._save_patch_patient_cache(results, split_name)
                patient_viz_dir = patch_split_dir / patient_id
                patient_viz_dir.mkdir(parents=True, exist_ok=True)
                self.save_visualization_patches(patches_data, patient_id, patient_viz_dir, ct_for_patch_visuals)
            
            if produce_images:
                patient_dir = self._get_image_patient_dir(split_name, patient_id, ensure=True)
                sitk.WriteImage(ct_image, str(patient_dir / f"{patient_id}_ct_processed.nrrd"))
                sitk.WriteImage(dose_image, str(patient_dir / f"{patient_id}_dose_processed.nrrd"))
                for organ_name, mask in lung_masks.items():
                    sitk.WriteImage(mask, str(patient_dir / f"{patient_id}_{organ_name}_mask.nrrd"))
                results['output_dir'] = str(patient_dir)
            
            # Final garbage collection
            gc.collect()
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing patient {patient_id}: {str(e)}")
            # Force cleanup on error
            gc.collect()
            raise
