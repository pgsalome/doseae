"""
HNC (Head and Neck Cancer) specific preprocessor for dose autoencoder training.
Handles HNC segmentation, dose normalization, and patch extraction.
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

from entities.base.base_preprocessor import BasePreprocessor


class HNCPreprocessor(BasePreprocessor):
    """
    HNC-specific preprocessor for dose autoencoder training.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize HNC preprocessor.
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for processed data
        """
        super().__init__(config, output_dir)
        
        # HNC-specific configuration
        self.hnc_config = config.get('hnc', {})
        self.patch_size = self.hnc_config.get('patch_size', 32)  # Larger patches for HNC
        self.overlap_percentage = self.hnc_config.get('overlap_percentage', 25)
        self.dose_threshold = self.hnc_config.get('dose_threshold', 0.2)
        
        # HNC-specific organs
        self.target_organs = self.hnc_config.get('target_organs', [
            'parotid_left', 'parotid_right', 'submandibular_left', 'submandibular_right',
            'oral_cavity', 'larynx', 'pharynx', 'spinal_cord', 'brainstem'
        ])
        
        # Preprocessing configuration
        self.preproc_config = config.get('preprocessing', {})
        self.target_spacing = tuple(self.preproc_config.get('voxel_spacing', [0.5, 0.5, 0.5]))
        self.ct_preprocessing = self.preproc_config.get('ct_preprocessing', {})
        self.dose_preprocessing = self.preproc_config.get('dose_preprocessing', {})
    
    def segment_organs(self, ct_image: sitk.Image) -> Dict[str, sitk.Image]:
        """
        Segment HNC organs from CT image.
        This is a placeholder implementation - in practice, you would use
        HNC-specific segmentation tools or pre-segmented data.
        
        Args:
            ct_image: CT image to segment
            
        Returns:
            Dictionary containing organ masks
        """
        try:
            # For now, create dummy masks based on anatomical regions
            # In practice, you would use HNC-specific segmentation tools
            organ_masks = self._create_dummy_hnc_masks(ct_image)
            
            self.logger.info(f"Successfully segmented {len(organ_masks)} HNC organs")
            return organ_masks
            
        except Exception as e:
            self.logger.error(f"HNC organ segmentation failed: {e}")
            raise
    
    def _create_dummy_hnc_masks(self, ct_image: sitk.Image) -> Dict[str, sitk.Image]:
        """
        Create dummy HNC organ masks for demonstration.
        In practice, replace with actual HNC segmentation.
        
        Args:
            ct_image: CT image
            
        Returns:
            Dictionary containing dummy organ masks
        """
        array = sitk.GetArrayFromImage(ct_image)
        organ_masks = {}
        
        # Create dummy masks based on anatomical regions
        for i, organ_name in enumerate(self.target_organs):
            mask_array = np.zeros_like(array, dtype=np.uint8)
            
            # Create different regions for different organs
            if 'parotid' in organ_name:
                # Parotid glands - lateral regions
                if 'left' in organ_name:
                    mask_array[:, :, :array.shape[2]//3] = 1
                else:
                    mask_array[:, :, 2*array.shape[2]//3:] = 1
            elif 'submandibular' in organ_name:
                # Submandibular glands - anterior regions
                if 'left' in organ_name:
                    mask_array[:, :array.shape[1]//2, :array.shape[2]//2] = 1
                else:
                    mask_array[:, :array.shape[1]//2, array.shape[2]//2:] = 1
            elif 'oral_cavity' in organ_name:
                # Oral cavity - central region
                mask_array[:, array.shape[1]//3:2*array.shape[1]//3, 
                          array.shape[2]//3:2*array.shape[2]//3] = 1
            elif 'larynx' in organ_name:
                # Larynx - central anterior region
                mask_array[:, 2*array.shape[1]//3:, array.shape[2]//3:2*array.shape[2]//3] = 1
            elif 'pharynx' in organ_name:
                # Pharynx - central region
                mask_array[:, array.shape[1]//3:2*array.shape[1]//3, 
                          array.shape[2]//3:2*array.shape[2]//3] = 1
            elif 'spinal_cord' in organ_name:
                # Spinal cord - central posterior region
                mask_array[:, :array.shape[1]//3, array.shape[2]//3:2*array.shape[2]//3] = 1
            elif 'brainstem' in organ_name:
                # Brainstem - central superior region
                mask_array[:array.shape[0]//3, array.shape[1]//3:2*array.shape[1]//3, 
                          array.shape[2]//3:2*array.shape[2]//3] = 1
            
            # Create mask image
            mask_image = sitk.GetImageFromArray(mask_array)
            mask_image.CopyInformation(ct_image)
            organ_masks[organ_name] = mask_image
        
        return organ_masks
    
    def normalize_dose(self, dose_image: sitk.Image, 
                      prescribed_dose: Optional[float] = None) -> sitk.Image:
        """
        Normalize dose distribution for HNC.
        
        Args:
            dose_image: Dose distribution image
            prescribed_dose: Prescribed dose value (if available)
            
        Returns:
            Normalized dose image
        """
        array = sitk.GetArrayFromImage(dose_image)
        
        if prescribed_dose is not None:
            # Normalize to prescribed dose
            normalized_array = array / prescribed_dose
            self.logger.info(f"Normalized dose to prescribed dose: {prescribed_dose} Gy")
        else:
            # For HNC, use 95th percentile as fallback (higher dose regions)
            p95 = np.percentile(array, 95)
            normalized_array = array / p95
            self.logger.info(f"Normalized dose to 95th percentile: {p95:.2f} Gy")
        
        # Clip to [0, 1] range
        normalized_array = np.clip(normalized_array, 0, 1)
        
        normalized_image = sitk.GetImageFromArray(normalized_array)
        normalized_image.CopyInformation(dose_image)
        
        return normalized_image
    
    def extract_patches(self, ct_image: sitk.Image, 
                       dose_image: sitk.Image,
                       organ_masks: Dict[str, sitk.Image]) -> Dict[str, Any]:
        """
        Extract patches from HNC regions.
        
        Args:
            ct_image: CT image
            dose_image: Dose image
            organ_masks: Dictionary containing organ masks
            
        Returns:
            Dictionary containing extracted patches and metadata
        """
        patches_data = {}
        
        for organ_name, organ_mask in organ_masks.items():
            self.logger.info(f"Extracting patches for {organ_name}")
            
            # Apply mask to images
            masked_ct = self.apply_mask(ct_image, organ_mask)
            masked_dose = self.apply_mask(dose_image, organ_mask)
            
            # Extract patches
            ct_patches, dose_patches, patch_locations = self._extract_patches_from_organ(
                masked_ct, masked_dose, organ_mask, organ_name
            )
            
            patches_data[organ_name] = {
                'ct_patches': ct_patches,
                'dose_patches': dose_patches,
                'patch_locations': patch_locations,
                'organ_mask': organ_mask,
                'masked_ct': masked_ct,
                'masked_dose': masked_dose
            }
        
        return patches_data
    
    def _extract_patches_from_organ(self, ct_image: sitk.Image, 
                                   dose_image: sitk.Image,
                                   organ_mask: sitk.Image,
                                   organ_name: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Extract patches from a specific HNC organ.
        
        Args:
            ct_image: Masked CT image
            dose_image: Masked dose image
            organ_mask: Organ mask
            organ_name: Name of the organ
            
        Returns:
            Tuple of (ct_patches, dose_patches, patch_locations)
        """
        ct_array = sitk.GetArrayFromImage(ct_image)
        dose_array = sitk.GetArrayFromImage(dose_image)
        mask_array = sitk.GetArrayFromImage(organ_mask)
        
        # Get organ bounding box
        coords = np.where(mask_array > 0)
        if len(coords[0]) == 0:
            self.logger.warning(f"No voxels found in {organ_name} mask")
            return [], [], []
        
        min_coords = [np.min(coords[i]) for i in range(3)]
        max_coords = [np.min([np.max(coords[i]) + 1, ct_array.shape[i]]) for i in range(3)]
        
        # Calculate patch grid
        patch_size = self.patch_size
        overlap = int(patch_size * self.overlap_percentage / 100)
        step_size = patch_size - overlap
        
        ct_patches = []
        dose_patches = []
        patch_locations = []
        
        for z in range(min_coords[0], max_coords[0] - patch_size + 1, step_size):
            for y in range(min_coords[1], max_coords[1] - patch_size + 1, step_size):
                for x in range(min_coords[2], max_coords[2] - patch_size + 1, step_size):
                    # Extract patch
                    ct_patch = ct_array[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    dose_patch = dose_array[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    
                    # Check if patch contains sufficient organ tissue
                    patch_mask = mask_array[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    organ_ratio = np.sum(patch_mask > 0) / (patch_size ** 3)
                    
                    if organ_ratio > 0.05:  # At least 5% organ tissue (lower threshold for HNC)
                        ct_patches.append(ct_patch)
                        dose_patches.append(dose_patch)
                        patch_locations.append({
                            'coordinates': (z, y, x),
                            'organ_ratio': organ_ratio,
                            'organ_name': organ_name
                        })
        
        self.logger.info(f"Extracted {len(ct_patches)} patches for {organ_name}")
        return ct_patches, dose_patches, patch_locations
    
    def apply_ct_preprocessing(self, ct_image: sitk.Image) -> sitk.Image:
        """
        Apply CT-specific preprocessing for HNC.
        
        Args:
            ct_image: Input CT image
            
        Returns:
            Preprocessed CT image
        """
        if self.ct_preprocessing.get('cOOpD_style', False):
            # Apply cOOpD-style HU windowing
            return self._apply_cOOpD_style_preprocessing(ct_image)
        else:
            # Apply standard normalization
            return self.normalize_image(ct_image, method='percentile', percentile=99.0)
    
    def _apply_cOOpD_style_preprocessing(self, ct_image: sitk.Image) -> sitk.Image:
        """
        Apply cOOpD-style HU windowing (-1000 to 400 HU).
        
        Args:
            ct_image: Input CT image
            
        Returns:
            Preprocessed CT image
        """
        array = sitk.GetArrayFromImage(ct_image)
        
        # Apply HU windowing
        array = np.clip(array, -1000, 400)
        array = (array + 1000) / 1400  # Normalize to [0, 1]
        
        processed_image = sitk.GetImageFromArray(array)
        processed_image.CopyInformation(ct_image)
        
        return processed_image
    
    def create_visualization(self, ct_image: sitk.Image,
                           dose_image: sitk.Image,
                           organ_masks: Dict[str, sitk.Image],
                           patient_id: str) -> str:
        """
        Create visualization of processed HNC data.
        
        Args:
            ct_image: CT image
            dose_image: Dose image
            organ_masks: Dictionary of organ masks
            patient_id: Patient identifier
            
        Returns:
            Path to saved visualization
        """
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle(f'HNC Preprocessing Results - Patient {patient_id}', fontsize=16)
        
        # Get middle slice
        ct_array = sitk.GetArrayFromImage(ct_image)
        dose_array = sitk.GetArrayFromImage(dose_image)
        middle_slice = ct_array.shape[0] // 2
        
        # CT slices
        axes[0, 0].imshow(ct_array[middle_slice], cmap='gray')
        axes[0, 0].set_title('CT - Axial')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(ct_array[:, middle_slice, :], cmap='gray')
        axes[0, 1].set_title('CT - Coronal')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(ct_array[:, :, middle_slice], cmap='gray')
        axes[0, 2].set_title('CT - Sagittal')
        axes[0, 2].axis('off')
        
        # Dose slices
        axes[1, 0].imshow(dose_array[middle_slice], cmap='hot', alpha=0.7)
        axes[1, 0].set_title('Dose - Axial')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(dose_array[:, middle_slice, :], cmap='hot', alpha=0.7)
        axes[1, 1].set_title('Dose - Coronal')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(dose_array[:, :, middle_slice], cmap='hot', alpha=0.7)
        axes[1, 2].set_title('Dose - Sagittal')
        axes[1, 2].axis('off')
        
        # Organ masks (show first 3 organs)
        organ_names = list(organ_masks.keys())[:3]
        colors = ['Reds', 'Blues', 'Greens']
        
        for i, (organ_name, color) in enumerate(zip(organ_names, colors)):
            if i < 3:
                organ_mask = sitk.GetArrayFromImage(organ_masks[organ_name])
                axes[2, i].imshow(organ_mask[middle_slice], cmap=color, alpha=0.7)
                axes[2, i].set_title(f'{organ_name.replace("_", " ").title()}')
                axes[2, i].axis('off')
        
        # Hide unused subplots
        for i in range(len(organ_names), 3):
            axes[2, i].axis('off')
        
        # Save visualization
        viz_path = self.output_dir / f"{patient_id}_hnc_preprocessing.png"
        plt.tight_layout()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(viz_path)
    
    def process_patient(self, patient_id: str, 
                       ct_path: str, 
                       dose_path: str,
                       prescribed_dose: Optional[float] = None) -> Dict[str, Any]:
        """
        Process a single HNC patient's data.
        
        Args:
            patient_id: Patient identifier
            ct_path: Path to CT file
            dose_path: Path to dose file
            prescribed_dose: Prescribed dose value (if available)
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing HNC patient {patient_id}")
        
        try:
            # Load images
            ct_image = sitk.ReadImage(ct_path)
            dose_image = sitk.ReadImage(dose_path)
            
            # Log original shapes
            self.logger.info(f"📏 Original CT shape: {ct_image.GetSize()}, spacing: {ct_image.GetSpacing()}")
            self.logger.info(f"📏 Original dose shape: {dose_image.GetSize()}, spacing: {dose_image.GetSpacing()}")
            
            # Check and resample dose to CT space if needed (before any resampling)
            dose_image = self.check_and_resample_dose_to_ct_space(ct_image, dose_image)
            
            # Segment HNC organs on original resolution (faster)
            self.logger.info("🫀 Running HNC organ segmentation on original resolution...")
            organ_masks = self.segment_organs(ct_image)
            
            # Resample both to common spacing
            ct_image = self.resample_to_common_spacing(ct_image, self.target_spacing)
            dose_image = self.resample_to_common_spacing(dose_image, self.target_spacing)
            
            # Resample organ masks to match final resolution
            self.logger.info("🔄 Resampling organ masks to final resolution...")
            resampled_organ_masks = {}
            for organ_name, mask in organ_masks.items():
                resampled_organ_masks[organ_name] = self.resample_to_common_spacing(mask, self.target_spacing)
            organ_masks = resampled_organ_masks
            
            # Resize to standardized size if specified
            resize_to = self.preproc_config.get('resize_to')
            if resize_to:
                self.logger.info(f"🔄 Resizing images to standardized size: {resize_to}")
                ct_image = self.resize_image(ct_image, resize_to)
                dose_image = self.resize_image(dose_image, resize_to)
                
                # Resize masks to match
                resized_organ_masks = {}
                for organ_name, mask in organ_masks.items():
                    resized_organ_masks[organ_name] = self.resize_image(mask, resize_to)
                organ_masks = resized_organ_masks
            
            # Apply CT preprocessing
            ct_image = self.apply_ct_preprocessing(ct_image)
            
            # Normalize dose
            dose_image = self.normalize_dose(dose_image, prescribed_dose)
            
            # Extract patches only for patch-based experiments
            experiment_type = self.config.get('dataset', {}).get('dataset_type', 'patches')
            if experiment_type == 'patches':
                patches_data = self.extract_patches(ct_image, dose_image, organ_masks)
            else:
                # For full_images experiments, skip patch extraction
                patches_data = {}
            
            # Create visualization
            viz_path = self.create_visualization(ct_image, dose_image, organ_masks, patient_id)
            
            # Save processed data
            patient_dir = self.output_dir / patient_id
            patient_dir.mkdir(parents=True, exist_ok=True)
            
            # Save images
            sitk.WriteImage(ct_image, str(patient_dir / f"{patient_id}_ct_processed.nrrd"))
            sitk.WriteImage(dose_image, str(patient_dir / f"{patient_id}_dose_processed.nrrd"))
            
            # Save masks
            for organ_name, mask in organ_masks.items():
                sitk.WriteImage(mask, str(patient_dir / f"{patient_id}_{organ_name}_mask.nrrd"))
            
            # Save patches
            with open(patient_dir / f"{patient_id}_patches.pkl", 'wb') as f:
                pickle.dump(patches_data, f)
            
            self.logger.info(f"Successfully processed HNC patient {patient_id}")
            
            return {
                'patient_id': patient_id,
                'ct_image': ct_image,
                'dose_image': dose_image,
                'organ_masks': organ_masks,
                'patches_data': patches_data,
                'visualization_path': viz_path,
                'output_dir': str(patient_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process HNC patient {patient_id}: {e}")
            raise
