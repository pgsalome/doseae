"""
HNC (Head and Neck Cancer) specific dataset for dose autoencoder training.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random

from entities.base.base_dataset import BaseDataset


class HNCDataset(BaseDataset):
    """
    Dataset for HNC dose autoencoder training.
    """
    
    def __init__(self, 
                 data_dir: str,
                 config: Dict[str, Any],
                 split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize HNC dataset.
        
        Args:
            data_dir: Directory containing processed HNC data
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformations
        """
        super().__init__(data_dir, config, split, transform)
        
        # HNC-specific configuration
        self.hnc_config = config.get('hnc', {})
        self.patch_size = self.hnc_config.get('patch_size', 32)
        self.use_attention = self.hnc_config.get('use_attention', True)
        self.attention_type = self.hnc_config.get('attention_type', 'high_dose')
        
        # HNC-specific organs
        self.target_organs = self.hnc_config.get('target_organs', [
            'parotid_left', 'parotid_right', 'submandibular_left', 'submandibular_right',
            'oral_cavity', 'larynx', 'pharynx', 'spinal_cord', 'brainstem'
        ])
        
        # Data configuration
        self.data_config = config.get('data', {})
        self.batch_size = self.data_config.get('batch_size', 1)
        self.num_patches_per_sample = self.data_config.get('num_patches_per_sample', 15)
        
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """
        Load paths to HNC data files for this split.
        
        Returns:
            List of dictionaries containing file paths for each sample
        """
        data_paths = []
        
        # Get all patient directories
        patient_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        
        # Split patients based on split
        random.seed(42)  # For reproducible splits
        patient_dirs = sorted(patient_dirs)
        random.shuffle(patient_dirs)
        
        n_patients = len(patient_dirs)
        if self.split == 'train':
            patient_dirs = patient_dirs[:int(0.7 * n_patients)]
        elif self.split == 'val':
            patient_dirs = patient_dirs[int(0.7 * n_patients):int(0.85 * n_patients)]
        else:  # test
            patient_dirs = patient_dirs[int(0.85 * n_patients):]
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            # Check if required files exist
            ct_path = patient_dir / f"{patient_id}_ct_processed.nrrd"
            dose_path = patient_dir / f"{patient_id}_dose_processed.nrrd"
            patches_path = patient_dir / f"{patient_id}_patches.pkl"
            
            if ct_path.exists() and dose_path.exists() and patches_path.exists():
                # Get organ mask paths
                organ_mask_paths = {}
                for organ_name in self.target_organs:
                    mask_path = patient_dir / f"{patient_id}_{organ_name}_mask.nrrd"
                    if mask_path.exists():
                        organ_mask_paths[organ_name] = str(mask_path)
                
                sample_paths = {
                    'patient_id': patient_id,
                    'ct_path': str(ct_path),
                    'dose_path': str(dose_path),
                    'patches_path': str(patches_path),
                    'organ_mask_paths': organ_mask_paths
                }
                
                data_paths.append(sample_paths)
            else:
                self.logger.warning(f"Missing files for HNC patient {patient_id}")
        
        self.logger.info(f"Loaded {len(data_paths)} HNC samples for {self.split} split")
        return data_paths
    
    def _load_sample(self, sample_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load a single HNC sample from file paths.
        
        Args:
            sample_paths: Dictionary mapping data keys to file paths
            
        Returns:
            Dictionary containing loaded data as tensors
        """
        # Load images
        ct_tensor = self.load_image(sample_paths['ct_path'])
        dose_tensor = self.load_image(sample_paths['dose_path'])
        
        # Load organ masks
        organ_masks = {}
        for organ_name, mask_path in sample_paths['organ_mask_paths'].items():
            organ_masks[organ_name] = self.load_image(mask_path)
        
        # Load patches
        patches_data = self._load_patches_data(sample_paths['patches_path'])
        
        return {
            'patient_id': sample_paths['patient_id'],
            'ct': ct_tensor,
            'dose': dose_tensor,
            'organ_masks': organ_masks,
            'patches_data': patches_data
        }
    
    def _load_patches_data(self, patches_path: str) -> Dict[str, Any]:
        """
        Load patches data from pickle file.
        
        Args:
            patches_path: Path to patches pickle file
            
        Returns:
            Dictionary containing patches data
        """
        with open(patches_path, 'rb') as f:
            patches_data = pickle.load(f)
        
        return patches_data
    
    def _apply_entity_specific_processing(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply HNC-specific processing to the sample.
        
        Args:
            sample: Dictionary containing loaded data
            
        Returns:
            Dictionary containing processed data
        """
        # Extract patches for training
        if self.split == 'train':
            # For training, extract random patches
            patches = self._extract_random_patches(sample)
            sample['patches'] = patches
        else:
            # For validation/test, use all patches
            patches = self._extract_all_patches(sample)
            sample['patches'] = patches
        
        # Create attention masks if needed
        if self.use_attention:
            attention_masks = self._create_attention_masks(sample)
            sample['attention_masks'] = attention_masks
        
        return sample
    
    def _extract_random_patches(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Extract random patches for training.
        
        Args:
            sample: Sample data
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        patches_data = sample['patches_data']
        
        # Collect all available patches
        all_patches = []
        for organ_name, organ_data in patches_data.items():
            ct_patches = organ_data['ct_patches']
            dose_patches = organ_data['dose_patches']
            patch_locations = organ_data['patch_locations']
            
            for i, (ct_patch, dose_patch, location) in enumerate(zip(ct_patches, dose_patches, patch_locations)):
                all_patches.append({
                    'ct_patch': torch.from_numpy(ct_patch).float().unsqueeze(0),
                    'dose_patch': torch.from_numpy(dose_patch).float().unsqueeze(0),
                    'location': location,
                    'organ_name': organ_name
                })
        
        # Sample random patches
        if len(all_patches) > self.num_patches_per_sample:
            patches = random.sample(all_patches, self.num_patches_per_sample)
        else:
            patches = all_patches
        
        return patches
    
    def _extract_all_patches(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Extract all patches for validation/test.
        
        Args:
            sample: Sample data
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        patches_data = sample['patches_data']
        
        for organ_name, organ_data in patches_data.items():
            ct_patches = organ_data['ct_patches']
            dose_patches = organ_data['dose_patches']
            patch_locations = organ_data['patch_locations']
            
            for i, (ct_patch, dose_patch, location) in enumerate(zip(ct_patches, dose_patches, patch_locations)):
                patches.append({
                    'ct_patch': torch.from_numpy(ct_patch).float().unsqueeze(0),
                    'dose_patch': torch.from_numpy(dose_patch).float().unsqueeze(0),
                    'location': location,
                    'organ_name': organ_name
                })
        
        return patches
    
    def _create_attention_masks(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Create attention masks for HNC regions.
        
        Args:
            sample: Sample data
            
        Returns:
            Dictionary containing attention masks
        """
        attention_masks = {}
        dose_tensor = sample['dose']
        organ_masks = sample['organ_masks']
        
        for organ_name, organ_mask in organ_masks.items():
            attention_mask = self.create_attention_mask(
                dose_tensor, organ_mask, self.attention_type
            )
            attention_masks[organ_name] = attention_mask
        
        return attention_masks
    
    def get_high_dose_patches(self, sample: Dict[str, torch.Tensor], 
                             threshold: float = 0.5) -> List[Dict[str, torch.Tensor]]:
        """
        Get patches with high dose values.
        
        Args:
            sample: Sample data
            threshold: Dose threshold for high dose patches
            
        Returns:
            List of high dose patches
        """
        high_dose_patches = []
        patches = sample['patches']
        
        for patch in patches:
            dose_patch = patch['dose_patch']
            max_dose = torch.max(dose_patch).item()
            
            if max_dose > threshold:
                high_dose_patches.append(patch)
        
        return high_dose_patches
    
    def get_organ_statistics(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each organ in the sample.
        
        Args:
            sample: Sample data
            
        Returns:
            Dictionary containing organ statistics
        """
        statistics = {}
        patches_data = sample['patches_data']
        
        for organ_name, organ_data in patches_data.items():
            ct_patches = organ_data['ct_patches']
            dose_patches = organ_data['dose_patches']
            
            # Calculate statistics
            all_ct_values = np.concatenate([patch.flatten() for patch in ct_patches])
            all_dose_values = np.concatenate([patch.flatten() for patch in dose_patches])
            
            statistics[organ_name] = {
                'ct_mean': float(np.mean(all_ct_values)),
                'ct_std': float(np.std(all_ct_values)),
                'ct_min': float(np.min(all_ct_values)),
                'ct_max': float(np.max(all_ct_values)),
                'dose_mean': float(np.mean(all_dose_values)),
                'dose_std': float(np.std(all_dose_values)),
                'dose_min': float(np.min(all_dose_values)),
                'dose_max': float(np.max(all_dose_values)),
                'num_patches': len(ct_patches)
            }
        
        return statistics
    
    def get_organ_dose_statistics(self, sample: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Get dose statistics for each organ (useful for HNC dose constraints).
        
        Args:
            sample: Sample data
            
        Returns:
            Dictionary containing organ dose statistics
        """
        organ_dose_stats = {}
        organ_masks = sample['organ_masks']
        dose_tensor = sample['dose']
        
        for organ_name, organ_mask in organ_masks.items():
            # Get dose values within organ
            organ_dose = dose_tensor * organ_mask
            organ_dose_values = organ_dose[organ_mask > 0]
            
            if len(organ_dose_values) > 0:
                organ_dose_stats[organ_name] = {
                    'mean_dose': float(torch.mean(organ_dose_values)),
                    'max_dose': float(torch.max(organ_dose_values)),
                    'min_dose': float(torch.min(organ_dose_values)),
                    'std_dose': float(torch.std(organ_dose_values)),
                    'dose_volume': float(torch.sum(organ_dose_values > 0.5) / len(organ_dose_values))  # D50
                }
        
        return organ_dose_stats


class HNCPatchDataset(BaseDataset):
    """
    Dataset for HNC patch-based training with on-the-fly patch generation.
    """
    
    def __init__(self, 
                 data_dir: str,
                 config: Dict[str, Any],
                 split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize HNC patch dataset.
        
        Args:
            data_dir: Directory containing processed HNC data
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformations
        """
        super().__init__(data_dir, config, split, transform)
        
        # HNC-specific configuration
        self.hnc_config = config.get('hnc', {})
        self.patch_size = self.hnc_config.get('patch_size', 32)
        self.use_attention = self.hnc_config.get('use_attention', True)
        self.attention_type = self.hnc_config.get('attention_type', 'high_dose')
        
        # HNC-specific organs
        self.target_organs = self.hnc_config.get('target_organs', [
            'parotid_left', 'parotid_right', 'submandibular_left', 'submandibular_right',
            'oral_cavity', 'larynx', 'pharynx', 'spinal_cord', 'brainstem'
        ])
        
        # Patch configuration
        self.patch_config = config.get('patch', {})
        self.num_patches_per_sample = self.patch_config.get('num_patches_per_sample', 15)
        self.high_dose_bias = self.patch_config.get('high_dose_bias', 0.8)  # Higher bias for HNC
        
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """
        Load paths to HNC data files for this split.
        
        Returns:
            List of dictionaries containing file paths for each sample
        """
        data_paths = []
        
        # Get all patient directories
        patient_dirs = [d for d in Path(self.data_dir).iterdir() if d.is_dir()]
        
        # Split patients based on split
        random.seed(42)  # For reproducible splits
        patient_dirs = sorted(patient_dirs)
        random.shuffle(patient_dirs)
        
        n_patients = len(patient_dirs)
        if self.split == 'train':
            patient_dirs = patient_dirs[:int(0.7 * n_patients)]
        elif self.split == 'val':
            patient_dirs = patient_dirs[int(0.7 * n_patients):int(0.85 * n_patients)]
        else:  # test
            patient_dirs = patient_dirs[int(0.85 * n_patients):]
        
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            
            # Check if required files exist
            ct_path = patient_dir / f"{patient_id}_ct_processed.nrrd"
            dose_path = patient_dir / f"{patient_id}_dose_processed.nrrd"
            
            if ct_path.exists() and dose_path.exists():
                # Get organ mask paths
                organ_mask_paths = {}
                for organ_name in self.target_organs:
                    mask_path = patient_dir / f"{patient_id}_{organ_name}_mask.nrrd"
                    if mask_path.exists():
                        organ_mask_paths[organ_name] = str(mask_path)
                
                sample_paths = {
                    'patient_id': patient_id,
                    'ct_path': str(ct_path),
                    'dose_path': str(dose_path),
                    'organ_mask_paths': organ_mask_paths
                }
                
                data_paths.append(sample_paths)
            else:
                self.logger.warning(f"Missing files for HNC patient {patient_id}")
        
        self.logger.info(f"Loaded {len(data_paths)} HNC samples for {self.split} split")
        return data_paths
    
    def _load_sample(self, sample_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load a single HNC sample from file paths.
        
        Args:
            sample_paths: Dictionary mapping data keys to file paths
            
        Returns:
            Dictionary containing loaded data as tensors
        """
        # Load images
        ct_tensor = self.load_image(sample_paths['ct_path'])
        dose_tensor = self.load_image(sample_paths['dose_path'])
        
        # Load organ masks
        organ_masks = {}
        for organ_name, mask_path in sample_paths['organ_mask_paths'].items():
            organ_masks[organ_name] = self.load_image(mask_path)
        
        return {
            'patient_id': sample_paths['patient_id'],
            'ct': ct_tensor,
            'dose': dose_tensor,
            'organ_masks': organ_masks
        }
    
    def _apply_entity_specific_processing(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply HNC-specific processing to the sample.
        
        Args:
            sample: Sample data
            
        Returns:
            Dictionary containing processed data
        """
        # Generate patches on-the-fly
        patches = self._generate_patches(sample)
        sample['patches'] = patches
        
        # Create attention masks if needed
        if self.use_attention:
            attention_masks = self._create_attention_masks(sample)
            sample['attention_masks'] = attention_masks
        
        return sample
    
    def _generate_patches(self, sample: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """
        Generate patches on-the-fly with high-dose bias for HNC.
        
        Args:
            sample: Sample data
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        ct_tensor = sample['ct']
        dose_tensor = sample['dose']
        organ_masks = sample['organ_masks']
        
        for organ_name, organ_mask in organ_masks.items():
            # Get high-dose regions
            high_dose_mask = self.get_high_dose_regions(dose_tensor, threshold=0.4)
            organ_high_dose_mask = high_dose_mask * organ_mask
            
            # Generate patches with bias towards high-dose regions
            organ_patches = self._generate_organ_patches(
                ct_tensor, dose_tensor, organ_mask, organ_high_dose_mask, organ_name
            )
            patches.extend(organ_patches)
        
        return patches
    
    def _generate_organ_patches(self, ct_tensor: torch.Tensor, 
                               dose_tensor: torch.Tensor,
                               organ_mask: torch.Tensor,
                               high_dose_mask: torch.Tensor,
                               organ_name: str) -> List[Dict[str, torch.Tensor]]:
        """
        Generate patches for a specific HNC organ with high-dose bias.
        
        Args:
            ct_tensor: CT tensor
            dose_tensor: Dose tensor
            organ_mask: Organ mask
            high_dose_mask: High-dose mask
            organ_name: Name of the organ
            
        Returns:
            List of patch dictionaries
        """
        patches = []
        
        # Get organ coordinates
        organ_coords = torch.nonzero(organ_mask.squeeze(), as_tuple=False)
        
        if len(organ_coords) == 0:
            return patches
        
        # Get high-dose coordinates
        high_dose_coords = torch.nonzero(high_dose_mask.squeeze(), as_tuple=False)
        
        # Calculate number of patches to generate
        num_patches = min(self.num_patches_per_sample, len(organ_coords) // (self.patch_size ** 3))
        
        for _ in range(num_patches):
            # Bias towards high-dose regions
            if len(high_dose_coords) > 0 and random.random() < self.high_dose_bias:
                # Sample from high-dose regions
                center_idx = random.randint(0, len(high_dose_coords) - 1)
                center = high_dose_coords[center_idx]
            else:
                # Sample from all organ regions
                center_idx = random.randint(0, len(organ_coords) - 1)
                center = organ_coords[center_idx]
            
            # Extract patch
            patch = self._extract_patch_at_center(ct_tensor, dose_tensor, center)
            if patch is not None:
                patch['organ_name'] = organ_name
                patches.append(patch)
        
        return patches
    
    def _extract_patch_at_center(self, ct_tensor: torch.Tensor, 
                                dose_tensor: torch.Tensor,
                                center: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """
        Extract patch at given center coordinates.
        
        Args:
            ct_tensor: CT tensor
            dose_tensor: Dose tensor
            center: Center coordinates
            
        Returns:
            Patch dictionary or None if extraction fails
        """
        z, y, x = center[0].item(), center[1].item(), center[2].item()
        half_size = self.patch_size // 2
        
        # Check bounds
        if (z - half_size < 0 or z + half_size >= ct_tensor.shape[2] or
            y - half_size < 0 or y + half_size >= ct_tensor.shape[3] or
            x - half_size < 0 or x + half_size >= ct_tensor.shape[4]):
            return None
        
        # Extract patches
        ct_patch = ct_tensor[:, :, z-half_size:z+half_size, y-half_size:y+half_size, x-half_size:x+half_size]
        dose_patch = dose_tensor[:, :, z-half_size:z+half_size, y-half_size:y+half_size, x-half_size:x+half_size]
        
        return {
            'ct_patch': ct_patch,
            'dose_patch': dose_patch,
            'center': center
        }
