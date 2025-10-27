"""
Base dataset class for all anatomical entities.
Provides common interface for data loading across different anatomical regions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from torch.utils.data import Dataset
import numpy as np
import SimpleITK as sitk
import os
import logging
from pathlib import Path
import pickle


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for entity-specific datasets.
    All anatomical entity datasets should inherit from this class.
    """
    
    def __init__(self, 
                 data_dir: str,
                 config: Dict[str, Any],
                 split: str = 'train',
                 transform: Optional[Any] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed data
            config: Configuration dictionary
            split: Data split ('train', 'val', 'test')
            transform: Optional data transformations
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
    @abstractmethod
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """
        Load paths to data files for this split.
        
        Returns:
            List of dictionaries containing file paths for each sample
        """
        pass
    
    @abstractmethod
    def _load_sample(self, sample_paths: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """
        Load a single sample from file paths.
        
        Args:
            sample_paths: Dictionary mapping data keys to file paths
            
        Returns:
            Dictionary containing loaded data as tensors
        """
        pass
    
    @abstractmethod
    def _apply_entity_specific_processing(self, sample: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply entity-specific processing to the sample.
        
        Args:
            sample: Dictionary containing loaded data
            
        Returns:
            Dictionary containing processed data
        """
        pass
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing sample data
        """
        if idx >= len(self.data_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data_paths)}")
        
        sample_paths = self.data_paths[idx]
        sample = self._load_sample(sample_paths)
        sample = self._apply_entity_specific_processing(sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def load_image(self, file_path: str) -> torch.Tensor:
        """
        Load image from file path.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Image as tensor
        """
        if file_path.endswith('.nrrd'):
            image = sitk.ReadImage(file_path)
            array = sitk.GetArrayFromImage(image)
        elif file_path.endswith('.npy'):
            array = np.load(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Convert to tensor and add channel dimension if needed
        tensor = torch.from_numpy(array).float()
        if tensor.dim() == 3:  # Add channel dimension
            tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def load_patches(self, file_path: str) -> List[torch.Tensor]:
        """
        Load patches from file path.
        
        Args:
            file_path: Path to patches file
            
        Returns:
            List of patch tensors
        """
        if file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                patches_data = pickle.load(f)
            
            patches = []
            for patch_data in patches_data:
                if isinstance(patch_data, dict) and 'patch' in patch_data:
                    patch = torch.from_numpy(patch_data['patch']).float()
                    if patch.dim() == 3:
                        patch = patch.unsqueeze(0)
                    patches.append(patch)
                else:
                    # Assume it's a numpy array
                    patch = torch.from_numpy(patch_data).float()
                    if patch.dim() == 3:
                        patch = patch.unsqueeze(0)
                    patches.append(patch)
            
            return patches
        else:
            raise ValueError(f"Unsupported patches file format: {file_path}")
    
    def get_high_dose_regions(self, dose_tensor: torch.Tensor, 
                             threshold: float = 0.5) -> torch.Tensor:
        """
        Get high dose regions from dose tensor.
        
        Args:
            dose_tensor: Dose distribution tensor
            threshold: Threshold for high dose regions
            
        Returns:
            Binary mask of high dose regions
        """
        return (dose_tensor > threshold).float()
    
    def create_attention_mask(self, dose_tensor: torch.Tensor,
                            organ_mask: torch.Tensor,
                            attention_type: str = 'high_dose') -> torch.Tensor:
        """
        Create attention mask based on dose and organ information.
        
        Args:
            dose_tensor: Dose distribution tensor
            organ_mask: Organ mask tensor
            attention_type: Type of attention ('high_dose', 'gradient', 'uniform')
            
        Returns:
            Attention mask tensor
        """
        if attention_type == 'high_dose':
            high_dose_mask = self.get_high_dose_regions(dose_tensor)
            attention_mask = high_dose_mask * organ_mask
        elif attention_type == 'gradient':
            # Create gradient-based attention
            grad_x = torch.abs(torch.diff(dose_tensor, dim=-1, prepend=dose_tensor[..., :1]))
            grad_y = torch.abs(torch.diff(dose_tensor, dim=-2, prepend=dose_tensor[..., :1, :]))
            grad_z = torch.abs(torch.diff(dose_tensor, dim=-3, prepend=dose_tensor[..., :1, :, :]))
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
            attention_mask = gradient_magnitude * organ_mask
        elif attention_type == 'uniform':
            attention_mask = organ_mask
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Normalize attention mask
        if attention_mask.sum() > 0:
            attention_mask = attention_mask / attention_mask.sum()
        
        return attention_mask
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function for batching samples.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data dictionary
        """
        batched_data = {}
        
        # Get all keys from first sample
        keys = batch[0].keys()
        
        for key in keys:
            if key in ['ct', 'dose', 'organ_mask']:
                # Stack tensors
                batched_data[key] = torch.stack([sample[key] for sample in batch])
            elif key in ['patches', 'patch_locations']:
                # Concatenate lists
                all_items = []
                for sample in batch:
                    all_items.extend(sample[key])
                batched_data[key] = all_items
            else:
                # Keep as list for other data types
                batched_data[key] = [sample[key] for sample in batch]
        
        return batched_data
