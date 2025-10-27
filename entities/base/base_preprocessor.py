"""
Base preprocessor class for all anatomical entities.
Provides common interface for preprocessing across different anatomical regions.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import SimpleITK as sitk
import os
import logging
from pathlib import Path


class BasePreprocessor(ABC):
    """
    Abstract base class for entity-specific preprocessing.
    All anatomical entity preprocessors should inherit from this class.
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Initialize the preprocessor.
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
            output_dir: Output directory for processed data
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def segment_organs(self, ct_image: sitk.Image) -> Dict[str, sitk.Image]:
        """
        Segment organs specific to this anatomical entity.
        
        Args:
            ct_image: CT image to segment
            
        Returns:
            Dictionary mapping organ names to segmented masks
        """
        pass
    
    @abstractmethod
    def normalize_dose(self, dose_image: sitk.Image, 
                      prescribed_dose: Optional[float] = None) -> sitk.Image:
        """
        Normalize dose distribution specific to this anatomical entity.
        
        Args:
            dose_image: Dose distribution image
            prescribed_dose: Prescribed dose value (if available)
            
        Returns:
            Normalized dose image
        """
        pass
    
    @abstractmethod
    def extract_patches(self, ct_image: sitk.Image, 
                       dose_image: sitk.Image,
                       organ_masks: Dict[str, sitk.Image]) -> Dict[str, Any]:
        """
        Extract patches specific to this anatomical entity.
        
        Args:
            ct_image: CT image
            dose_image: Dose image
            organ_masks: Dictionary of organ masks
            
        Returns:
            Dictionary containing extracted patches and metadata
        """
        pass
    
    def check_and_resample_dose_to_ct_space(self, ct_image: sitk.Image, 
                                           dose_image: sitk.Image) -> sitk.Image:
        """
        Check if CT and dose images are in the same space, and resample dose to CT space if needed.
        
        Args:
            ct_image: CT image (reference space)
            dose_image: Dose image to potentially resample
            
        Returns:
            Dose image in CT space
        """
        # Check if images are already in the same space
        ct_spacing = ct_image.GetSpacing()
        dose_spacing = dose_image.GetSpacing()
        ct_origin = ct_image.GetOrigin()
        dose_origin = dose_image.GetOrigin()
        ct_direction = ct_image.GetDirection()
        dose_direction = dose_image.GetDirection()
        ct_size = ct_image.GetSize()
        dose_size = dose_image.GetSize()
        
        # Tolerance for floating point comparisons
        tolerance = 1e-6
        
        # Check if spacing, origin, direction, and size match
        spacing_match = all(abs(ct_spacing[i] - dose_spacing[i]) < tolerance for i in range(3))
        origin_match = all(abs(ct_origin[i] - dose_origin[i]) < tolerance for i in range(3))
        direction_match = all(abs(ct_direction[i] - dose_direction[i]) < tolerance for i in range(9))
        size_match = ct_size == dose_size
        
        if spacing_match and origin_match and direction_match and size_match:
            self.logger.info("✅ CT and dose images are already in the same space")
            return dose_image
        else:
            self.logger.warning("⚠️  CT and dose images are NOT in the same space!")
            self.logger.warning(f"   CT spacing: {ct_spacing}, dose spacing: {dose_spacing}")
            self.logger.warning(f"   CT origin: {ct_origin}, dose origin: {dose_origin}")
            self.logger.warning(f"   CT size: {ct_size}, dose size: {dose_size}")
            self.logger.info("🔄 Resampling dose to CT space...")
            
            # Resample dose to match CT space exactly
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(ct_spacing)
            resampler.SetOutputOrigin(ct_origin)
            resampler.SetOutputDirection(ct_direction)
            resampler.SetSize(ct_size)
            
            # Use nearest neighbor for dose to preserve dose values
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0.0)
            
            resampled_dose = resampler.Execute(dose_image)
            
            self.logger.info("✅ Dose successfully resampled to CT space")
            return resampled_dose

    def resample_to_common_spacing(self, image: sitk.Image, 
                                  target_spacing: Tuple[float, float, float]) -> sitk.Image:
        """
        Resample image to common spacing.
        
        Args:
            image: Input image
            target_spacing: Target voxel spacing (x, y, z)
            
        Returns:
            Resampled image
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # Calculate new size
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) 
                   for i in range(3)]
        
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(new_size)
        
        return resampler.Execute(image)
    
    def resize_image(self, image: sitk.Image, target_size: List[int]) -> sitk.Image:
        """
        Resize image to target size.
        
        Args:
            image: Input image
            target_size: Target size [width, height, depth]
            
        Returns:
            Resized image
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(target_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        # Calculate new spacing to maintain aspect ratio
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        
        new_spacing = [
            original_spacing[0] * original_size[0] / target_size[0],
            original_spacing[1] * original_size[1] / target_size[1],
            original_spacing[2] * original_size[2] / target_size[2]
        ]
        
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        
        return resampler.Execute(image)
    
    def apply_mask(self, image: sitk.Image, mask: sitk.Image) -> sitk.Image:
        """
        Apply mask to image.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Masked image
        """
        # Ensure mask is binary
        mask_filter = sitk.BinaryThresholdImageFilter()
        mask_filter.SetLowerThreshold(1)
        mask_filter.SetUpperThreshold(1)
        mask_filter.SetInsideValue(1)
        mask_filter.SetOutsideValue(0)
        binary_mask = mask_filter.Execute(mask)
        
        # Apply mask
        masked_image = sitk.MaskImageFilter().Execute(image, binary_mask)
        return masked_image
    
    def normalize_image(self, image: sitk.Image, 
                       method: str = 'percentile',
                       percentile: float = 99.0) -> sitk.Image:
        """
        Normalize image using specified method.
        
        Args:
            image: Input image
            method: Normalization method ('percentile', 'minmax', 'zscore')
            percentile: Percentile for percentile-based normalization
            
        Returns:
            Normalized image
        """
        array = sitk.GetArrayFromImage(image)
        
        if method == 'percentile':
            p_value = np.percentile(array, percentile)
            array = np.clip(array / p_value, 0, 1)
        elif method == 'minmax':
            array = (array - array.min()) / (array.max() - array.min())
        elif method == 'zscore':
            array = (array - array.mean()) / array.std()
        
        normalized_image = sitk.GetImageFromArray(array)
        normalized_image.CopyInformation(image)
        return normalized_image
    
    def save_processed_data(self, data: Dict[str, Any], 
                           patient_id: str, 
                           organ_name: str) -> Dict[str, str]:
        """
        Save processed data to disk.
        
        Args:
            data: Dictionary containing processed data
            patient_id: Patient identifier
            organ_name: Organ name
            
        Returns:
            Dictionary mapping data keys to file paths
        """
        patient_dir = self.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        for key, value in data.items():
            if isinstance(value, sitk.Image):
                file_path = patient_dir / f"{organ_name}_{key}.nrrd"
                sitk.WriteImage(value, str(file_path))
                saved_paths[key] = str(file_path)
            elif isinstance(value, np.ndarray):
                file_path = patient_dir / f"{organ_name}_{key}.npy"
                np.save(str(file_path), value)
                saved_paths[key] = str(file_path)
            else:
                # Save as pickle for other data types
                import pickle
                file_path = patient_dir / f"{organ_name}_{key}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                saved_paths[key] = str(file_path)
        
        return saved_paths
    
    def create_visualization(self, ct_image: sitk.Image,
                           dose_image: sitk.Image,
                           organ_masks: Dict[str, sitk.Image],
                           patient_id: str) -> str:
        """
        Create visualization of processed data.
        
        Args:
            ct_image: CT image
            dose_image: Dose image
            organ_masks: Dictionary of organ masks
            patient_id: Patient identifier
            
        Returns:
            Path to saved visualization
        """
        # This will be implemented by entity-specific preprocessors
        pass
