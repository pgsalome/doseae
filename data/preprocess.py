import os
import glob
import pickle
import json
import yaml
import argparse
import numpy as np
import nrrd
import torch
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random
import matplotlib.pyplot as plt

# Import your existing functionality
from utils.patch_extraction import extract_informative_patches_2d, extract_patches_3d

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """
    Load configuration from YAML file.

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def debug_nrrd_file(file_path):
    """
    Debug an NRRD file to see if it contains valid data.

    Args:
        file_path (str): Path to NRRD file

    Returns:
        bool: True if file was examined successfully
    """
    try:
        data, header = nrrd.read(file_path)
        print(f"File: {file_path}")
        print(f"Shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Mean value: {np.mean(data)}")
        print(
            f"Non-zero values: {np.count_nonzero(data)}/{data.size} ({np.count_nonzero(data) / data.size * 100:.2f}%)")

        # Visualize a middle slice if it's 3D
        if len(data.shape) == 3:
            mid_slice = data.shape[0] // 2
            plt.figure(figsize=(10, 8))
            plt.imshow(data[mid_slice], cmap='viridis')
            plt.colorbar()
            plt.title(f"{os.path.basename(file_path)} - Slice {mid_slice}")
            plt.savefig(f"debug_{os.path.basename(file_path)}.png", dpi=300)
            plt.close()
        return True
    except Exception as e:
        print(f"Error examining {file_path}: {e}")
        return False


def resample_volume(volume, original_spacing, target_spacing):
    """
    Resample a volume to a target spacing.

    Args:
        volume (numpy.ndarray): Input volume
        original_spacing (tuple): Original voxel spacing (mm)
        target_spacing (tuple): Target voxel spacing (mm)

    Returns:
        numpy.ndarray: Resampled volume
    """
    # Convert to SimpleITK image
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(original_spacing)

    # Calculate target size
    original_size = sitk_image.GetSize()
    target_size = [int(original_size[i] * original_spacing[i] / target_spacing[i]) for i in range(3)]

    # Set up resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())

    # Resample image
    resampled_image = resampler.Execute(sitk_image)

    # Convert back to numpy array
    resampled_volume = sitk.GetArrayFromImage(resampled_image)

    return resampled_volume


def load_nrrd_file(file_path):
    """
    Load NRRD file.

    Args:
        file_path (str): Path to NRRD file

    Returns:
        tuple: (data, header)
    """
    try:
        data, header = nrrd.read(file_path)
        return data, header
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None, None


def min_max_normalize(data):
    """
    Normalize data to range [0, 1] using min-max normalization.

    Args:
        data: Input data

    Returns:
        Normalized data
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data

    min_val = np.min(data_np)
    max_val = np.max(data_np)

    if max_val > min_val:
        normalized = (data_np - min_val) / (max_val - min_val)
    else:
        normalized = data_np

    if isinstance(data, torch.Tensor):
        return torch.from_numpy(normalized).to(data.device)
    return normalized


def normalize_tensor(data):
    """
    Normalize data using Z-score normalization.

    Args:
        data: Input data

    Returns:
        Normalized data
    """
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    else:
        data_np = data

    mean = np.mean(data_np)
    std = np.std(data_np)

    if std > 0:
        normalized = (data_np - mean) / std
    else:
        normalized = data_np - mean

    if isinstance(data, torch.Tensor):
        return torch.from_numpy(normalized).to(data.device)
    return normalized


def process_patient_folder(patient_folder, config):
    """
    Process a patient folder containing dose distribution images and masks.

    Args:
        patient_folder (str): Path to patient folder
        config (dict): Configuration dictionary

    Returns:
        list: List of processed data dictionaries
    """
    processed_items = []

    # Check for dose image file
    dose_file = os.path.join(patient_folder, 'image_dd.nrrd')  # Despite name, it's dose distribution
    if not os.path.exists(dose_file):
        logger.warning(f"No dose distribution image found in {patient_folder}")
        return []

    # Load the dose image
    dose_data, dose_header = load_nrrd_file(dose_file)
    if dose_data is None:
        logger.warning(f"Failed to load dose distribution from {dose_file}")
        return []

    # Check for non-zero values in dose image
    if np.count_nonzero(dose_data) == 0:
        logger.warning(f"Dose distribution in {patient_folder} has no non-zero values")
        return []

    # Get spacing information
    if 'spacing' in dose_header:
        spacing = dose_header['spacing']
    else:
        logger.warning(f"No spacing information found in {dose_file}, assuming isotropic 1mm spacing")
        spacing = (1.0, 1.0, 1.0)

    # Get patient ID
    patient_id = os.path.basename(patient_folder)

    # Process each mask type - we know we're looking for lungctr and lungipsi
    for mask_type in ['lungctr', 'lungipsi']:
        mask_file = os.path.join(patient_folder, f'mask_{mask_type}.nrrd')
        if not os.path.exists(mask_file):
            logger.warning(f"No {mask_type} mask found in {patient_folder}")
            continue

        # Load the mask
        mask_data, mask_header = load_nrrd_file(mask_file)
        if mask_data is None:
            logger.warning(f"Failed to load mask from {mask_file}")
            continue

        # Check for non-zero values in mask
        if np.count_nonzero(mask_data) == 0:
            logger.warning(f"Mask {mask_type} in {patient_folder} has no non-zero values")
            continue

        # Ensure mask and image are the same shape
        if mask_data.shape != dose_data.shape:
            logger.warning(
                f"Mask shape {mask_data.shape} does not match dose shape {dose_data.shape} for {mask_file}")
            continue

        # Extract dose data within mask (set everything outside the mask to zero)
        masked_dose = np.zeros_like(dose_data)
        masked_dose[mask_data > 0] = dose_data[mask_data > 0]

        # Debug - verify there's data in the masked image
        if np.count_nonzero(masked_dose) == 0:
            logger.warning(f"Masked dose for {patient_id}, {mask_type} has no non-zero values")

            # Try to see if there are overlapping non-zero values in both image and mask
            overlap = np.logical_and(dose_data != 0, mask_data > 0)
            if np.any(overlap):
                logger.warning(f"There are overlapping non-zero values, but masking failed")
            else:
                logger.warning(f"No overlapping non-zero values between dose and mask")
            continue

        # Process the masked dose data
        result = process_masked_data(
            masked_dose,
            mask_data,
            spacing,
            patient_id,
            mask_type,
            dose_file,
            mask_file,
            config
        )

        if result is not None:
            processed_items.append(result)

    return processed_items


def process_masked_data(data, mask, spacing, patient_id, mask_type, dose_path, mask_path, config):
    """
    Process masked dose data.

    Args:
        data (numpy.ndarray): Masked dose data
        mask (numpy.ndarray): Mask data
        spacing (tuple): Voxel spacing
        patient_id (str): Patient ID
        mask_type (str): Type of mask (ctr or ipsi)
        dose_path (str): Path to dose image
        mask_path (str): Path to mask
        config (dict): Configuration dictionary

    Returns:
        dict: Processed data or None if processing failed
    """
    try:
        # Save original data for visualization
        original_data = data.copy()

        # Get dimensionality and size settings
        is_2d = len(data.shape) == 2 or config.get('preprocessing', {}).get('extract_slices', False)
        unit_type = config.get('preprocessing', {}).get('unit_type', 'voxels')

        # Get target size - check if empty or None
        resize_to = config.get('preprocessing', {}).get('resize_to', None)
        should_resize = resize_to is not None and resize_to != [] and resize_to != ''

        # Only process sizing and spacing if resize_to is specified
        if should_resize:
            # Get spacing settings
            target_spacing = config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
            if isinstance(target_spacing, (int, float)):
                target_spacing = (target_spacing, target_spacing, target_spacing)
            elif isinstance(target_spacing, list):
                target_spacing = tuple(target_spacing)

            # Get target size
            target_size = resize_to
            if isinstance(target_size, (int, float)):
                if not is_2d:
                    target_size = (target_size, target_size, target_size)
                else:
                    target_size = (target_size, target_size)
            elif isinstance(target_size, str):
                # Parse string format like "128,128,64"
                target_size = tuple(map(int, target_size.split(',')))
            elif isinstance(target_size, list):
                target_size = tuple(target_size)

            # Process based on dimensionality
            if is_2d:
                # If input is 3D but we want 2D, extract middle slice
                if len(data.shape) == 3:
                    # Find a non-empty slice
                    non_zero_slices = [i for i in range(data.shape[0]) if np.any(data[i])]
                    if non_zero_slices:
                        middle_slice_idx = non_zero_slices[len(non_zero_slices) // 2]
                    else:
                        middle_slice_idx = data.shape[0] // 2

                    data = data[middle_slice_idx]
                    mask = mask[middle_slice_idx]

                # Reshape 2D data if needed
                if data.shape != target_size:
                    # Create SimpleITK images for both data and mask
                    sitk_data = sitk.GetImageFromArray(data)
                    sitk_mask = sitk.GetImageFromArray(mask)

                    # Set spacing
                    if unit_type == 'mm':
                        sitk_data.SetSpacing(target_spacing[:2])
                        sitk_mask.SetSpacing(target_spacing[:2])

                    # Create resamplers
                    resampler_data = sitk.ResampleImageFilter()
                    resampler_data.SetInterpolator(sitk.sitkLinear)
                    resampler_data.SetSize(target_size)
                    resampler_data.SetOutputDirection(sitk_data.GetDirection())
                    resampler_data.SetOutputOrigin(sitk_data.GetOrigin())

                    resampler_mask = sitk.ResampleImageFilter()
                    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for mask
                    resampler_mask.SetSize(target_size)
                    resampler_mask.SetOutputDirection(sitk_mask.GetDirection())
                    resampler_mask.SetOutputOrigin(sitk_mask.GetOrigin())

                    # Calculate new spacing if needed
                    if unit_type == 'voxels':
                        new_spacing = [data.shape[i] * spacing[i] / target_size[i] for i in range(2)]
                        resampler_data.SetOutputSpacing(new_spacing)
                        resampler_mask.SetOutputSpacing(new_spacing)
                    else:
                        resampler_data.SetOutputSpacing(target_spacing[:2])
                        resampler_mask.SetOutputSpacing(target_spacing[:2])

                    # Resample
                    resampled_data = resampler_data.Execute(sitk_data)
                    resampled_mask = resampler_mask.Execute(sitk_mask)

                    # Convert back to numpy arrays
                    data = sitk.GetArrayFromImage(resampled_data)
                    mask = sitk.GetArrayFromImage(resampled_mask)

                    # Re-apply mask in case interpolation created values outside the mask
                    data[mask == 0] = 0

            else:  # 3D processing
                # Process 3D data
                if data.shape != target_size:
                    # Create SimpleITK images
                    sitk_data = sitk.GetImageFromArray(data)
                    sitk_mask = sitk.GetImageFromArray(mask)

                    # Set spacing
                    if unit_type == 'mm':
                        sitk_data.SetSpacing(target_spacing)
                        sitk_mask.SetSpacing(target_spacing)

                    # Create resamplers
                    resampler_data = sitk.ResampleImageFilter()
                    resampler_data.SetInterpolator(sitk.sitkLinear)
                    resampler_data.SetSize(target_size[::-1])  # SimpleITK uses XYZ order
                    resampler_data.SetOutputDirection(sitk_data.GetDirection())
                    resampler_data.SetOutputOrigin(sitk_data.GetOrigin())

                    resampler_mask = sitk.ResampleImageFilter()
                    resampler_mask.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for mask
                    resampler_mask.SetSize(target_size[::-1])  # SimpleITK uses XYZ order
                    resampler_mask.SetOutputDirection(sitk_mask.GetDirection())
                    resampler_mask.SetOutputOrigin(sitk_mask.GetOrigin())

                    # Calculate new spacing if needed
                    if unit_type == 'voxels':
                        new_spacing = [data.shape[i] * spacing[i] / target_size[i] for i in range(3)]
                        resampler_data.SetOutputSpacing(new_spacing)
                        resampler_mask.SetOutputSpacing(new_spacing)
                    else:
                        resampler_data.SetOutputSpacing(target_spacing)
                        resampler_mask.SetOutputSpacing(target_spacing)

                    # Resample
                    resampled_data = resampler_data.Execute(sitk_data)
                    resampled_mask = resampler_mask.Execute(sitk_mask)

                    # Convert back to numpy arrays
                    data = sitk.GetArrayFromImage(resampled_data)
                    mask = sitk.GetArrayFromImage(resampled_mask)

                    # Re-apply mask in case interpolation created values outside the mask
                    data[mask == 0] = 0

            # Use the target spacing for the processed data if using mm
            if unit_type == 'mm':
                spacing_to_use = target_spacing
            else:
                # If using voxels, update the spacing to reflect the new voxel sizes
                if is_2d:
                    spacing_to_use = [data.shape[i] * spacing[i] / target_size[i] for i in range(2)]
                else:
                    spacing_to_use = [data.shape[i] * spacing[i] / target_size[i] for i in range(3)]
        else:
            # If not resizing, keep original spacing
            spacing_to_use = spacing

        # Check if there's still data after resizing
        if np.count_nonzero(data) == 0:
            logger.warning(f"No non-zero values in resized data for {patient_id}, mask {mask_type}")
            return None

        # Normalize if requested
        if config.get('preprocessing', {}).get('normalize', False):
            # Store normalization method
            normalize_method = config.get('preprocessing', {}).get('normalize_method', '95percentile')

            # Only consider non-zero values (where mask is applied)
            non_zero_indices = mask > 0
            non_zero_data = data[non_zero_indices]

            if len(non_zero_data) == 0:
                logger.warning(f"No non-zero values in masked data for {patient_id}, mask {mask_type}")
                return None

            # Log value ranges before normalization for debugging
            logger.info(
                f"Patient {patient_id}, Mask {mask_type} - Value range before normalization: {np.min(data):.4f} to {np.max(data):.4f}")

            if normalize_method == "95percentile":
                percentile_val = np.percentile(non_zero_data,
                                               config.get('preprocessing', {}).get('percentile_norm', 95))
                if percentile_val > 0:
                    # Normalize only non-zero values
                    data[non_zero_indices] = data[non_zero_indices] / percentile_val
            elif normalize_method == "minmax":
                min_val = np.min(non_zero_data)
                max_val = np.max(non_zero_data)
                if max_val > min_val:
                    # Normalize only non-zero values
                    data[non_zero_indices] = (data[non_zero_indices] - min_val) / (max_val - min_val)
            elif normalize_method == "zscore":
                mean = np.mean(non_zero_data)
                std = np.std(non_zero_data)
                if std > 0:
                    # Normalize only non-zero values
                    data[non_zero_indices] = (data[non_zero_indices] - mean) / std

            # Log value ranges after normalization for debugging
            logger.info(
                f"Patient {patient_id}, Mask {mask_type} - Value range after normalization: {np.min(data):.4f} to {np.max(data):.4f}")
        else:
            normalize_method = "none"

        # Calculate zero count (non-zero voxels in the mask)
        zc_value = np.count_nonzero(mask)

        return {
            "data": data,
            "mask": mask,
            "original_data": original_data,
            "patient_id": patient_id,
            "mask_type": mask_type,
            "zc_value": zc_value,
            "is_2d": len(data.shape) == 2,
            "dose_path": dose_path,
            "mask_path": mask_path,
            "spacing": spacing_to_use,
            "original_shape": data.shape,
            "was_resized": should_resize,
            "normalize_method": normalize_method
        }

    except Exception as e:
        logger.error(f"Error processing masked data for {patient_id}, mask {mask_type}: {e}")
        return None


def extract_patches_from_data(volume_data, config):
    """
    Extract patches from volume data based on configuration.

    Args:
        volume_data (dict): Volume data
        config (dict): Configuration dictionary

    Returns:
        list: List of patch dictionaries
    """
    data = volume_data["data"]
    mask = volume_data["mask"]
    patches = []

    # Get patch extraction settings
    patch_dimension = config.get('patch_extraction', {}).get('patch_dimension', '3D')
    patch_size = config.get('patch_extraction', {}).get('patch_size', [50, 50, 50])
    patch_unit_type = config.get('patch_extraction', {}).get('patch_unit_type', 'voxels')

    # Convert patch size to tuple if needed
    if isinstance(patch_size, (int, float)):
        if patch_dimension == '3D':
            patch_size = (patch_size, patch_size, patch_size)
        else:
            patch_size = (patch_size, patch_size)
    elif isinstance(patch_size, str):
        patch_size = tuple(map(int, patch_size.split(',')))
    elif isinstance(patch_size, list):
        patch_size = tuple(patch_size)

    # Get spacing information
    spacing = volume_data.get("spacing", (1.0, 1.0, 1.0))

    # Handle different dimensionality and unit types
    if patch_dimension == '2D':
        # Extract 2D patches
        if volume_data["is_2d"]:
            # From 2D data
            if patch_unit_type == 'mm' and spacing:
                # Convert physical size to voxel size
                voxel_patch_size = (
                    int(patch_size[0] / spacing[0]),
                    int(patch_size[1] / spacing[1])
                )
            else:
                voxel_patch_size = patch_size

            patches_arr = extract_informative_patches_2d(
                data,
                patch_size=voxel_patch_size,
                threshold=config.get('patch_extraction', {}).get('threshold', 0.01)
            )

            # Create corresponding mask patches
            mask_patches = []
            for patch in patches_arr:
                # Find the location of this patch in the original data
                for i in range(0, data.shape[0] - voxel_patch_size[0] + 1, voxel_patch_size[0]):
                    for j in range(0, data.shape[1] - voxel_patch_size[1] + 1, voxel_patch_size[1]):
                        if np.array_equal(data[i:i + voxel_patch_size[0], j:j + voxel_patch_size[1]], patch):
                            mask_patch = mask[i:i + voxel_patch_size[0], j:j + voxel_patch_size[1]]
                            mask_patches.append(mask_patch)
                            break
                    else:
                        continue
                    break

            # Check if we found mask patches for all data patches
            if len(mask_patches) != len(patches_arr):
                logger.warning(
                    f"Could not find mask patches for all data patches: {len(mask_patches)} vs {len(patches_arr)}")
                # Create empty mask patches for missing ones
                while len(mask_patches) < len(patches_arr):
                    mask_patches.append(np.zeros_like(patches_arr[len(mask_patches)]))

            # Create patch dictionaries
            for i, (patch, mask_patch) in enumerate(zip(patches_arr, mask_patches)):
                patch_dict = {
                    "data": patch,
                    "mask": mask_patch,
                    "patient_id": volume_data["patient_id"],
                    "mask_type": volume_data["mask_type"],
                    "zc_value": np.count_nonzero(patch),
                    "is_2d": True,
                    "original_data": patch.copy(),  # Store original patch
                    "normalize_method": volume_data.get("normalize_method", "none"),
                    "was_resized": volume_data.get("was_resized", False),
                    "dose_path": volume_data["dose_path"],
                    "mask_path": volume_data["mask_path"]
                }
                patches.append(patch_dict)
        else:
            # Extract 2D patches from 3D data (slice first)
            for i in range(data.shape[0]):
                slice_data = data[i]
                slice_mask = mask[i]

                if patch_unit_type == 'mm' and spacing:
                    # Convert physical size to voxel size
                    voxel_patch_size = (
                        int(patch_size[0] / spacing[1]),
                        int(patch_size[1] / spacing[2])
                    )
                else:
                    voxel_patch_size = patch_size

                slice_patches = extract_informative_patches_2d(
                    slice_data,
                    patch_size=voxel_patch_size,
                    threshold=config.get('patch_extraction', {}).get('threshold', 0.01)
                )

                # Create corresponding mask patches using same approach as above
                mask_patches = []
                for patch in slice_patches:
                    found = False
                    for i_patch in range(0, slice_data.shape[0] - voxel_patch_size[0] + 1, voxel_patch_size[0]):
                        for j_patch in range(0, slice_data.shape[1] - voxel_patch_size[1] + 1, voxel_patch_size[1]):
                            if np.array_equal(slice_data[i_patch:i_patch + voxel_patch_size[0],
                                              j_patch:j_patch + voxel_patch_size[1]], patch):
                                mask_patch = slice_mask[i_patch:i_patch + voxel_patch_size[0],
                                             j_patch:j_patch + voxel_patch_size[1]]
                                mask_patches.append(mask_patch)
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        mask_patches.append(np.zeros_like(patch))

                # Create patch dictionaries
                for j, (patch, mask_patch) in enumerate(zip(slice_patches, mask_patches)):
                    patch_dict = {
                        "data": patch,
                        "mask": mask_patch,
                        "patient_id": volume_data["patient_id"],
                        "mask_type": volume_data["mask_type"],
                        "zc_value": np.count_nonzero(patch),
                        "is_2d": True,
                        "original_data": patch.copy(),  # Store original patch
                        "normalize_method": volume_data.get("normalize_method", "none"),
                        "was_resized": volume_data.get("was_resized", False),
                        "slice_idx": i,
                        "dose_path": volume_data["dose_path"],
                        "mask_path": volume_data["mask_path"]
                    }
                    patches.append(patch_dict)
    else:  # 3D patches
        if not volume_data["is_2d"]:
            # Extract 3D patches
            if patch_unit_type == 'mm' and spacing:
                # Convert physical size to voxel size
                voxel_patch_size = (
                    int(patch_size[0] / spacing[0]),
                    int(patch_size[1] / spacing[1]),
                    int(patch_size[2] / spacing[2])
                )
            else:
                voxel_patch_size = patch_size

            # Use patch sizes from config if specified
            if config.get('dataset', {}).get('patch_size_3d') and patch_unit_type == 'voxels':
                voxel_patch_size = tuple(config['dataset']['patch_size_3d'])

            patches_arr = extract_patches_3d(
                data,
                patch_size=voxel_patch_size,
                max_patches=config.get('patch_extraction', {}).get('max_patches_per_volume', 100),
                random_state=config.get('patch_extraction', {}).get('random_state', 42)
            )

            # For 3D, we'll just use a mask of ones since extraction_patches_3d doesn't give us patch locations
            for patch in patches_arr:
                # Create a mask of ones (assuming the patch is already masked)
                mask_patch = np.ones_like(patch)

                patch_dict = {
                    "data": patch,
                    "mask": mask_patch,
                    "patient_id": volume_data["patient_id"],
                    "mask_type": volume_data["mask_type"],
                    "zc_value": np.count_nonzero(patch),
                    "is_2d": False,
                    "original_data": patch.copy(),  # Store original patch
                    "normalize_method": volume_data.get("normalize_method", "none"),
                    "was_resized": volume_data.get("was_resized", False),
                    "dose_path": volume_data["dose_path"],
                    "mask_path": volume_data["mask_path"]
                }
                patches.append(patch_dict)
        else:
            logger.warning(f"Cannot extract 3D patches from 2D data for {volume_data['patient_id']}")

    return patches


def extract_slices(volume_data, config):
    """
    Extract 2D slices from a 3D volume.

    Args:
        volume_data (dict): Volume data
        config (dict): Configuration dictionary

    Returns:
        list: List of slice data dictionaries
    """
    data = volume_data["data"]
    mask = volume_data["mask"]
    original_data = volume_data.get("original_data", data)

    # If already 2D, return as is
    if volume_data["is_2d"]:
        return [volume_data]

    slices = []
    slice_extraction = config.get('preprocessing', {}).get('slice_extraction', 'all')

    # Extract slices based on strategy
    if slice_extraction == "all":
        for i in range(data.shape[0]):
            slice_data = data[i]
            slice_mask = mask[i]
            slice_original = original_data[i] if original_data is not None else slice_data

            # Only include slices that have mask content
            if np.any(slice_mask):
                slice_dict = {
                    "data": slice_data,
                    "mask": slice_mask,
                    "original_data": slice_original,
                    "patient_id": volume_data["patient_id"],
                    "mask_type": volume_data["mask_type"],
                    "zc_value": np.count_nonzero(slice_mask),
                    "is_2d": True,
                    "slice_idx": i,
                    "normalize_method": volume_data.get("normalize_method", "none"),
                    "was_resized": volume_data.get("was_resized", False),
                    "dose_path": volume_data["dose_path"],
                    "mask_path": volume_data["mask_path"]
                }
                slices.append(slice_dict)
    elif slice_extraction == "center":
        # Find center slice that has mask content
        non_zero_slices = [i for i in range(mask.shape[0]) if np.any(mask[i])]
        if non_zero_slices:
            i = non_zero_slices[len(non_zero_slices) // 2]
            slice_data = data[i]
            slice_mask = mask[i]
            slice_original = original_data[i] if original_data is not None else slice_data

            slice_dict = {
                "data": slice_data,
                "mask": slice_mask,
                "original_data": slice_original,
                "patient_id": volume_data["patient_id"],
                "mask_type": volume_data["mask_type"],
                "zc_value": np.count_nonzero(slice_mask),
                "is_2d": True,
                "slice_idx": i,
                "normalize_method": volume_data.get("normalize_method", "none"),
                "was_resized": volume_data.get("was_resized", False),
                "dose_path": volume_data["dose_path"],
                "mask_path": volume_data["mask_path"]
            }
            slices.append(slice_dict)
    elif slice_extraction == "informative":
        threshold = config.get('preprocessing', {}).get('non_zero_threshold', 0.05)
        for i in range(data.shape[0]):
            slice_mask = mask[i]
            if np.count_nonzero(slice_mask) / slice_mask.size > threshold:
                slice_data = data[i]
                slice_original = original_data[i] if original_data is not None else slice_data

                slice_dict = {
                    "data": slice_data,
                    "mask": slice_mask,
                    "original_data": slice_original,
                    "patient_id": volume_data["patient_id"],
                    "mask_type": volume_data["mask_type"],
                    "zc_value": np.count_nonzero(slice_mask),
                    "is_2d": True,
                    "slice_idx": i,
                    "normalize_method": volume_data.get("normalize_method", "none"),
                    "was_resized": volume_data.get("was_resized", False),
                    "dose_path": volume_data["dose_path"],
                    "mask_path": volume_data["mask_path"]
                }
                slices.append(slice_dict)

    return slices


def visualize_samples(dataset, output_dir, num_samples=2):
    """
    Visualize sample images from the dataset showing original image and normalized.
    (No middle image as requested)

    Args:
        dataset (list): List of processed data items
        output_dir (Path): Output directory
        num_samples (int): Number of samples to visualize
    """
    if not dataset:
        logger.warning("No samples to visualize")
        return

    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True, parents=True)

    # Select random samples
    if len(dataset) > num_samples:
        sample_indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[i] for i in sample_indices]
    else:
        samples = dataset[:min(num_samples, len(dataset))]

    for i, sample in enumerate(samples):
        data = sample["data"]  # Processed data
        original_data = sample.get("original_data", data)  # Original data
        mask = sample.get("mask", np.ones_like(data))  # Mask data
        patient_id = sample["patient_id"]
        mask_type = sample.get("mask_type", "unknown")
        normalize_method = sample.get("normalize_method", "unknown")

        # Find middle slice with content for 3D data
        if len(data.shape) == 3:
            non_zero_indices = []
            for idx in range(data.shape[0]):
                if np.any(data[idx]):
                    non_zero_indices.append(idx)

            if non_zero_indices:
                # Take the middle of non-zero slices
                slice_idx = non_zero_indices[len(non_zero_indices) // 2]
            else:
                # If no non-zero slices, just take the middle slice
                slice_idx = data.shape[0] // 2

            # Extract slices
            data_slice = data[slice_idx]
            mask_slice = mask[slice_idx]
            original_slice = original_data[slice_idx] if original_data is not None else data_slice
            slice_info = f" (Slice {slice_idx})"
        else:
            # 2D data
            data_slice = data
            mask_slice = mask
            original_slice = original_data
            slice_info = ""

        # Create figure with images (2 rows: original and normalized)
        plt.figure(figsize=(10, 10))

        # Row 1: Original image
        plt.subplot(2, 1, 1)
        plt.imshow(original_slice, cmap='viridis')  # Changed to viridis for dose visualization
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Original Dose Distribution - Patient {patient_id}{slice_info}")
        plt.axis("off")

        # Row 2: Normalized image
        plt.subplot(2, 1, 2)
        plt.imshow(data_slice, cmap='viridis')  # Changed to viridis for dose visualization
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.title(f"Normalized Dose ({normalize_method}) - {mask_type} mask")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(viz_dir / f"sample_{i + 1}_patient_{patient_id}.png", dpi=300)
        plt.close()

        # Create separate histogram figure
        plt.figure(figsize=(10, 5))

        # Only include non-zero values in the histograms
        # For original data, don't mask it
        orig_values = original_slice.flatten()
        orig_values = orig_values[orig_values != 0]  # Filter zeros

        # For normalized, only look in the mask area
        norm_values = data_slice.flatten()
        norm_values = norm_values[norm_values != 0]  # Filter zeros

        # Plot histograms - check if there's any data first
        plt.subplot(1, 2, 1)
        if len(orig_values) > 0:
            plt.hist(orig_values, bins=50, alpha=0.7)
            plt.title(f"Original Dose Distribution\nPatient: {patient_id}")
            plt.xlabel("Dose Values")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No non-zero values",
                     horizontalalignment='center', verticalalignment='center')
            plt.title("Original Dose Histogram (Empty)")

        plt.subplot(1, 2, 2)
        if len(norm_values) > 0:
            plt.hist(norm_values, bins=50, alpha=0.7)
            plt.title(f"Normalized ({normalize_method}) Histogram")
            plt.xlabel("Normalized Dose Values")
            plt.ylabel("Frequency")
        else:
            plt.text(0.5, 0.5, "No non-zero values",
                     horizontalalignment='center', verticalalignment='center')
            plt.title("Normalized Histogram (Empty)")

        plt.suptitle(f"Dose Distribution Histogram for Patient {patient_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(viz_dir / f"histogram_{i + 1}_patient_{patient_id}.png", dpi=300)
        plt.close()

        # Also save a visualization of the mask itself
        plt.figure(figsize=(8, 8))
        plt.imshow(mask_slice, cmap='binary')
        plt.colorbar()
        plt.title(f"Mask ({mask_type}) - Patient {patient_id}{slice_info}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(viz_dir / f"mask_{i + 1}_patient_{patient_id}.png", dpi=300)
        plt.close()

    logger.info(f"Sample visualizations saved to {viz_dir}")


def create_dataset(config):
    """
    Create dataset from patient folders containing dose distribution images and masks.

    Args:
        config (dict): Configuration dictionary
    """
    # Get directories from config
    data_dir = Path(config['dataset']['data_dir'])

    # Debug - check if the directory exists
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    logger.info(f"Data directory: {data_dir}")

    # Debug the first few patient folders
    patient_folders = [f for f in data_dir.glob("*") if f.is_dir()]
    if not patient_folders:
        logger.error(f"No patient folders found in {data_dir}")
        return

    logger.info(f"Found {len(patient_folders)} patient folders")

    # Debug first patient
    if patient_folders:
        first_patient = patient_folders[0]
        logger.info(f"Examining first patient folder: {first_patient}")

        # Check dose file
        dose_file = first_patient / "image_ct.nrrd"  # Despite name, it's dose distribution
        if dose_file.exists():
            logger.info(f"Found dose distribution image: {dose_file}")
            debug_nrrd_file(str(dose_file))
        else:
            logger.warning(f"No dose distribution image found for first patient")

        # Check mask files
        for mask_type in ['lungctr', 'lungipsi']:
            mask_file = first_patient / f"mask_{mask_type}.nrrd"
            if mask_file.exists():
                logger.info(f"Found {mask_type} mask: {mask_file}")
                debug_nrrd_file(str(mask_file))
            else:
                logger.warning(f"No {mask_type} mask found for first patient")

    # Use output_dir from dataset if specified, otherwise use default
    if 'output_dir' in config['dataset']:
        output_dir = Path(config['dataset']['output_dir'])
    else:
        output_dir = Path(config['output']['results_dir']) / "datasets"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if we're in test mode
    test_mode = config.get('dataset', {}).get('test_mode', False) or config.get('preprocessing', {}).get('test_mode',
                                                                                                         False)
    if test_mode:
        n_test_samples = config.get('dataset', {}).get('n_test_samples', 20) or config.get('preprocessing', {}).get(
            'n_test_samples', 20)
        if len(patient_folders) > n_test_samples:
            # Randomly select n_test_samples folders
            random.seed(config.get('training', {}).get('seed', 42))
            patient_folders = random.sample(patient_folders, n_test_samples)
            logger.info(f"Test mode enabled: Selected {n_test_samples} patients randomly")

            # Update the output directory to indicate test mode
            output_dir = output_dir / f"test_{n_test_samples}"
            output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(patient_folders)} patient folders")

    # Process patient folders
    processed_data = []
    for folder in tqdm(patient_folders, desc="Processing patients"):
        items = process_patient_folder(str(folder), config)
        processed_data.extend(items)

    if not processed_data:
        logger.error("No data was successfully processed")
        return

    logger.info(f"Successfully processed {len(processed_data)} masked regions")

    # Visualize sample cases
    visualize_samples(processed_data, output_dir, num_samples=2)

    # Check if we need to extract patches or slices
    if config.get('patch_extraction', {}).get('enable', False):
        # Extract patches
        all_patches = []
        for item in tqdm(processed_data, desc="Extracting patches"):
            patches = extract_patches_from_data(item, config)
            all_patches.extend(patches)

        logger.info(f"Extracted {len(all_patches)} patches")
        output_data = all_patches
        output_format = "patches"
    elif config.get('preprocessing', {}).get('extract_slices', False):
        # Extract slices
        all_slices = []
        for item in tqdm(processed_data, desc="Extracting slices"):
            slices = extract_slices(item, config)
            all_slices.extend(slices)

        logger.info(f"Extracted {len(all_slices)} slices")
        output_data = all_slices
        output_format = "slices"
    else:
        # Use full volumes
        output_data = processed_data
        output_format = "volumes"

    # Save in batches
    batch_size = config.get('preprocessing', {}).get('batch_size', 1000)
    batch_dir = output_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Save in batches
    batches = []
    for i in range(0, len(output_data), batch_size):
        batch = output_data[i:i + batch_size]
        batch_file = batch_dir / f"batch_{i // batch_size}.pkl"
        with open(batch_file, "wb") as f:
            pickle.dump(batch, f)
        batches.append(batch_file)

    logger.info(f"Saved {len(batches)} batches")

    # Create metadata
    metadata = {
        "num_samples": len(output_data),
        "format": output_format,
        "patients": sorted(list(set(item["patient_id"] for item in output_data))),
        "mask_types": sorted(list(set(item.get("mask_type", "unknown") for item in output_data))),
        "is_2d": all(item["is_2d"] for item in output_data),
        "data_shape": output_data[0]["data"].shape if output_data else None,
        "mean_zc_value": float(np.mean([item["zc_value"] for item in output_data])),
        "min_zc_value": int(min([item["zc_value"] for item in output_data])),
        "max_zc_value": int(max([item["zc_value"] for item in output_data])),
        "test_mode": test_mode,
        "n_test_samples": n_test_samples if test_mode else None
    }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Create a single merged dataset file
    dataset_path = output_dir / f"{output_format}_dataset.pkl"
    with open(dataset_path, "wb") as f:
        pickle.dump(output_data, f)

    # Update config with dataset path
    config['dataset']['data_pkl'] = str(dataset_path)

    # If in test mode, update the configuration to indicate this
    if test_mode:
        config['preprocessing']['is_test_dataset'] = True
        config['preprocessing']['test_dataset_path'] = str(dataset_path)

    logger.info(f"Dataset created successfully in {output_dir}")
    logger.info(f"Final dataset shape: {metadata['data_shape']}")
    logger.info(f"Total samples: {metadata['num_samples']}")
    logger.info(f"Visualizations saved in {output_dir}/visualizations")

    # Return the path to the dataset
    return str(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dose distribution dataset")
    parser.add_argument("--config", type=str, default="/home/e210/git/doseae/config/config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create dataset
    dataset_path = create_dataset(config)

    # Save updated configuration
    output_config_path = os.path.join(config['output']['results_dir'], "config_with_dataset.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Updated configuration saved to {output_config_path}")