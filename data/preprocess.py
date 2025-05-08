import os
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
import logging
import random
import matplotlib.pyplot as plt
import glob

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
        logger.warning(f"No spacing information found in {dose_file}, assuming isotropic 2mm spacing")
        spacing = (2.0, 2.0, 2.0)  # Default to 2mm spacing instead of 1mm

    # Log the volume dimensions and spacing
    logger.info(f"Patient {os.path.basename(patient_folder)} - Volume shape: {dose_data.shape}, Spacing: {spacing}")

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
        enable_patches = config.get('patch_extraction', {}).get('enable', False)

        # If patch extraction is enabled, we don't want to resize the data before extracting patches
        if not enable_patches:
            # Only resize and normalize if patch extraction is not enabled
            unit_type = config.get('preprocessing', {}).get('unit_type', 'voxels')
            resize_to = config.get('preprocessing', {}).get('resize_to', None)
            should_resize = resize_to is not None and resize_to != [] and resize_to != ''

            # Only process sizing and spacing if resize_to is specified
            if should_resize:
                # Get spacing settings
                target_spacing = config.get('preprocessing', {}).get('voxel_spacing', [2.0, 2.0, 2.0])  # Default to 2mm
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
        else:
            # For patch extraction, keep original spacing
            spacing_to_use = spacing

        # Normalize if requested - always normalize even if patch extraction is enabled
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
            "was_resized": not enable_patches and should_resize,
            "normalize_method": normalize_method
        }

    except Exception as e:
        logger.error(f"Error processing masked data for {patient_id}, mask {mask_type}: {e}")
        return None


def resample_patch(patch, original_size, target_spacing, target_size):
    """
    Resample a patch to a target spacing and target size.

    Args:
        patch (numpy.ndarray): Input patch
        original_size (tuple): Original size (voxel dimensions)
        target_spacing (tuple): Target voxel spacing (mm)
        target_size (tuple): Target size (voxel dimensions)

    Returns:
        numpy.ndarray: Resampled patch
    """
    # Use SimpleITK for high-quality resampling
    import SimpleITK as sitk

    # Convert patch to SimpleITK image
    patch_sitk = sitk.GetImageFromArray(patch)

    # Set original spacing - assuming 2mm if not specified otherwise
    original_spacing = [2.0, 2.0, 2.0]  # Default
    patch_sitk.SetSpacing(original_spacing)

    # First resample to target spacing (e.g., 1mm)
    intermediate_size = [
        int(round(patch.shape[0] * original_spacing[0] / target_spacing[0])),
        int(round(patch.shape[1] * original_spacing[1] / target_spacing[1])),
        int(round(patch.shape[2] * original_spacing[2] / target_spacing[2]))
    ]

    # Create resample filter for first step
    resampler1 = sitk.ResampleImageFilter()
    resampler1.SetInterpolator(sitk.sitkLinear)
    resampler1.SetOutputSpacing(target_spacing)

    # SimpleITK expects size in (x,y,z) order but numpy uses (z,y,x)
    sitk_size = (intermediate_size[2], intermediate_size[1], intermediate_size[0])
    resampler1.SetSize(sitk_size)
    resampler1.SetOutputDirection(patch_sitk.GetDirection())
    resampler1.SetOutputOrigin(patch_sitk.GetOrigin())

    # Resample to target spacing
    intermediate_sitk = resampler1.Execute(patch_sitk)

    # If target size is different from intermediate size, resample again
    if target_size != tuple(intermediate_size):
        # Create resample filter for second step
        resampler2 = sitk.ResampleImageFilter()
        resampler2.SetInterpolator(sitk.sitkLinear)
        resampler2.SetOutputSpacing(target_spacing)

        # SimpleITK expects size in (x,y,z) order
        sitk_target_size = (target_size[2], target_size[1], target_size[0])
        resampler2.SetSize(sitk_target_size)
        resampler2.SetOutputDirection(intermediate_sitk.GetDirection())
        resampler2.SetOutputOrigin(intermediate_sitk.GetOrigin())

        # Resample to target size
        final_sitk = resampler2.Execute(intermediate_sitk)
    else:
        final_sitk = intermediate_sitk

    # Convert back to numpy array
    resampled_patch = sitk.GetArrayFromImage(final_sitk)

    return resampled_patch


def extract_patches_from_data(volume_data, config):
    """
    Extract patches from volume data based on configuration.
    """
    data = volume_data["data"]
    mask = volume_data["mask"]
    patches = []

    # Get patch extraction settings from config
    patch_dimension = config.get('patch_extraction', {}).get('patch_dimension', '3D')
    patch_size = config.get('patch_extraction', {}).get('patch_size', [50, 50, 50])
    patch_unit_type = config.get('patch_extraction', {}).get('patch_unit_type', 'mm')
    threshold = config.get('patch_extraction', {}).get('threshold', 0.01)
    max_patches = config.get('patch_extraction', {}).get('max_patches_per_volume', 500)
    random_state = config.get('patch_extraction', {}).get('random_state', 42)

    # Get spacing information
    spacing = volume_data.get("spacing", (2.0, 2.0, 2.0))  # Default to 2mm spacing if not available

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

    # Get target spacing and check if resize_to is null
    target_spacing = config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
    if isinstance(target_spacing, list):
        target_spacing = tuple(target_spacing)
    elif isinstance(target_spacing, (int, float)):
        target_spacing = (target_spacing, target_spacing, target_spacing)

    # Check if resize_to is null (to skip resampling)
    resize_to = config.get('preprocessing', {}).get('resize_to')
    should_resize = resize_to is not None

    # Format resize_to if it's specified
    if should_resize:
        if isinstance(resize_to, list):
            resize_to = tuple(resize_to)
        elif isinstance(resize_to, str):
            resize_to = tuple(map(int, resize_to.split(',')))
        elif isinstance(resize_to, (int, float)):
            resize_to = (resize_to, resize_to, resize_to)

    # Log the volume shape and resampling targets for debugging
    logger.info(f"Volume shape for patient {volume_data['patient_id']}: {data.shape}, spacing: {spacing}")
    logger.info(f"Target settings: spacing={target_spacing}, size={resize_to}")
    logger.info(f"Will resize patches: {should_resize}")

    # Handle 3D patches
    if patch_dimension == '3D' and not volume_data["is_2d"]:
        # Convert physical patch size to voxel dimensions based on original spacing
        if patch_unit_type == 'mm' and spacing:
            voxel_patch_size = (
                max(8, int(round(patch_size[0] / spacing[0]))),
                max(8, int(round(patch_size[1] / spacing[1]))),
                max(8, int(round(patch_size[2] / spacing[2])))
            )
            logger.info(f"Converting {patch_size} mm to {voxel_patch_size} voxels using spacing {spacing}")
        else:
            voxel_patch_size = patch_size

        # Check if patch size is too large for the volume and adjust if needed
        if (voxel_patch_size[0] >= data.shape[0] or
                voxel_patch_size[1] >= data.shape[1] or
                voxel_patch_size[2] >= data.shape[2]):
            # Adjust patch size to be at most 80% of the volume size
            adjusted_patch_size = (
                min(voxel_patch_size[0], max(8, int(data.shape[0] * 0.8))),
                min(voxel_patch_size[1], max(8, int(data.shape[1] * 0.8))),
                min(voxel_patch_size[2], max(8, int(data.shape[2] * 0.8)))
            )

            logger.warning(f"Adjusted patch size from {voxel_patch_size} to {adjusted_patch_size} " +
                           f"for volume of shape {data.shape}")

            voxel_patch_size = adjusted_patch_size

        try:
            # Extract patches from the original resolution data
            patches_arr = extract_patches_3d(
                data,
                patch_size=voxel_patch_size,
                max_patches=max_patches,
                threshold=threshold,
                random_state=random_state
            )

            # Process each extracted patch
            for i, patch in enumerate(patches_arr):
                # Create a mask patch (assuming the patch is already masked)
                mask_patch = np.ones_like(patch)

                # Keep a copy of the original patch
                original_patch = patch.copy()

                if should_resize:
                    # Only resample if resize_to is specified
                    try:
                        resampled_patch = resample_patch(
                            patch,
                            voxel_patch_size,
                            target_spacing,
                            resize_to
                        )
                    except Exception as e:
                        logger.error(f"Error resampling patch {i} for patient {volume_data['patient_id']}: {e}")
                        continue  # Skip this patch and continue with the next one
                else:
                    # Keep the original patch without resampling
                    resampled_patch = patch

                # Create the patch dictionary
                patch_dict = {
                    "data": resampled_patch,  # Either resampled or original patch
                    "mask": mask_patch if not should_resize else np.ones_like(resampled_patch),
                    "patient_id": volume_data["patient_id"],
                    "mask_type": volume_data["mask_type"],
                    "zc_value": np.count_nonzero(patch),
                    "is_2d": False,
                    "original_spacing": spacing,
                    "normalize_method": volume_data.get("normalize_method", "none"),
                    "patch_size_mm": patch_size if patch_unit_type == 'mm' else None,
                    "patch_size_voxels": voxel_patch_size,
                    "target_spacing": target_spacing if should_resize else spacing,
                    "resized": should_resize,
                    "final_shape": resampled_patch.shape  # Store the final shape
                }
                patches.append(patch_dict)

        except Exception as e:
            logger.error(f"Error extracting patches from volume for patient {volume_data['patient_id']}: {e}")
            # Continue with the next patient rather than failing

    # Handle 2D patch extraction
    elif patch_dimension == '2D':
        # [Code for 2D patches would go here]
        pass

    if not patches:
        logger.warning(f"No patches extracted for patient {volume_data['patient_id']}")
    else:
        # Print the shape of the first patch's data
        logger.info(f"First patch shape: {patches[0]['data'].shape}, Patient: {volume_data['patient_id']}")
        logger.info(f"Total patches extracted for patient {volume_data['patient_id']}: {len(patches)}")

    return patches


def extract_patches_3d(volume, patch_size, max_patches=None, threshold=0.01, random_state=None):
    """
    Extract patches from a 3D volume.

    Args:
        volume (numpy.ndarray): Input volume (D, H, W)
        patch_size (tuple): Size of the patches (depth, height, width)
        max_patches (int or float, optional): Maximum number of patches to extract
        threshold (float): Minimum standard deviation threshold for a patch to be considered informative
        random_state (int or RandomState, optional): Random seed or state

    Returns:
        numpy.ndarray: Extracted patches
    """
    i_d, i_h, i_w = volume.shape[:3]
    p_d, p_h, p_w = patch_size

    # Verify volume shape
    logger.debug(f"Volume shape: {volume.shape}, Patch size: {patch_size}")

    # Check if patch size is valid
    if p_d >= i_d or p_h >= i_h or p_w >= i_w:
        raise ValueError(
            f"Patch dimensions {patch_size} should be less than the corresponding volume dimensions {volume.shape}.")

    # Reshape to add a channel dimension if needed
    if volume.ndim == 3:
        volume_with_channel = volume.reshape((i_d, i_h, i_w, 1))
    else:
        volume_with_channel = volume

    n_channels = volume_with_channel.shape[3]

    # Calculate the total number of possible patches with 50% overlap
    stride_d = max(1, p_d // 2)
    stride_h = max(1, p_h // 2)
    stride_w = max(1, p_w // 2)

    d_indices = range(0, i_d - p_d + 1, stride_d)
    h_indices = range(0, i_h - p_h + 1, stride_h)
    w_indices = range(0, i_w - p_w + 1, stride_w)

    # Extract all patches
    patches = []
    for d in d_indices:
        for h in h_indices:
            for w in w_indices:
                patch = volume_with_channel[d:d + p_d, h:h + p_h, w:w + p_w, :]
                # Only keep patches with information (std > threshold)
                if np.std(patch) > threshold:
                    patches.append(patch)

    # Convert to numpy array
    if patches:
        patches = np.array(patches)

        # Reshape to remove channel dimension if it's 1
        if n_channels == 1:
            patches = patches.reshape(-1, p_d, p_h, p_w)

        # If max_patches is provided and less than the total number of patches,
        # randomly select a subset
        if max_patches and max_patches < len(patches):
            rng = np.random.RandomState(random_state)
            indices = rng.choice(len(patches), size=max_patches, replace=False)
            patches = patches[indices]
    else:
        # Return empty array with correct shape
        if n_channels == 1:
            patches = np.zeros((0, p_d, p_h, p_w))
        else:
            patches = np.zeros((0, p_d, p_h, p_w, n_channels))

    logger.info(f"Extracted {len(patches)} patches of size {patch_size}")
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
        normalize_method = sample.get("normalize_method", "none")

        # Find a safe slice index for 3D data
        if len(data.shape) == 3:
            # Make sure we don't go out of bounds for either array
            max_idx = min(data.shape[0], original_data.shape[0]) - 1
            slice_idx = min(max_idx // 2, max_idx)  # Use the middle slice or max available

            # Extract slices
            data_slice = data[slice_idx]
            mask_slice = mask[slice_idx] if slice_idx < mask.shape[0] else np.ones_like(data_slice)
            original_slice = original_data[slice_idx] if slice_idx < original_data.shape[0] else data_slice
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
    """
    # Get directories from config
    data_dir = Path(config['dataset']['data_dir'])

    # Record the pipeline steps
    pipeline_steps = []
    pipeline_steps.append("1. Load patient data from NRRD files")

    # Debug - check if the directory exists
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    logger.info(f"Data directory: {data_dir}")

    # Get patient folders
    patient_folders = [f for f in data_dir.glob("*") if f.is_dir()]
    if not patient_folders:
        logger.error(f"No patient folders found in {data_dir}")
        return

    logger.info(f"Found {len(patient_folders)} patient folders")

    # Use output_dir from dataset if specified, otherwise use default
    if 'output_dir' in config['dataset']:
        output_dir = Path(config['dataset']['output_dir'])
    else:
        output_dir = Path(config['output']['results_dir']) / "datasets"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create patients directory to store patient-specific files
    patients_dir = output_dir / "patients"
    patients_dir.mkdir(exist_ok=True, parents=True)

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
            pipeline_steps.append(f"2. Test mode: Selected {n_test_samples} patients randomly")

            # Update the output directory to indicate test mode
            output_dir = output_dir / f"test_{n_test_samples}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Update patients directory
            patients_dir = output_dir / "patients"
            patients_dir.mkdir(exist_ok=True, parents=True)

    logger.info(f"Processing {len(patient_folders)} patient folders")
    pipeline_steps.append("3. Apply mask to isolate dose distributions in regions of interest")

    # Add normalization step if enabled
    if config.get('preprocessing', {}).get('normalize', False):
        normalize_method = config.get('preprocessing', {}).get('normalize_method', '95percentile')
        pipeline_steps.append(f"4. Normalize data using {normalize_method} method")

    # Determine extraction type based on configuration
    patch_extraction = config.get('patch_extraction', {}).get('enable', False)
    extract_slices_enabled = config.get('preprocessing', {}).get('extract_slices', False)

    # Record details of what we're extracting
    total_items = 0
    if patch_extraction:
        patch_dimension = config.get('patch_extraction', {}).get('patch_dimension', '3D')
        patch_size = config.get('patch_extraction', {}).get('patch_size', [50, 50, 50])
        patch_unit_type = config.get('patch_extraction', {}).get('patch_unit_type', 'mm')
        target_spacing = config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
        target_size = config.get('preprocessing', {}).get('resize_to', [64, 64, 64])

        pipeline_steps.append(f"5. Extract {patch_dimension} patches of size {patch_size} {patch_unit_type}")
        pipeline_steps.append(f"6. Resample patches to {target_spacing}mm spacing and size {target_size} voxels")
        format_type = "patches"
    elif extract_slices_enabled:
        slice_extraction = config.get('preprocessing', {}).get('slice_extraction', 'all')
        pipeline_steps.append(f"5. Extract 2D slices using {slice_extraction} method")
        format_type = "slices"
    else:
        format_type = "volumes"
        pipeline_steps.append("5. Process full volumes")

    # List to keep track of patient files
    patient_files = []

    # Process each patient folder
    for folder_idx, folder in enumerate(tqdm(patient_folders, desc="Processing patients")):
        patient_id = folder.name

        # Process the patient folder to get masked data
        processed_items = process_patient_folder(str(folder), config)

        # Skip if no items were processed
        if not processed_items:
            continue

        # Extract patches or slices as needed
        extracted_data = []
        if patch_extraction:
            for item in processed_items:
                patches = extract_patches_from_data(item, config)
                extracted_data.extend(patches)
                total_items += len(patches)
        elif extract_slices_enabled:
            for item in processed_items:
                slices = extract_slices(item, config)
                extracted_data.extend(slices)
                total_items += len(slices)
        else:
            extracted_data = processed_items
            total_items += len(processed_items)

        # Save patient data if we have any
        if extracted_data:
            patient_file = patients_dir / f"patient_{patient_id}.pkl"
            with open(patient_file, "wb") as f:
                pickle.dump(extracted_data, f)
            patient_files.append(str(patient_file))

            # Print what we extracted
            if patch_extraction:
                logger.info(f"Patient {patient_id}: Extracted {len(extracted_data)} patches")
            elif extract_slices_enabled:
                logger.info(f"Patient {patient_id}: Extracted {len(extracted_data)} slices")
            else:
                logger.info(f"Patient {patient_id}: Processed {len(extracted_data)} volumes")

        # Force garbage collection
        import gc
        gc.collect()

    # Check if we have any data
    if not patient_files:
        logger.error("No data was successfully processed")
        return

    logger.info(f"Successfully processed {total_items} items across {len(patient_files)} patient files")

    # Sample visualization from the first patient file
    if patient_files:
        try:
            # Load just the first patient file for visualization
            with open(patient_files[0], "rb") as f:
                first_batch = pickle.load(f)

            # Visualize a few samples
            if first_batch:
                visualize_samples(first_batch[:min(2, len(first_batch))], output_dir, num_samples=2)
        except Exception as e:
            logger.error(f"Error visualizing samples: {e}")

    # Now combine all into a single dataset file for the training script
    logger.info(f"Combining all patient files into a single dataset file...")
    combined_data = []

    for patient_file in tqdm(patient_files, desc="Combining data"):
        try:
            with open(patient_file, "rb") as f:
                patient_data = pickle.load(f)
                combined_data.extend(patient_data)
        except Exception as e:
            logger.error(f"Error loading patient file {patient_file}: {e}")

    # Print final counts and shapes
    if combined_data:
        logger.info(f"Combined dataset has {len(combined_data)} items")
        logger.info(f"First item shape: {combined_data[0]['data'].shape}")

    # Save the combined dataset
    combined_dataset_path = output_dir / f"{format_type}_dataset.pkl"
    with open(combined_dataset_path, "wb") as f:
        pickle.dump(combined_data, f)

    logger.info(f"Created combined dataset with {len(combined_data)} items at {combined_dataset_path}")

    # Create metadata
    metadata = {
        "num_samples": total_items,
        "format": format_type,
        "patient_files": patient_files,
        "num_patients": len(patient_files),
        "test_mode": test_mode,
        "n_test_samples": n_test_samples if test_mode else None,
        "pipeline": pipeline_steps,
        "combined_dataset_path": str(combined_dataset_path)
    }

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    pipeline_steps.append("7. Save metadata about processed dataset")

    # Update config with dataset path to the combined file
    config['dataset']['data_pkl'] = str(combined_dataset_path)
    config['dataset']['data_patients_dir'] = str(patients_dir)

    # If in test mode, update the configuration to indicate this
    if test_mode:
        config['preprocessing']['is_test_dataset'] = True
        config['preprocessing']['test_dataset_path'] = str(combined_dataset_path)

    # Save pipeline steps to a text file
    pipeline_path = output_dir / "preprocessing_pipeline.txt"
    with open(pipeline_path, "w") as f:
        f.write("Preprocessing Pipeline Steps:\n")
        f.write("\n".join(pipeline_steps))

    logger.info(f"Dataset processing completed successfully")
    logger.info(f"Total samples: {total_items}")
    logger.info(f"Patient files saved in: {patients_dir}")
    logger.info(f"Combined dataset saved to: {combined_dataset_path}")
    logger.info(f"Preprocessing pipeline saved to: {pipeline_path}")

    # Return the path to the combined dataset
    return str(combined_dataset_path)


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