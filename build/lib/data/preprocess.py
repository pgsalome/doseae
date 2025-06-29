import os
import pickle
import json
import yaml
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import logging
import random
import matplotlib.pyplot as plt
import glob
import math
import subprocess
import sys
from os.path import dirname

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def find_rtdose_dirs(base_dir):
    """
    Find the appropriate RTDOSE directories based on the specified pattern.
    Look for RTDOSE-*-MPT first, then fallback to RTDOSE-*-MP1 if needed.

    Args:
        base_dir (str): Base directory containing TCIA subjects

    Returns:
        list: List of paths to the appropriate RTDOSE directories
    """
    # Pattern to match directories with the -MPT flag
    rtdose_pattern = os.path.join(base_dir, "*", "*", "*", "*", "RTDOSE*-MPT", "*")
    mpt_dirs = glob.glob(rtdose_pattern)

    logger.info(f"Found {len(mpt_dirs)} RTDOSE directories with -MPT flag")

    # If no MPT directories found, use MP1 as fallback
    if not mpt_dirs:
        rtdose_pattern = os.path.join(base_dir, "*", "*", "*", "*", "RTDOSE*-MP1", "*")
        mpt_dirs = glob.glob(rtdose_pattern)
        logger.info(f"Fallback: Found {len(mpt_dirs)} RTDOSE directories with -MP1 flag")

    # Get the parent directories which are the RTDOSE-* directories
    rtdose_dirs = [os.path.dirname(d) for d in mpt_dirs]

    # Remove duplicates while preserving order
    rtdose_dirs = list(dict.fromkeys(rtdose_dirs))

    logger.info(f"Final list contains {len(rtdose_dirs)} unique RTDOSE directories")
    return rtdose_dirs


def convert_dicom_to_nifti(rtdose_dir):
    """
    Convert DICOM files to NIfTI format using pycurt.

    Args:
        rtdose_dir (str): RTDOSE directory path

    Returns:
        tuple: Paths to CT and dose in CT space NIfTI files
    """
    # Import pycurtv2 here
    import pycurtv2
    from pycurtv2.converters.dicom import DicomConverter

    # Get the first directory in the RTDOSE directory (there should be only one)
    dicom_subdirs = [d for d in os.listdir(rtdose_dir) if os.path.isdir(os.path.join(rtdose_dir, d))]

    if not dicom_subdirs:
        logger.warning(f"No subdirectories found in {rtdose_dir}")
        return None, None

    # Use the first subdirectory
    dicom_dir = os.path.join(rtdose_dir, dicom_subdirs[0])

    # Find reference CT using the exact pattern with the correct number of levels
    ref_ct_pattern = dirname(dirname(dirname(dirname(rtdose_dir)))) + '/*/*/*/*CT-*MP1*/*'
    ref_ct_dirs = [x for x in glob.glob(ref_ct_pattern) if 'RTSTRUCT' not in x and os.path.isdir(x)]

    if not ref_ct_dirs:
        logger.warning(f"No reference CT directories found using pattern: {ref_ct_pattern}")
        return None, None

    # Use the first CT directory
    ct_dicom_dir = ref_ct_dirs[0]

    # Get the base names of the DICOM directories
    rtdose_base_name = os.path.basename(dicom_dir)

    # Check for existing NIfTI files in the RTDOSE directory
    rt_dir_path = os.path.dirname(dicom_dir)

    # Look for the dose files in the RTDOSE directory
    # _ct.nii.gz is dose in CT space, _dd.nii.gz is just the dose
    ct_space_dose_files = glob.glob(os.path.join(rt_dir_path, f"{rtdose_base_name}_ct.nii.gz"))

    # Also check for the actual CT file
    ct_files = glob.glob(os.path.join(os.path.dirname(ct_dicom_dir), "*.nii.gz"))
    ct_files = [f for f in ct_files if not f.endswith("_ct.nii.gz") and not f.endswith("_dd.nii.gz")]

    # Set paths based on found files
    ct_file = None
    if ct_files:
        ct_file = ct_files[0]
        logger.info(f"Found existing CT NIFTI: {ct_file}")

    dose_in_ct_space = None
    if ct_space_dose_files:
        dose_in_ct_space = ct_space_dose_files[0]
        logger.info(f"Found existing dose in CT space: {dose_in_ct_space}")
    else:
        dose_in_ct_space = os.path.join(rt_dir_path, f"{rtdose_base_name}_ct.nii.gz")

    # If both files exist, return them
    if os.path.exists(ct_file) and os.path.exists(dose_in_ct_space):
        logger.info(f"Using existing NIfTI files")
        return ct_file, dose_in_ct_space

    # If either file doesn't exist, convert from DICOM
    try:
        # Convert RTDOSE if needed
        if not os.path.exists(dose_in_ct_space):
            logger.info(f"Converting RTDOSE: {dicom_dir}")
            dd = DicomConverter(toConvert=dicom_dir)
            dd.convert_ps()

            # Check if conversion worked
            ct_space_dose_files = glob.glob(os.path.join(rt_dir_path, f"{rtdose_base_name}_ct.nii.gz"))
            if ct_space_dose_files:
                dose_in_ct_space = ct_space_dose_files[0]

        # Convert CT if needed
        if ct_file is None or not os.path.exists(ct_file):
            logger.info(f"Converting CT: {ct_dicom_dir}")
            dd = DicomConverter(toConvert=ct_dicom_dir)
            dd.convert_ps()

            # Check if conversion worked
            ct_files = glob.glob(os.path.join(os.path.dirname(ct_dicom_dir), "*.nii.gz"))
            ct_files = [f for f in ct_files if not f.endswith("_ct.nii.gz") and not f.endswith("_dd.nii.gz")]
            if ct_files:
                ct_file = ct_files[0]

        # Check if files were created or found
        if os.path.exists(ct_file) and os.path.exists(dose_in_ct_space):
            logger.info(f"Successfully located or converted NIfTI files")
            return ct_file, dose_in_ct_space
        else:
            logger.warning(f"Conversion completed but NIfTI files not found")
            return None, None

    except Exception as e:
        logger.error(f"Error converting to NIfTI: {e}")
        return None, None


def process_rtdose(rtdose_dir, output_dir, config):
    """
    Process a single RTDOSE directory. Extract patches from left and right lung separately.
    Apply lung mask to dose before patch extraction.

    Args:
        rtdose_dir (str): RTDOSE directory path
        output_dir (str): Output directory for processed data
        config (dict): Configuration dictionary

    Returns:
        list: List of processed patches
    """
    # Create subject output directory
    subject_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(rtdose_dir)))))
    subject_output_dir = os.path.join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)

    # Convert DICOM to NIfTI
    # ct_nifti is the actual CT image, dose_ct_nifti is dose in CT space (_ct.nii.gz)
    ct_nifti, dose_ct_nifti = convert_dicom_to_nifti(rtdose_dir)

    if ct_nifti is None or dose_ct_nifti is None:
        logger.warning(f"Failed to convert DICOM for {subject_id}")
        return []

    # Get base filenames without extensions for the output paths
    ct_base_name = os.path.basename(ct_nifti)
    if ct_base_name.endswith('.nii.gz'):
        ct_base_name = ct_base_name[:-7]  # Remove .nii.gz
    elif ct_base_name.endswith('.nii'):
        ct_base_name = ct_base_name[:-4]  # Remove .nii

    dose_base_name = os.path.basename(dose_ct_nifti)
    if dose_base_name.endswith('.nii.gz'):
        dose_base_name = dose_base_name[:-7]  # Remove .nii.gz
    elif dose_base_name.endswith('.nii'):
        dose_base_name = dose_base_name[:-4]  # Remove .nii

    # Set explicit output paths
    resampled_ct_path = os.path.join(subject_output_dir, f"{ct_base_name}_resampled.nii.gz")
    resampled_dose_path = os.path.join(subject_output_dir, f"{dose_base_name}_resampled.nii.gz")

    # Resample CT
    resampled_ct_image, _ = resample_to_isotropic(
        ct_nifti,
        resampled_ct_path,
        config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
    )

    # Resample the dose in CT space to isotropic spacing
    resampled_dose_image, _ = resample_to_isotropic(
        dose_ct_nifti,
        resampled_dose_path,
        config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
    )

    # Segment lungs using TotalSegmentator on the actual CT image
    left_lung_path, right_lung_path, combined_lung_path = segment_lungs_with_totalsegmentator(
        resampled_ct_path,
        os.path.join(subject_output_dir, "segmentation")
    )

    if combined_lung_path is None:
        logger.warning(f"Lung segmentation failed for {subject_id}")
        return []

    # Extract patch parameters
    target_spacing = config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0])
    if isinstance(target_spacing, list):
        target_spacing = tuple(target_spacing)

    patch_size_mm = config.get('patch_extraction', {}).get('patch_size', [50, 50, 50])
    if isinstance(patch_size_mm, list):
        patch_size_mm = tuple(patch_size_mm)

    zero_pad_to = config.get('preprocessing', {}).get('resize_to', [64, 64, 64])
    if isinstance(zero_pad_to, list):
        zero_pad_to = tuple(zero_pad_to)

    threshold = config.get('patch_extraction', {}).get('threshold', 0.01)
    min_lung_percentage = config.get('patch_extraction', {}).get('min_lung_percentage', 0.25)
    min_dose_percentage = config.get('patch_extraction', {}).get('min_dose_percentage', 0.01)
    norm_method = config.get('preprocessing', {}).get('normalize_method', '95percentile')

    # Get the new parameter for random patch visualization
    patches_per_lobe = config.get('logging', {}).get('random_patches_per_lobe', 0)

    processed_data = []
    left_lung_patches = []
    right_lung_patches = []

    # Process left lung patches if left lung segmentation is available
    if left_lung_path and os.path.exists(left_lung_path):
        logger.info(f"Processing left lung for {subject_id}")
        left_lung_mask = sitk.ReadImage(left_lung_path)

        # Create a masked dose image for left lung
        left_lung_array = sitk.GetArrayFromImage(left_lung_mask)
        resampled_dose_array = sitk.GetArrayFromImage(resampled_dose_image)

        # Apply mask directly to dose array - this is the key change
        left_masked_dose_array = resampled_dose_array.copy()
        left_masked_dose_array[left_lung_array == 0] = 0

        # Create SimpleITK image from masked array
        left_masked_dose_image = sitk.GetImageFromArray(left_masked_dose_array)
        left_masked_dose_image.CopyInformation(resampled_dose_image)

        # Save masked dose for inspection if needed
        left_masked_dose_path = os.path.join(subject_output_dir, f"{dose_base_name}_left_lung_masked.nii.gz")
        sitk.WriteImage(left_masked_dose_image, left_masked_dose_path)
        logger.info(f"Saved left lung masked dose to {left_masked_dose_path}")

        # Extract patches from left masked dose
        left_patches = extract_patches_3d(
            left_masked_dose_image,  # Using the directly masked dose
            patch_size_mm,
            target_spacing,
            zero_pad_to,
            threshold,
            min_dose_percentage
        )

        # Normalize patches
        left_normalized_patches = normalize_dose(left_patches, method=norm_method)

        # Add left lung patches to processed data
        for patch_dict in left_normalized_patches:
            patch = patch_dict['data']
            lung_percentage = patch_dict.get('lung_percentage', 1.0)  # Default to 1.0 as we masked directly
            dose_percentage = patch_dict.get('dose_percentage', 0.0)

            patch_data = {
                "data": patch,
                "patient_id": subject_id,
                "lung_side": "left",
                "zc_value": int(np.count_nonzero(patch)),
                "is_2d": False,
                "normalize_method": norm_method,
                "final_shape": list(patch.shape),
                "lung_percentage": lung_percentage,
                "dose_percentage": dose_percentage
            }

            processed_data.append(patch_data)
            left_lung_patches.append(patch_data)

        logger.info(f"Extracted {len(left_normalized_patches)} patches from left lung")

    # Process right lung patches if right lung segmentation is available
    if right_lung_path and os.path.exists(right_lung_path):
        logger.info(f"Processing right lung for {subject_id}")
        right_lung_mask = sitk.ReadImage(right_lung_path)

        # Create a masked dose image for right lung
        right_lung_array = sitk.GetArrayFromImage(right_lung_mask)
        resampled_dose_array = sitk.GetArrayFromImage(resampled_dose_image)

        # Apply mask directly to dose array - this is the key change
        right_masked_dose_array = resampled_dose_array.copy()
        right_masked_dose_array[right_lung_array == 0] = 0

        # Create SimpleITK image from masked array
        right_masked_dose_image = sitk.GetImageFromArray(right_masked_dose_array)
        right_masked_dose_image.CopyInformation(resampled_dose_image)

        # Save masked dose for inspection if needed
        right_masked_dose_path = os.path.join(subject_output_dir, f"{dose_base_name}_right_lung_masked.nii.gz")
        sitk.WriteImage(right_masked_dose_image, right_masked_dose_path)
        logger.info(f"Saved right lung masked dose to {right_masked_dose_path}")

        # Extract patches from right masked dose
        right_patches = extract_patches_3d(
            right_masked_dose_image,  # Using the directly masked dose
            patch_size_mm,
            target_spacing,
            zero_pad_to,
            threshold,
            min_dose_percentage
        )

        # Normalize patches
        right_normalized_patches = normalize_dose(right_patches, method=norm_method)

        # Add right lung patches to processed data
        for patch_dict in right_normalized_patches:
            patch = patch_dict['data']
            lung_percentage = patch_dict.get('lung_percentage', 1.0)  # Default to 1.0 as we masked directly
            dose_percentage = patch_dict.get('dose_percentage', 0.0)

            patch_data = {
                "data": patch,
                "patient_id": subject_id,
                "lung_side": "right",
                "zc_value": int(np.count_nonzero(patch)),
                "is_2d": False,
                "normalize_method": norm_method,
                "final_shape": list(patch.shape),
                "lung_percentage": lung_percentage,
                "dose_percentage": dose_percentage
            }

            processed_data.append(patch_data)
            right_lung_patches.append(patch_data)

        logger.info(f"Extracted {len(right_normalized_patches)} patches from right lung")

    # Visualize random patches per lobe if configured
    if patches_per_lobe > 0:
        visualize_random_patches_per_lobe(left_lung_patches, right_lung_patches, subject_output_dir,
                                          subject_id, patches_per_lobe)

    # Save patches for this subject
    if processed_data:
        output_path = os.path.join(subject_output_dir, f"{subject_id}_patches.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(processed_data, f)

        logger.info(f"Saved {len(processed_data)} total patches for {subject_id} to {output_path}")
    else:
        logger.warning(f"No patches were extracted for {subject_id}")

    return processed_data


def visualize_random_patches_per_lobe(left_patches, right_patches, output_dir, subject_id, num_patches=10):
    """
    Visualize random patches from each lung lobe.

    Args:
        left_patches (list): List of patches from the left lung
        right_patches (list): List of patches from the right lung
        output_dir (str): Output directory for visualizations
        subject_id (str): Subject ID
        num_patches (int): Number of random patches to visualize per lobe
    """
    # Create visualization directory
    viz_dir = os.path.join(output_dir, "patch_visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Process left lung patches
    if left_patches:
        left_viz_dir = os.path.join(viz_dir, "left_lung")
        os.makedirs(left_viz_dir, exist_ok=True)

        # Select random patches
        num_to_select = min(num_patches, len(left_patches))
        if num_to_select > 0:
            selected_indices = random.sample(range(len(left_patches)), num_to_select)

            for i, idx in enumerate(selected_indices):
                patch = left_patches[idx]["data"]
                dose_percentage = left_patches[idx].get("dose_percentage", 0)

                # Create plot for center slices
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Get center slices
                z_center = patch.shape[0] // 2
                y_center = patch.shape[1] // 2
                x_center = patch.shape[2] // 2

                # Plot slices
                im1 = axes[0].imshow(patch[z_center, :, :], cmap='viridis')
                axes[0].set_title(f"Axial (Z={z_center})")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                im2 = axes[1].imshow(patch[:, y_center, :], cmap='viridis')
                axes[1].set_title(f"Coronal (Y={y_center})")
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                im3 = axes[2].imshow(patch[:, :, x_center], cmap='viridis')
                axes[2].set_title(f"Sagittal (X={x_center})")
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                plt.suptitle(f"{subject_id} - Left Lung - Patch {i + 1} - Dose%: {dose_percentage:.4f}", fontsize=16)
                plt.tight_layout()

                # Save figure
                plt.savefig(os.path.join(left_viz_dir, f"{subject_id}_left_patch_{i + 1}.png"), dpi=300)
                plt.close()

            logger.info(f"Visualized {num_to_select} random patches from left lung for {subject_id}")

    # Process right lung patches
    if right_patches:
        right_viz_dir = os.path.join(viz_dir, "right_lung")
        os.makedirs(right_viz_dir, exist_ok=True)

        # Select random patches
        num_to_select = min(num_patches, len(right_patches))
        if num_to_select > 0:
            selected_indices = random.sample(range(len(right_patches)), num_to_select)

            for i, idx in enumerate(selected_indices):
                patch = right_patches[idx]["data"]
                dose_percentage = right_patches[idx].get("dose_percentage", 0)

                # Create plot for center slices
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))

                # Get center slices
                z_center = patch.shape[0] // 2
                y_center = patch.shape[1] // 2
                x_center = patch.shape[2] // 2

                # Plot slices
                im1 = axes[0].imshow(patch[z_center, :, :], cmap='viridis')
                axes[0].set_title(f"Axial (Z={z_center})")
                axes[0].axis('off')
                plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

                im2 = axes[1].imshow(patch[:, y_center, :], cmap='viridis')
                axes[1].set_title(f"Coronal (Y={y_center})")
                axes[1].axis('off')
                plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

                im3 = axes[2].imshow(patch[:, :, x_center], cmap='viridis')
                axes[2].set_title(f"Sagittal (X={x_center})")
                axes[2].axis('off')
                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                plt.suptitle(f"{subject_id} - Right Lung - Patch {i + 1} - Dose%: {dose_percentage:.4f}", fontsize=16)
                plt.tight_layout()

                # Save figure
                plt.savefig(os.path.join(right_viz_dir, f"{subject_id}_right_patch_{i + 1}.png"), dpi=300)
                plt.close()

            logger.info(f"Visualized {num_to_select} random patches from right lung for {subject_id}")


def extract_patches_3d(image, patch_size_mm=(50, 50, 50), voxel_spacing=(1, 1, 1),
                       zero_pad_to=(64, 64, 64), threshold=0.01, min_dose_percentage=0.001):
    """
    Extract patches of specified size in mm from a 3D volume with overlap, from regions
    where there are non-zero values (already masked with lung).

    Args:
        image (SimpleITK.Image): Input 3D image (already masked with lung)
        patch_size_mm (tuple): Patch size in mm
        voxel_spacing (tuple): Voxel spacing in mm
        zero_pad_to (tuple): Output size after zero padding
        threshold (float): Minimum standard deviation threshold
        min_dose_percentage (float): Minimum percentage of non-zero dose voxels required in a patch

    Returns:
        list: List of extracted patches as NumPy arrays
    """
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)

    # Debug: Check if image has any non-zero values
    image_nonzero = np.count_nonzero(array)
    logger.info(f"Masked dose volume shape: {array.shape}, non-zero voxels: {image_nonzero}")

    if image_nonzero == 0:
        logger.warning("No non-zero values in masked dose!")
        return []

    # Calculate patch size in voxels
    patch_size_voxels = [int(round(patch_size_mm[i] / voxel_spacing[i])) for i in range(3)]

    # Determine step size (50% overlap)
    step_size = [max(1, ps // 2) for ps in patch_size_voxels]

    # Extract patches
    patches = []
    shape = array.shape

    # Calculate ranges for patch extraction
    ranges = [range(0, shape[i] - patch_size_voxels[i] + 1, step_size[i]) for i in range(3)]

    total_potential_patches = len(ranges[0]) * len(ranges[1]) * len(ranges[2])
    logger.info(f"Total potential patches to evaluate: {total_potential_patches}")

    # Track reasons for rejection
    rejected_reasons = {
        "dose_percentage": 0,
        "std_threshold": 0
    }

    for z in ranges[0]:
        for y in ranges[1]:
            for x in ranges[2]:
                # Extract image patch
                image_patch = array[
                              z:z + patch_size_voxels[0],
                              y:y + patch_size_voxels[1],
                              x:x + patch_size_voxels[2]
                              ]

                # Calculate percentage of non-zero dose voxels in the patch
                dose_percentage = np.count_nonzero(image_patch) / image_patch.size

                # Skip patches with too few dose values
                if dose_percentage < min_dose_percentage:
                    rejected_reasons["dose_percentage"] += 1
                    continue

                # Check if patch has information (non-zero standard deviation)
                if np.std(image_patch) > threshold:
                    # Zero pad to target size if needed
                    if image_patch.shape != zero_pad_to:
                        padded_patch = np.zeros(zero_pad_to, dtype=image_patch.dtype)

                        # Calculate padding
                        z_pad = (zero_pad_to[0] - image_patch.shape[0]) // 2
                        y_pad = (zero_pad_to[1] - image_patch.shape[1]) // 2
                        x_pad = (zero_pad_to[2] - image_patch.shape[2]) // 2

                        # Copy patch to center of padded array
                        padded_patch[
                        z_pad:z_pad + image_patch.shape[0],
                        y_pad:y_pad + image_patch.shape[1],
                        x_pad:x_pad + image_patch.shape[2]
                        ] = image_patch

                        image_patch = padded_patch

                    patches.append({
                        'patch': image_patch,
                        'lung_percentage': 1.0,  # Always 1.0 since we're only using lung regions
                        'dose_percentage': dose_percentage
                    })
                else:
                    rejected_reasons["std_threshold"] += 1

    logger.info(f"Patch extraction summary:")
    logger.info(f"  - Total potential patches: {total_potential_patches}")
    logger.info(f"  - Rejected due to low dose percentage: {rejected_reasons['dose_percentage']}")
    logger.info(f"  - Rejected due to low standard deviation: {rejected_reasons['std_threshold']}")
    logger.info(f"  - Accepted patches: {len(patches)}")

    logger.info(
        f"Extracted {len(patches)} patches of size {patch_size_mm}mm (padded to {zero_pad_to} voxels)")
    return patches


def resample_to_isotropic(nifti_path, output_path=None, target_spacing=(1.0, 1.0, 1.0)):
    """
    Resample a NIfTI volume to isotropic voxel spacing.

    Args:
        nifti_path (str): Path to input NIfTI file
        output_path (str, optional): Path to save resampled volume
        target_spacing (tuple): Target voxel spacing in mm

    Returns:
        SimpleITK.Image: Resampled image and path
    """
    if output_path is None:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(nifti_path))[0]
        # Handle double extensions (.nii.gz)
        if base_name.endswith('.nii'):
            base_name = os.path.splitext(base_name)[0]
        # Create output path in the same directory
        output_dir = os.path.dirname(nifti_path)
        output_path = os.path.join(output_dir, f"{base_name}_resampled.nii.gz")

    # Load image with SimpleITK
    image = sitk.ReadImage(nifti_path)

    # Get original spacing
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # Calculate new size
    new_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(3)]

    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    # Use linear interpolation for continuous values like dose
    resampler.SetInterpolator(sitk.sitkLinear)

    # Resample
    resampled_image = resampler.Execute(image)

    # Save resampled image
    sitk.WriteImage(resampled_image, output_path)

    logger.info(f"Resampled {nifti_path} to spacing {target_spacing}")
    return resampled_image, output_path


def segment_lungs_with_totalsegmentator(ct_nifti_path, output_dir):
    """
    Segment lung lobes using TotalSegmentator and create separate masks for left and right lungs.

    Args:
        ct_nifti_path (str): Path to CT NIfTI file
        output_dir (str): Output directory for segmentation results

    Returns:
        tuple: Paths to left lung, right lung, and combined lung segmentation masks
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define output paths
    left_lung_path = os.path.join(output_dir, "lung_left.nii.gz")
    right_lung_path = os.path.join(output_dir, "lung_right.nii.gz")
    combined_lung_path = os.path.join(output_dir, "lung_segmentation.nii.gz")

    # Check if segmentation already exists
    if os.path.exists(left_lung_path) and os.path.exists(right_lung_path) and os.path.exists(combined_lung_path):
        logger.info(f"Lung segmentations already exist")
        return left_lung_path, right_lung_path, combined_lung_path

    # Run TotalSegmentator to segment lung lobes
    try:
        logger.info(f"Running TotalSegmentator on {ct_nifti_path}")
        subprocess.run([
            "TotalSegmentator",
            "-i", ct_nifti_path,
            "-o", output_dir,
            "--fast",  # Use fast mode for efficiency
            "--roi_subset", "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"
        ], check=True)

        # Define left and right lobe files
        left_lobe_files = [
            os.path.join(output_dir, "lung_upper_lobe_left.nii.gz"),
            os.path.join(output_dir, "lung_lower_lobe_left.nii.gz")
        ]

        right_lobe_files = [
            os.path.join(output_dir, "lung_upper_lobe_right.nii.gz"),
            os.path.join(output_dir, "lung_middle_lobe_right.nii.gz"),
            os.path.join(output_dir, "lung_lower_lobe_right.nii.gz")
        ]

        # Check which files exist
        existing_left_files = [f for f in left_lobe_files if os.path.exists(f)]
        existing_right_files = [f for f in right_lobe_files if os.path.exists(f)]

        # Log found files
        for f in left_lobe_files + right_lobe_files:
            if os.path.exists(f):
                logger.info(f"Found: {f}")
            else:
                logger.warning(f"Missing: {f}")

        # Make sure we have at least some files
        if not existing_left_files and not existing_right_files:
            logger.error("No lung segmentations were created")
            return None, None, None

        # Get a reference image for metadata
        reference_image = None
        for f in existing_left_files + existing_right_files:
            reference_image = sitk.ReadImage(f)
            break

        if reference_image is None:
            logger.error("No valid reference image found")
            return None, None, None

        # Create and save left lung mask
        if existing_left_files:
            # Combine all left lobe masks
            left_lung_array = None
            for lobe_file in existing_left_files:
                lobe_image = sitk.ReadImage(lobe_file)
                lobe_array = sitk.GetArrayFromImage(lobe_image) > 0

                if left_lung_array is None:
                    left_lung_array = lobe_array
                else:
                    left_lung_array = left_lung_array | lobe_array

            # Save left lung mask
            left_lung_image = sitk.GetImageFromArray(left_lung_array.astype(np.uint8))
            left_lung_image.CopyInformation(reference_image)
            sitk.WriteImage(left_lung_image, left_lung_path)
            logger.info(f"Created left lung segmentation at {left_lung_path}")
        else:
            left_lung_array = None
            left_lung_path = None
            logger.warning("No left lung segmentation was created")

        # Create and save right lung mask
        if existing_right_files:
            # Combine all right lobe masks
            right_lung_array = None
            for lobe_file in existing_right_files:
                lobe_image = sitk.ReadImage(lobe_file)
                lobe_array = sitk.GetArrayFromImage(lobe_image) > 0

                if right_lung_array is None:
                    right_lung_array = lobe_array
                else:
                    right_lung_array = right_lung_array | lobe_array

            # Save right lung mask
            right_lung_image = sitk.GetImageFromArray(right_lung_array.astype(np.uint8))
            right_lung_image.CopyInformation(reference_image)
            sitk.WriteImage(right_lung_image, right_lung_path)
            logger.info(f"Created right lung segmentation at {right_lung_path}")
        else:
            right_lung_array = None
            right_lung_path = None
            logger.warning("No right lung segmentation was created")

        # Create and save combined lung mask
        combined_lung_array = None
        if left_lung_array is not None and right_lung_array is not None:
            combined_lung_array = left_lung_array | right_lung_array
        elif left_lung_array is not None:
            combined_lung_array = left_lung_array
        elif right_lung_array is not None:
            combined_lung_array = right_lung_array

        if combined_lung_array is not None:
            combined_lung_image = sitk.GetImageFromArray(combined_lung_array.astype(np.uint8))
            combined_lung_image.CopyInformation(reference_image)
            sitk.WriteImage(combined_lung_image, combined_lung_path)
            logger.info(f"Created combined lung segmentation at {combined_lung_path}")
        else:
            combined_lung_path = None
            logger.error("Failed to create any lung masks")

        # Return the paths to the segmentations
        return left_lung_path, right_lung_path, combined_lung_path

    except Exception as e:
        logger.error(f"Error running TotalSegmentator: {e}")
        return None, None, None


def normalize_dose(patches, method='95percentile'):
    """
    Normalize dose patches.

    Args:
        patches (list): List of patch dictionaries
        method (str): Normalization method ('95percentile', 'minmax', or 'zscore')

    Returns:
        list: List of normalized patches
    """
    normalized_patches = []

    for patch_dict in patches:
        patch = patch_dict['patch']
        lung_percentage = patch_dict.get('lung_percentage', 1.0)
        dose_percentage = patch_dict.get('dose_percentage', 0.0)

        # Skip empty patches (no non-zero values)
        if np.count_nonzero(patch) == 0:
            logger.debug("Skipping patch with no dose values")
            continue

        if method == '95percentile':
            # Normalize to 95th percentile
            nonzero_values = patch[patch > 0]
            if len(nonzero_values) > 0:
                percentile_95 = np.percentile(nonzero_values, 95)
                if percentile_95 > 0:
                    normalized = patch / percentile_95
                else:
                    logger.debug("Skipping patch with zero 95th percentile")
                    continue
            else:
                logger.debug("Skipping patch with no positive values")
                continue

        elif method == 'minmax':
            # Min-max normalization
            min_val = np.min(patch)
            max_val = np.max(patch)
            if max_val > min_val:
                normalized = (patch - min_val) / (max_val - min_val)
            else:
                logger.debug("Skipping patch with equal min and max values")
                continue

        elif method == 'zscore':
            # Z-score normalization
            nonzero_values = patch[patch > 0]
            if len(nonzero_values) > 0:
                mean = np.mean(nonzero_values)
                std = np.std(nonzero_values)
                if std > 0:
                    normalized = (patch - mean) / std
                else:
                    logger.debug("Skipping patch with zero standard deviation")
                    continue
            else:
                logger.debug("Skipping patch with no positive values")
                continue
        else:
            normalized = patch

        normalized_patches.append({
            'data': normalized,  # Key changed to 'data' to match dataloader expectations
            'lung_percentage': lung_percentage,
            'dose_percentage': dose_percentage
        })

    logger.info(f"Normalized {len(normalized_patches)} patches using {method} method")
    return normalized_patches


def combine_patches(all_patches, output_dir):
    """
    Combine all patches into a single dataset file.

    Args:
        all_patches (list): List of all patch dictionaries
        output_dir (str): Output directory

    Returns:
        str: Path to the combined dataset
    """
    combined_path = os.path.join(output_dir, "patches_dataset.pkl")

    # Save in batches to manage memory
    batch_size = 5000
    total_batches = math.ceil(len(all_patches) / batch_size)

    # Initialize with empty list
    with open(combined_path, "wb") as f:
        pickle.dump([], f)

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(all_patches))

        batch = all_patches[start_idx:end_idx]

        # Read current data
        with open(combined_path, "rb") as f:
            try:
                combined_data = pickle.load(f)
            except:
                combined_data = []

        # Append batch
        combined_data.extend(batch)

        # Write updated data
        with open(combined_path, "wb") as f:
            pickle.dump(combined_data, f)

        logger.info(f"Written batch {batch_idx + 1}/{total_batches}, total items: {len(combined_data)}")

        # Force garbage collection
        combined_data = None
        batch = None
        import gc
        gc.collect()

    logger.info(f"Combined dataset saved to {combined_path}")
    return combined_path


def visualize_samples(patches, output_dir, num_samples=5):
    """
    Visualize sample patches.

    Args:
        patches (list): List of patch dictionaries
        output_dir (str): Output directory
        num_samples (int): Number of samples to visualize
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Select random samples
    if len(patches) > num_samples:
        indices = random.sample(range(len(patches)), num_samples)
    else:
        indices = range(min(len(patches), num_samples))

    for i, idx in enumerate(indices):
        # Key is now 'data' to match dataloader expectations
        patch = patches[idx]["data"]
        patient_id = patches[idx]["patient_id"]
        lung_side = patches[idx].get("lung_side", "unknown")
        lung_percentage = patches[idx].get("lung_percentage", "N/A")
        dose_percentage = patches[idx].get("dose_percentage", "N/A")

        # Create a figure for center slices along each axis
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get center slices
        z_center = patch.shape[0] // 2
        y_center = patch.shape[1] // 2
        x_center = patch.shape[2] // 2

        # Plot slices
        im1 = axes[0].imshow(patch[z_center, :, :], cmap='viridis')
        axes[0].set_title(f"Axial (Z={z_center})")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(patch[:, y_center, :], cmap='viridis')
        axes[1].set_title(f"Coronal (Y={y_center})")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        im3 = axes[2].imshow(patch[:, :, x_center], cmap='viridis')
        axes[2].set_title(f"Sagittal (X={x_center})")
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        if isinstance(lung_percentage, float) and isinstance(dose_percentage, float):
            plt.suptitle(
                f"Patient {patient_id} - {lung_side.capitalize()} Lung - Patch {idx} - Lung %: {lung_percentage:.2f} - Dose %: {dose_percentage:.2f}",
                fontsize=16)
        else:
            plt.suptitle(f"Patient {patient_id} - {lung_side.capitalize()} Lung - Patch {idx}", fontsize=16)
        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(viz_dir, f"patch_{i + 1}_patient_{patient_id}_{lung_side}.png"), dpi=300)
        plt.close()

    logger.info(f"Sample visualizations saved to {viz_dir}")


def create_dataset(config):
    """
    Create dataset from RTDOSE DICOM data.

    Args:
        config (dict): Configuration dictionary

    Returns:
        str: Path to the combined dataset
    """
    # Get base directory from config
    base_dir = config['dataset']['data_dir']

    # Set output directory
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Find all RTDOSE directories
    rtdose_dirs = find_rtdose_dirs(base_dir)

    if not rtdose_dirs:
        logger.error(f"No RTDOSE directories found in {base_dir}")
        return None

    # Limit subjects in test mode if specified in config
    test_mode = config.get('preprocessing', {}).get('test_mode', False) or config.get('dataset', {}).get('test_mode',
                                                                                                         False)
    if test_mode:
        num_subjects = config.get('preprocessing', {}).get('n_test_samples', 5) or config.get('dataset', {}).get(
            'n_test_samples', 5)
        if len(rtdose_dirs) > num_subjects:
            # Set seed for reproducibility
            random.seed(config.get('training', {}).get('seed', 42))
            rtdose_dirs = random.sample(rtdose_dirs, num_subjects)
            logger.info(f"Test mode: processing {len(rtdose_dirs)} subjects")

    # Process each RTDOSE directory
    all_patches = []
    for rtdose_dir in tqdm(rtdose_dirs, desc="Processing RTDOSE directories"):
        patches = process_rtdose(rtdose_dir, output_dir, config)
        all_patches.extend(patches)

    # Combine all patches
    if all_patches:
        combined_path = combine_patches(all_patches, output_dir)

        # Verify dataset format is compatible with dataloader
        with open(combined_path, 'rb') as f:
            data_sample = pickle.load(f)
            if len(data_sample) > 0:
                sample = data_sample[0]
                logger.info(f"Sample data keys: {list(sample.keys())}")
                logger.info(f"Sample data shape: {sample['data'].shape}")
                logger.info(f"Sample zc_value: {sample['zc_value']}")

                # Verify format compatibility with dataloader
                if 'data' in sample and 'zc_value' in sample:
                    logger.info("Dataset format is compatible with dataloader")
                else:
                    logger.warning("Dataset format may not be compatible with dataloader!")

        # Update config with dataset path
        config['dataset']['data_pkl'] = combined_path

        # Visualize some samples
        num_samples_to_visualize = config.get('logging', {}).get('num_samples_to_log', 10)
        visualize_samples(all_patches[:num_samples_to_visualize], output_dir, num_samples=num_samples_to_visualize)

        logger.info(f"Created dataset with {len(all_patches)} patches")
        return combined_path
    else:
        logger.warning("No patches were extracted from any RTDOSE directories")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dose distribution dataset")
    parser.add_argument("--config", type=str, default="/home/patrick/projects/git/doseae/config/config.yaml",
                        help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create dataset
    dataset_path = create_dataset(config)

    # Save updated configuration
    if dataset_path:
        output_config_path = os.path.join(config['output']['results_dir'], "config_with_dataset.json")
        with open(output_config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Updated configuration saved to {output_config_path}")