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

# Import your existing functionality
from data.transforms import NormalizeTo95PercentDose, ZscoreNormalization, ToTensor
from data.preprocess import normalize_tensor, min_max_normalize
from utils.patch_extraction import extract_informative_patches_2d, extract_patches_3d
from utils.utils import ensure_dir

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


def process_file(file_path, config):
    """
    Process a single NRRD dose distribution file.

    Args:
        file_path (str): Path to file
        config (dict): Configuration dictionary

    Returns:
        dict: Processed data or None if processing failed
    """
    try:
        data, header = load_nrrd_file(file_path)
        if data is None:
            return None

        # Get spacing information from header
        if 'spacing' in header:
            original_spacing = header['spacing']
        else:
            logger.warning(f"No spacing information found in {file_path}, assuming isotropic 1mm spacing")
            original_spacing = (1.0, 1.0, 1.0)

        # Extract metadata
        patient_id = Path(file_path).stem.split('_')[0] if '_' in Path(file_path).stem else "unknown"
        region = os.path.basename(os.path.dirname(file_path))

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
                    middle_slice_idx = data.shape[0] // 2
                    data = data[middle_slice_idx]

                # Reshape 2D data if needed
                if data.shape != target_size:
                    if unit_type == 'mm':
                        # Calculate voxel-based size from physical size
                        voxel_size = [int(target_size[i] / target_spacing[i]) for i in range(2)]
                        # Create SimpleITK image and resample
                        sitk_image = sitk.GetImageFromArray(data)
                        sitk_image.SetSpacing(target_spacing[:2])

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampler.SetSize(voxel_size)
                        resampler.SetOutputDirection(sitk_image.GetDirection())
                        resampler.SetOutputOrigin(sitk_image.GetOrigin())
                        resampler.SetOutputSpacing(target_spacing[:2])

                        resampled_image = resampler.Execute(sitk_image)
                        data = sitk.GetArrayFromImage(resampled_image)
                    else:
                        # Direct voxel-based resize
                        sitk_image = sitk.GetImageFromArray(data)

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampler.SetSize(target_size)
                        resampler.SetOutputDirection(sitk_image.GetDirection())
                        resampler.SetOutputOrigin(sitk_image.GetOrigin())

                        # Calculate new spacing
                        new_spacing = [data.shape[i] * original_spacing[i] / target_size[i] for i in range(2)]
                        resampler.SetOutputSpacing(new_spacing)

                        resampled_image = resampler.Execute(sitk_image)
                        data = sitk.GetArrayFromImage(resampled_image)
            else:  # 3D processing
                # Ensure data is 3D
                if len(data.shape) == 2:
                    # Convert 2D to 3D by adding a dimension
                    data = data.reshape(1, *data.shape)

                # Resample to target spacing or size
                if data.shape != target_size:
                    if unit_type == 'mm':
                        # Calculate voxel-based size from physical size
                        voxel_size = [int(target_size[i] / target_spacing[i]) for i in range(3)]
                        data = resample_volume(data, original_spacing, target_spacing)

                        # If the resampled volume doesn't match the target voxel size, resize it
                        if data.shape != tuple(voxel_size):
                            sitk_image = sitk.GetImageFromArray(data)
                            sitk_image.SetSpacing(target_spacing)

                            resampler = sitk.ResampleImageFilter()
                            resampler.SetInterpolator(sitk.sitkLinear)
                            resampler.SetSize(voxel_size)
                            resampler.SetOutputDirection(sitk_image.GetDirection())
                            resampler.SetOutputOrigin(sitk_image.GetOrigin())
                            resampler.SetOutputSpacing(target_spacing)

                            resampled_image = resampler.Execute(sitk_image)
                            data = sitk.GetArrayFromImage(resampled_image)
                    else:
                        # Direct voxel-based resize
                        sitk_image = sitk.GetImageFromArray(data)

                        resampler = sitk.ResampleImageFilter()
                        resampler.SetInterpolator(sitk.sitkLinear)
                        resampler.SetSize(target_size[::-1])  # SimpleITK uses XYZ order
                        resampler.SetOutputDirection(sitk_image.GetDirection())
                        resampler.SetOutputOrigin(sitk_image.GetOrigin())

                        # Calculate new spacing
                        new_spacing = [data.shape[i] * original_spacing[i] / target_size[i] for i in range(3)]
                        resampler.SetOutputSpacing(new_spacing)

                        resampled_image = resampler.Execute(sitk_image)
                        data = sitk.GetArrayFromImage(resampled_image)

            # Use the target spacing for the processed data if using mm
            if unit_type == 'mm':
                spacing_to_use = target_spacing
            else:
                # If using voxels, update the spacing to reflect the new voxel sizes
                spacing_to_use = original_spacing
        else:
            # If not resizing, keep original spacing
            spacing_to_use = original_spacing

        # Normalize if requested
        if config.get('preprocessing', {}).get('normalize', False):
            # Check if prescription dose is provided
            prescription_dose = None
            if 'prescription_dose' in config.get('preprocessing', {}):
                prescription_doses = config['preprocessing']['prescription_dose']
                if isinstance(prescription_doses, (int, float)):
                    # Single dose for all data
                    prescription_dose = prescription_doses
                elif isinstance(prescription_doses, dict) and patient_id in prescription_doses:
                    # Patient-specific doses from dictionary
                    prescription_dose = prescription_doses[patient_id]

            normalize_method = config.get('preprocessing', {}).get('normalize_method', '95percentile')

            if normalize_method == "95percentile":
                percentile_val = np.percentile(data, config.get('preprocessing', {}).get('percentile_norm', 95))
                if percentile_val > 0:
                    data = data / percentile_val
                    # If prescription dose is provided, rescale to make 100% = prescription dose
                    if prescription_dose is not None:
                        data = data * prescription_dose / 100.0
            elif normalize_method == "prescription" and prescription_dose is not None:
                # Normalize directly to prescription dose
                max_val = np.max(data)
                if max_val > 0:
                    data = data * prescription_dose / max_val
            elif normalize_method == "minmax":
                data = min_max_normalize(torch.from_numpy(data)).numpy()
            elif normalize_method == "zscore":
                data = normalize_tensor(torch.from_numpy(data)).numpy()

        # Calculate zero count
        zc_value = np.count_nonzero(data)

        return {
            "data": data,
            "metadata": header,
            "patient_id": patient_id,
            "region": region,
            "zc_value": zc_value,
            "is_2d": len(data.shape) == 2,
            "file_path": file_path,
            "spacing": spacing_to_use,
            "prescription_dose": prescription_dose if 'prescription_dose' in locals() else None,
            "was_resized": should_resize
        }

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
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

            for patch in patches_arr:
                patch_dict = {
                    "data": patch,
                    "patient_id": volume_data["patient_id"],
                    "region": volume_data["region"],
                    "zc_value": np.count_nonzero(patch),
                    "is_2d": True,
                    "original_file": volume_data["file_path"]
                }
                patches.append(patch_dict)
        else:
            # Extract 2D patches from 3D data (slice first)
            for i in range(data.shape[0]):
                slice_data = data[i]

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

                for patch in slice_patches:
                    patch_dict = {
                        "data": patch,
                        "patient_id": volume_data["patient_id"],
                        "region": volume_data["region"],
                        "zc_value": np.count_nonzero(patch),
                        "is_2d": True,
                        "slice_idx": i,
                        "original_file": volume_data["file_path"]
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

            for patch in patches_arr:
                patch_dict = {
                    "data": patch,
                    "patient_id": volume_data["patient_id"],
                    "region": volume_data["region"],
                    "zc_value": np.count_nonzero(patch),
                    "is_2d": False,
                    "original_file": volume_data["file_path"]
                }
                patches.append(patch_dict)
        else:
            logger.warning(f"Cannot extract 3D patches from 2D data: {volume_data['file_path']}")

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

    # If already 2D, return as is
    if volume_data["is_2d"]:
        return [volume_data]

    slices = []
    slice_extraction = config.get('preprocessing', {}).get('slice_extraction', 'all')

    # Extract slices based on strategy
    if slice_extraction == "all":
        for i in range(data.shape[0]):
            slice_data = data[i]
            if np.any(slice_data):  # Skip empty slices
                slice_dict = {
                    "data": slice_data,
                    "patient_id": volume_data["patient_id"],
                    "region": volume_data["region"],
                    "zc_value": np.count_nonzero(slice_data),
                    "is_2d": True,
                    "slice_idx": i,
                    "original_file": volume_data["file_path"]
                }
                slices.append(slice_dict)
    elif slice_extraction == "center":
        i = data.shape[0] // 2
        slice_data = data[i]
        if np.any(slice_data):
            slice_dict = {
                "data": slice_data,
                "patient_id": volume_data["patient_id"],
                "region": volume_data["region"],
                "zc_value": np.count_nonzero(slice_data),
                "is_2d": True,
                "slice_idx": i,
                "original_file": volume_data["file_path"]
            }
            slices.append(slice_dict)
    elif slice_extraction == "informative":
        threshold = config.get('preprocessing', {}).get('non_zero_threshold', 0.05)
        for i in range(data.shape[0]):
            slice_data = data[i]
            if np.count_nonzero(slice_data) / slice_data.size > threshold:
                slice_dict = {
                    "data": slice_data,
                    "patient_id": volume_data["patient_id"],
                    "region": volume_data["region"],
                    "zc_value": np.count_nonzero(slice_data),
                    "is_2d": True,
                    "slice_idx": i,
                    "original_file": volume_data["file_path"]
                }
                slices.append(slice_dict)

    return slices


def create_dataset(config):
    """
    Create dataset from dose distribution files.

    Args:
        config (dict): Configuration dictionary
    """
    # Get directories from config
    data_dir = Path(config['dataset']['data_dir'])
    output_dir = Path(config['output']['results_dir']) / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of NRRD files
    file_list = list(data_dir.glob("**/*.nrrd"))

    if not file_list:
        logger.error(f"No NRRD files found in {data_dir}")
        return

    # Check if we're in test mode
    test_mode = config.get('preprocessing', {}).get('test_mode', False)
    if test_mode:
        n_test_samples = config.get('preprocessing', {}).get('n_test_samples', 20)
        if len(file_list) > n_test_samples:
            # Randomly select n_test_samples files
            random.seed(config.get('training', {}).get('seed', 42))
            file_list = random.sample(file_list, n_test_samples)
            logger.info(f"Test mode enabled: Selected {n_test_samples} files randomly")

            # Update the output directory to indicate test mode
            output_dir = output_dir / f"test_{n_test_samples}"
            output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(file_list)} NRRD files")

    # Process files in parallel
    num_workers = config.get('preprocessing', {}).get('num_workers', 8)
    processed_data = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, str(file_path), config): file_path for file_path in file_list}
        for future in tqdm(as_completed(future_to_file), total=len(file_list), desc="Processing files"):
            file_path = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    processed_data.append(result)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")

    if not processed_data:
        logger.error("No files were successfully processed")
        return

    logger.info(f"Successfully processed {len(processed_data)} files")

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
        "regions": sorted(list(set(item["region"] for item in output_data))),
        "patients": sorted(list(set(item["patient_id"] for item in output_data))),
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

    # Return the path to the dataset
    return str(dataset_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dose distribution dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
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