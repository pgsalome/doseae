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
import time
import gc
from os.path import dirname, join, basename, splitext
import h5py
import re

# It's good practice to handle potential import errors for optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

# Assuming pycurtv2 is in the environment. If not, this will raise an ImportError.
try:
    from pycurtv2.converters.dicom import DicomConverter
    PYCURT_AVAILABLE = True
except ImportError:
    PYCURT_AVAILABLE = False
    DicomConverter = None

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
PROBLEMATIC_PATIENTS = []
ZERO_DOSE_PATIENTS = []  # Will be populated from file if it exists


# --- Core Functions (Refactored for Memory Efficiency) ---

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_file_extension(config):
    """Get file extension from config."""
    preproc_config = config.get('preprocessing', {})
    file_format = preproc_config.get('file_format', 'nifti')
    if file_format.lower() == 'nrrd':
        return preproc_config.get('nrrd_extension', '.nrrd')
    return preproc_config.get('nifti_extension', '.nii.gz')


def get_subject_id_from_path(path, use_pycurt_format=True):
    """Extracts a unique subject ID from a given directory path."""
    if use_pycurt_format:
        return basename((dirname(dirname(dirname(dirname(dirname(path)))))))
    else:
        # For simplified format, patient ID is the directory name itself
        return basename(path.rstrip('/'))


def load_zero_dose_patients(output_dir):
    """Load list of patients with zero doses from file."""
    global ZERO_DOSE_PATIENTS
    zero_dose_file = join(output_dir, "zero_dose_patients.json")

    if os.path.exists(zero_dose_file):
        try:
            with open(zero_dose_file, 'r') as f:
                ZERO_DOSE_PATIENTS = json.load(f)
            logger.info(f"Loaded {len(ZERO_DOSE_PATIENTS)} patients with zero doses from {zero_dose_file}")
        except Exception as e:
            logger.error(f"Failed to load zero dose patients file: {e}")
            ZERO_DOSE_PATIENTS = []
    else:
        ZERO_DOSE_PATIENTS = []


def save_zero_dose_patients(output_dir):
    """Save list of patients with zero doses to file."""
    zero_dose_file = join(output_dir, "zero_dose_patients.json")

    try:
        with open(zero_dose_file, 'w') as f:
            json.dump(ZERO_DOSE_PATIENTS, f, indent=4)
        logger.info(f"Saved {len(ZERO_DOSE_PATIENTS)} zero dose patients to {zero_dose_file}")
    except Exception as e:
        logger.error(f"Failed to save zero dose patients file: {e}")


def should_skip_patient(subject_id):
    """Check if a patient should be skipped due to known issues or zero doses."""
    if subject_id in PROBLEMATIC_PATIENTS:
        logger.warning(f"Skipping known problematic patient: {subject_id}")
        return True
    if subject_id in ZERO_DOSE_PATIENTS:
        logger.warning(f"Skipping patient with zero doses: {subject_id}")
        return True
    return False


def is_patient_processable(ct_dir_or_file, rtdose_dir_or_file, max_size_gb=3.0, use_pycurt_format=True):
    """Check if a patient's data is within a reasonable size limit."""
    try:
        if use_pycurt_format:
            # Original logic for directories
            subject_id = get_subject_id_from_path(rtdose_dir_or_file, use_pycurt_format)
            ct_size = sum(f.stat().st_size for f in Path(ct_dir_or_file).rglob('*') if f.is_file())
            rtdose_size = sum(f.stat().st_size for f in Path(rtdose_dir_or_file).rglob('*') if f.is_file())
        else:
            # Simplified format for files
            subject_id = get_subject_id_from_path(dirname(ct_dir_or_file), use_pycurt_format)
            ct_size = os.path.getsize(ct_dir_or_file) if os.path.exists(ct_dir_or_file) else 0
            rtdose_size = os.path.getsize(rtdose_dir_or_file) if os.path.exists(rtdose_dir_or_file) else 0

        total_size_gb = (ct_size + rtdose_size) / (1024 ** 3)

        logger.info(f"Patient {subject_id} data size: {total_size_gb:.2f} GB")
        if total_size_gb > max_size_gb:
            logger.warning(
                f"Skipping patient {subject_id} with data size {total_size_gb:.2f} GB (exceeds limit of {max_size_gb} GB)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking patient processability: {e}")
        return False


def check_memory(mem_threshold_gb=2.0, swap_threshold_pct=80.0):
    """Monitors system memory and swap usage, waiting if thresholds are exceeded."""
    if not psutil:
        logger.warning("psutil not installed, cannot monitor memory.")
        return True

    try:
        # Check memory multiple times to ensure stability
        for i in range(3):
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            mem_available_gb = mem.available / (1024 ** 3)

            logger.info(f"Memory: {mem_available_gb:.2f}GB available, Swap: {swap.percent}% used")

            if mem_available_gb < mem_threshold_gb or swap.percent > swap_threshold_pct:
                wait_time = 30 + (i * 15)  # Increase wait time on subsequent failures
                logger.warning(
                    f"Low memory detected (Available: {mem_available_gb:.2f}GB, Swap: {swap.percent}%). "
                    f"Waiting for {wait_time}s to allow system recovery."
                )
                gc.collect()
                time.sleep(wait_time)
            else:
                return True  # Memory is sufficient

        logger.error(
            "Memory and/or swap space remains critically low after multiple checks. Skipping current operation.")
        return False

    except Exception as e:
        logger.error(f"Error monitoring memory: {e}")
        return True  # Fail safe


def run_dicom_conversion(dicom_dir, convert_to='nifti_gz'):
    """Runs DICOM conversion for a given directory."""
    if not PYCURT_AVAILABLE:
        logger.error("pycurtv2 is not available, cannot perform DICOM conversion")
        return False

    logger.info(f"Converting DICOMs in {dicom_dir} to {convert_to}...")
    converter = DicomConverter(toConvert=dicom_dir, convert_to=convert_to)
    converter.convert_ps()
    del converter  # Explicitly delete the object
    gc.collect()
    return True


def resample_image(input_path, output_path, target_spacing, precision=sitk.sitkFloat32):
    """Resamples an image to a new spacing and saves it, returning the path."""
    try:
        image = sitk.ReadImage(input_path, precision)

        original_spacing = image.GetSpacing()
        original_size = image.GetSize()

        new_size = [int(round(orig_sz * orig_sp / target_sp)) for orig_sz, orig_sp, target_sp in
                    zip(original_size, original_spacing, target_spacing)]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        resampler.SetInterpolator(sitk.sitkLinear)

        resampled_image = resampler.Execute(image)
        sitk.WriteImage(resampled_image, output_path)

        logger.info(
            f"Resampled {basename(input_path)} to spacing {target_spacing} and saved to {basename(output_path)}")

        # Clean up
        del image
        del resampled_image
        gc.collect()

        return output_path
    except Exception as e:
        logger.error(f"Failed to resample {input_path}: {e}")
        return None


def run_lung_segmentation(ct_path, seg_output_dir, file_ext):
    """
    Runs lungmask to get a combined lung mask and then splits it into
    separate left and right lung masks.
    """
    os.makedirs(seg_output_dir, exist_ok=True)

    # --- Define final output paths ---
    left_lung_path = join(seg_output_dir, f"lung_left{file_ext}")
    right_lung_path = join(seg_output_dir, f"lung_right{file_ext}")

    # --- Check if masks already exist ---
    if os.path.exists(left_lung_path) and os.path.exists(right_lung_path):
        logger.info("Lung segmentations already exist, skipping.")
        return left_lung_path, right_lung_path

    # Path for the temporary combined mask file that lungmask will create
    combined_mask_path = join(seg_output_dir, f"temp_combined_mask{file_ext}")

    if not check_memory(): return None, None

    try:
        # 1. RUN LUNGMASK
        cmd = ["lungmask", ct_path, combined_mask_path]
        logger.info(f"Running lungmask: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # 2. SPLIT THE COMBINED MASK
        combined_mask_image = sitk.ReadImage(combined_mask_path)

        # Create Right Lung Mask (select label 1)
        right_lung_img = sitk.BinaryThreshold(combined_mask_image, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
        right_lung_img.CopyInformation(combined_mask_image)
        sitk.WriteImage(right_lung_img, right_lung_path)
        logger.info(f"Created right lung mask: {right_lung_path}")

        # Create Left Lung Mask (select label 2)
        left_lung_img = sitk.BinaryThreshold(combined_mask_image, lowerThreshold=2, upperThreshold=2, insideValue=1, outsideValue=0)
        left_lung_img.CopyInformation(combined_mask_image)
        sitk.WriteImage(left_lung_img, left_lung_path)
        logger.info(f"Created left lung mask: {left_lung_path}")

        # --- Cleanup ---
        os.remove(combined_mask_path)
        del combined_mask_image, right_lung_img, left_lung_img
        gc.collect()

        return left_lung_path, right_lung_path

    except subprocess.CalledProcessError as e:
        logger.error(f"lungmask failed with exit code {e.returncode}.")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None, None
    except Exception as e:
        logger.error(f"An error occurred during segmentation with lungmask: {e}")
        return None, None


def apply_mask_and_normalize_dose(dose_path, mask_path, output_path):
    """Applies a lung mask to a dose file and normalizes it by the 95th percentile."""
    try:
        dose_image = sitk.ReadImage(dose_path)
        mask_image = sitk.ReadImage(mask_path)

        # Ensure mask is correct type
        mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)

        # Apply mask
        masked_dose_image = sitk.Mask(dose_image, mask_image)

        # Normalize
        dose_array = sitk.GetArrayViewFromImage(masked_dose_image)
        nonzero_dose = dose_array[dose_array > 0]

        if nonzero_dose.size == 0:
            logger.warning("No non-zero dose values found in the masked region.")
            return None

        percentile_95 = np.percentile(nonzero_dose, 95)

        if percentile_95 <= 1e-6:
            logger.warning(f"95th percentile is {percentile_95}, too low to normalize. Saving unnormalized masked dose.")
            sitk.WriteImage(masked_dose_image, output_path)
            return output_path

        normalized_image = sitk.Cast(masked_dose_image, sitk.sitkFloat32) / percentile_95
        sitk.WriteImage(normalized_image, output_path)

        logger.info(
            f"Applied mask, normalized by 95th percentile ({percentile_95:.2f}), and saved to {basename(output_path)}")

        del dose_image, mask_image, masked_dose_image, normalized_image
        gc.collect()

        return output_path
    except Exception as e:
        logger.error(f"Failed during masking and normalization for {basename(dose_path)}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def extract_and_process_patches(normalized_masked_dose_path, config, subject_id, lung_side):
    """Extracts, normalizes, and yields patches one by one to save memory."""
    try:
        # --- Parameters ---
        patch_params = config.get('patch_extraction', {})
        preproc_params = config.get('preprocessing', {})

        # CORRECTED LOGIC: Directly read the patch size in voxels from your config.
        patch_size_voxels = tuple(patch_params.get('patch_size_voxels', [50, 50, 50]))

        # Get other parameters
        min_dose_percentage = patch_params.get('min_dose_percentage', 0.01)
        threshold = patch_params.get('threshold', 0.01)
        zero_pad_to = tuple(preproc_params.get('resize_to', [64, 64, 64]))

        image = sitk.ReadImage(normalized_masked_dose_path)
        array = sitk.GetArrayFromImage(image)

        if np.count_nonzero(array) == 0:
            logger.warning(f"Input image for patch extraction is all zeros for {subject_id} {lung_side} lung.")
            return

        step_size = [max(1, ps // 2) for ps in patch_size_voxels]
        shape = array.shape

        for z in range(0, shape[0] - patch_size_voxels[0] + 1, step_size[0]):
            for y in range(0, shape[1] - patch_size_voxels[1] + 1, step_size[1]):
                for x in range(0, shape[2] - patch_size_voxels[2] + 1, step_size[2]):
                    patch = array[z:z + patch_size_voxels[0], y:y + patch_size_voxels[1], x:x + patch_size_voxels[2]]

                    # --- Filtering ---
                    if np.count_nonzero(patch) / patch.size < min_dose_percentage:
                        continue
                    if np.std(patch) <= threshold:
                        continue

                    # --- DL Normalization (Min-Max) ---
                    min_val, max_val = np.min(patch), np.max(patch)
                    if max_val > min_val:
                        normalized_patch = (patch - min_val) / (max_val - min_val)
                    else:
                        continue

                    # --- Padding ---
                    if normalized_patch.shape != zero_pad_to:
                        padded_patch = np.zeros(zero_pad_to, dtype=np.float32)
                        pad_z, pad_y, pad_x = [(zp - ps) // 2 for zp, ps in zip(zero_pad_to, normalized_patch.shape)]

                        # Check for negative padding which indicates patch is larger than target
                        if any(p < 0 for p in [pad_z, pad_y, pad_x]):
                            logger.error(f"Patch size {normalized_patch.shape} is larger than target padded size {zero_pad_to}. Check config.")
                            continue # Skip this invalid patch

                        padded_patch[pad_z:pad_z + patch.shape[0], pad_y:pad_y + patch.shape[1],
                        pad_x:pad_x + patch.shape[2]] = normalized_patch
                        final_patch = padded_patch
                    else:
                        final_patch = normalized_patch

                    # --- Yield Patch Data ---
                    yield {
                        "data": final_patch.astype(np.float32),
                        "patient_id": subject_id,
                        "lung_side": lung_side,
                        "final_shape": list(final_patch.shape)
                    }

    except Exception as e:
        logger.error(f"Error extracting patches for {subject_id} {lung_side} lung: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Ensure memory is released
        if 'image' in locals(): del image
        if 'array' in locals(): del array
        gc.collect()


def append_to_hdf5(file_path, data_list, compression='gzip', compression_opts=9):
    """Appends a list of items to an HDF5 file with compression."""
    if not data_list:
        return
    try:
        with h5py.File(file_path, 'a') as f:
            # Get starting index for new patches
            existing_patches = [key for key in f.keys() if key.startswith('patch_')]
            start_idx = len(existing_patches)

            for i, item in enumerate(data_list):
                group_name = f'patch_{start_idx + i:06d}'
                grp = f.create_group(group_name)

                # Store the patch data with compression
                grp.create_dataset('data',
                                   data=item['data'],
                                   compression=compression,
                                   compression_opts=compression_opts,
                                   dtype=np.float32)

                # Store metadata as attributes
                grp.attrs['patient_id'] = item['patient_id']
                grp.attrs['lung_side'] = item['lung_side']
                grp.attrs['final_shape'] = item['final_shape']

        logger.info(f"Appended {len(data_list)} items to {file_path}")
    except Exception as e:
        logger.error(f"Failed to append to HDF5 file {file_path}: {e}")


def find_patient_data(patient_dir, use_pycurt_format=True):
    """
    Find CT and dose files for a patient.

    Args:
        patient_dir: Patient directory path (for pycurt: rtdose_dir, for simplified: patient_dir)
        use_pycurt_format: If True, use original pycurt structure, else use simplified format

    Returns:
        tuple: (ct_path, dose_path) or (None, None) if not found
    """
    if use_pycurt_format:
        # Original logic for pycurt format
        ct_dicom_dirs = [x for x in glob.glob(join(dirname(dirname(dirname(dirname(patient_dir)))), '*/*/*CT-*MP1*/*')) if 'RTSTRUCT' not in x]
        ct_dicom_dir = next((d for d in ct_dicom_dirs if os.path.isdir(d)), None)
        return ct_dicom_dir, patient_dir  # patient_dir is the rtdose_dir in pycurt format
    else:
        # Simplified format: look for CT_rt.nii.gz and RTDOSE_rt_ct.nii.gz in patient_dir
        ct_path = join(patient_dir, "CT_rt.nii.gz")
        dose_path = join(patient_dir, "RTDOSE_rt_ct.nii.gz")

        if os.path.exists(ct_path) and os.path.exists(dose_path):
            return ct_path, dose_path
        else:
            logger.warning(f"Missing files in {patient_dir}: CT_rt.nii.gz or RTDOSE_rt_ct.nii.gz")
            return None, None


def process_patient(patient_dir, output_dir, config, use_pycurt_format=True):
    """
    Complete processing pipeline for a single patient, with added logic
    to determine ipsilateral/contralateral side based on mean dose.
    MODIFIED TO SAVE AS HDF5 and handle both pycurt and simplified formats.
    """
    subject_id = get_subject_id_from_path(patient_dir, use_pycurt_format)
    if should_skip_patient(subject_id): return 0

    subject_output_dir = join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)

    # Check for both .pkl and .h5 files, use .h5 for new saves
    final_patches_file_pkl = join(subject_output_dir, f"{subject_id}_patches.pkl")
    final_patches_file_h5 = join(subject_output_dir, f"{subject_id}_patches.h5")

    if os.path.exists(final_patches_file_pkl) or os.path.exists(final_patches_file_h5):
        logger.info(f"Patches for subject {subject_id} already exist. Skipping.")
        return -1

    if not check_memory(): return 0

    # 1. Find CT and dose files based on format
    if use_pycurt_format:
        ct_source, dose_source = find_patient_data(patient_dir, use_pycurt_format)
        if not ct_source or not dose_source:
            logger.warning(f"No valid CT or dose data found for {subject_id}")
            return 0

        # Check processability for pycurt format (directories)
        if not is_patient_processable(ct_source, dose_source, use_pycurt_format=True): return 0

        # Convert DICOMs
        file_ext = get_file_extension(config)
        ext_map = {'.nii.gz': 'nifti_gz', '.nii': 'nifti', '.nrrd': 'nrrd'}
        convert_to_keyword = ext_map.get(file_ext, 'nifti_gz')

        ct_conv_path = join(ct_source + file_ext)
        dose_conv_path = join(dose_source + "_ct" + file_ext)

        if not os.path.exists(ct_conv_path):
            if not run_dicom_conversion(ct_source, convert_to_keyword): return 0
        if not os.path.exists(dose_conv_path):
            if not run_dicom_conversion(dose_source, convert_to_keyword): return 0

        ct_path = ct_conv_path
        dose_path = dose_conv_path

    else:
        # Simplified format: files are already converted
        ct_path, dose_path = find_patient_data(patient_dir, use_pycurt_format)
        if not ct_path or not dose_path:
            logger.warning(f"No valid CT or dose files found for {subject_id}")
            return 0

        # Check processability for simplified format (files)
        if not is_patient_processable(ct_path, dose_path, use_pycurt_format=False): return 0

    # 2. Resample CT and Dose
    spacing = tuple(config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0]))
    file_ext = get_file_extension(config)

    resampled_ct_path = resample_image(ct_path, join(subject_output_dir, f"ct_resampled{file_ext}"), spacing)
    resampled_dose_path = resample_image(dose_path, join(subject_output_dir, f"dose_resampled{file_ext}"), spacing)
    if not resampled_ct_path or not resampled_dose_path: return 0

    # 3. Segment Lungs
    seg_dir = join(subject_output_dir, "segmentation")
    left_lung_mask_path, right_lung_mask_path = run_lung_segmentation(resampled_ct_path, seg_dir, file_ext)
    if not left_lung_mask_path or not right_lung_mask_path: return 0

    # --- LOGIC: DETERMINE IPSI/CONTRA SIDE AND CHECK FOR ZERO DOSES ---
    try:
        logger.info(f"Determining ipsi/contra side for patient {subject_id} based on mean dose.")
        dose_image = sitk.ReadImage(resampled_dose_path, sitk.sitkFloat32)
        left_mask = sitk.ReadImage(left_lung_mask_path)
        right_mask = sitk.ReadImage(right_lung_mask_path)

        stats_filter = sitk.LabelStatisticsImageFilter()

        # Calculate mean dose for left lung
        stats_filter.Execute(dose_image, left_mask)
        left_mean_dose = stats_filter.GetMean(1) if stats_filter.HasLabel(1) else 0

        # Calculate mean dose for right lung
        stats_filter.Execute(dose_image, right_mask)
        right_mean_dose = stats_filter.GetMean(1) if stats_filter.HasLabel(1) else 0

        logger.info(f"Mean Doses - Left: {left_mean_dose:.2f}, Right: {right_mean_dose:.2f}")

        # CHECK FOR ZERO DOSES AND ADD TO SKIP LIST
        if left_mean_dose == 0.0 and right_mean_dose == 0.0:
            logger.warning(f"Patient {subject_id} has zero doses in both lungs. Adding to skip list.")
            if subject_id not in ZERO_DOSE_PATIENTS:
                ZERO_DOSE_PATIENTS.append(subject_id)
                save_zero_dose_patients(output_dir)  # Save immediately
            return 0  # Return 0 to indicate no patches processed

        # Create a mapping from anatomical side to clinical side
        if left_mean_dose > right_mean_dose:
            laterality_map = {"left": "ipsi", "right": "contra"}
            logger.info("Left lung determined to be IPSILATERAL.")
        else:
            laterality_map = {"left": "contra", "right": "ipsi"}
            logger.info("Right lung determined to be IPSILATERAL.")

        del dose_image, left_mask, right_mask # Clean up memory

    except Exception as e:
        logger.error(f"Could not determine ipsi/contra side for patient {subject_id}: {e}. Defaulting to left/right.")
        laterality_map = {"left": "left", "right": "right"}
    # --- END OF LOGIC ---

    # 4. Process each lung (MODIFIED to use HDF5 and batched saving)
    total_patches = 0
    batch_size = 50  # Process in smaller batches to save memory

    for lung_side, mask_path in [("left", left_lung_mask_path), ("right", right_lung_mask_path)]:
        clinical_side = laterality_map[lung_side] # Use the map to get "ipsi" or "contra"
        logger.info(f"--- Processing {lung_side.upper()} ({clinical_side.upper()}) lung for patient {subject_id} ---")

        norm_masked_dose_path = join(subject_output_dir, f"dose_{lung_side}_norm_masked{file_ext}")
        processed_dose_path = apply_mask_and_normalize_dose(resampled_dose_path, mask_path, norm_masked_dose_path)
        if not processed_dose_path: continue

        # MODIFIED: Process patches in batches and save immediately to HDF5
        lung_patches = []
        for patch_data in extract_and_process_patches(processed_dose_path, config, subject_id, clinical_side):
            lung_patches.append(patch_data)

            # Save batch when it reaches batch_size
            if len(lung_patches) >= batch_size:
                append_to_hdf5(final_patches_file_h5, lung_patches)
                total_patches += len(lung_patches)
                lung_patches = []  # Clear batch
                gc.collect()  # Force garbage collection

        # Save remaining patches
        if lung_patches:
            append_to_hdf5(final_patches_file_h5, lung_patches)
            total_patches += len(lung_patches)

        logger.info(f"Extracted patches from {lung_side} ({clinical_side}) lung.")

    # 5. Log final results
    if total_patches > 0:
        logger.info(f"Saved {total_patches} patches for patient {subject_id} to {final_patches_file_h5}")
    else:
        logger.warning(f"No patches were extracted for patient {subject_id}.")

    # 6. Final cleanup of large intermediate files
    for f in [resampled_ct_path, resampled_dose_path, left_lung_mask_path, right_lung_mask_path,
              norm_masked_dose_path]:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                logger.warning(f"Could not remove intermediate file {f}: {e}")
    gc.collect()

    return total_patches


def find_patient_dirs(base_dir, use_pycurt_format=True):
    """
    Finds patient directories based on the format being used.

    For pycurt format: Finds RTDOSE directories, renames special cases from '-MP*' to '-MPT'
    For simplified format: Finds patient directories containing CT_rt.nii.gz and RTDOSE_rt_ct.nii.gz

    Returns a unique list of patient directory paths.
    """
    if use_pycurt_format:
        return find_rtdose_dirs(base_dir)
    else:
        return find_patient_dirs_simplified(base_dir)


def find_patient_dirs_simplified(base_dir):
    """
    Find patient directories in simplified format.
    Each patient directory should contain CT_rt.nii.gz and RTDOSE_rt_ct.nii.gz
    """
    patient_dirs = []

    # Look for directories that contain both required files
    for item in os.listdir(base_dir):
        patient_path = join(base_dir, item)
        if os.path.isdir(patient_path):
            ct_file = join(patient_path, "CT_rt.nii.gz")
            dose_file = join(patient_path, "RTDOSE_rt_ct.nii.gz")

            if os.path.exists(ct_file) and os.path.exists(dose_file):
                patient_dirs.append(patient_path)
                logger.debug(f"Found valid patient directory: {patient_path}")
            else:
                logger.debug(f"Skipping {patient_path} - missing required files")

    logger.info(f"Found {len(patient_dirs)} patient directories in simplified format.")
    return patient_dirs


def find_rtdose_dirs(base_dir):
    """
    Finds RTDOSE directories, renames special cases from '-MP*' to '-MPT',
    and returns a unique list of final directory paths.
    """
    special_pids = ["0617548432","0617601841","0617795029","0617766921","0617385185","0617789552","0617698612","0617692367","0617466966","0617544655","0617744077","0617693006","0617589358","0617592801","0617673893", "0617713112","0617757588","0617544655", "0617744077","0617661400", "0617876923", "0617706363", "0617553904", "0617517268", "0617455683",
                    "0617450775", "0617706363", "0617876923","0617661400","0617548432","0617698612","0617789552","0617385185","0617417018","0617544856"]

    rename = False

    # Check if the base directory corresponds to a special PID
    if os.path.basename(base_dir) in special_pids:
        # This pattern finds the target directory inside a parent named like 'RTDOSE*-MP*'
        patterns = [
            join(base_dir, "*", "*", "*", "*", "RTDOSE*-MP*", "RTDOSETOTALHETERO"),
        ]
        rename = True
    else:
        # For other PIDs, find directories that already match the desired names
        patterns = [
            join(base_dir, "*", "*", "*", "*", "RTDOSE*-MPT", "*"),
            join(base_dir, "*", "*", "*", "*", "RTDOSE*-MP1", "*")
        ]

    all_dirs = []
    for pattern in patterns:
        found_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]

        # If the rename flag is set, process and rename the parent directories
        if rename:
            processed_dirs = []
            for inner_dir in found_dirs:
                parent_dir_path = os.path.dirname(inner_dir)
                old_name = os.path.basename(parent_dir_path)

                # Create the new name by replacing '-MP' followed by characters with '-MPT'
                new_name = re.sub(r'-MP\w+', '-MPT', old_name)

                # Start with the original path
                final_inner_dir_path = inner_dir

                # Only proceed if a rename is actually needed
                if old_name != new_name:
                    grandparent_path = os.path.dirname(parent_dir_path)
                    new_parent_path = os.path.join(grandparent_path, new_name)

                    # --- Print the old and new paths ---
                    print(f"--- Renaming directory ---\n  Old Path: {parent_dir_path}\n  New Path: {new_parent_path}\n")

                    try:
                        # Perform the rename
                        os.rename(parent_dir_path, new_parent_path)
                        # IMPORTANT: Update the path to the inner directory for the return list
                        final_inner_dir_path = os.path.join(new_parent_path, os.path.basename(inner_dir))
                    except OSError as e:
                        logger.error(f"Failed to rename directory: {e}")

                processed_dirs.append(final_inner_dir_path)

            all_dirs.extend(processed_dirs)
        else:
            # If not renaming, just add the directories as found
            all_dirs.extend(found_dirs)

    # Remove duplicates while preserving order
    unique_dirs = list(dict.fromkeys(all_dirs))
    if unique_dirs:
        logger.info(f"Found {len(unique_dirs)} unique RTDOSE directories to process.")

    return unique_dirs


def combine_hdf5_files(output_dir):
    """Combine all individual HDF5 files into one final dataset."""
    logger.info("Combining all individual patient HDF5 files into a single dataset...")

    # Look for both .h5 and .pkl files (in case some are already converted)
    all_h5_files = list(Path(output_dir).rglob("*_patches.h5"))
    all_pkl_files = list(Path(output_dir).rglob("*_patches.pkl"))

    combined_dataset_path = join(output_dir, "patches_dataset_final.h5")

    # Remove existing combined file
    if os.path.exists(combined_dataset_path):
        os.remove(combined_dataset_path)

    total_patches = 0
    patch_counter = 0

    with h5py.File(combined_dataset_path, 'w') as combined_f:
        # Process HDF5 files
        for h5_file in tqdm(all_h5_files, desc="Combining HDF5 Files"):
            try:
                with h5py.File(h5_file, 'r') as source_f:
                    for patch_name in source_f.keys():
                        if patch_name.startswith('patch_'):
                            # Copy group to combined file with new name
                            new_name = f'patch_{patch_counter:08d}'
                            source_f.copy(patch_name, combined_f, name=new_name)
                            patch_counter += 1
                            total_patches += 1

            except Exception as e:
                logger.error(f"Could not process {h5_file}: {e}")

        # Process remaining pickle files (if any)
        for pkl_file in tqdm(all_pkl_files, desc="Converting remaining pickle files"):
            try:
                with open(pkl_file, 'rb') as f:
                    pickle_data = pickle.load(f)

                # Convert pickle data to HDF5 format in the combined file
                for item in pickle_data:
                    group_name = f'patch_{patch_counter:08d}'
                    grp = combined_f.create_group(group_name)

                    grp.create_dataset('data',
                                       data=item['data'],
                                       compression='gzip',
                                       compression_opts=9,
                                       dtype=np.float32)

                    grp.attrs['patient_id'] = item['patient_id']
                    grp.attrs['lung_side'] = item['lung_side']
                    grp.attrs['final_shape'] = item['final_shape']

                    patch_counter += 1
                    total_patches += 1

            except Exception as e:
                logger.error(f"Could not process {pkl_file}: {e}")

    if total_patches > 0:
        logger.info(f"Successfully combined {total_patches} patches into {combined_dataset_path}")

        # Report file size
        file_size_gb = Path(combined_dataset_path).stat().st_size / (1024**3)
        logger.info(f"Combined dataset size: {file_size_gb:.2f} GB")
    else:
        logger.warning("No patch data found to combine.")


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dose distribution dataset from medical image files.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--start", type=int, default=0, help="Start index of patients to process")
    parser.add_argument("--end", type=int, default=None, help="End index of patients to process")
    parser.add_argument("--pycurt-format", action="store_true",
                        help="Use pycurt directory structure (default: simplified format)")
    args = parser.parse_args()

    # Load Config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        exit(1)

    # Determine format to use
    use_pycurt_format = args.pycurt_format
    if use_pycurt_format and not PYCURT_AVAILABLE:
        logger.error("pycurt format requested but pycurtv2 is not available. Please install pycurtv2 or use simplified format.")
        exit(1)

    logger.info(f"Using {'pycurt' if use_pycurt_format else 'simplified'} format for processing")

    # Setup directories - use hardcoded output dir as requested
    base_dir = config['dataset']['data_dir']
    output_dir = "/data/pgsal/ldc_doseae"  # Hardcoded output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load zero dose patients list
    load_zero_dose_patients(output_dir)

    # --- MODIFIED LOGIC TO FILTER PATIENTS ---

    # 1. Find all available patient directories in the base directory
    all_patient_dirs = find_patient_dirs(base_dir, use_pycurt_format)

    # 2. Check the config for a specific list of patients to run
    # .get() is used to avoid errors if the keys don't exist
    specific_patients_to_run = config.get('processing_scope', {}).get('specific_patients')

    patient_dirs_to_process = []
    if specific_patients_to_run:
        logger.info(f"Found a specific list of {len(specific_patients_to_run)} patients to process in the config file.")
        # Create a set for fast lookups
        target_patient_set = {str(p) for p in specific_patients_to_run}

        # Filter the full list of directories to keep only the ones we want
        for patient_dir in all_patient_dirs:
            subject_id = get_subject_id_from_path(patient_dir, use_pycurt_format)
            if subject_id in target_patient_set:
                patient_dirs_to_process.append(patient_dir)

        logger.info(f"Matched {len(patient_dirs_to_process)} of the specified patients in the data directory.")
    else:
        logger.info("No specific patient list found in config. The script will process all available patients.")
        patient_dirs_to_process = all_patient_dirs

    # --- END OF MODIFIED LOGIC ---

    # Slice for batch processing, now using the potentially filtered list
    process_list = patient_dirs_to_process[args.start:args.end]

    logger.info(
        f"Starting processing run. Total patients to process: {len(process_list)} (from index {args.start} to {args.end or len(patient_dirs_to_process)})")

    failed_patients = []
    processed_count = 0

    with tqdm(process_list, desc="Processing Patients") as pbar:
        for patient_dir in pbar:
            subject_id = get_subject_id_from_path(patient_dir, use_pycurt_format)
            pbar.set_description(f"Processing Patient {subject_id}")
            try:
                num_patches = process_patient(patient_dir, output_dir, config, use_pycurt_format)
                if num_patches > 0:
                    processed_count += 1
                elif num_patches == 0:
                    failed_patients.append(
                        {"patient_id": subject_id, "reason": "Processing failed or yielded no patches."})
            except Exception as e:
                logger.error(f"--- Unhandled exception for patient {subject_id}: {e} ---")
                import traceback
                logger.error(traceback.format_exc())
                failed_patients.append({"patient_id": subject_id, "reason": str(e)})

    logger.info("--- PROCESSING RUN COMPLETE ---")
    logger.info(f"Successfully processed {processed_count} new patients.")
    logger.info(f"Failed to process {len(failed_patients)} patients.")

    # Save the list of failed patients
    if failed_patients:
        failed_patients_path = join(output_dir, "failed_patients.json")
        with open(failed_patients_path, "w") as f:
            json.dump(failed_patients, f, indent=4)
        logger.info(f"List of failed patients saved to {failed_patients_path}")

    # --- Final Step: Combine all individual patient files into one (HDF5) ---
    combine_hdf5_files(output_dir)