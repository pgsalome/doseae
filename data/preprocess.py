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

# It's good practice to handle potential import errors for optional dependencies
try:
    import psutil
except ImportError:
    psutil = None

# Assuming pycurtv2 is in the environment. If not, this will raise an ImportError.
from pycurtv2.converters.dicom import DicomConverter

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
PROBLEMATIC_PATIENTS = ["0617535250", "0617445118"]


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


def get_subject_id_from_path(path):
    """Extracts a unique subject ID from a given directory path."""
    return basename((dirname(dirname(dirname(dirname(path))))))


def should_skip_patient(subject_id):
    """Check if a patient should be skipped due to known issues."""
    if subject_id in PROBLEMATIC_PATIENTS:
        logger.warning(f"Skipping known problematic patient: {subject_id}")
        return True
    return False


def is_patient_processable(ct_dir, rtdose_dir, max_size_gb=3.0):
    """Check if a patient's data is within a reasonable size limit."""
    try:
        subject_id = get_subject_id_from_path(rtdose_dir)
        ct_size = sum(f.stat().st_size for f in Path(ct_dir).rglob('*') if f.is_file())
        rtdose_size = sum(f.stat().st_size for f in Path(rtdose_dir).rglob('*') if f.is_file())
        total_size_gb = (ct_size + rtdose_size) / (1024 ** 3)

        logger.info(f"Patient {subject_id} data size: {total_size_gb:.2f} GB")
        if total_size_gb > max_size_gb:
            logger.warning(
                f"Skipping patient {subject_id} with data size {total_size_gb:.2f} GB (exceeds limit of {max_size_gb} GB)")
            return False
        return True
    except Exception as e:
        logger.error(f"Error checking patient processability for {rtdose_dir}: {e}")
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


def run_dicom_conversion(dicom_dir, convert_to='nrrd'):
    """Runs DICOM conversion for a given directory."""
    try:
        logger.info(f"Converting DICOMs in {dicom_dir} to {convert_to}...")
        converter = DicomConverter(toConvert=dicom_dir, convert_to=convert_to)
        converter.convert_ps()
        del converter  # Explicitly delete the object
        gc.collect()
        return True
    except Exception as e:
        logger.error(f"DICOM conversion failed for {dicom_dir}: {e}")
        return False


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
    """Runs TotalSegmentator to get lung masks."""
    os.makedirs(seg_output_dir, exist_ok=True)

    # Define final output paths
    left_lung_path = join(seg_output_dir, f"lung_left{file_ext}")
    right_lung_path = join(seg_output_dir, f"lung_right{file_ext}")

    if os.path.exists(left_lung_path) and os.path.exists(right_lung_path):
        logger.info("Lung segmentations already exist, skipping.")
        return left_lung_path, right_lung_path

    if not check_memory(): return None, None

    try:
        # Command for TotalSegmentator
        cmd = [
            "TotalSegmentator",
            "-i", ct_path,
            "-o", seg_output_dir,
            "--fast",
            "--roi_subset", "lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right"
        ]
        logger.info(f"Running TotalSegmentator: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # --- Combine lobes into single left/right lung masks ---
        reference_image = sitk.ReadImage(ct_path)

        # Combine Left Lung
        left_lobe_files = glob.glob(join(seg_output_dir, "lung_*_left.nii.gz"))
        if left_lobe_files:
            left_mask_arr = np.zeros(sitk.GetArrayViewFromImage(reference_image).shape, dtype=np.uint8)
            for f in left_lobe_files:
                lobe_img = sitk.ReadImage(f)
                left_mask_arr = np.logical_or(left_mask_arr, sitk.GetArrayViewFromImage(lobe_img))
            left_lung_img = sitk.GetImageFromArray(left_mask_arr)
            left_lung_img.CopyInformation(reference_image)
            sitk.WriteImage(left_lung_img, left_lung_path)
            logger.info(f"Created left lung mask: {left_lung_path}")

        # Combine Right Lung
        right_lobe_files = glob.glob(join(seg_output_dir, "lung_*_right.nii.gz"))
        if right_lobe_files:
            right_mask_arr = np.zeros(sitk.GetArrayViewFromImage(reference_image).shape, dtype=np.uint8)
            for f in right_lobe_files:
                lobe_img = sitk.ReadImage(f)
                right_mask_arr = np.logical_or(right_mask_arr, sitk.GetArrayViewFromImage(lobe_img))
            right_lung_img = sitk.GetImageFromArray(right_mask_arr)
            right_lung_img.CopyInformation(reference_image)
            sitk.WriteImage(right_lung_img, right_lung_path)
            logger.info(f"Created right lung mask: {right_lung_path}")

        # Clean up individual lobe files
        for f in left_lobe_files + right_lobe_files:
            os.remove(f)

        del reference_image, left_lung_img, right_lung_img, left_mask_arr, right_mask_arr
        gc.collect()

        return left_lung_path, right_lung_path

    except subprocess.CalledProcessError as e:
        logger.error(f"TotalSegmentator failed with exit code {e.returncode}.")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return None, None
    except Exception as e:
        logger.error(f"An error occurred during segmentation: {e}")
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
        stats = sitk.LabelStatisticsImageFilter()
        stats.Execute(masked_dose_image, sitk.BinaryThreshold(masked_dose_image, 0.001, 1e9, 1, 0))

        dose_array = sitk.GetArrayViewFromImage(masked_dose_image)
        nonzero_dose = dose_array[dose_array > 0]

        if nonzero_dose.size == 0:
            logger.warning("No non-zero dose values found in the masked region.")
            return None

        percentile_95 = np.percentile(nonzero_dose, 95)

        if percentile_95 <= 0:
            logger.warning(f"95th percentile is {percentile_95}, cannot normalize. Saving unnormalized masked dose.")
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
        return None


def extract_and_process_patches(normalized_masked_dose_path, config, subject_id, lung_side):
    """Extracts, normalizes, and yields patches one by one to save memory."""
    try:
        # Patch Extraction Parameters
        patch_params = config.get('patch_extraction', {})
        patch_size_mm = tuple(patch_params.get('patch_size', [50, 50, 50]))
        min_dose_percentage = patch_params.get('min_dose_percentage', 0.01)
        threshold = patch_params.get('threshold', 0.01)

        # Resizing/Padding parameters
        preproc_params = config.get('preprocessing', {})
        target_spacing = tuple(preproc_params.get('voxel_spacing', [1.0, 1.0, 1.0]))
        zero_pad_to = tuple(preproc_params.get('resize_to', [64, 64, 64]))

        image = sitk.ReadImage(normalized_masked_dose_path)
        array = sitk.GetArrayFromImage(image)

        if np.count_nonzero(array) == 0:
            logger.warning(f"Input image for patch extraction is all zeros for {subject_id} {lung_side} lung.")
            return

        patch_size_voxels = [int(round(psm / ts)) for psm, ts in zip(patch_size_mm, target_spacing)]
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
    finally:
        # Ensure memory is released
        if 'image' in locals(): del image
        if 'array' in locals(): del array
        gc.collect()


def append_to_pickle(file_path, data_list):
    """Appends a list of items to a pickle file without loading the whole file."""
    if not data_list:
        return
    try:
        with open(file_path, 'ab') as f:
            for item in data_list:
                pickle.dump(item, f)
        logger.info(f"Appended {len(data_list)} items to {file_path}")
    except Exception as e:
        logger.error(f"Failed to append to pickle file {file_path}: {e}")


def process_patient(rtdose_dir, output_dir, config):
    """
    Complete processing pipeline for a single patient with a focus on memory efficiency.
    Returns the number of patches successfully processed.
    """
    subject_id = get_subject_id_from_path(rtdose_dir)
    if should_skip_patient(subject_id): return 0

    # Define subject-specific output directory
    subject_output_dir = join(output_dir, f"subject_{subject_id}")
    os.makedirs(subject_output_dir, exist_ok=True)

    # Check if final patch file exists and skip if so
    final_patches_file = join(subject_output_dir, f"{subject_id}_patches.pkl")
    if os.path.exists(final_patches_file):
        logger.info(f"Patches for subject {subject_id} already exist. Skipping.")
        return -1  # Use a special value to indicate already processed

    if not check_memory(): return 0

    # 1. Find DICOM directories
    ct_dicom_dirs = glob.glob(join(dirname(dirname(dirname(dirname(rtdose_dir)))), '*/*/*CT-*MP1*/*'))
    ct_dicom_dir = next((d for d in ct_dicom_dirs if os.path.isdir(d)), None)
    if not ct_dicom_dir:
        logger.warning(f"No valid CT DICOM directory found for {subject_id}")
        return 0

    if not is_patient_processable(ct_dicom_dir, rtdose_dir): return 0

    # 2. Convert DICOM to NRRD/NIfTI
    file_ext = get_file_extension(config)
    ct_conv_path = join(ct_dicom_dir + file_ext)
    dose_conv_path = join(rtdose_dir + "_ct" + file_ext)

    if not os.path.exists(ct_conv_path):
        if not run_dicom_conversion(ct_dicom_dir, file_ext.replace('.', '')): return 0
    if not os.path.exists(dose_conv_path):
        if not run_dicom_conversion(rtdose_dir, file_ext.replace('.', '')): return 0

    # 3. Resample CT and Dose
    spacing = tuple(config.get('preprocessing', {}).get('voxel_spacing', [1.0, 1.0, 1.0]))
    resampled_ct_path = resample_image(ct_conv_path, join(subject_output_dir, f"ct_resampled{file_ext}"), spacing)
    resampled_dose_path = resample_image(dose_conv_path, join(subject_output_dir, f"dose_resampled{file_ext}"), spacing)
    if not resampled_ct_path or not resampled_dose_path: return 0

    # 4. Segment Lungs
    seg_dir = join(subject_output_dir, "segmentation")
    left_lung_mask_path, right_lung_mask_path = run_lung_segmentation(resampled_ct_path, seg_dir, file_ext)
    if not left_lung_mask_path or not right_lung_mask_path: return 0

    # 5. Process each lung
    patient_patches = []
    total_patches = 0

    for lung_side, mask_path in [("left", left_lung_mask_path), ("right", right_lung_mask_path)]:
        logger.info(f"--- Processing {lung_side.upper()} lung for patient {subject_id} ---")

        # 5a. Apply mask and normalize dose
        norm_masked_dose_path = join(subject_output_dir, f"dose_{lung_side}_norm_masked{file_ext}")
        processed_dose_path = apply_mask_and_normalize_dose(resampled_dose_path, mask_path, norm_masked_dose_path)
        if not processed_dose_path: continue

        # 5b. Extract patches
        lung_patches = []
        for patch_data in extract_and_process_patches(processed_dose_path, config, subject_id, lung_side):
            lung_patches.append(patch_data)

        logger.info(f"Extracted {len(lung_patches)} patches from {lung_side} lung.")
        patient_patches.extend(lung_patches)
        total_patches += len(lung_patches)

    # 6. Save patches for this patient to a single file
    if total_patches > 0:
        with open(final_patches_file, 'wb') as f:
            pickle.dump(patient_patches, f)
        logger.info(f"Saved {total_patches} patches for patient {subject_id} to {final_patches_file}")
    else:
        logger.warning(f"No patches were extracted for patient {subject_id}.")

    # 7. Final cleanup of large intermediate files
    for f in [resampled_ct_path, resampled_dose_path, left_lung_mask_path, right_lung_mask_path, norm_masked_dose_path]:
        if f and os.path.exists(f):
            try:
                os.remove(f)
            except OSError as e:
                logger.warning(f"Could not remove intermediate file {f}: {e}")
    gc.collect()

    return total_patches


def find_rtdose_dirs(base_dir):
    """Finds all RTDOSE directories, preferring '-MPT' over '-MP1'."""
    patterns = [
        join(base_dir, "*", "*", "*", "*", "RTDOSE*-MPT", "*"),
        join(base_dir, "*", "*", "*", "*", "RTDOSE*-MP1", "*")
    ]
    all_dirs = []
    for pattern in patterns:
        # dirname gets the parent RTDOSE-* directory
        found_dirs = [dirname(d) for d in glob.glob(pattern) if os.path.isdir(d)]
        all_dirs.extend(found_dirs)

    # Remove duplicates while preserving order (MPT will come first)
    unique_dirs = list(dict.fromkeys(all_dirs))
    logger.info(f"Found {len(unique_dirs)} unique RTDOSE directories to process.")
    return unique_dirs


# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dose distribution dataset from DICOM files.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--start", type=int, default=0, help="Start index of patients to process")
    parser.add_argument("--end", type=int, default=None, help="End index of patients to process")
    args = parser.parse_args()

    # Load Config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at {args.config}")
        exit(1)

    # Setup directories
    base_dir = config['dataset']['data_dir']
    output_dir = config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Find all patient directories
    rtdose_dirs = find_rtdose_dirs(base_dir)

    # Slice for batch processing
    process_list = rtdose_dirs[args.start:args.end]

    logger.info(
        f"Starting processing run. Total patients to process: {len(process_list)} (from index {args.start} to {args.end or len(rtdose_dirs)})")

    failed_patients = []
    processed_count = 0

    with tqdm(process_list, desc="Processing Patients") as pbar:
        for rtdose_dir in pbar:
            subject_id = get_subject_id_from_path(rtdose_dir)
            pbar.set_description(f"Processing Patient {subject_id}")
            try:
                num_patches = process_patient(rtdose_dir, output_dir, config)
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

    # --- Final Step: Combine all individual patient pickle files into one ---
    logger.info("Combining all individual patient patch files into a single dataset...")
    all_patch_files = glob.glob(join(output_dir, "subject_*", "*_patches.pkl"))
    combined_dataset_path = join(output_dir, "patches_dataset_final.pkl")

    all_patches_data = []
    for f in tqdm(all_patch_files, desc="Combining Files"):
        try:
            with open(f, 'rb') as pkl_file:
                patient_data = pickle.load(pkl_file)
                all_patches_data.extend(patient_data)
        except Exception as e:
            logger.error(f"Could not load or read {f}: {e}")

    if all_patches_data:
        try:
            with open(combined_dataset_path, 'wb') as f:
                pickle.dump(all_patches_data, f)
            logger.info(
                f"Successfully combined {len(all_patches_data)} patches from {len(all_patch_files)} patients into {combined_dataset_path}")
        except Exception as e:
            logger.error(f"Failed to write final combined dataset: {e}")
    else:
        logger.warning("No patch data found to combine.")