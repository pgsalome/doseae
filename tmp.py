import os
import glob
import argparse
from os.path import dirname, basename
from collections import defaultdict
from tqdm import tqdm
import logging

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_subject_id_from_path(path):
    """
    Extracts a unique subject ID from a given directory path.
    This function is copied directly from your original script to ensure
    the data structure is interpreted identically.
    """
    # Navigates up the directory tree from a given path to find the patient ID.
    # The number of 'dirname' calls is based on your specific folder hierarchy.
    try:
        # Assuming the path is to the RTDOSE directory itself.
        # We go up 4 levels from the RTDOSE directory to get the patient folder.
        # e.g., patient_id/study/series/RTDOSE-DIR -> patient_id
        return basename(dirname(dirname(dirname(dirname(path)))))
    except Exception:
        return None


def verify_mpt_uniqueness(base_dir):
    """
    Verifies if patients have a unique RTDOSE-MPT directory.

    Args:
        base_dir (str): The root directory containing the patient data.

    Returns:
        dict: A dictionary containing the analysis results.
    """
    if not os.path.isdir(base_dir):
        logger.error(f"Error: Base directory not found at '{base_dir}'")
        return None

    logger.info("Searching for all RTDOSE-MPT directories...")
    # Find all directories named 'RTDOSE*-MPT'
    mpt_series_paths = glob.glob(os.path.join(base_dir, '**', 'RTDOSE*-MPT'), recursive=True)

    if not mpt_series_paths:
        logger.warning("No RTDOSE-MPT directories were found in the dataset.")
        return None

    logger.info(f"Found {len(mpt_series_paths)} total MPT dose directories. Grouping by patient...")

    # Group the found MPT directories by their patient ID
    # The key is the patient_id, the value is a list of their MPT directory paths
    patient_mpt_map = defaultdict(list)
    for path in tqdm(mpt_series_paths, desc="Grouping MPT Series by Patient"):
        patient_id = get_subject_id_from_path(path)
        if patient_id:
            patient_mpt_map[patient_id].append(path)

    # --- Analyze the grouped map ---
    single_mpt_patients = 0
    multiple_mpt_patients = 0
    problematic_patients = {} # Store patients with multiple MPTs to display them

    for patient_id, paths in patient_mpt_map.items():
        if len(paths) == 1:
            single_mpt_patients += 1
        elif len(paths) > 1:
            multiple_mpt_patients += 1
            problematic_patients[patient_id] = paths

    return {
        "total_with_mpt": len(patient_mpt_map),
        "single_mpt": single_mpt_patients,
        "multiple_mpt": multiple_mpt_patients,
        "problem_cases": problematic_patients
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify that each patient has only one RTDOSE-MPT directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to the base directory containing all patient data."
    )

    args = parser.parse_args()
    results = verify_mpt_uniqueness(args.base_dir)

    if results:
        print("\n" + "="*45)
        print("=== MPT Uniqueness Verification Results ===")
        print("="*45)
        print(f"Total Unique Patients with MPT Series: {results['total_with_mpt']}")
        print("-"*45)
        print(f"Patients with EXACTLY ONE MPT series:  {results['single_mpt']} ✅")
        print(f"Patients with MULTIPLE MPT series:   {results['multiple_mpt']} ⚠️")
        print("="*45)

        if results['problem_cases']:
            print("\nDetails of Patients with Multiple MPT Series:")
            print("-------------------------------------------\n")
            for patient_id, paths in results['problem_cases'].items():
                print(f"🚨 Patient ID: {patient_id}")
                for path in paths:
                    print(f"   - Found Series: {path}")
                print("") # Add a blank line for readability