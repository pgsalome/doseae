#!/usr/bin/env python3
"""
Script to convert existing pickle files to HDF5 format for space savings.

Usage:
    python convert_pickle_to_hdf5.py /path/to/output_directory
    python convert_pickle_to_hdf5.py /path/to/output_directory --dry-run
    python convert_pickle_to_hdf5.py /path/to/output_directory --keep-originals
"""

import os
import sys
import pickle
import h5py
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import gc
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pickle_to_hdf5_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def check_disk_space(path, min_gb_required=1):
    """Check if enough disk space is available."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)

    logger.info(f"Available disk space: {free_gb:.2f} GB")

    if free_gb < min_gb_required:
        logger.warning(f"⚠️  Low disk space! Only {free_gb:.2f}GB available")
        return False
    return True


def append_to_hdf5(file_path, data_list, compression='gzip', compression_opts=9):
    """Append patch data to HDF5 file with compression."""
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

        logger.debug(f"Successfully appended {len(data_list)} patches to {file_path}")

    except Exception as e:
        logger.error(f"Failed to append to HDF5 file {file_path}: {e}")
        raise


def verify_hdf5_conversion(pickle_path, hdf5_path):
    """Verify that HDF5 file contains the same data as pickle file."""
    try:
        # Load original pickle data
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)

        # Load HDF5 data and count patches
        with h5py.File(hdf5_path, 'r') as f:
            hdf5_patch_count = len([key for key in f.keys() if key.startswith('patch_')])

        # Compare counts
        if hdf5_patch_count == len(pickle_data):
            logger.debug(f"✓ Verification passed: {len(pickle_data)} patches in both files")
            return True
        else:
            logger.error(f"✗ Verification failed: pickle has {len(pickle_data)} patches, HDF5 has {hdf5_patch_count}")
            return False

    except Exception as e:
        logger.error(f"Verification failed for {pickle_path}: {e}")
        return False


def convert_single_pickle_file(pickle_path, hdf5_path, keep_original=False):
    """Convert a single pickle file to HDF5."""
    try:
        # Load pickle data
        logger.info(f"Loading {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)

        # Remove existing HDF5 file if it exists
        if Path(hdf5_path).exists():
            logger.info(f"Removing existing HDF5 file: {hdf5_path}")
            Path(hdf5_path).unlink()

        # Convert to HDF5
        logger.info(f"Converting to HDF5: {hdf5_path}")
        append_to_hdf5(hdf5_path, data)

        # Verify conversion
        if not verify_hdf5_conversion(pickle_path, hdf5_path):
            logger.error(f"Conversion verification failed for {pickle_path}")
            return None

        # Calculate size savings
        pickle_size = Path(pickle_path).stat().st_size
        hdf5_size = Path(hdf5_path).stat().st_size
        savings_mb = (pickle_size - hdf5_size) / (1024**2)
        compression_ratio = (1 - hdf5_size / pickle_size) * 100

        logger.info(f"✓ Conversion successful!")
        logger.info(f"  Original: {pickle_size / (1024**2):.1f} MB")
        logger.info(f"  HDF5:     {hdf5_size / (1024**2):.1f} MB")
        logger.info(f"  Saved:    {savings_mb:.1f} MB ({compression_ratio:.1f}% reduction)")

        # Delete original pickle file if requested
        if not keep_original:
            os.remove(pickle_path)
            logger.info(f"  Deleted original pickle file")

        # Force garbage collection
        del data
        gc.collect()

        return savings_mb

    except Exception as e:
        logger.error(f"Failed to convert {pickle_path}: {e}")
        # Clean up partial HDF5 file on failure
        if Path(hdf5_path).exists():
            Path(hdf5_path).unlink()
        return None


def find_pickle_files(base_directory):
    """Find all pickle files in the directory structure."""
    pickle_files = []

    # Look for individual patient files
    patient_pickle_files = list(Path(base_directory).rglob("*_patches.pkl"))
    pickle_files.extend(patient_pickle_files)

    # Look for combined dataset files
    combined_pickle_files = list(Path(base_directory).glob("patches_dataset_final.pkl"))
    pickle_files.extend(combined_pickle_files)

    return pickle_files


def estimate_savings(pickle_files):
    """Estimate total space savings from conversion."""
    total_size = 0
    for pickle_file in pickle_files:
        total_size += pickle_file.stat().st_size

    total_size_gb = total_size / (1024**3)
    estimated_savings_gb = total_size_gb * 0.7  # Conservative 70% compression

    logger.info(f"Found {len(pickle_files)} pickle files")
    logger.info(f"Total size: {total_size_gb:.2f} GB")
    logger.info(f"Estimated savings: {estimated_savings_gb:.2f} GB")

    return total_size_gb, estimated_savings_gb


def convert_all_pickle_files(base_directory, dry_run=False, keep_originals=False):
    """Convert all pickle files in the directory to HDF5."""

    # Check initial disk space
    if not check_disk_space(base_directory, min_gb_required=1):
        logger.error("Insufficient disk space to proceed safely!")
        return False

    # Find all pickle files
    pickle_files = find_pickle_files(base_directory)

    if not pickle_files:
        logger.info("No pickle files found to convert.")
        return True

    # Estimate savings
    total_size_gb, estimated_savings_gb = estimate_savings(pickle_files)

    if dry_run:
        logger.info("DRY RUN - No files will be modified")
        logger.info("Files that would be converted:")
        for pickle_file in pickle_files:
            hdf5_file = pickle_file.with_suffix('.h5')
            logger.info(f"  {pickle_file} → {hdf5_file}")
        return True

    # Confirm with user
    response = input(f"\nConvert {len(pickle_files)} files? This will save ~{estimated_savings_gb:.1f}GB. (y/N): ")
    if response.lower() != 'y':
        logger.info("Conversion cancelled by user.")
        return False

    # Convert files
    total_savings = 0
    successful_conversions = 0

    for pickle_file in tqdm(pickle_files, desc="Converting files"):
        hdf5_file = pickle_file.with_suffix('.h5')

        logger.info(f"\n--- Converting {pickle_file.name} ---")

        savings = convert_single_pickle_file(
            str(pickle_file),
            str(hdf5_file),
            keep_original=keep_originals
        )

        if savings is not None:
            total_savings += savings
            successful_conversions += 1
        else:
            logger.error(f"Failed to convert {pickle_file}")

    # Summary
    logger.info("\n" + "="*50)
    logger.info("CONVERSION SUMMARY")
    logger.info("="*50)
    logger.info(f"Total files processed: {len(pickle_files)}")
    logger.info(f"Successful conversions: {successful_conversions}")
    logger.info(f"Failed conversions: {len(pickle_files) - successful_conversions}")
    logger.info(f"Total space saved: {total_savings:.1f} MB ({total_savings/1024:.2f} GB)")

    if successful_conversions > 0:
        logger.info("✅ Conversion completed successfully!")

        # Check final disk space
        check_disk_space(base_directory)

        return True
    else:
        logger.error("❌ All conversions failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert pickle files to HDF5 for space savings")
    parser.add_argument("directory", help="Directory containing pickle files to convert")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be converted without doing it")
    parser.add_argument("--keep-originals", action="store_true", help="Keep original pickle files after conversion")

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        logger.error(f"Directory does not exist: {args.directory}")
        sys.exit(1)

    logger.info(f"Starting pickle to HDF5 conversion in: {args.directory}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Keep originals: {args.keep_originals}")

    # Perform conversion
    success = convert_all_pickle_files(
        args.directory,
        dry_run=args.dry_run,
        keep_originals=args.keep_originals
    )

    if success:
        logger.info("Script completed successfully!")
        sys.exit(0)
    else:
        logger.error("Script completed with errors!")
        sys.exit(1)


if __name__ == "__main__":
    main()