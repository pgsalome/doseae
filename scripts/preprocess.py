#!/usr/bin/env python3
"""
New preprocessing script that supports both image and patch experiments
with the new HDF5 structure.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
import gc

from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_data_splits(splits_path: str) -> Dict[str, List[Dict]]:
    """Load data splits from JSON file."""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return splits

def process_patients(patients: List[Dict], config: Dict[str, Any], 
                    experiment_type: str, output_dir: Path, split_name: str,
                    workers: int):
    """Process a list of patients."""
    logger.info(f"Processing {len(patients)} patients for {experiment_type} experiments")
    
    # Set output directory based on experiment type
    preprocessor = LungPreprocessor(config, str(output_dir))
    patient_ids = [patient['patient_id'] for patient in patients]
    extra_patient_ids: List[str] = []
    
    results: List[Dict[str, Any]] = []
    collect_results = experiment_type != 'patch'

    def _process_single(item):
        index, patient = item
        patient_id = patient['patient_id']
        ct_path = patient['ct_path']
        dose_path = patient['dose_path']
        prescribed_dose = patient.get('prescribed_dose', 0.0)
        
        logger.info(f"Processing patient {index+1}/{len(patients)}: {patient_id}")
        
        if preprocessor.has_processed_patient(split_name, patient_id, experiment_type):
            logger.info(f"⏭️  Skipping {patient_id} (cache already exists)")
            return None
        
        try:
            result = preprocessor.process_patient(
                patient_id=patient_id,
                ct_path=ct_path,
                dose_path=dose_path,
                prescribed_dose=prescribed_dose,
                split_name=split_name,
                experiment_type=experiment_type
            )
            # Force garbage collection after each patient to free memory
            gc.collect()
            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            synthetic_results = result.pop('synthetic_results', []) if isinstance(result, dict) else []
            combined = [result] + synthetic_results if isinstance(result, dict) else []
            if not combined:
                return []
            if collect_results:
                return combined
            return [{'patient_id': entry.get('patient_id')} for entry in combined if entry.get('patient_id')]
        except Exception as e:
            logger.error(f"❌ Failed to process {patient_id}: {e}")
            # Clean up memory even on failure
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            return []
    
    if workers <= 1:
        for idx, patient in enumerate(patients):
            try:
                entries = _process_single((idx, patient))
                if entries:
                    results.extend(entries)
                    for entry in entries:
                        pid = entry.get('patient_id')
                        if pid and pid not in patient_ids and pid not in extra_patient_ids:
                            extra_patient_ids.append(pid)
            except Exception as e:
                logger.error(f"❌ Unexpected error processing patient {idx+1}: {e}")
                # Continue with next patient
                continue
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_patient = {
                executor.submit(_process_single, (idx, patient)): patient
                for idx, patient in enumerate(patients)
            }
            for future in as_completed(future_to_patient):
                try:
                    entries = future.result()
                    if entries:
                        results.extend(entries)
                        for entry in entries:
                            pid = entry.get('patient_id')
                            if pid and pid not in patient_ids and pid not in extra_patient_ids:
                                extra_patient_ids.append(pid)
                except Exception as e:
                    patient = future_to_patient[future]
                    logger.error(f"❌ Unexpected error processing patient {patient['patient_id']}: {e}")
                    # Continue with next patient
                    continue
    
    patient_ids.extend(extra_patient_ids)
    return preprocessor, results, patient_ids

def main():
    parser = argparse.ArgumentParser(description='Preprocess data with new pipeline')
    parser.add_argument('--config', required=True, help='Path to experiment config file')
    parser.add_argument('--splits', required=True, help='Path to data splits JSON file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--n_patients', type=int, default=5, help='Number of patients to process')
    parser.add_argument('--experiment_type', choices=['image', 'patch'], required=True,
                       help='Type of experiment: image or patch')
    parser.add_argument('--workers', type=int, default=1, help='Number of patients to process in parallel')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    augmentation_cfg = config.get('augmentation', {})
    if augmentation_cfg.get('enable_synthetic_dose', False):
        synthetic_count = int(augmentation_cfg.get('synthetic_doses_per_patient', 0) or 0)
        logger.info(
            "Synthetic dose augmentation enabled: targeting %d synthetic dose volumes per patient.",
            synthetic_count,
        )
    else:
        logger.info("Synthetic dose augmentation disabled; proceeding with original cohort only.")
    
    # Load data splits
    splits = load_data_splits(args.splits)
    
    # Set experiment type in config
    if args.experiment_type == 'patch':
        config['dataset']['dataset_type'] = 'patches_only'
    else:
        config['dataset']['dataset_type'] = 'full_images'
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Decide which split-level HDF5 files must exist to consider preprocessing complete.
    expected_splits = [name for name in ['train', 'val', 'test'] if name in splits and splits[name]]
    if args.experiment_type == 'patch':
        required_paths = [
            output_dir / 'processed_patches' / f'{split_name}.h5'
            for split_name in expected_splits
        ]
    else:
        required_paths = [
            output_dir / 'processed_images' / f'{split_name}.h5'
            for split_name in expected_splits
        ]

    if expected_splits and all(path.exists() for path in required_paths):
        logger.info("⏭️  Skipping preprocessing - all required HDF5 files already exist:")
        for path in required_paths:
            logger.info("    %s", path)
        logger.info("🎉 Preprocessing complete!")
        return
    
    # Process patients from each split
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            logger.warning(f"Split {split_name} not found in data splits")
            continue
        
        patients = splits[split_name][:args.n_patients]  # Limit to n_patients
        
        if not patients:
            logger.warning(f"No patients in {split_name} split")
            continue
        
        logger.info(f"Processing {split_name} split with {len(patients)} patients")
        
        # Process patients
        preprocessor, results, patient_ids = process_patients(
            patients, config, args.experiment_type, output_dir, split_name, args.workers
        )
        patch_h5_path = output_dir / 'processed_patches' / f'{split_name}.h5'
        image_h5_path = output_dir / 'processed_images' / f'{split_name}.h5'
        
        if args.experiment_type == 'patch':
            # Check if HDF5 file already exists
            if patch_h5_path.exists():
                logger.info(f"⏭️  Skipping HDF5 generation for {split_name} (file already exists: {patch_h5_path})")
            elif results:
                preprocessor.consolidate_patches_to_hdf5(
                    results, str(patch_h5_path), split_name=split_name, all_patient_ids=patient_ids
                )
            else:
                preprocessor.consolidate_patches_from_cache(split_name, patient_ids, str(patch_h5_path))
        
        if args.experiment_type == 'image':
            # Check if HDF5 file already exists
            if image_h5_path.exists():
                logger.info(f"⏭️  Skipping HDF5 generation for {split_name} (file already exists: {image_h5_path})")
            elif results:
                preprocessor.consolidate_images_to_hdf5(results, str(image_h5_path))
            else:
                preprocessor.consolidate_images_from_disk(split_name, patient_ids, str(image_h5_path))
                logger.info(f"✅ Image cache updated at {image_h5_path}")

    logger.info("🎉 Preprocessing complete!")

if __name__ == '__main__':
    main()
