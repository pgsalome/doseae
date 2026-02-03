#!/usr/bin/env python3
"""
Smoke-test utility for the OpenTPS synthetic dose generator.

This bypasses the full preprocessing pipeline: it loads a single CT/dose pair,
derives a target mask from the clinical dose, and asks the OpenTPS augmentor to
generate a synthetic plan.  Useful for validating environment/setup before
running full preprocessing.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import SimpleITK as sitk
import yaml

from entities.lung.preprocessing.lung_preprocessor import LungPreprocessor

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict:
    with path.open("r") as handle:
        return yaml.safe_load(handle)


def resolve_patient_paths(mapping_path: Path, patient_id: Optional[str]) -> Dict:
    with mapping_path.open("r") as handle:
        mapping = json.load(handle)
    if patient_id:
        if patient_id not in mapping:
            raise KeyError(f"Patient {patient_id} not found in {mapping_path}")
        return {patient_id: mapping[patient_id]}
    # fall back to first entry
    first_id = next(iter(mapping))
    return {first_id: mapping[first_id]}


def main():
    parser = argparse.ArgumentParser(description="Test OpenTPS synthetic dose generation.")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config with augmentation/opentps settings",
    )
    parser.add_argument(
        "--mapping",
        default="data/ct_dose_file_mapping.json",
        help="JSON mapping from patient_id to ct/dose paths",
    )
    parser.add_argument(
        "--patient_id",
        default=None,
        help="Optional patient ID; defaults to the first entry in mapping.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/opentps_smoke_test",
        help="Directory to store synthetic dose exports.",
    )
    parser.add_argument(
        "--retain_raw",
        action="store_true",
        help="Persist the original CT/dose alongside synthetic outputs for inspection.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        help="Logging verbosity (DEBUG, INFO, WARNING, ...).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config_path = Path(args.config).resolve()
    mapping_path = Path(args.mapping).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    patient_entries = resolve_patient_paths(mapping_path, args.patient_id)
    patient_id, patient_meta = next(iter(patient_entries.items()))
    logger.info("Testing OpenTPS augmentation for patient %s", patient_id)

    preprocessor = LungPreprocessor(config, output_dir=str(output_dir / "cache"))
    if preprocessor.synthetic_augmentor is None:
        raise RuntimeError(
            "Synthetic dose augmentation is disabled or OpenTPS is unavailable. "
            "Check the config's augmentation block."
        )

    ct_path = Path(patient_meta["ct_path"]).expanduser().resolve()
    dose_path = Path(patient_meta["dose_path"]).expanduser().resolve()
    prescribed_dose = patient_meta.get("prescribed_dose") or None

    logger.info("Loading CT from %s", ct_path)
    ct_image = sitk.ReadImage(str(ct_path))
    logger.info("Loading dose from %s", dose_path)
    dose_image = sitk.ReadImage(str(dose_path))

    dose_image = preprocessor.check_and_resample_dose_to_ct_space(ct_image, dose_image)
    normalized_dose, scale_info = preprocessor.normalize_dose(dose_image, prescribed_dose)

    logger.info("Normalisation info: %s", scale_info)
    synthetic_cases = preprocessor.synthetic_augmentor.generate(
        patient_id=patient_id,
        ct_image_hu=ct_image,
        normalized_dose_image=normalized_dose,
        normalization_fn=preprocessor.normalize_dose,
        base_prescription=scale_info.get("scale") if isinstance(scale_info, dict) else None,
    )

    if not synthetic_cases:
        logger.warning("No synthetic dose generated.")
        return

    logger.info("Generated %d synthetic dose volume(s).", len(synthetic_cases))
    for synth in synthetic_cases:
        synth_id = synth["patient_id"]
        synth_dir = output_dir / synth_id
        synth_dir.mkdir(parents=True, exist_ok=True)
        dose_image_out: sitk.Image = synth["dose_image"]
        dose_path_out = synth_dir / f"{synth_id}_synthetic_dose.nrrd"
        sitk.WriteImage(dose_image_out, str(dose_path_out))
        meta_path_out = synth_dir / f"{synth_id}_metadata.json"
        meta_payload = {
            "scale_info": synth.get("scale_info"),
            "augmentation_metadata": synth.get("metadata"),
        }
        meta_path_out.write_text(json.dumps(meta_payload, indent=2))
        logger.info("Wrote synthetic dose to %s", dose_path_out)

    if args.retain_raw:
        reference_dir = output_dir / f"{patient_id}_reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct_image, str(reference_dir / f"{patient_id}_ct.nrrd"))
        sitk.WriteImage(normalized_dose, str(reference_dir / f"{patient_id}_dose_normalized.nrrd"))
        logger.info("Stored reference CT and normalised dose in %s", reference_dir)


if __name__ == "__main__":
    main()
