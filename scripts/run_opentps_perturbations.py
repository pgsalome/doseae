#!/usr/bin/env python3
"""
Generate clinically-perturbed synthetic dose plans using OpenTPS.

For a fresh checkout, apply the tracked OpenTPS DICOM compatibility patch with
``scripts/apply_opentps_vmat_sbrt_patch.sh`` before processing dynamic plans.

Example:
    python scripts/run_opentps_perturbations.py \
        --config config/single_patient_synth.yaml \
        --mapping data/ct_dose_file_mapping.json \
        --patient_id 0617697905 \
        --n_samples 3 \
        --output_dir outputs/perturbations
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import yaml

from entities.lung.augmentation import ClinicalPlanPerturbator

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as handle:
        return yaml.safe_load(handle)


def pick_patient(mapping_path: Path, requested_id: str | None) -> Tuple[str, Dict]:
    with mapping_path.open("r") as handle:
        mapping = json.load(handle)
    if requested_id:
        if requested_id not in mapping:
            raise KeyError(f"Patient {requested_id} not found in {mapping_path}")
        return requested_id, mapping[requested_id]
    patient_id = next(iter(mapping))
    return patient_id, mapping[patient_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenTPS clinical perturbations.")
    parser.add_argument("--config", required=True, help="Path to YAML config with augmentation settings.")
    parser.add_argument(
        "--mapping",
        default="data/ct_dose_file_mapping.json",
        help="JSON mapping from patient_id to CT / dose / RTPLAN paths.",
    )
    parser.add_argument("--patient_id", default=None, help="Patient ID to process (defaults to first entry).")
    parser.add_argument("--n_samples", type=int, default=3, help="Number of synthetic variants to generate.")
    parser.add_argument(
        "--output_dir",
        default="outputs/perturbations",
        help="Directory where synthetic doses will be stored.",
    )
    parser.add_argument(
        "--retain_reference",
        action="store_true",
        help="Persist CT and normalised dose alongside synthetic outputs.",
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

    config_path = Path(args.config).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    augmentation_cfg = config.get("augmentation", {})
    if not augmentation_cfg:
        raise RuntimeError("Config is missing the 'augmentation' block required for perturbations.")

    patient_id, meta = pick_patient(mapping_path, args.patient_id)
    ct_path = Path(meta["ct_path"])
    dose_path = Path(meta["dose_path"])
    rtplan_path = Path(meta.get("rtplan_path", meta.get("plan_path", "")))
    if not rtplan_path:
        raise RuntimeError(f"No RTPLAN path provided for patient {patient_id}.")

    logger.info("Generating %d variants for patient %s", args.n_samples, patient_id)
    perturbator = ClinicalPlanPerturbator(augmentation_cfg)
    variants = perturbator.generate_from_paths(
        patient_id=patient_id,
        ct_path=ct_path,
        dose_path=dose_path,
        rtplan_path=rtplan_path,
        n_samples=args.n_samples,
        prescribed_dose=meta.get("prescribed_dose"),
        output_dir=output_dir,
        retain_reference=args.retain_reference,
    )

    logger.info("Generated %d variants:", len(variants))
    for item in variants:
        logger.info("  %s -> stored in %s", item["patient_id"], output_dir / item["patient_id"])


if __name__ == "__main__":
    main()
