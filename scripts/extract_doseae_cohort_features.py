#!/usr/bin/env python3
"""Extract reusable DoseAE features for every patient in a cohort manifest.

The input CSV must contain ``patient_id`` and either:

- ``h5_path`` for an existing per-patient patch cache, or
- ``ct_path`` and ``dose_path`` for direct preprocessing and extraction.

Optional columns are ``prescribed_dose`` and ``preprocessed_dir``. Outcomes are
deliberately not consumed; labels should be joined to the exported feature table
only in a downstream supervised analysis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = REPO_ROOT / "scripts" / "extract_doseae_latents.py"
AGGREGATIONS = (
    "mean",
    "ipsilateral_mean",
    "std",
    "mean_std",
    "q95",
    "max",
    "dose_weighted_mean",
)
SAFE_PATIENT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Cohort CSV manifest.")
    parser.add_argument("--config", type=Path, required=True, help="DoseAE model config.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Frozen DoseAE checkpoint.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=AGGREGATIONS,
        default=["mean", "ipsilateral_mean"],
    )
    parser.add_argument("--device", default=None, help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--segmentation-device", default=None, help="TotalSegmentator device.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows.")
    parser.add_argument("--patch-limit", type=int, default=None, help="Optional patch limit per patient for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Re-extract patients with complete outputs.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--use-cache", action="store_true", help="Allow extractor cache lookup for CT+dose rows.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path, limit: int | None) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cohort manifest: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "patient_id" not in rows[0]:
        raise ValueError("Manifest must contain a patient_id column")
    if limit is not None:
        rows = rows[:limit]

    seen = set()
    for row_number, row in enumerate(rows, start=2):
        patient_id = str(row.get("patient_id", "")).strip()
        if not patient_id:
            raise ValueError(f"Manifest row {row_number} has an empty patient_id")
        if not SAFE_PATIENT_ID.fullmatch(patient_id):
            raise ValueError(
                f"Unsafe patient_id at row {row_number}: {patient_id!r}; "
                "use letters, numbers, period, underscore, or hyphen"
            )
        if patient_id in seen:
            raise ValueError(f"Duplicate patient_id in manifest: {patient_id}")
        seen.add(patient_id)

        h5_path = str(row.get("h5_path", "")).strip()
        ct_path = str(row.get("ct_path", "")).strip()
        dose_path = str(row.get("dose_path", "")).strip()
        if not h5_path and not (ct_path and dose_path):
            raise ValueError(
                f"Manifest row {row_number} ({patient_id}) needs h5_path or both ct_path and dose_path"
            )
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_complete(patient_dir: Path, aggregations: Sequence[str]) -> bool:
    if not any(patient_dir.glob("*_latents.pt")):
        return False
    return all(
        any(patient_dir.glob(f"*_patient_features_{aggregation}.csv"))
        and any(patient_dir.glob(f"*_patient_features_{aggregation}.npz"))
        for aggregation in aggregations
    )


def validate_resume(
    output_dir: Path,
    patient_root: Path,
    config_hash: str,
    checkpoint_hash: str,
    aggregations: Sequence[str],
    force: bool,
) -> None:
    if force:
        return
    provenance_path = output_dir / "cohort_feature_provenance.json"
    patient_outputs_exist = patient_root.exists() and any(patient_root.iterdir())
    if not provenance_path.exists():
        if patient_outputs_exist:
            raise RuntimeError(
                f"Existing patient outputs in {patient_root} have no provenance record. "
                "Use --force or a new --output-dir."
            )
        return

    previous = json.loads(provenance_path.read_text(encoding="utf-8"))
    mismatches = []
    if previous.get("config_sha256") != config_hash:
        mismatches.append("config")
    if previous.get("checkpoint_sha256") != checkpoint_hash:
        mismatches.append("checkpoint")
    if list(previous.get("aggregations", [])) != list(aggregations):
        mismatches.append("aggregations")
    if mismatches:
        raise RuntimeError(
            "Refusing to reuse cohort features generated with different "
            f"{', '.join(mismatches)}. Use --force or a new --output-dir."
        )


def require_path(raw: str, label: str, patient_id: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{patient_id}: missing {label}: {path}")
    return path


def build_command(args: argparse.Namespace, row: Dict[str, str], patient_dir: Path) -> List[str]:
    patient_id = str(row["patient_id"]).strip()
    command = [
        sys.executable,
        str(EXTRACTOR),
        "--config",
        str(args.config.resolve()),
        "--checkpoint",
        str(args.checkpoint.resolve()),
        "--patient-id",
        patient_id,
        "--output-dir",
        str(patient_dir),
        "--aggregations",
        *args.aggregations,
    ]

    h5_path = str(row.get("h5_path", "")).strip()
    if h5_path:
        command.extend(["--h5-path", str(require_path(h5_path, "h5_path", patient_id))])
    else:
        command.extend(
            [
                "--ct-path",
                str(require_path(str(row["ct_path"]).strip(), "ct_path", patient_id)),
                "--dose-path",
                str(require_path(str(row["dose_path"]).strip(), "dose_path", patient_id)),
                "--segmentation-cache-dir",
                str(args.output_dir.resolve() / "segmentations" / patient_id),
            ]
        )
        prescribed_dose = str(row.get("prescribed_dose", "")).strip()
        if prescribed_dose:
            command.extend(["--prescribed-dose", prescribed_dose])
        preprocessed_dir = str(row.get("preprocessed_dir", "")).strip()
        if preprocessed_dir:
            command.extend(
                ["--preprocessed-dir", str(require_path(preprocessed_dir, "preprocessed_dir", patient_id))]
            )
        if args.use_cache:
            command.append("--use-cache")

    if args.device:
        command.extend(["--device", args.device])
    if args.segmentation_device:
        command.extend(["--segmentation-device", args.segmentation_device])
    if args.batch_size is not None:
        command.extend(["--batch-size", str(args.batch_size)])
    if args.patch_limit is not None:
        command.extend(["--limit", str(args.patch_limit)])
    return command


def write_status(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    fieldnames = ["patient_id", "status", "message", "output_dir"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def combine_features(
    successful_dirs: Sequence[Path],
    aggregations: Sequence[str],
    output_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    for aggregation in aggregations:
        combined_rows: List[Dict[str, str]] = []
        fieldnames: List[str] | None = None
        for patient_dir in successful_dirs:
            matches = sorted(patient_dir.glob(f"*_patient_features_{aggregation}.csv"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {aggregation} feature CSV in {patient_dir}, found {len(matches)}"
                )
            with matches[0].open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                if len(rows) != 1:
                    raise RuntimeError(f"Expected one patient row in {matches[0]}, found {len(rows)}")
                if fieldnames is None:
                    fieldnames = list(reader.fieldnames or [])
                elif list(reader.fieldnames or []) != fieldnames:
                    raise RuntimeError(f"Feature schema mismatch in {matches[0]}")
                combined_rows.extend(rows)

        if not combined_rows or fieldnames is None:
            continue
        output = output_dir / f"cohort_patient_features_{aggregation}.csv"
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        outputs.append(output)
        print(f"[saved] {output}")
    return outputs


def main() -> None:
    args = parse_args()
    args.config = require_path(str(args.config), "config", "cohort")
    args.checkpoint = require_path(str(args.checkpoint), "checkpoint", "cohort")
    rows = read_manifest(args.manifest, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patient_root = args.output_dir / "patients"
    patient_root.mkdir(parents=True, exist_ok=True)
    config_hash = sha256(args.config)
    checkpoint_hash = sha256(args.checkpoint)
    validate_resume(
        args.output_dir,
        patient_root,
        config_hash,
        checkpoint_hash,
        args.aggregations,
        args.force,
    )

    statuses: List[Dict[str, object]] = []
    successful_dirs: List[Path] = []
    for index, row in enumerate(rows, start=1):
        patient_id = str(row["patient_id"]).strip()
        patient_dir = patient_root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(rows)}] {patient_id}", flush=True)

        if patient_complete(patient_dir, args.aggregations) and not args.force:
            statuses.append(
                {
                    "patient_id": patient_id,
                    "status": "cached",
                    "message": "complete outputs already exist",
                    "output_dir": str(patient_dir),
                }
            )
            successful_dirs.append(patient_dir)
            continue

        try:
            command = build_command(args, row, patient_dir)
            if args.dry_run:
                print("[dry-run] " + " ".join(command))
                status = "dry_run"
            else:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                if not patient_complete(patient_dir, args.aggregations):
                    raise RuntimeError("extractor completed without all requested outputs")
                successful_dirs.append(patient_dir)
                status = "completed"
            statuses.append(
                {
                    "patient_id": patient_id,
                    "status": status,
                    "message": "",
                    "output_dir": str(patient_dir),
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "patient_id": patient_id,
                    "status": "failed",
                    "message": str(exc),
                    "output_dir": str(patient_dir),
                }
            )
            if not args.continue_on_error:
                write_status(args.output_dir / "cohort_extraction_status.csv", statuses)
                raise

    status_path = args.output_dir / "cohort_extraction_status.csv"
    write_status(status_path, statuses)
    print(f"[saved] {status_path}")

    combined_outputs = [] if args.dry_run else combine_features(successful_dirs, args.aggregations, args.output_dir)
    failures = [row for row in statuses if row["status"] == "failed"]
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config),
        "config_sha256": config_hash,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "aggregations": list(args.aggregations),
        "n_requested": len(rows),
        "n_successful": len(successful_dirs),
        "n_failed": len(failures),
        "combined_outputs": [str(path) for path in combined_outputs],
    }
    provenance_path = args.output_dir / "cohort_feature_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {provenance_path}")

    if failures:
        raise SystemExit(f"Feature extraction failed for {len(failures)} patient(s); see {status_path}")


if __name__ == "__main__":
    main()
