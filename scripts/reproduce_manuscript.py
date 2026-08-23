#!/usr/bin/env python3
"""Rebuild the manuscript figures and tables from exported analysis data.

The default workflow is deliberately cache-only: it does not train models,
extract embeddings, or recompute dose reconstructions. Source-derived Table 1
and supplementary imaging require explicit flags because they depend on local
clinical or imaging files that must not be committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO_ROOT / "oliver_paper"
FIGURE_DIR = PAPER_ROOT / "figures"
FIGURE_DATA_DIR = PAPER_ROOT / "figure_data"
TABLE_DIR = PAPER_ROOT / "tables"
MANIFEST_PATH = PAPER_ROOT / "manifests" / "reproducibility_manifest.json"


@dataclass
class TaskResult:
    name: str
    status: str
    command: list[str] = field(default_factory=list)
    missing_inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("figures", "tables"),
        default=("figures", "tables"),
        help="Core manuscript sections to rebuild.",
    )
    parser.add_argument(
        "--include-source-derived",
        action="store_true",
        help="Also rebuild Table 1 from local clinical/imaging sources.",
    )
    parser.add_argument(
        "--include-supplementary",
        action="store_true",
        help="Also rebuild workflow and synthetic-plan supplementary figures.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--strict", action="store_true", help="Fail when any requested dependency is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    return parser.parse_args()


def absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def run_task(
    *,
    name: str,
    command: Sequence[str],
    required: Sequence[Path],
    expected_outputs: Sequence[Path],
    dry_run: bool,
    strict: bool,
) -> TaskResult:
    missing = [str(path) for path in required if not path.exists()]
    result = TaskResult(
        name=name,
        status="pending",
        command=[str(value) for value in command],
        missing_inputs=missing,
        outputs=[str(path) for path in expected_outputs],
    )
    if missing:
        result.status = "missing_inputs"
        message = f"[{name}] missing inputs: {missing}"
        if strict:
            raise FileNotFoundError(message)
        print(f"[skip] {message}")
        return result
    if dry_run:
        result.status = "dry_run"
        print(f"[dry-run] {name}: {' '.join(result.command)}")
        return result

    print(f"[run] {name}")
    subprocess.run(result.command, cwd=REPO_ROOT, check=True)
    absent_outputs = [str(path) for path in expected_outputs if not path.exists()]
    if absent_outputs:
        raise RuntimeError(f"{name} completed but outputs are missing: {absent_outputs}")
    result.status = "completed"
    return result


def copy_outputs(mapping: Sequence[tuple[Path, Path]], *, dry_run: bool) -> None:
    for source, destination in mapping:
        if dry_run:
            print(f"[dry-run] copy {source} -> {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"[copied] {destination}")


def reproduce_figures(args: argparse.Namespace, results: list[TaskResult]) -> None:
    python = str(Path(sys.executable).resolve())
    metrics_dir = REPO_ROOT / "outputs" / "paper" / "cohort_figure2"
    figure1_inputs = [
        metrics_dir / "rtog-imrt_combined.csv",
        metrics_dir / "rtog-conv_combined.csv",
        metrics_dir / "ukhd-sbrt_combined.csv",
        metrics_dir / "ukhd-other-train_combined.csv",
        REPO_ROOT
        / "outputs/paper/figures/figure_full_ukhd_ANON0024_slice429_dense_reconstruction_gamma_CDE_dense_slice_cache.npz",
        REPO_ROOT
        / "outputs/paper/figures/figure_full_ukhd_ANON0029_slice440_dense_reconstruction_gamma_FGH_dense_slice_cache.npz",
    ]

    if args.dry_run:
        figure1_temp = Path("<temporary-directory>") / "figure1"
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="doseae_manuscript_figure1_")
        figure1_temp = Path(temp_context.name)
    try:
        result = run_task(
            name="Figure 1: DoseAE reconstruction summary",
            command=[
                python,
                "figure_scripts/figure_all_cohorts_summary.py",
                "--skip-compute",
                "--metrics-dir",
                str(metrics_dir),
                "--output-dir",
                str(figure1_temp),
            ],
            required=[REPO_ROOT / "figure_scripts/figure_all_cohorts_summary.py", *figure1_inputs],
            expected_outputs=[
                figure1_temp / "figure_main_summary_all_cohorts.png",
                figure1_temp / "figure_main_summary_all_cohorts.pdf",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
        results.append(result)
        if result.status in {"completed", "dry_run"}:
            copy_outputs(
                [
                    (
                        figure1_temp / "figure_main_summary_all_cohorts.png",
                        FIGURE_DIR / "Figure1_doseae_reconstruction_summary.png",
                    ),
                    (
                        figure1_temp / "figure_main_summary_all_cohorts.pdf",
                        FIGURE_DIR / "Figure1_doseae_reconstruction_summary.pdf",
                    ),
                ],
                dry_run=args.dry_run,
            )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    figure2_data = FIGURE_DATA_DIR / "figureS_dose_patch_aggregation_sweep_heatmaps_data.csv"
    results.append(
        run_task(
            name="Figure 2: validation sweep heatmaps",
            command=[
                python,
                "figure_scripts/recreate_figure2_from_sweep_csv.py",
                "--input-csv",
                str(figure2_data),
                "--output-base",
                str(FIGURE_DIR / "Figure2_validation_sweep_heatmaps"),
                "--dpi",
                str(args.dpi),
            ],
            required=[
                REPO_ROOT / "figure_scripts/recreate_figure2_from_sweep_csv.py",
                figure2_data,
            ],
            expected_outputs=[
                FIGURE_DIR / "Figure2_validation_sweep_heatmaps.png",
                FIGURE_DIR / "Figure2_validation_sweep_heatmaps.pdf",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )

    figure3_input = FIGURE_DATA_DIR / "figure3_recreate_actual_pred_by_panel.csv"
    if args.dry_run:
        figure3_temp = Path("<temporary-directory>") / "figure3"
        temp_context = None
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="doseae_manuscript_figure3_")
        figure3_temp = Path(temp_context.name)
    try:
        result = run_task(
            name="Figure 3: classification",
            command=[
                python,
                "figure_scripts/recreate_figure3_from_actual_pred_csv.py",
                "--input-csv",
                str(figure3_input),
                "--out-dir",
                str(figure3_temp),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--seed",
                str(args.seed),
                "--dpi",
                str(args.dpi),
            ],
            required=[
                REPO_ROOT / "figure_scripts/recreate_figure3_from_actual_pred_csv.py",
                figure3_input,
            ],
            expected_outputs=[
                figure3_temp / "figure3_classification.png",
                figure3_temp / "figure3_classification.pdf",
                figure3_temp / "figure3_from_recreate_csv_summary.csv",
                figure3_temp / "figure3_from_recreate_csv_bootstrap_boxplots.csv",
                figure3_temp / "figure3_from_recreate_csv_roc_curves.csv",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
        results.append(result)
        if result.status in {"completed", "dry_run"}:
            copy_outputs(
                [
                    (figure3_temp / "figure3_classification.png", FIGURE_DIR / "Figure3_classification.png"),
                    (figure3_temp / "figure3_classification.pdf", FIGURE_DIR / "Figure3_classification.pdf"),
                    (
                        figure3_temp / "figure3_from_recreate_csv_summary.csv",
                        FIGURE_DATA_DIR / "figure3_from_recreate_csv_summary.csv",
                    ),
                    (
                        figure3_temp / "figure3_from_recreate_csv_bootstrap_boxplots.csv",
                        FIGURE_DATA_DIR / "figure3_from_recreate_csv_bootstrap_boxplots.csv",
                    ),
                    (
                        figure3_temp / "figure3_from_recreate_csv_roc_curves.csv",
                        FIGURE_DATA_DIR / "figure3_from_recreate_csv_roc_curves.csv",
                    ),
                ],
                dry_run=args.dry_run,
            )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    figure4_patient = FIGURE_DIR / "Figure4_ukhd_sbrt_km_risk_groups_patient_data.csv"
    figure4_summary = FIGURE_DIR / "Figure4_ukhd_sbrt_km_risk_groups_summary.csv"
    figure4_base = FIGURE_DIR / "Figure4_ukhd_sbrt_km_risk_groups"
    results.append(
        run_task(
            name="Figure 4: UKHD-SBRT Kaplan-Meier analysis",
            command=[
                python,
                "figure_scripts/recreate_figure4_km_from_patient_data.py",
                "--patient-data",
                str(figure4_patient),
                "--summary-csv",
                str(figure4_summary),
                "--out-base",
                str(figure4_base),
                "--dpi",
                str(args.dpi),
            ],
            required=[
                REPO_ROOT / "figure_scripts/recreate_figure4_km_from_patient_data.py",
                figure4_patient,
                figure4_summary,
            ],
            expected_outputs=[figure4_base.with_suffix(".png"), figure4_base.with_suffix(".pdf")],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )


def reproduce_tables(args: argparse.Namespace, results: list[TaskResult]) -> None:
    python = str(Path(sys.executable).resolve())
    metrics = REPO_ROOT / "outputs/paper/cohort_figure2/all_cohorts_per_patient_metrics.csv"
    results.append(
        run_task(
            name="DoseAE reconstruction tables",
            command=[
                python,
                "figure_scripts/create_doseae_mse_gamma_table.py",
                "--input-csv",
                str(metrics),
                "--output-dir",
                str(TABLE_DIR),
                "--n-bootstrap",
                "10000",
                "--seed",
                str(args.seed),
            ],
            required=[REPO_ROOT / "figure_scripts/create_doseae_mse_gamma_table.py", metrics],
            expected_outputs=[
                TABLE_DIR / "table_doseae_mse_gamma_by_model.csv",
                TABLE_DIR / "table_doseae_mse_gamma_by_model_cohort.csv",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )

    table1_source = TABLE_DIR / "table1_patient_level_characteristics_source.csv"
    results.append(
        run_task(
            name="Table 1 p-values",
            command=[
                python,
                "scripts/compute_table1_pvalues.py",
                "--csv",
                str(table1_source),
                "--out",
                str(TABLE_DIR / "table1_pvalues.csv"),
            ],
            required=[REPO_ROOT / "scripts/compute_table1_pvalues.py", table1_source],
            expected_outputs=[TABLE_DIR / "table1_pvalues.csv"],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )

    combined_predictions = TABLE_DIR / "table2_predictions_used.csv"
    results.append(
        run_task(
            name="Table 2 discrimination and DeLong results",
            command=[
                python,
                "figure_scripts/export_table2_main_results.py",
                "--combined-predictions",
                str(combined_predictions),
                "--out-dir",
                str(TABLE_DIR),
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                "--seed",
                str(args.seed),
            ],
            required=[REPO_ROOT / "figure_scripts/export_table2_main_results.py", combined_predictions],
            expected_outputs=[
                TABLE_DIR / "table2_main_data.csv",
                TABLE_DIR / "table2_delong_pvalues.csv",
                TABLE_DIR / "table2_predictions_used.csv",
                TABLE_DIR / "table2_summary.txt",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )


def reproduce_source_derived(args: argparse.Namespace, results: list[TaskResult]) -> None:
    python = str(Path(sys.executable).resolve())
    script = REPO_ROOT / "figure_scripts/export_table1_cohort_characteristics.py"
    required = [
        script,
        REPO_ROOT / "clinical_features/NCT00533949-D1-Dataset.csv",
        REPO_ROOT / "clinical_features/test_subset_early_sbrt_tp1_ct_dose_anonymized_id_map_private.csv",
        Path("/home/pgsalome/projects/git/predictR-main/clidmeraddos_features/nsclc/all/features_cli_ohe.csv"),
        Path("/home/pgsalome/projects/git/predictR-main/clidmeraddos_features/nsclc/other/features_dm_all.csv"),
    ]
    results.append(
        run_task(
            name="Table 1 source-derived characteristics",
            command=[python, str(script), "--out-dir", str(TABLE_DIR)],
            required=required,
            expected_outputs=[
                TABLE_DIR / "table1_cohort_characteristics.csv",
                TABLE_DIR / "table1_patient_level_characteristics_source.csv",
                TABLE_DIR / "table1_cohort_definitions.txt",
                TABLE_DIR / "table1_missingness_flags.csv",
                TABLE_DIR / "table1_dose_qc_exclusions.csv",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )


def reproduce_supplementary(args: argparse.Namespace, results: list[TaskResult]) -> None:
    python = str(Path(sys.executable).resolve())
    workflow_cache = FIGURE_DATA_DIR / "Figure1_representative_planning_input.npz"
    workflow_base = FIGURE_DIR / "Figure1_study_workflow"
    results.append(
        run_task(
            name="Supplementary study workflow",
            command=[
                python,
                "figure_scripts/create_figure1_study_workflow.py",
                "--image-cache",
                str(workflow_cache),
                "--output-base",
                str(workflow_base),
                "--dpi",
                str(args.dpi),
            ],
            required=[REPO_ROOT / "figure_scripts/create_figure1_study_workflow.py", workflow_cache],
            expected_outputs=[workflow_base.with_suffix(".png"), workflow_base.with_suffix(".pdf")],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )

    mapping = REPO_ROOT / "outputs/paper/supplementary_synthetic_plan_examples_ukhd/opentps_inputs/cropped_mapping.json"
    cache = REPO_ROOT / "outputs/paper/supplementary_synthetic_plan_examples_ukhd/opentps_cache"
    synthetic_base = FIGURE_DIR / "supplementary_synthetic_plan_examples"
    results.append(
        run_task(
            name="Supplementary synthetic-plan examples",
            command=[
                python,
                "figure_scripts/supplementary_synthetic_plan_examples.py",
                "--mapping",
                str(mapping),
                "--synthetic-dir",
                str(cache),
                "--figure-dir",
                str(FIGURE_DIR),
                "--table-dir",
                str(TABLE_DIR),
            ],
            required=[
                REPO_ROOT / "figure_scripts/supplementary_synthetic_plan_examples.py",
                mapping,
                cache,
                TABLE_DIR / "table1_patient_level_characteristics_source.csv",
            ],
            expected_outputs=[
                synthetic_base.with_suffix(".png"),
                synthetic_base.with_suffix(".pdf"),
                TABLE_DIR / "supplementary_synthetic_plan_examples_metadata.csv",
            ],
            dry_run=args.dry_run,
            strict=args.strict,
        )
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path: Path, results: Sequence[TaskResult], *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] write manifest {path}")
        return
    files = []
    for root in (FIGURE_DIR, FIGURE_DATA_DIR, TABLE_DIR):
        if not root.exists():
            continue
        for file_path in sorted(path for path in root.rglob("*") if path.is_file()):
            files.append(
                {
                    "path": str(file_path.relative_to(PAPER_ROOT)),
                    "bytes": file_path.stat().st_size,
                    "sha256": sha256(file_path),
                }
            )
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(REPO_ROOT),
        "paper_root": str(PAPER_ROOT),
        "tasks": [result.__dict__ for result in results],
        "files": files,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {path}")


def main() -> None:
    args = parse_args()
    for directory in (FIGURE_DIR, FIGURE_DATA_DIR, TABLE_DIR):
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)

    results: list[TaskResult] = []
    if "figures" in args.sections:
        reproduce_figures(args, results)
    if args.include_source_derived:
        reproduce_source_derived(args, results)
    if "tables" in args.sections:
        reproduce_tables(args, results)
    if args.include_supplementary:
        reproduce_supplementary(args, results)
    write_manifest(absolute(args.manifest), results, dry_run=args.dry_run)

    completed = sum(result.status == "completed" for result in results)
    skipped = sum(result.status == "missing_inputs" for result in results)
    print(f"[summary] completed={completed} skipped={skipped} total={len(results)}")
    if args.strict and skipped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
