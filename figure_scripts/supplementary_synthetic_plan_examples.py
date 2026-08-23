#!/usr/bin/env python3
"""Render representative RTPLAN-based OpenTPS dose perturbations."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy import ndimage

from red_journal_style import apply_red_journal_style, save_publication_figure

warnings.filterwarnings("ignore", message="Invalid value for VR IS.*", category=UserWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path(
            "outputs/paper/supplementary_synthetic_plan_examples_ukhd/"
            "opentps_inputs/cropped_mapping.json"
        ),
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=Path(
            "outputs/paper/supplementary_synthetic_plan_examples_ukhd/opentps_cache"
        ),
    )
    parser.add_argument(
        "--lung-mask-root",
        type=Path,
        default=Path(
            "outputs/paper/supplementary_synthetic_plan_examples_ukhd/no_lung_masks"
        ),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("oliver_paper/figures"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("oliver_paper/tables"),
    )
    parser.add_argument(
        "--clinical-table",
        type=Path,
        default=Path("oliver_paper/tables/table1_patient_level_characteristics_source.csv"),
    )
    parser.add_argument(
        "--hypofractionated-patient",
        help="Optional source patient key. By default, select the first eligible case with <=10 fractions.",
    )
    parser.add_argument(
        "--conventional-patient",
        help="Optional source patient key. By default, select the first eligible case with >10 fractions.",
    )
    return parser.parse_args()


def read_on_reference(path: Path, reference: sitk.Image) -> sitk.Image:
    image = sitk.ReadImage(str(path))
    if (
        image.GetSize() == reference.GetSize()
        and np.allclose(image.GetSpacing(), reference.GetSpacing())
        and np.allclose(image.GetOrigin(), reference.GetOrigin())
        and np.allclose(image.GetDirection(), reference.GetDirection())
    ):
        return image
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )


def largest_components(mask: np.ndarray, count: int = 2) -> np.ndarray:
    labels, n_labels = ndimage.label(mask)
    if n_labels == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n_labels + 1))
    keep = np.argsort(sizes)[-count:] + 1
    return np.isin(labels, keep)


def approximate_lung_mask(ct: np.ndarray) -> np.ndarray:
    # Used only for the descriptive MLD quality-control values in the CSV.
    air_like = (ct >= -1024.0) & (ct <= -250.0)
    mask = largest_components(air_like, count=2)
    structure = ndimage.generate_binary_structure(3, 1)
    return ndimage.binary_closing(mask, structure=structure, iterations=2)


def plan_metadata(path: Path) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = pydicom.dcmread(str(path), stop_before_pixels=True)
    fractions = None
    if getattr(ds, "FractionGroupSequence", None):
        raw = getattr(ds.FractionGroupSequence[0], "NumberOfFractionsPlanned", None)
        fractions = int(float(raw)) if raw is not None else None

    prescription = None
    for dose_ref in getattr(ds, "DoseReferenceSequence", []):
        value = getattr(dose_ref, "TargetPrescriptionDose", None)
        if value is not None and float(value) > 0:
            prescription = float(value)
            break
    return {
        "fractions_planned": fractions,
        "clinical_prescription_gy": prescription,
        "clinical_beam_count": len(getattr(ds, "BeamSequence", [])),
    }


def select_cases(
    mapping: dict[str, dict],
    synthetic_dir: Path,
    hypofractionated_patient: str | None,
    conventional_patient: str | None,
) -> list[dict[str, str]]:
    """Select one source-domain example from each fractionation category."""

    def is_renderable(patient_id: str) -> bool:
        source = mapping.get(patient_id, {})
        rtplan_path = Path(str(source.get("rtplan_path", "")))
        if not rtplan_path.exists():
            return False
        return all(
            (
                synthetic_dir
                / f"{patient_id}_variant_{index:02d}"
                / f"{patient_id}_variant_{index:02d}_dose.nrrd"
            ).exists()
            for index in (1, 2)
        )

    eligible: list[tuple[str, int]] = []
    for patient_id in sorted(mapping):
        if not is_renderable(patient_id):
            continue
        fractions = plan_metadata(Path(mapping[patient_id]["rtplan_path"]))["fractions_planned"]
        if fractions is not None:
            eligible.append((patient_id, int(fractions)))

    def choose(explicit: str | None, *, low_fraction: bool) -> tuple[str, int]:
        if explicit is not None:
            match = next((item for item in eligible if item[0] == explicit), None)
            if match is None:
                raise ValueError(f"Requested patient is not renderable: {explicit}")
            return match
        match = next((item for item in eligible if (item[1] <= 10) == low_fraction), None)
        if match is None:
            category = "<=10" if low_fraction else ">10"
            raise ValueError(f"No renderable source patient with {category} fractions was found")
        return match

    hypofractionated = choose(hypofractionated_patient, low_fraction=True)
    conventional = choose(conventional_patient, low_fraction=False)
    return [
        {
            "patient_id": hypofractionated[0],
            "display_label": "Patient A",
            "fractionation_category": "Hypofractionated",
        },
        {
            "patient_id": conventional[0],
            "display_label": "Patient B",
            "fractionation_category": "Conventionally fractionated",
        },
    ]


def select_slice(dose: np.ndarray, scale: float) -> int:
    high_dose_area = np.sum(dose >= 0.80 * scale, axis=(1, 2))
    if np.any(high_dose_area):
        return int(np.argmax(high_dose_area))
    return int(np.unravel_index(np.argmax(dose), dose.shape)[0])


def display_bounds(ct_slice: np.ndarray, dose_slice: np.ndarray) -> tuple[slice, slice]:
    del dose_slice  # Display cropping is anatomy-based so beam spill does not enlarge the panel.
    body = ct_slice > -700.0
    body = ndimage.binary_closing(body, iterations=3)
    body = ndimage.binary_fill_holes(body)
    labels, n_labels = ndimage.label(body)
    if n_labels:
        sizes = ndimage.sum(body, labels, index=np.arange(1, n_labels + 1))
        body = labels == (int(np.argmax(sizes)) + 1)
        body = ndimage.binary_fill_holes(body)
    ys, xs = np.where(body)
    if len(xs) == 0:
        return slice(0, ct_slice.shape[0]), slice(0, ct_slice.shape[1])
    pad = 5
    y0, y1 = max(0, int(ys.min()) - pad), min(ct_slice.shape[0], int(ys.max()) + pad + 1)
    x0, x1 = max(0, int(xs.min()) - pad), min(ct_slice.shape[1], int(xs.max()) + pad + 1)
    return slice(y0, y1), slice(x0, x1)


def load_lung_mask(patient_id: str, root: Path, reference: sitk.Image) -> tuple[np.ndarray, str]:
    patient_dir = root / patient_id
    paths = [
        patient_dir / f"{patient_id}_left_lung_mask.nrrd",
        patient_dir / f"{patient_id}_right_lung_mask.nrrd",
    ]
    if all(path.exists() for path in paths):
        masks = []
        for path in paths:
            mask = sitk.ReadImage(str(path))
            mask = sitk.Resample(
                mask,
                reference,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                sitk.sitkUInt8,
            )
            masks.append(sitk.GetArrayFromImage(mask) > 0)
        combined = masks[0] | masks[1]
        if np.any(combined):
            return combined, "cached TotalSegmentator left+right lung masks"
    ct = sitk.GetArrayFromImage(reference).astype(np.float32)
    return approximate_lung_mask(ct), "CT-threshold fallback"


def load_case(
    case: dict[str, str],
    mapping: dict[str, dict],
    synthetic_dir: Path,
    lung_mask_root: Path,
    clinical_lookup: dict[str, dict[str, str]],
) -> dict:
    patient_id = case["patient_id"]
    source = mapping[patient_id]
    ct_image = sitk.ReadImage(source["ct_path"])
    clinical_image = read_on_reference(Path(source["dose_path"]), ct_image)
    ct = sitk.GetArrayFromImage(ct_image).astype(np.float32)
    clinical = sitk.GetArrayFromImage(clinical_image).astype(np.float32)

    variant_ids = [f"{patient_id}_variant_{idx:02d}" for idx in range(1, 3)]
    variant_paths = [
        synthetic_dir / variant_id / f"{variant_id}_dose.nrrd"
        for variant_id in variant_ids
    ]
    missing = [str(path) for path in variant_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing synthetic dose files:\n" + "\n".join(missing))
    synthetic = [
        sitk.GetArrayFromImage(read_on_reference(path, ct_image)).astype(np.float32)
        for path in variant_paths
    ]

    pmeta = plan_metadata(Path(source["rtplan_path"]))
    clinical_meta = clinical_lookup.get(patient_id, {})
    if pmeta["clinical_prescription_gy"] is None and clinical_meta.get("total_dose_gy"):
        pmeta["clinical_prescription_gy"] = float(clinical_meta["total_dose_gy"])
    if clinical_meta.get("rt_technique_display"):
        pmeta["rt_technique"] = clinical_meta["rt_technique_display"]
    else:
        pmeta["rt_technique"] = "not available"
    positive = clinical[clinical > 0]
    clinical_scale = float(pmeta["clinical_prescription_gy"] or np.percentile(positive, 99.0))
    slice_index = select_slice(clinical, clinical_scale)
    bounds = display_bounds(ct[slice_index], clinical[slice_index])
    lung_mask, lung_mask_source = load_lung_mask(patient_id, lung_mask_root, ct_image)

    perturbations = []
    for variant_id in variant_ids:
        meta_path = synthetic_dir / variant_id / f"{variant_id}_metadata.json"
        perturbations.append(json.loads(meta_path.read_text())["perturbation"])

    return {
        **case,
        **pmeta,
        "ct": ct,
        "clinical": clinical,
        "synthetic": synthetic,
        "variant_ids": variant_ids,
        "perturbations": perturbations,
        "clinical_scale": clinical_scale,
        "slice_index": slice_index,
        "bounds": bounds,
        "lung_mask": lung_mask,
        "lung_mask_source": lung_mask_source,
        "spacing_xyz": ct_image.GetSpacing(),
    }


def render(cases: list[dict], output_base: Path) -> None:
    apply_red_journal_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.65), facecolor="white")
    titles = ("Clinical dose", "Synthetic 1", "Synthetic 2")
    cmap = plt.get_cmap("turbo").copy()
    cmap.set_bad(alpha=0.0)
    norm = Normalize(0.0, 1.0)
    contour_level = 0.80
    contour_color = "#202020"

    for row, case in enumerate(cases):
        z = case["slice_index"]
        ys, xs = case["bounds"]
        ct_slice = case["ct"][z, ys, xs]
        dose_slices = [case["clinical"][z, ys, xs], *[d[z, ys, xs] for d in case["synthetic"]]]
        aspect = float(case["spacing_xyz"][1] / case["spacing_xyz"][0])

        for col, ax in enumerate(axes[row]):
            ax.imshow(ct_slice, cmap="gray", vmin=-1000, vmax=300, origin="lower", aspect=aspect)
            normalized = np.clip(dose_slices[col] / case["clinical_scale"], 0.0, 1.0)
            overlay = np.ma.masked_where(normalized < 0.08, normalized)
            ax.imshow(overlay, cmap=cmap, norm=norm, alpha=0.34, origin="lower", aspect=aspect)
            if float(np.nanmax(normalized)) >= contour_level:
                ax.contour(
                    normalized,
                    levels=(contour_level,),
                    colors=contour_color,
                    linestyles="-",
                    linewidths=0.60,
                    alpha=0.95,
                    origin="lower",
                )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color("black")
            if row == 0:
                ax.set_title(titles[col], fontsize=8.5, pad=3, fontweight="normal")

    fig.subplots_adjust(left=0.025, right=0.99, top=0.86, bottom=0.20, wspace=0.022, hspace=0.30)
    top_position = axes[0, 0].get_position()
    bottom_position = axes[1, 0].get_position()
    fig.text(
        0.025,
        min(0.98, top_position.y1 + 0.105),
        f"A. SBRT / hypofractionated RT ({cases[0]['fractions_planned']} fractions)",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
    )
    fig.text(
        0.025,
        bottom_position.y1 + 0.020,
        f"B. Conventional RT ({cases[1]['fractions_planned']} fractions)",
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    colorbar_y = max(0.08, bottom_position.y0 - 0.105)
    colorbar_ax = fig.add_axes([0.28, colorbar_y, 0.30, 0.017])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
        orientation="horizontal",
    )
    colorbar.set_label("Normalized dose", fontsize=9, labelpad=2)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=8, width=0.8, length=2.5)
    contour_handle = Line2D([0], [0], color=contour_color, linewidth=0.8, linestyle="-")
    fig.legend(
        handles=[contour_handle],
        labels=["80% isodose"],
        loc="center left",
        bbox_to_anchor=(0.63, colorbar_y + 0.008),
        frameon=False,
        fontsize=8,
        handlelength=1.8,
        borderaxespad=0.0,
    )
    save_publication_figure(fig, output_base, dpi=600, formats=("png", "pdf"), bbox_inches=None)
    plt.close(fig)


def metadata_table(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        lung = case["lung_mask"]
        synthetic_max = [float(np.max(dose)) for dose in case["synthetic"]]
        synthetic_mld = [float(np.mean(dose[lung])) for dose in case["synthetic"]]
        gantry_jitter = [float(np.max(np.abs(p["gantry_jitter_pct"]))) * 100.0 for p in case["perturbations"]]
        couch_jitter = [float(np.max(np.abs(p["couch_jitter_pct"]))) * 100.0 for p in case["perturbations"]]
        iso_shift = [float(np.linalg.norm(p["isocenter_shift_mm"])) for p in case["perturbations"]]
        mu_scale = [float(p["mu_scale"]) for p in case["perturbations"]]
        rows.append(
            {
                "patient_id": case["patient_id"],
                "anonymized_figure_label": case["display_label"],
                "fractionation_category": case["fractionation_category"],
                "fractions_planned": case["fractions_planned"],
                "clinical_beam_count": case["clinical_beam_count"],
                "rt_technique": case["rt_technique"],
                "clinical_prescription_gy": case["clinical_prescription_gy"],
                "normalization_reference_gy": case["clinical_scale"],
                "number_synthetic_plans_displayed": len(case["synthetic"]),
                "selected_synthetic_plan_ids": ";".join(case["variant_ids"]),
                "clinical_max_dose_gy": float(np.max(case["clinical"])),
                "synthetic_max_dose_gy": ";".join(f"{value:.4f}" for value in synthetic_max),
                "clinical_mean_lung_dose_gy": float(np.mean(case["clinical"][lung])),
                "synthetic_mean_lung_dose_gy": ";".join(f"{value:.4f}" for value in synthetic_mld),
                "lung_mask_source": case["lung_mask_source"],
                "selected_axial_slice_index": case["slice_index"],
                "max_abs_gantry_jitter_pct": ";".join(f"{value:.3f}" for value in gantry_jitter),
                "max_abs_couch_jitter_pct": ";".join(f"{value:.3f}" for value in couch_jitter),
                "isocenter_shift_norm_mm": ";".join(f"{value:.3f}" for value in iso_shift),
                "effective_prescription_scale": ";".join(f"{value:.4f}" for value in mu_scale),
                "illustrative_opentps_calculation_spacing_mm": 2.5,
                "illustrative_opentps_beamlet_spacing_mm": 20.0,
                "illustrative_optimizer_max_iterations": 8,
                "generation_note": (
                    "Illustrative OpenTPS recomputation on source-domain UKHD-train cases; "
                    "dynamic delivery represented by sampled clinical control-point directions"
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text())
    clinical_table = pd.read_csv(args.clinical_table, dtype={"patient_key": str})
    clinical_lookup = clinical_table.set_index("patient_key").fillna("").to_dict("index")
    case_specs = select_cases(
        mapping,
        args.synthetic_dir,
        args.hypofractionated_patient,
        args.conventional_patient,
    )
    cases = [
        load_case(case, mapping, args.synthetic_dir, args.lung_mask_root, clinical_lookup)
        for case in case_specs
    ]

    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    output_base = args.figure_dir / "supplementary_synthetic_plan_examples"
    render(cases, output_base)

    table = metadata_table(cases)
    table_path = args.table_dir / "supplementary_synthetic_plan_examples_metadata.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    print(f"[saved] {output_base.with_suffix('.png')}")
    print(f"[saved] {output_base.with_suffix('.pdf')}")
    print(f"[saved] {table_path}")


if __name__ == "__main__":
    main()
