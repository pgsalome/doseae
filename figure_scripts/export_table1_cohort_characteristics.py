#!/usr/bin/env python3
"""Export Table 1 cohort definitions and patient/treatment characteristics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "new_fig"

RTOG_CLINICAL_CSV = REPO_ROOT / "clinical_features" / "NCT00533949-D1-Dataset.csv"
UKHD_CLINICAL_CSV = Path(
    "/home/pgsalome/projects/git/predictR-main/clidmeraddos_features/nsclc/all/features_cli_ohe.csv"
)
UKHD_DOSE_FEATURE_CSV = Path(
    "/home/pgsalome/projects/git/predictR-main/clidmeraddos_features/nsclc/other/features_dm_all.csv"
)
UKHD_ANON_MAP_CSV = REPO_ROOT / "clinical_features" / "test_subset_early_sbrt_tp1_ct_dose_anonymized_id_map_private.csv"
RTOG_MAPPING_JSON = REPO_ROOT / "data" / "ct_dose_file_mapping.json"

COHORT_SPECS = [
    {
        "key": "ukhd-train",
        "label": "UKHD-train",
        "available_csv": Path("/data/pgsal/doseae/test_cohorts/ukhd-other-train/available_patients.csv"),
        "patients_csv": Path("/data/pgsal/doseae/test_cohorts/ukhd-other-train/patients.csv"),
        "source_h5": Path("/data/pgsal/doseae/ukhd_train/train_subset_all_tp1_ct_dose.h5"),
        "role": "Classifier source-domain training; institutional UKHD mixed-fractionation cohort.",
    },
    {
        "key": "rtog-other",
        "label": "RTOG-other",
        "available_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-other/available_patients.csv"),
        "patients_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-other/patients.csv"),
        "role": "Additional classifier source-domain/background cohort; RTOG-0617 patients in ct_dose_file_mapping not assigned to the available RTOG-conv or RTOG-IMRT monitor/test cohorts.",
    },
    {
        "key": "rtog-conv",
        "label": "RTOG-conv",
        "available_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-conv/available_patients.csv"),
        "patients_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-conv/patients.csv"),
        "role": "Validation/model-selection monitor cohort; available RTOG-0617 3D-CRT subset with matched toxicity/non-toxicity labels.",
    },
    {
        "key": "rtog-imrt",
        "label": "RTOG-IMRT",
        "available_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-imrt/available_patients.csv"),
        "patients_csv": Path("/data/pgsal/doseae/test_cohorts/rtog-imrt/patients.csv"),
        "role": "Held-out external test cohort; available RTOG-0617 IMRT subset with matched toxicity/non-toxicity labels.",
    },
    {
        "key": "ukhd-sbrt",
        "label": "UKHD-SBRT",
        "available_csv": Path("/data/pgsal/doseae/test_cohorts/ukhd-imrt/available_patients.csv"),
        "patients_csv": Path("/data/pgsal/doseae/test_cohorts/ukhd-imrt/patients.csv"),
        "source_h5": Path("/data/pgsal/doseae/test_subset_early_sbrt_tp1_ct_dose_anonymized.h5"),
        "role": "Held-out external test cohort; institutional UKHD early-stage SBRT cohort.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def norm_rtog_id(value: object) -> str:
    digits = re.sub(r"\D", "", str(value).strip())
    if len(digits) == 9 and digits.startswith("617"):
        digits = "0" + digits
    return digits


def norm_ukhd_id(value: object) -> str:
    return str(value).strip()


def fmt_num(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def median_range(series: pd.Series, digits: int = 1, suffix: str = "") -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "Not available"
    return f"{fmt_num(values.median(), digits)} ({fmt_num(values.min(), digits)}-{fmt_num(values.max(), digits)}){suffix}; n={len(values)}"


def count_pct(n: int, denom: int) -> str:
    if denom <= 0:
        return f"{n} (NA)"
    return f"{n} ({100.0 * n / denom:.1f}%)"


def categorical_counts(
    series: pd.Series,
    mapping: Optional[Dict[object, str]] = None,
    *,
    categories: Optional[Iterable[str]] = None,
    denom: Optional[int] = None,
    missing_label: str = "Unknown",
) -> str:
    mapped = series.copy()
    if mapping is not None:
        mapped = mapped.map(mapping)
    mapped = mapped.where(mapped.notna(), missing_label).astype(str)
    if categories is None:
        cats = [c for c in sorted(mapped.unique()) if c != missing_label]
        if missing_label in set(mapped):
            cats.append(missing_label)
    else:
        cats = list(categories)
    total = int(denom if denom is not None else len(mapped))
    parts = []
    for cat in cats:
        n = int((mapped == cat).sum())
        if n or cat == missing_label:
            parts.append(f"{cat}: {count_pct(n, total)}")
    return "; ".join(parts) if parts else "NA"


def scheme_counts(total_dose: pd.Series, fractions: pd.Series) -> str:
    dose = pd.to_numeric(total_dose, errors="coerce")
    fx = pd.to_numeric(fractions, errors="coerce")
    labels = []
    for d, f in zip(dose, fx):
        if pd.isna(d) or pd.isna(f):
            labels.append("Unknown")
        else:
            labels.append(f"{float(d):g} Gy/{int(round(float(f)))} fx")
    s = pd.Series(labels)
    counts = s.value_counts(dropna=False)
    ordered = [x for x in counts.index if x != "Unknown"] + (["Unknown"] if "Unknown" in counts.index else [])
    return "; ".join(f"{label}: {int(counts[label])}" for label in ordered)


def bed10(total_dose: pd.Series, fractions: pd.Series) -> pd.Series:
    dose = pd.to_numeric(total_dose, errors="coerce")
    fx = pd.to_numeric(fractions, errors="coerce")
    dpf = dose / fx
    return dose * (1.0 + dpf / 10.0)


def load_ukhd_sources() -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    clinical = pd.read_csv(UKHD_CLINICAL_CSV, dtype={"sub": str}, low_memory=False)
    clinical["_ukhd_join_id"] = clinical["sub"].map(norm_ukhd_id)
    dose_cols = [
        "sub",
        "mean-ipsi",
        "mean-contra",
        "v20-ipsi",
        "v20-contra",
        "volume-ptv",
        "volume-ipsi",
        "volume-contra",
    ]
    dose = pd.read_csv(UKHD_DOSE_FEATURE_CSV, usecols=lambda c: c in dose_cols, dtype={"sub": str}, low_memory=False)
    dose["_ukhd_join_id"] = dose["sub"].map(norm_ukhd_id)
    anon = pd.read_csv(UKHD_ANON_MAP_CSV, dtype={"original_patient_id": str, "anon_id": str}, low_memory=False)
    anon_map = dict(zip(anon["anon_id"].map(norm_ukhd_id), anon["original_patient_id"].map(norm_ukhd_id)))
    return clinical, dose, anon_map


def load_rtog_clinical() -> pd.DataFrame:
    clinical = pd.read_csv(RTOG_CLINICAL_CSV, dtype={"patid": str}, low_memory=False)
    clinical["_rtog_join_id"] = clinical["patid"].map(norm_rtog_id)
    return clinical


def load_cohort_patient_frame(spec: dict, rtog: pd.DataFrame, ukhd_clin: pd.DataFrame, ukhd_dose: pd.DataFrame, anon_map: Dict[str, str]) -> pd.DataFrame:
    cohort = pd.read_csv(spec["available_csv"], dtype=str, low_memory=False)
    if Path(spec["patients_csv"]).exists():
        full = pd.read_csv(spec["patients_csv"], dtype=str, low_memory=False)
        join_col = "patient_id" if "patient_id" in cohort.columns and "patient_id" in full.columns else None
        if join_col is None and "patid" in cohort.columns and "patid" in full.columns:
            join_col = "patid"
        if join_col is not None:
            extra_cols = [c for c in full.columns if c != join_col and c not in cohort.columns]
            if extra_cols:
                cohort = cohort.merge(full[[join_col] + extra_cols], on=join_col, how="left")
    cohort["cohort_key"] = spec["key"]
    cohort["cohort"] = spec["label"]
    if spec["key"].startswith("rtog"):
        cohort["patient_key"] = cohort["patient_id"].map(norm_rtog_id)
        out = cohort.merge(rtog, left_on="patient_key", right_on="_rtog_join_id", how="left", suffixes=("", "_rtog"))
        out["metadata_source"] = "RTOG-0617 public D1 dataset"
        if spec["key"] in {"rtog-conv", "rtog-imrt"}:
            out["rili_label"] = out["label"].map(lambda x: np.nan if pd.isna(x) else 0 if str(x) == "non" else 1)
            out["phenotype_label"] = out["label"].fillna("unknown")
        else:
            out["rili_label"] = pd.to_numeric(out.get("toxicity_label_0p70", 0), errors="coerce").fillna(0)
            out["phenotype_label"] = "not phenotyped; set non-RILI for classifier"
        return out

    cohort["patient_key"] = cohort["patient_id"].map(norm_ukhd_id)
    cohort["_ukhd_join_id"] = cohort["patient_key"]
    if spec["key"] == "ukhd-sbrt":
        cohort["_ukhd_join_id"] = cohort["_ukhd_join_id"].map(lambda key: anon_map.get(str(key), str(key)))
    out = cohort.merge(ukhd_clin, on="_ukhd_join_id", how="left", suffixes=("", "_ukhd"))
    out = out.merge(ukhd_dose, on="_ukhd_join_id", how="left", suffixes=("", "_dose"))
    out["metadata_source"] = "UKHD clinical/radiomics feature tables"
    out["rili_label"] = pd.to_numeric(out["toxicity_label_0p70"], errors="coerce")
    out["phenotype_label"] = out["rili_label"].map({0.0: "none", 1.0: "RILI"}).fillna("unknown")
    return out


def add_derived_treatment_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key = str(out["cohort_key"].iloc[0])
    if key.startswith("rtog"):
        arm_source = pd.to_numeric(out.get("arm"), errors="coerce") if "arm" in out.columns else pd.Series(np.nan, index=out.index)
        arm_rtog = pd.to_numeric(out.get("arm_rtog"), errors="coerce") if "arm_rtog" in out.columns else pd.Series(np.nan, index=out.index)
        arm = arm_source.where(arm_source.notna(), arm_rtog)
        out["total_dose_gy"] = np.where(arm.isin([1, 3]), 60.0, np.where(arm.isin([2, 4]), 74.0, np.nan))
        out["num_fractions"] = np.where(out["total_dose_gy"].eq(60.0), 30.0, np.where(out["total_dose_gy"].eq(74.0), 37.0, np.nan))
        out["bed10_gy"] = bed10(out["total_dose_gy"], out["num_fractions"])
        out["rt_technique_display"] = pd.to_numeric(out["rt_technique"], errors="coerce").map({1.0: "3D-CRT", 2.0: "IMRT"})
        if key == "rtog-conv":
            out["rt_technique_display"] = "3D-CRT"
        elif key == "rtog-imrt":
            out["rt_technique_display"] = "IMRT"
        out["v20_lung_percent"] = pd.to_numeric(out.get("v20_lung"), errors="coerce")
        out["mean_lung_dose_gy"] = pd.to_numeric(out.get("dmean_lung"), errors="coerce")
        out["ptv_volume_cc"] = pd.to_numeric(out.get("volume_ptv"), errors="coerce")
    else:
        source_total = out.get("Total.dose..Gy.")
        source_fx = out.get("Number.of.fractions")
        out["total_dose_gy"] = pd.to_numeric(source_total, errors="coerce")
        out["total_dose_gy"] = out["total_dose_gy"].where(
            out["total_dose_gy"].notna(), pd.to_numeric(out.get("prescribed_dose"), errors="coerce")
        )
        out["num_fractions"] = pd.to_numeric(source_fx, errors="coerce")
        out["num_fractions"] = out["num_fractions"].where(
            out["num_fractions"].notna(), pd.to_numeric(out.get("number_of_fractions_planned"), errors="coerce")
        )
        out["bed10_gy"] = pd.to_numeric(out.get("BED..Biological.effective.dose."), errors="coerce")
        out["bed10_gy"] = out["bed10_gy"].where(out["bed10_gy"].notna(), bed10(out["total_dose_gy"], out["num_fractions"]))
        out["rt_technique_display"] = "SBRT" if key == "ukhd-sbrt" else "Institutional mixed RT"
        vol_i = pd.to_numeric(out.get("volume-ipsi"), errors="coerce")
        vol_c = pd.to_numeric(out.get("volume-contra"), errors="coerce")
        denom = vol_i + vol_c
        out["v20_lung_percent"] = 100.0 * (
            pd.to_numeric(out.get("v20-ipsi"), errors="coerce") * vol_i
            + pd.to_numeric(out.get("v20-contra"), errors="coerce") * vol_c
        ) / denom
        out["mean_lung_dose_gy"] = (
            pd.to_numeric(out.get("mean-ipsi"), errors="coerce") * vol_i
            + pd.to_numeric(out.get("mean-contra"), errors="coerce") * vol_c
        ) / denom
        out["ptv_volume_cc"] = pd.to_numeric(out.get("volume-ptv"), errors="coerce") / 1000.0
    return out


def apply_table1_dose_qc(frames: Dict[str, pd.DataFrame]) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """Recode implausible Table 1 dose-summary values to missing.

    This does not modify the raw source files. It prevents source extraction
    failures or clinically implausible sentinels from appearing as real Table 1
    ranges while retaining a machine-readable audit trail.
    """

    cleaned: Dict[str, pd.DataFrame] = {}
    qc_rows: List[dict] = []

    def flag_value(df: pd.DataFrame, mask: pd.Series, variable: str, reason: str) -> None:
        if variable not in df.columns:
            return
        mask = mask.fillna(False)
        for idx, raw_value in df.loc[mask, variable].items():
            qc_rows.append(
                {
                    "cohort": df.at[idx, "cohort"],
                    "patient_key": df.at[idx, "patient_key"],
                    "variable": variable,
                    "raw_value": raw_value,
                    "qc_action": "set_to_NA_for_table1",
                    "reason": reason,
                    "metadata_source": df.at[idx, "metadata_source"],
                }
            )
        df.loc[mask, variable] = np.nan

    for cohort, source_df in frames.items():
        df = source_df.copy()
        key = str(df["cohort_key"].iloc[0])
        prescription = pd.to_numeric(df.get("total_dose_gy"), errors="coerce")
        v20 = pd.to_numeric(df.get("v20_lung_percent"), errors="coerce")
        mld = pd.to_numeric(df.get("mean_lung_dose_gy"), errors="coerce")
        ptv = pd.to_numeric(df.get("ptv_volume_cc"), errors="coerce")

        # V20 exactly zero is not clinically credible for conventionally
        # fractionated stage-III RTOG plans prescribed well above 20 Gy. In the
        # public D1 source, these rows also have problematic RT review flags.
        if key.startswith("rtog"):
            suspicious_v20_zero = prescription.ge(20.0) & v20.eq(0.0)
            flag_value(
                df,
                suspicious_v20_zero,
                "v20_lung_percent",
                "RTOG lung V20=0 despite prescription dose >=20 Gy; treated as likely DVH extraction/review issue.",
            )
            flag_value(
                df,
                suspicious_v20_zero,
                "mean_lung_dose_gy",
                "RTOG lung DVH row paired with implausible V20=0; MLD excluded with the same DVH QC flag.",
            )

            high_mld = mld.gt(35.0)
            flag_value(
                df,
                high_mld,
                "v20_lung_percent",
                "RTOG lung DVH row paired with whole-lung mean dose >35 Gy; V20 excluded with the same DVH QC flag.",
            )
            flag_value(
                df,
                high_mld,
                "mean_lung_dose_gy",
                "Whole-lung mean dose >35 Gy is clinically implausible for Table 1 summary; treated as likely DVH/structure issue.",
            )

        # Early-stage SBRT PTVs should not approach locally advanced volumes.
        # The flagged patient has PTV=773 cc and V20=0 from the UKHD radiomics
        # source, so both are excluded from descriptive ranges.
        if key == "ukhd-sbrt":
            very_large_sbrt_ptv = ptv.gt(500.0)
            flag_value(
                df,
                very_large_sbrt_ptv,
                "ptv_volume_cc",
                "UKHD-SBRT PTV volume >500 cc is implausible for early-stage SBRT; treated as structure extraction issue.",
            )
            flag_value(
                df,
                very_large_sbrt_ptv & v20.eq(0.0),
                "v20_lung_percent",
                "UKHD-SBRT patient has implausibly large PTV and V20=0; V20 treated as likely extraction issue.",
            )

        cleaned[cohort] = df

    return cleaned, pd.DataFrame(qc_rows)


def summarize_cohort(df: pd.DataFrame) -> Dict[str, str]:
    n = int(len(df))
    key = str(df["cohort_key"].iloc[0])
    out: Dict[str, str] = {}
    out["n"] = str(n)
    out["Age, years"] = median_range(df.get("age", pd.Series(dtype=float)), digits=1)

    if key.startswith("rtog"):
        out["Sex"] = categorical_counts(
            pd.to_numeric(df.get("gender"), errors="coerce"),
            {1.0: "Male", 2.0: "Female"},
            categories=["Male", "Female", "Unknown"],
            denom=n,
        )
        out["Performance status"] = categorical_counts(
            pd.to_numeric(df.get("zubrod"), errors="coerce"),
            {0.0: "Zubrod 0", 1.0: "Zubrod 1"},
            categories=["Zubrod 0", "Zubrod 1", "Unknown"],
            denom=n,
        )
        out["Smoking"] = categorical_counts(
            pd.to_numeric(df.get("smoke_hx"), errors="coerce"),
            {
                1.0: "Never",
                2.0: "Former light",
                3.0: "Former heavy",
                4.0: "Current",
                9.0: "Unknown",
            },
            categories=["Never", "Former light", "Former heavy", "Current", "Unknown"],
            denom=n,
        )
        out["COPD"] = "NA; not in RTOG public D1 table"
        out["FEV1 pre-RT"] = "NA; not in RTOG public D1 table"
        out["Stage"] = categorical_counts(
            pd.to_numeric(df.get("ajcc_stage_grp"), errors="coerce"),
            {1.0: "IIIA/N2", 2.0: "IIIB/N3"},
            categories=["IIIA/N2", "IIIB/N3", "Unknown"],
            denom=n,
        )
        out["Histology"] = categorical_counts(
            pd.to_numeric(df.get("nonsquam_squam"), errors="coerce"),
            {1.0: "Non-squamous", 2.0: "Squamous"},
            categories=["Squamous", "Non-squamous", "Unknown"],
            denom=n,
        )
        out["Location"] = "NA; central/peripheral not in RTOG public D1 table"
        out["Concurrent chemotherapy"] = categorical_counts(
            pd.to_numeric(df.get("received_conc_chemo"), errors="coerce"),
            {0.0: "No", 1.0: "Yes"},
            categories=["Yes", "No", "Unknown"],
            denom=n,
        )
    else:
        out["Sex"] = categorical_counts(
            pd.to_numeric(df.get("Gender"), errors="coerce"),
            {0.0: "Gender=0", 1.0: "Gender=1"},
            categories=["Gender=0", "Gender=1", "Unknown"],
            denom=n,
        )
        out["Performance status"] = categorical_counts(
            pd.to_numeric(df.get("KPS"), errors="coerce").map(lambda x: f"KPS {int(x)}" if pd.notna(x) else np.nan),
            None,
            denom=n,
        )
        out["Smoking"] = categorical_counts(
            pd.to_numeric(df.get("cliohe_Smoker"), errors="coerce"),
            {0.0: "Smoker=0", 1.0: "Smoker=1"},
            categories=["Smoker=0", "Smoker=1", "Unknown"],
            denom=n,
        )
        out["COPD"] = categorical_counts(
            pd.to_numeric(df.get("COPD"), errors="coerce"),
            {0.0: "No", 1.0: "Yes"},
            categories=["Yes", "No", "Unknown"],
            denom=n,
        )
        out["FEV1 pre-RT"] = (
            f"FEV1 L: {median_range(df.get('FEV1', pd.Series(dtype=float)), digits=2)}; "
            f"FEV1%: {median_range(df.get('FEV1.', pd.Series(dtype=float)), digits=1)}"
        )
        t_stage = []
        for _, row in df.iterrows():
            label = np.nan
            for col in ("T1", "T2", "T3"):
                if pd.to_numeric(pd.Series([row.get(col)]), errors="coerce").iloc[0] == 1:
                    label = col
                    break
            t_stage.append(label)
        out["Stage"] = categorical_counts(pd.Series(t_stage), None, categories=["T1", "T2", "T3", "Unknown"], denom=n)
        out["Histology"] = categorical_counts(
            pd.to_numeric(df.get("AD.vs.SC"), errors="coerce"),
            {0.0: "AD.vs.SC=0", 1.0: "AD.vs.SC=1"},
            categories=["AD.vs.SC=0", "AD.vs.SC=1", "Unknown"],
            denom=n,
        )
        out["Location"] = categorical_counts(
            pd.to_numeric(df.get("RTOG_Central"), errors="coerce"),
            {0.0: "Peripheral", 1.0: "Central"},
            categories=["Central", "Peripheral", "Unknown"],
            denom=n,
        )
        out["Concurrent chemotherapy"] = "NA; not available/applicable in UKHD feature table"

    out["RT technique"] = categorical_counts(df["rt_technique_display"], None, denom=n)
    out["Fractionation"] = scheme_counts(df["total_dose_gy"], df["num_fractions"])
    out["Total dose, Gy"] = median_range(df["total_dose_gy"], digits=1)
    out["BED10, Gy"] = median_range(df["bed10_gy"], digits=1)
    out["Lung V20, %"] = median_range(df["v20_lung_percent"], digits=1)
    out["Mean lung dose, Gy"] = median_range(df["mean_lung_dose_gy"], digits=1)
    out["PTV volume, cc"] = median_range(df["ptv_volume_cc"], digits=1)

    labeled = pd.to_numeric(df["rili_label"], errors="coerce").dropna()
    events = int(labeled.sum()) if not labeled.empty else 0
    out["Any-RILI endpoint"] = f"{events}/{len(labeled)} positive ({100 * events / len(labeled):.1f}%)" if len(labeled) else "NA"
    out["Phenotype breakdown"] = categorical_counts(
        df["phenotype_label"], None, categories=["non", "pneumonitis", "fibrosis", "both", "none", "RILI", "unknown", "not phenotyped; set non-RILI for classifier"], denom=n
    )
    out["Median follow-up, months"] = median_range(df.get("survival_months", pd.Series(dtype=float)), digits=1)
    return out


def missingness_rows(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = {
        "Age": "age",
        "Sex/Gender": lambda d: "gender" if d["cohort_key"].iloc[0].startswith("rtog") else "Gender",
        "Performance status": lambda d: "zubrod" if d["cohort_key"].iloc[0].startswith("rtog") else "KPS",
        "Smoking": lambda d: "smoke_hx" if d["cohort_key"].iloc[0].startswith("rtog") else "cliohe_Smoker",
        "COPD": lambda d: None if d["cohort_key"].iloc[0].startswith("rtog") else "COPD",
        "FEV1": lambda d: None if d["cohort_key"].iloc[0].startswith("rtog") else "FEV1",
        "Stage": lambda d: "ajcc_stage_grp" if d["cohort_key"].iloc[0].startswith("rtog") else "T1",
        "Histology": lambda d: "nonsquam_squam" if d["cohort_key"].iloc[0].startswith("rtog") else "AD.vs.SC",
        "Location": lambda d: None if d["cohort_key"].iloc[0].startswith("rtog") else "RTOG_Central",
        "V20 lung": "v20_lung_percent",
        "Mean lung dose": "mean_lung_dose_gy",
        "PTV volume": "ptv_volume_cc",
        "Follow-up": lambda d: "survival_months" if d["cohort_key"].iloc[0].startswith("rtog") else None,
    }
    rows: List[dict] = []
    for cohort, df in frames.items():
        n = len(df)
        for feature, col_spec in checks.items():
            col = col_spec(df) if callable(col_spec) else col_spec
            if col is None or col not in df.columns:
                missing = n
                source_status = "not_available"
            else:
                missing = int(df[col].isna().sum())
                source_status = "available"
            pct = 100.0 * missing / n if n else np.nan
            rows.append(
                {
                    "cohort": cohort,
                    "feature": feature,
                    "n": n,
                    "n_missing": missing,
                    "missing_percent": round(pct, 1),
                    "flag_gt20pct": bool(pct > 20.0),
                    "source_status": source_status,
                }
            )
    return pd.DataFrame(rows)


def build_definitions(frames: Dict[str, pd.DataFrame]) -> str:
    lines: List[str] = []
    lines.append("Table 1 cohort definitions")
    lines.append("")
    for spec in COHORT_SPECS:
        df = frames[spec["label"]]
        lines.append(f"{spec['label']}: n={len(df)}. {spec['role']}")
        if spec["key"] == "rtog-conv":
            requested_n = len(pd.read_csv(spec["patients_csv"]))
            lines.append(f"  Original manual 3D-CRT list contained {requested_n}; {len(df)} were available in the processed cache and used in classifier analyses.")
        if spec["key"] == "rtog-imrt":
            requested_n = len(pd.read_csv(spec["patients_csv"]))
            lines.append(f"  Original manual IMRT list contained {requested_n}; {len(df)} were available in the processed cache and used in classifier analyses.")
        if spec["key"] == "rtog-other":
            lines.append("  Constructed from ct_dose_file_mapping.json by excluding the available RTOG-conv and RTOG-IMRT patients.")
        if spec.get("source_h5"):
            lines.append(f"  Source H5: {spec['source_h5']}")
    lines.append("")
    lines.append("Patient overlap check:")
    rtog_sets = {
        spec["label"]: set(frames[spec["label"]]["patient_key"])
        for spec in COHORT_SPECS
        if spec["key"].startswith("rtog")
    }
    ukhd_sets = {
        spec["label"]: set(frames[spec["label"]]["patient_key"])
        for spec in COHORT_SPECS
        if spec["key"].startswith("ukhd")
    }
    overlaps = []
    for name_a, ids_a in rtog_sets.items():
        for name_b, ids_b in rtog_sets.items():
            if name_a < name_b and ids_a & ids_b:
                overlaps.append(f"{name_a} vs {name_b}: {len(ids_a & ids_b)}")
    for name_a, ids_a in ukhd_sets.items():
        for name_b, ids_b in ukhd_sets.items():
            if name_a < name_b and ids_a & ids_b:
                overlaps.append(f"{name_a} vs {name_b}: {len(ids_a & ids_b)}")
    if overlaps:
        lines.append("  Overlap detected: " + "; ".join(overlaps))
    else:
        lines.append("  No patient ID overlap detected within RTOG cohorts or within UKHD cohorts.")
    lines.append("  RTOG and UKHD identifiers are from distinct institutions/namespaces.")
    lines.append("")
    lines.append("DoseAE pretraining cohort:")
    lines.append("  In the current classifier workflow, UKHD-train and RTOG-other are the source-domain cohorts used for feature/model development. No additional distinct DoseAE pretraining cohort was identified from the active LODO classifier scripts; older/synthetic DoseAE pretraining caches are separate from the five Table 1 analysis cohorts.")
    lines.append("")
    lines.append("Endpoint notes:")
    lines.append("  RTOG-conv and RTOG-IMRT phenotype labels come from matched non-toxicity vs pneumonitis/fibrosis/both cohort lists.")
    lines.append("  RTOG-other is set to non-RILI for the classifier protocol and is not manually phenotyped into pneumonitis/fibrosis/both.")
    lines.append("  UKHD-train labels use the existing toxicity_label_0p70 threshold field; UKHD-SBRT uses the available binary toxicity labels, with one unlabeled patient retained in n but excluded from event percentage denominator.")
    lines.append("")
    lines.append("Data quality concerns:")
    lines.append("  UKHD-train has substantial clinical metadata missingness: only 37/145 matched the available clinical feature table; lung DVH/PTV radiomics dose descriptors were available for 34/145.")
    lines.append("  UKHD-SBRT clinical metadata matched 55/56 patients; UKHD radiomics lung/PTV dose descriptors matched 49/56 patients.")
    lines.append("  UKHD sex/gender and AD.vs.SC histology are binary source-code fields; no local codebook was found, so Table 1 reports source codes rather than male/female or squamous/non-squamous labels for UKHD.")
    lines.append("  RTOG public D1 metadata does not include COPD, FEV1, or central/peripheral tumor location.")
    lines.append("  RTOG-other is used as non-RILI/background for the classifier protocol but is not manually phenotype-labeled; do not interpret it as a curated no-toxicity cohort.")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rtog = load_rtog_clinical()
    ukhd_clin, ukhd_dose, anon_map = load_ukhd_sources()

    frames: Dict[str, pd.DataFrame] = {}
    for spec in COHORT_SPECS:
        df = load_cohort_patient_frame(spec, rtog, ukhd_clin, ukhd_dose, anon_map)
        frames[spec["label"]] = add_derived_treatment_fields(df)
    frames, dose_qc = apply_table1_dose_qc(frames)

    feature_order = [
        ("Clinical", "n"),
        ("Clinical", "Age, years"),
        ("Clinical", "Sex"),
        ("Clinical", "Performance status"),
        ("Clinical", "Smoking"),
        ("Clinical", "COPD"),
        ("Clinical", "FEV1 pre-RT"),
        ("Tumor", "Stage"),
        ("Tumor", "Histology"),
        ("Tumor", "Location"),
        ("Treatment", "RT technique"),
        ("Treatment", "Fractionation"),
        ("Treatment", "Total dose, Gy"),
        ("Treatment", "BED10, Gy"),
        ("Treatment", "Lung V20, %"),
        ("Treatment", "Mean lung dose, Gy"),
        ("Treatment", "PTV volume, cc"),
        ("Treatment", "Concurrent chemotherapy"),
        ("Outcome", "Any-RILI endpoint"),
        ("Outcome", "Phenotype breakdown"),
        ("Outcome", "Median follow-up, months"),
    ]

    summaries = {cohort: summarize_cohort(df) for cohort, df in frames.items()}
    rows = []
    for group, feature in feature_order:
        row = {"category": group, "feature": feature}
        for spec in COHORT_SPECS:
            row[spec["label"]] = summaries[spec["label"]].get(feature, "NA")
        rows.append(row)
    table = pd.DataFrame(rows)

    notes = {
        "n": "Counts are actually available/cache-used patients for LODO analyses.",
        "Sex": "UKHD sex/gender binary coding is reported as source codes because no local codebook was found.",
        "Smoking": "RTOG smoking categories follow public D1 dictionary. UKHD is binary source smoker code, not never/former/current.",
        "COPD": "COPD is not available in the RTOG public D1 table used here.",
        "FEV1 pre-RT": "FEV1 is available only for matched UKHD clinical-feature rows.",
        "Location": "Central/peripheral is available only in UKHD feature table via RTOG_Central source column.",
        "Total dose, Gy": "RTOG total dose/fractionation are assigned arm-derived values; UKHD uses clinical feature table when matched, otherwise cohort H5 metadata.",
        "Lung V20, %": "RTOG uses public whole-lung V20. UKHD uses volume-weighted ipsilateral/contralateral radiomics dose-feature V20 where available.",
        "Mean lung dose, Gy": "RTOG uses public whole-lung mean lung dose. UKHD uses volume-weighted ipsilateral/contralateral radiomics dose-feature mean where available.",
        "PTV volume, cc": "RTOG uses public volume_ptv in cc. UKHD uses radiomics volume-ptv converted from mm3 to cc where available.",
        "Any-RILI endpoint": "Classifier endpoint definitions differ by source; see table1_cohort_definitions.txt.",
        "Median follow-up, months": "Follow-up is available from RTOG public survival_months only.",
    }
    table["notes"] = table["feature"].map(notes).fillna("")

    patient_table = pd.concat(frames.values(), ignore_index=True, sort=False)
    missing = missingness_rows(frames)
    quality = missing.loc[missing["flag_gt20pct"]].copy()

    table_path = args.out_dir / "table1_cohort_characteristics.csv"
    definitions_path = args.out_dir / "table1_cohort_definitions.txt"
    missing_path = args.out_dir / "table1_missingness_flags.csv"
    patient_path = args.out_dir / "table1_patient_level_characteristics_source.csv"
    dose_qc_path = args.out_dir / "table1_dose_qc_exclusions.csv"

    table.to_csv(table_path, index=False)
    definitions_path.write_text(build_definitions(frames))
    missing.to_csv(missing_path, index=False)
    dose_qc.to_csv(dose_qc_path, index=False)

    keep_cols = [
        "cohort",
        "cohort_key",
        "patient_key",
        "metadata_source",
        "rili_label",
        "phenotype_label",
        "age",
        "gender",
        "Gender",
        "zubrod",
        "KPS",
        "smoke_hx",
        "cliohe_Smoker",
        "COPD",
        "FEV1",
        "FEV1.",
        "ajcc_stage_grp",
        "T1",
        "T2",
        "T3",
        "nonsquam_squam",
        "AD.vs.SC",
        "RTOG_Central",
        "rt_technique_display",
        "total_dose_gy",
        "num_fractions",
        "bed10_gy",
        "v20_lung_percent",
        "mean_lung_dose_gy",
        "ptv_volume_cc",
        "received_conc_chemo",
        "survival_months",
    ]
    existing_keep = [c for c in keep_cols if c in patient_table.columns]
    patient_table[existing_keep].to_csv(patient_path, index=False)

    print(f"[saved] {table_path}")
    print(f"[saved] {definitions_path}")
    print(f"[saved] {missing_path}")
    print(f"[saved] {dose_qc_path}")
    print(f"[saved] {patient_path}")
    if not dose_qc.empty:
        print("[dose QC exclusions]")
        print(dose_qc.to_string(index=False))
    if not quality.empty:
        print("[missingness >20%]")
        print(quality[["cohort", "feature", "missing_percent", "source_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
