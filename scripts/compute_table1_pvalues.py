#!/usr/bin/env python3
"""Compute Table 1 baseline p-values for the four-cohort NSCLC paper table.

The script uses available-case analysis on a per-patient CSV and compares
UKHD-train against UKHD-SBRT, RTOG-conv, and RTOG-IMRT.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu


REFERENCE = "UKHD-train"
COMPARATORS = [
    ("UKHD-SBRT", "p_vs_SBRT"),
    ("RTOG-conv", "p_vs_conv"),
    ("RTOG-IMRT", "p_vs_IMRT"),
]
KEEP_COHORTS = [REFERENCE] + [c for c, _ in COMPARATORS]


@dataclass(frozen=True)
class VariableSpec:
    name: str
    kind: str
    builder: Callable[[pd.DataFrame], pd.Series]
    source_columns: str
    assumption: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("oliver_paper/tables/table1_patient_level_characteristics_source.csv"),
        help="Per-patient Table 1 source CSV.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("oliver_paper/tables/table1_pvalues.csv"),
        help="Output CSV for p-values.",
    )
    return parser.parse_args()


def col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return df[name]


def numeric_col(name: str) -> Callable[[pd.DataFrame], pd.Series]:
    def _build(df: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(col(df, name), errors="coerce")

    return _build


def sex_harmonized(df: pd.DataFrame) -> pd.Series:
    # Assumption intentionally explicit:
    # RTOG public D1 coding is assumed gender 1=male, 2=female.
    # UKHD binary Gender lacks a local codebook in this repository, so it is
    # NOT silently harmonized; sex is set missing for UKHD to avoid invalid
    # cross-source tests. This yields NA for all UKHD-train reference tests.
    out = pd.Series(np.nan, index=df.index, dtype=object)
    rtog = df["cohort"].astype(str).str.startswith("RTOG", na=False)
    g = pd.to_numeric(col(df, "gender"), errors="coerce")
    out.loc[rtog & (g == 1)] = "male"
    out.loc[rtog & (g == 2)] = "female"
    return out


def good_ps(df: pd.DataFrame) -> pd.Series:
    # Harmonized good performance status:
    # UKHD: KPS >= 80. RTOG: Zubrod/ECOG 0.
    out = pd.Series(np.nan, index=df.index, dtype=object)
    ukhd = df["cohort"].astype(str).str.startswith("UKHD", na=False)
    rtog = df["cohort"].astype(str).str.startswith("RTOG", na=False)
    kps = pd.to_numeric(col(df, "KPS"), errors="coerce")
    zubrod = pd.to_numeric(col(df, "zubrod"), errors="coerce")
    out.loc[ukhd & kps.notna()] = np.where(kps.loc[ukhd & kps.notna()] >= 80, "good", "not_good")
    out.loc[rtog & zubrod.notna()] = np.where(zubrod.loc[rtog & zubrod.notna()] == 0, "good", "not_good")
    return out


def smoker_binary(df: pd.DataFrame) -> pd.Series:
    # UKHD cliohe_Smoker is assumed binary 1=ever/current smoker, 0=non-smoker.
    # RTOG smoke_hx follows D1 source coding used in Table 1:
    # 1=never, 2/3=former, 4=current, 9=unknown.
    out = pd.Series(np.nan, index=df.index, dtype=object)
    ukhd = df["cohort"].astype(str).str.startswith("UKHD", na=False)
    rtog = df["cohort"].astype(str).str.startswith("RTOG", na=False)
    s_ukhd = pd.to_numeric(col(df, "cliohe_Smoker"), errors="coerce")
    s_rtog = pd.to_numeric(col(df, "smoke_hx"), errors="coerce")
    out.loc[ukhd & (s_ukhd == 0)] = "never_or_non"
    out.loc[ukhd & (s_ukhd == 1)] = "ever_or_current"
    out.loc[rtog & (s_rtog == 1)] = "never_or_non"
    out.loc[rtog & s_rtog.isin([2, 3, 4])] = "ever_or_current"
    return out


def copd_binary(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(col(df, "COPD"), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out.loc[c == 0] = "no"
    out.loc[c == 1] = "yes"
    return out


def stage_source(df: pd.DataFrame) -> pd.Series:
    # Stage is not perfectly harmonized across sources:
    # UKHD uses T-stage one-hot columns (T1/T2/T3), while RTOG uses AJCC stage
    # group source codes. We keep source-specific labels instead of pretending
    # they are the same ordinal scale.
    out = pd.Series(np.nan, index=df.index, dtype=object)
    for t in ["T1", "T2", "T3"]:
        v = pd.to_numeric(col(df, t), errors="coerce")
        out.loc[v == 1] = t
    ajcc = pd.to_numeric(col(df, "ajcc_stage_grp"), errors="coerce")
    out.loc[ajcc == 1] = "AJCC_1"
    out.loc[ajcc == 2] = "AJCC_2"
    return out


def squamous_histology(df: pd.DataFrame) -> pd.Series:
    # RTOG nonsquam_squam is mapped using Table 1 source convention:
    # 1=non-squamous, 2=squamous. UKHD AD.vs.SC lacks a verified codebook here,
    # so use source labels AD.vs.SC=0/1 rather than silently mapping to squamous.
    out = pd.Series(np.nan, index=df.index, dtype=object)
    rtog = df["cohort"].astype(str).str.startswith("RTOG", na=False)
    h_rtog = pd.to_numeric(col(df, "nonsquam_squam"), errors="coerce")
    out.loc[rtog & (h_rtog == 1)] = "non_squamous"
    out.loc[rtog & (h_rtog == 2)] = "squamous"
    ukhd = df["cohort"].astype(str).str.startswith("UKHD", na=False)
    h_ukhd = pd.to_numeric(col(df, "AD.vs.SC"), errors="coerce")
    out.loc[ukhd & h_ukhd.notna()] = "AD.vs.SC=" + h_ukhd.loc[ukhd & h_ukhd.notna()].astype(int).astype(str)
    return out


def central_peripheral(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(col(df, "RTOG_Central"), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out.loc[c == 0] = "peripheral"
    out.loc[c == 1] = "central"
    return out


def concurrent_chemo(df: pd.DataFrame) -> pd.Series:
    c = pd.to_numeric(col(df, "received_conc_chemo"), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out.loc[c == 0] = "no"
    out.loc[c == 1] = "yes"
    return out


def any_rili(df: pd.DataFrame) -> pd.Series:
    y = pd.to_numeric(col(df, "rili_label"), errors="coerce")
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out.loc[y == 0] = "no"
    out.loc[y == 1] = "yes"
    return out


SPECS = [
    VariableSpec("age", "continuous", numeric_col("age"), "age"),
    VariableSpec("FEV1", "continuous", numeric_col("FEV1"), "FEV1", "FEV1 is liters; FEV1. percent-predicted is not used."),
    VariableSpec("total_dose", "continuous", numeric_col("total_dose_gy"), "total_dose_gy"),
    VariableSpec("BED10", "continuous", numeric_col("bed10_gy"), "bed10_gy"),
    VariableSpec("V20", "continuous", numeric_col("v20_lung_percent"), "v20_lung_percent"),
    VariableSpec("mean_lung_dose", "continuous", numeric_col("mean_lung_dose_gy"), "mean_lung_dose_gy"),
    VariableSpec("PTV_volume", "continuous", numeric_col("ptv_volume_cc"), "ptv_volume_cc"),
    VariableSpec("followup_months", "continuous", numeric_col("survival_months"), "survival_months", "UKHD follow-up is not in this per-patient CSV; UKHD comparisons may be NA."),
    VariableSpec("sex", "categorical", sex_harmonized, "gender/Gender", "UKHD Gender codebook unavailable; sex tests set NA rather than guessed."),
    VariableSpec("good_PS", "categorical", good_ps, "KPS/zubrod", "Harmonized as KPS>=80 for UKHD and Zubrod/ECOG 0 for RTOG."),
    VariableSpec("smoker", "categorical", smoker_binary, "cliohe_Smoker/smoke_hx", "Binary ever/current vs never/non-smoker; RTOG unknown smoke_hx=9 dropped."),
    VariableSpec("COPD", "categorical", copd_binary, "COPD"),
    VariableSpec("stage", "categorical", stage_source, "T1/T2/T3/ajcc_stage_grp", "Source-specific stage labels; not treated as a common ordinal scale."),
    VariableSpec("squamous_histology", "categorical", squamous_histology, "AD.vs.SC/nonsquam_squam", "UKHD AD.vs.SC source codes not relabeled without codebook."),
    VariableSpec("central_vs_peripheral", "categorical", central_peripheral, "RTOG_Central", "Available in UKHD feature source only."),
    VariableSpec("concurrent_chemo", "categorical", concurrent_chemo, "received_conc_chemo", "Available in RTOG source only; UKHD comparisons will be NA."),
    VariableSpec("any_RILI", "categorical", any_rili, "rili_label"),
]


def format_p(p: float | None, bold: bool = False) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    text = "<0.001" if p < 0.001 else f"{p:.3f}"
    if bold and p < 0.05:
        return f"**{text}**"
    return text


def categorical_p(train: pd.Series, comp: pd.Series) -> tuple[float | None, str]:
    a = train.dropna().astype(str)
    b = comp.dropna().astype(str)
    if a.empty or b.empty:
        return None, "NA_empty_cohort"
    categories = sorted(set(a.unique()).union(set(b.unique())))
    if len(categories) < 2:
        return None, "NA_single_category"
    table = np.vstack([
        [int((a == cat).sum()) for cat in categories],
        [int((b == cat).sum()) for cat in categories],
    ])
    try:
        chi2, p_chi, _, expected = chi2_contingency(table, correction=False)
        if table.shape == (2, 2) and np.any(expected < 5):
            _, p_fisher = fisher_exact(table)
            return float(p_fisher), "fisher_exact"
        return float(p_chi), "chi_square"
    except Exception as exc:  # keep table generation robust for manuscript workflow
        return None, f"NA_error:{type(exc).__name__}"


def continuous_p(train: pd.Series, comp: pd.Series) -> tuple[float | None, str]:
    a = pd.to_numeric(train, errors="coerce").dropna()
    b = pd.to_numeric(comp, errors="coerce").dropna()
    if a.empty or b.empty:
        return None, "NA_empty_cohort"
    try:
        res = mannwhitneyu(a.to_numpy(dtype=float), b.to_numpy(dtype=float), alternative="two-sided")
        return float(res.pvalue), "mann_whitney_u"
    except Exception as exc:
        return None, f"NA_error:{type(exc).__name__}"


def bh_adjust(p_values: list[float | None]) -> list[float | None]:
    arr = np.asarray([np.nan if p is None else p for p in p_values], dtype=float)
    finite = np.isfinite(arr)
    adjusted = np.full_like(arr, np.nan, dtype=float)
    if finite.sum() == 0:
        return [None for _ in p_values]
    idx = np.where(finite)[0]
    p = arr[idx]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    adjusted[idx[order]] = q
    return [None if not np.isfinite(x) else float(x) for x in adjusted]


def main() -> None:
    args = parse_args()
    df_raw = pd.read_csv(args.csv)
    df = df_raw.copy()
    print(f"[input] {args.csv}")
    print(f"[shape] {df.shape}")
    print("[df.columns]")
    print(list(df.columns))

    if "cohort" not in df.columns:
        raise KeyError("Input CSV must contain a cohort column.")
    missing_cohorts = sorted(set(KEEP_COHORTS) - set(df["cohort"].dropna().astype(str).unique()))
    if missing_cohorts:
        raise ValueError(f"Missing required cohorts: {missing_cohorts}")

    extra = sorted(set(df["cohort"].dropna().astype(str).unique()) - set(KEEP_COHORTS))
    if extra:
        print(f"[info] Filtering out non-requested cohorts: {extra}")
    df = df[df["cohort"].isin(KEEP_COHORTS)].copy()

    # For the binary endpoint only, compare UKHD-train against all technique-
    # defined RTOG patients by folding unmatched/background RTOG patients into
    # the corresponding RTOG-conv/RTOG-IMRT technique cohort. Those unmatched
    # RTOG patients carry rili_label=0 in the source table.
    endpoint_df = df_raw.copy()
    endpoint_df["_comparison_cohort"] = endpoint_df["cohort"].astype(object)
    rtog_other = endpoint_df["cohort"].eq("RTOG-other")
    endpoint_df.loc[
        rtog_other & endpoint_df["rt_technique_display"].eq("3D-CRT"),
        "_comparison_cohort",
    ] = "RTOG-conv"
    endpoint_df.loc[
        rtog_other & endpoint_df["rt_technique_display"].eq("IMRT"),
        "_comparison_cohort",
    ] = "RTOG-IMRT"
    endpoint_df = endpoint_df[endpoint_df["_comparison_cohort"].isin(KEEP_COHORTS)].copy()

    print("\n[column mapping / assumptions]")
    for spec in SPECS:
        msg = f"- {spec.name} ({spec.kind}) <- {spec.source_columns}"
        if spec.assumption:
            msg += f" | assumption/warning: {spec.assumption}"
        print(msg)

    rows: list[dict[str, object]] = []
    flat_raw_p: list[float | None] = []
    flat_refs: list[tuple[int, str]] = []

    for spec in SPECS:
        if spec.name == "any_RILI":
            working_df = endpoint_df
            cohort_col = "_comparison_cohort"
        else:
            working_df = df
            cohort_col = "cohort"
        values = spec.builder(working_df)
        train_values = values[working_df[cohort_col] == REFERENCE]
        row: dict[str, object] = {
            "variable": spec.name,
            "type": spec.kind,
            "source_columns": spec.source_columns,
            "assumption": spec.assumption,
            "n_train": int(train_values.notna().sum()),
        }
        n_parts = []
        tests = []
        raw_ps: dict[str, float | None] = {}
        for comp_name, p_col in COMPARATORS:
            comp_values = values[working_df[cohort_col] == comp_name]
            n_comp = int(comp_values.notna().sum())
            n_key = {
                "UKHD-SBRT": "n_SBRT",
                "RTOG-conv": "n_conv",
                "RTOG-IMRT": "n_IMRT",
            }[comp_name]
            row[n_key] = n_comp
            n_parts.append(f"{comp_name}={n_comp}")
            if spec.kind == "continuous":
                p, test = continuous_p(train_values, comp_values)
            else:
                p, test = categorical_p(train_values, comp_values)
            raw_ps[p_col] = p
            row[p_col] = format_p(p)
            row[p_col + "_raw"] = np.nan if p is None else p
            row[p_col + "_test"] = test
            tests.append(f"{p_col}:{test}")
            flat_refs.append((len(rows), p_col))
            flat_raw_p.append(p)
        row["n_comparator"] = "; ".join(n_parts)
        row["tests_used"] = "; ".join(tests)
        row["word_string"] = " / ".join(format_p(raw_ps[p_col], bold=True) for _, p_col in COMPARATORS)
        rows.append(row)

    adj = bh_adjust(flat_raw_p)
    for (row_idx, p_col), q in zip(flat_refs, adj):
        rows[row_idx][p_col.replace("p_vs", "bh_vs")] = format_p(q)
        rows[row_idx][p_col.replace("p_vs", "bh_vs") + "_raw"] = np.nan if q is None else q

    out = pd.DataFrame(rows)
    first_cols = [
        "variable",
        "type",
        "n_train",
        "n_comparator",
        "p_vs_SBRT",
        "p_vs_conv",
        "p_vs_IMRT",
        "word_string",
        "n_SBRT",
        "n_conv",
        "n_IMRT",
    ]
    remaining = [c for c in out.columns if c not in first_cols]
    out = out[first_cols + remaining]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\n[preview]")
    preview_cols = [
        "variable",
        "type",
        "n_train",
        "n_SBRT",
        "n_conv",
        "n_IMRT",
        "p_vs_SBRT",
        "p_vs_conv",
        "p_vs_IMRT",
        "word_string",
        "tests_used",
    ]
    print(out[preview_cols].to_string(index=False))
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
