"""Descriptive statistics tables for MEPS and BRFSS analytic datasets.

Produces two table types per dataset:
  Table A — all-sample summary (outcomes + predictors)
  Table B — stratified by a chosen sub-group, with statistical tests

Conventions:
  - Continuous variables : report median (IQR) — robust to skew
  - Binary variables     : report n (%) for "Yes" (==1)
  - Categorical variables: report n (%) per category
  - Unweighted estimates are the default; pass weights= to weight
  - Statistical tests: chi-square (categorical), Mann-Whitney U (continuous,
    2 groups), Kruskal-Wallis (continuous, 3+ groups)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ===========================================================================
# Variable specifications
# ===========================================================================
# Each entry maps variable_name -> dict with at least:
#   "label" : human-readable label
#   "type"  : "continuous" | "binary" | "categorical"

MEPS_VARS = {
    # Outcomes (continuous + binary)
    "total_medical_expense": {"label": "Total medical expense ($)",   "type": "continuous"},
    "out_of_pocket_expense": {"label": "Out-of-pocket spending ($)",  "type": "continuous"},
    "prescription_expense":  {"label": "Prescription spending ($)",   "type": "continuous"},
    "high_spender":          {"label": "High spender (top 25%)",      "type": "binary"},
    "any_er_visit":          {"label": "Any ER visit",                "type": "binary"},
    "any_inpatient":         {"label": "Any inpatient stay",          "type": "binary"},
    # Demographics
    "age":                   {"label": "Age (years)",                 "type": "continuous"},
    "age_band":              {"label": "Age band",                    "type": "categorical"},
    "sex_label":             {"label": "Sex",                         "type": "categorical"},
    "race_ethnicity":        {"label": "Race / ethnicity",            "type": "categorical"},
    # Socioeconomic
    "insurance_label":       {"label": "Insurance coverage",          "type": "categorical"},
    "poverty_label":         {"label": "Poverty tier",                "type": "categorical"},
    "is_low_income":         {"label": "<200% FPL",                   "type": "binary"},
    # Comorbidities
    "comorbidity_count":     {"label": "Comorbidity count (0-4)",     "type": "continuous"},
}

BRFSS_VARS = {
    # Outcomes
    "missed_care_cost":      {"label": "Missed care due to cost",     "type": "binary"},
    "uninsured":             {"label": "Uninsured",                   "type": "binary"},
    "has_personal_doctor":   {"label": "Has personal doctor",         "type": "binary"},
    "recent_checkup":        {"label": "Recent (past-year) checkup",  "type": "binary"},
    # Demographics
    "age_band_collapsed":    {"label": "Age band",                    "type": "categorical"},
    "sex_label":             {"label": "Sex",                         "type": "categorical"},
    "race_ethnicity":        {"label": "Race / ethnicity",            "type": "categorical"},
    "education_label":       {"label": "Education",                   "type": "categorical"},
    "income_tier":           {"label": "Income tier",                 "type": "categorical"},
    "is_low_income":         {"label": "Low income (<$50K)",          "type": "binary"},
    # Risk behaviors
    "obese":                 {"label": "Obese (BMI ≥30)",             "type": "binary"},
    "current_smoker":        {"label": "Current smoker",              "type": "binary"},
    "meets_pa_guideline":    {"label": "Meets PA guideline",          "type": "binary"},
    "poor_health":           {"label": "Fair/poor self-rated health", "type": "binary"},
    # Comorbidities + health
    "comorbidity_count":     {"label": "Comorbidity count (0-8)",     "type": "continuous"},
    "physhlth_days":         {"label": "Days physical health not good","type": "continuous"},
    "menthlth_days":         {"label": "Days mental health not good",  "type": "continuous"},
}

# Separate dict for variables that only exist in the diabetes self-management
# module (ME + NH only). Use this for a sub-table or for Table A's footnote section.
BRFSS_DIABETES_MODULE_VARS = {
    "takes_insulin":            {"label": "Takes insulin",                "type": "binary"},
    "recent_a1c":               {"label": "Recent A1C check (past year)", "type": "binary"},
    "recent_eye_exam":          {"label": "Recent dilated eye exam",      "type": "binary"},
    "preventive_care_index":    {"label": "Preventive care index (0-2)",  "type": "continuous"},
}


# ===========================================================================
# Summary helpers (unweighted)
# ===========================================================================

def _fmt_n_pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{n:,} ({n / total * 100:.1f}%)"


def _fmt_median_iqr(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "—"
    med = s.median()
    q1, q3 = s.quantile([0.25, 0.75])
    # Use thousands separator and no decimals for large dollar values
    if med >= 1000:
        return f"{med:,.0f} ({q1:,.0f}–{q3:,.0f})"
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


def summarize_continuous(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n_valid = s.notna().sum()
    n_total = len(s)
    return {
        "n_valid": n_valid,
        "statistic": _fmt_median_iqr(s),
        "missing_pct": f"{(n_total - n_valid) / n_total * 100:.1f}%",
    }


def summarize_binary(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n_valid = s.notna().sum()
    n_total = len(s)
    n_yes = int((s == 1).sum())
    return {
        "n_valid": n_valid,
        "statistic": _fmt_n_pct(n_yes, n_valid),
        "missing_pct": f"{(n_total - n_valid) / n_total * 100:.1f}%",
    }


def summarize_categorical(df: pd.DataFrame, col: str) -> list[dict]:
    """Return one row per category."""
    s = df[col]
    n_valid = s.notna().sum()
    n_total = len(s)
    rows = []
    counts = s.value_counts(dropna=True).sort_index()
    for cat, n in counts.items():
        rows.append({
            "category": str(cat),
            "n_valid": n_valid,
            "statistic": _fmt_n_pct(int(n), n_valid),
            "missing_pct": f"{(n_total - n_valid) / n_total * 100:.1f}%",
        })
    return rows

def brfss_diabetes_module_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of BRFSS respondents who were asked the diabetes
    self-management module (Maine + New Hampshire only in 2023).
    """
    if "preventive_care_eligible" in df.columns:
        return df.loc[df["preventive_care_eligible"] == 1].copy()
    # Fallback if the eligible flag is missing
    return df.loc[df["state_name"].isin(["Maine", "New Hampshire"])].copy()

# ===========================================================================
# Table A — all-sample summary
# ===========================================================================

def build_table_a(df: pd.DataFrame, vars_dict: dict, dataset_label: str = "") -> pd.DataFrame:
    """All-sample descriptive table.
    
    Returns a DataFrame with columns: Variable, Category, N (valid), Statistic, Missing %.
    Continuous and binary variables produce one row each;
    categorical variables produce one row per category.
    """
    rows = []
    for col, spec in vars_dict.items():
        if col not in df.columns:
            rows.append({
                "Variable": spec["label"],
                "Category": "—",
                "N (valid)": 0,
                "Statistic": "(variable not in dataset)",
                "Missing %": "100.0%",
            })
            continue

        vtype = spec["type"]
        label = spec["label"]

        if vtype == "continuous":
            s = summarize_continuous(df, col)
            rows.append({
                "Variable": label,
                "Category": "median (IQR)",
                "N (valid)": s["n_valid"],
                "Statistic": s["statistic"],
                "Missing %": s["missing_pct"],
            })
        elif vtype == "binary":
            s = summarize_binary(df, col)
            rows.append({
                "Variable": label,
                "Category": "Yes",
                "N (valid)": s["n_valid"],
                "Statistic": s["statistic"],
                "Missing %": s["missing_pct"],
            })
        elif vtype == "categorical":
            cat_rows = summarize_categorical(df, col)
            for i, cr in enumerate(cat_rows):
                rows.append({
                    "Variable": label if i == 0 else "",
                    "Category": cr["category"],
                    "N (valid)": cr["n_valid"],
                    "Statistic": cr["statistic"],
                    "Missing %": cr["missing_pct"] if i == 0 else "",
                })

    table = pd.DataFrame(rows)
    if dataset_label:
        table.attrs["dataset"] = dataset_label
    return table


# ===========================================================================
# Statistical tests for Table B
# ===========================================================================

def _test_continuous(df: pd.DataFrame, col: str, group_col: str) -> tuple[str, float]:
    """Mann-Whitney U for 2 groups; Kruskal-Wallis for 3+. Returns (test_name, p)."""
    groups = []
    for g in df[group_col].dropna().unique():
        vals = df.loc[df[group_col] == g, col].dropna()
        if len(vals) > 0:
            groups.append(vals)
    if len(groups) < 2:
        return ("—", np.nan)
    if len(groups) == 2:
        _, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        return ("Mann-Whitney U", p)
    else:
        _, p = stats.kruskal(*groups)
        return ("Kruskal-Wallis", p)


def _test_categorical(df: pd.DataFrame, col: str, group_col: str) -> tuple[str, float]:
    """Chi-square test of independence. Returns (test_name, p)."""
    valid = df[[col, group_col]].dropna()
    if valid.empty:
        return ("—", np.nan)
    table = pd.crosstab(valid[col], valid[group_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return ("—", np.nan)
    chi2, p, _, _ = stats.chi2_contingency(table)
    return ("Chi-square", p)


def _format_p(p: float) -> str:
    if pd.isna(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


# ===========================================================================
# Table B — stratified by sub-group
# ===========================================================================

def build_table_b(
    df: pd.DataFrame,
    vars_dict: dict,
    group_col: str,
    group_label: str = None,
    dataset_label: str = "",
) -> pd.DataFrame:
    """Stratified descriptive table with statistical tests.

    Returns a DataFrame with columns:
      Variable, Category, Overall, <group1>, <group2>, ..., Test, p-value
    """
    if group_col not in df.columns:
        raise KeyError(f"Group column '{group_col}' not in dataframe")

    group_label = group_label or group_col
    groups = sorted([g for g in df[group_col].dropna().unique()])
    n_overall = len(df)
    n_by_group = df[group_col].value_counts().to_dict()

    # Build header row showing group sizes
    rows = []
    header_row = {
        "Variable": "",
        "Category": "(n)",
        "Overall": f"{n_overall:,}",
    }
    for g in groups:
        header_row[str(g)] = f"{n_by_group.get(g, 0):,}"
    header_row["Test"] = ""
    header_row["p-value"] = ""
    rows.append(header_row)

    for col, spec in vars_dict.items():
        if col not in df.columns or col == group_col:
            continue

        vtype = spec["type"]
        label = spec["label"]

        if vtype == "continuous":
            test_name, p = _test_continuous(df, col, group_col)
            row = {
                "Variable": label,
                "Category": "median (IQR)",
                "Overall": _fmt_median_iqr(df[col]),
            }
            for g in groups:
                sub = df.loc[df[group_col] == g, col]
                row[str(g)] = _fmt_median_iqr(sub)
            row["Test"] = test_name
            row["p-value"] = _format_p(p)
            rows.append(row)

        elif vtype == "binary":
            test_name, p = _test_categorical(df, col, group_col)
            n_valid_overall = df[col].notna().sum()
            n_yes_overall = int((df[col] == 1).sum())
            row = {
                "Variable": label,
                "Category": "Yes",
                "Overall": _fmt_n_pct(n_yes_overall, n_valid_overall),
            }
            for g in groups:
                sub = df.loc[df[group_col] == g, col]
                n_valid_g = sub.notna().sum()
                n_yes_g = int((sub == 1).sum())
                row[str(g)] = _fmt_n_pct(n_yes_g, n_valid_g)
            row["Test"] = test_name
            row["p-value"] = _format_p(p)
            rows.append(row)

        elif vtype == "categorical":
            test_name, p = _test_categorical(df, col, group_col)
            cats = sorted(df[col].dropna().unique())
            n_valid_overall = df[col].notna().sum()
            for i, cat in enumerate(cats):
                row = {
                    "Variable": label if i == 0 else "",
                    "Category": str(cat),
                    "Overall": _fmt_n_pct(int((df[col] == cat).sum()), n_valid_overall),
                }
                for g in groups:
                    sub = df.loc[df[group_col] == g, col]
                    n_valid_g = sub.notna().sum()
                    n_cat_g = int((sub == cat).sum())
                    row[str(g)] = _fmt_n_pct(n_cat_g, n_valid_g)
                row["Test"] = test_name if i == 0 else ""
                row["p-value"] = _format_p(p) if i == 0 else ""
                rows.append(row)

    table = pd.DataFrame(rows)
    # Rename group columns to be more readable
    table.columns = [
        f"{group_label}: {c}" if c in [str(g) for g in groups] else c
        for c in table.columns
    ]
    if dataset_label:
        table.attrs["dataset"] = dataset_label
        table.attrs["group_col"] = group_col
    return table


# ===========================================================================
# Excel export with formatting
# ===========================================================================

def save_table(table: pd.DataFrame, output_path, sheet_name: str = "Table") -> None:
    """Save a descriptive table to Excel with reasonable column widths."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=sheet_name, index=False)
        # Auto-size column widths based on header + data length
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(table.columns, start=1):
            max_len = max(
                len(str(col)),
                table[col].astype(str).str.len().max() if len(table) else 0,
            )
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max_len + 2, 40)