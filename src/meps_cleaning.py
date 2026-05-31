"""MEPS HC-251 (2023) cleaning pipeline.

Produces a person-level analytic dataset for adults aged 45-64 with
diagnosed diabetes. Includes flags for income tier and Northeast region
so subsequent analyses can subset without re-cleaning.

Strategy (Option A from project plan):
  - Primary sample = age 45-64 + DIABDX_M18 == 1 (national, ~1,000 people)
  - Sub-flags tagged for low-income (~330) and Northeast (~98)
  - MEPS gives the national cost/utilization picture; BRFSS handles
    the New England access-to-care picture in notebook 02.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: required-column check
# ---------------------------------------------------------------------------

def require_columns(df: pd.DataFrame, columns: list, dataset_name: str = "dataset") -> None:
    """Raise KeyError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {dataset_name}: {missing}")


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_meps(meps: pd.DataFrame) -> pd.DataFrame:
    """Clean MEPS 2023 person-level data for adults 45-64 with diabetes.

    Parameters
    ----------
    meps : pd.DataFrame
        Raw MEPS HC-251 dataframe loaded from h251.sas7bdat.

    Returns
    -------
    pd.DataFrame
        Cleaned, filtered analytic dataset with derived flags.
    """
    df = meps.copy()
    n_start = len(df)

    # -----------------------------------------------------------------------
    # 1. Verify required variables exist
    # -----------------------------------------------------------------------
    required = [
        "DUPERSID", "AGE23X", "DIABDX_M18",
        "TOTEXP23", "TOTSLF23", "RXEXP23",
        "ERTEXP23", "OBVEXP23", "IPTEXP23",
        "ERTOT23", "OBTOTV23", "IPDIS23", "RXTOT23",
        "INSCOV23", "SEX", "RACETHX", "POVCAT23",
        "REGION23", "PERWT23F", "VARSTR", "VARPSU",
        "HIBPDX", "CHDDX", "STRKDX", "CHOLDX",
    ]
    require_columns(df, required, "MEPS HC-251")

    # -----------------------------------------------------------------------
    # 2. Apply primary filters (Option A: national, no income filter)
    # -----------------------------------------------------------------------
    df = df[(df["AGE23X"] >= 45) & (df["AGE23X"] <= 64)].copy()
    n_after_age = len(df)
    print(f"  Step 2a — after age 45-64 filter: {n_after_age:,} (removed {n_start - n_after_age:,})")

    df = df[df["DIABDX_M18"] == 1].copy()
    n_after_diab = len(df)
    print(f"  Step 2b — after diabetes filter:  {n_after_diab:,} (removed {n_after_age - n_after_diab:,})")

    # -----------------------------------------------------------------------
    # 3. Recode missing/refused codes to NaN
    # -----------------------------------------------------------------------
    # MEPS uses -1 (inapplicable), -7 (refused), -8 (don't know), -9 (not asked)
    # for non-response. Convert to NaN before computing summaries.
    missing_codes = [-1, -7, -8, -9, -15]

    for col in ["HIBPDX", "CHDDX", "STRKDX", "CHOLDX"]:
        df[col] = df[col].replace(missing_codes, np.nan)

    # -----------------------------------------------------------------------
    # 4. Convert expenditures to clean numeric (handle any stray non-numerics)
    # -----------------------------------------------------------------------
    exp_cols = [
        "TOTEXP23", "TOTSLF23", "RXEXP23", "ERTEXP23",
        "OBVEXP23", "IPTEXP23",
    ]
    for col in exp_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------------------------
    # 5. Rename to snake_case + add canonical names alongside MEPS originals
    # -----------------------------------------------------------------------
    df = df.rename(columns={
        "AGE23X": "age",
        "TOTEXP23": "total_medical_expense",
        "TOTSLF23": "out_of_pocket_expense",
        "RXEXP23": "prescription_expense",
        "ERTEXP23": "er_expense",
        "OBVEXP23": "office_visit_expense",
        "IPTEXP23": "inpatient_expense",
        "ERTOT23": "n_er_visits",
        "OBTOTV23": "n_office_visits",
        "IPDIS23": "n_inpatient_stays",
        "RXTOT23": "n_rx_fills",
        "INSCOV23": "insurance_coverage",
        "POVCAT23": "poverty_category",
        "REGION23": "census_region",
        "PERWT23F": "person_weight",
    })

    # -----------------------------------------------------------------------
    # 6. Derived flags and engineered features
    # -----------------------------------------------------------------------
    # ER utilization
    df["any_er_visit"] = (df["n_er_visits"] > 0).astype(int)

    # Inpatient utilization
    df["any_inpatient"] = (df["n_inpatient_stays"] > 0).astype(int)

    # High-spender flag = top quartile within the analytic sample (75th percentile)
    high_cutoff = df["total_medical_expense"].quantile(0.75)
    df["high_spender"] = (df["total_medical_expense"] >= high_cutoff).astype(int)
    df.attrs["high_spender_cutoff"] = float(high_cutoff)

    # Rx share of total spend (handle division by zero / missing)
    df["rx_share"] = np.where(
        df["total_medical_expense"] > 0,
        df["prescription_expense"] / df["total_medical_expense"],
        np.nan,
    )

    # Comorbidity count (sum of 4 core flags; NaN propagates)
    comorb_cols = ["HIBPDX", "CHDDX", "STRKDX", "CHOLDX"]
    # MEPS codes condition flags as 1 = Yes, 2 = No.
    # Convert to 1/0 before summing.
    for col in comorb_cols:
        df[col + "_flag"] = (df[col] == 1).astype("Int64")
        # Preserve NaN where original was NaN
        df.loc[df[col].isna(), col + "_flag"] = pd.NA

    df["comorbidity_count"] = df[[c + "_flag" for c in comorb_cols]].sum(axis=1, skipna=False)

    # Age band
    df["age_band"] = pd.cut(
        df["age"],
        bins=[44, 54, 64],
        labels=["45-54", "55-64"],
    )

    # Sex label
    df["sex_label"] = df["SEX"].map({1: "Male", 2: "Female"})

    # Race/ethnicity label
    df["race_ethnicity"] = df["RACETHX"].map({
        1: "Hispanic",
        2: "NH White",
        3: "NH Black",
        4: "NH Asian",
        5: "NH Other/Multiple",
    })

    # Insurance category label
    df["insurance_label"] = df["insurance_coverage"].map({
        1: "Any private",
        2: "Public only",
        3: "Uninsured",
    })

    # Low-income flag (POVCAT 1, 2, 3 = <200% FPL)
    df["is_low_income"] = df["poverty_category"].isin([1, 2, 3]).astype(int)

    # Northeast region flag
    df["is_northeast"] = (df["census_region"] == 1).astype(int)

    # Poverty tier label
    df["poverty_label"] = df["poverty_category"].map({
        1: "Poor (<100% FPL)",
        2: "Near Poor (100-124%)",
        3: "Low Income (125-199%)",
        4: "Middle Income (200-399%)",
        5: "High Income (400%+)",
    })

    # -----------------------------------------------------------------------
    # 7. Reorder columns for readability
    # -----------------------------------------------------------------------
    final_cols = [
        # IDs and weights
        "DUPERSID", "person_weight", "VARSTR", "VARPSU",
        # Demographics
        "age", "age_band", "sex_label", "race_ethnicity",
        # Income / region
        "poverty_category", "poverty_label", "is_low_income",
        "census_region", "is_northeast",
        # Insurance
        "insurance_coverage", "insurance_label",
        # Outcomes
        "total_medical_expense", "out_of_pocket_expense",
        "prescription_expense", "er_expense",
        "office_visit_expense", "inpatient_expense",
        "high_spender", "rx_share",
        # Utilization
        "n_er_visits", "n_office_visits", "n_inpatient_stays", "n_rx_fills",
        "any_er_visit", "any_inpatient",
        # Comorbidities
        "HIBPDX_flag", "CHDDX_flag", "STRKDX_flag", "CHOLDX_flag",
        "comorbidity_count",
    ]
    # Keep only columns that exist in case of partial input
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()

    # -----------------------------------------------------------------------
    # 8. Final summary
    # -----------------------------------------------------------------------
    print(f"\n  Final analytic sample: {len(df):,} people")
    print(f"  High-spender cutoff (75th pct): ${high_cutoff:,.0f}")
    print(f"  Columns: {len(df.columns)}")

    return df