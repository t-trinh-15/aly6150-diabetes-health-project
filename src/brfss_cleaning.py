"""BRFSS LLCP2023 cleaning pipeline.

Produces a person-level analytic dataset for New England adults aged 45-64
with diagnosed diabetes. Mirrors the structure of clean_meps() but with
BRFSS-specific recoding (7/9 -> NaN, 88-as-valid-zero for day-count items).

Filters applied:
  - _STATE in {9, 23, 25, 33, 44, 50}  (CT, ME, MA, NH, RI, VT)
  - _AGEG5YR in {6, 7, 8, 9}            (45-49, 50-54, 55-59, 60-64)
  - DIABETE4 == 1                       (diagnosed diabetes only;
                                         excludes prediabetes, gestational)

BRFSS coding conventions (vs. MEPS):
  Binary items   : 1 = Yes, 2 = No, 7 = DK, 9 = Refused
  Day-count items: 1-30 = days, 77 = DK, 88 = None (VALID 0), 99 = Refused
  Logical skip   : NaN (e.g., POORHLTH only asked if PHYS/MENT > 0)

Module limitations in 2023 for the six NE states:
  - Flu-shot module (_FLSHOT7): NOT run in any NE state
    -> flu_shot recoded for completeness but dropped from preventive index
  - Diabetes self-management module (CHKHEMO3, EYEEXAM1): only Maine and NH
    -> preventive_care_index reportable only for ~680 ME+NH respondents
    -> preventive_care_eligible flag identifies that subgroup
"""

"""BRFSS LLCP2023 cleaning pipeline.
 
Produces a person-level analytic dataset for adults aged 45-64 with
diagnosed diabetes across two regions:
  - New England        : CT, ME, MA, NH, RI, VT  (FIPS 9,23,25,33,44,50)
  - Lower-income South : AL, AR, LA, MS, WV      (FIPS 1,5,22,28,54)
 
A `region` column distinguishes the two groups for all downstream comparisons.
 
BRFSS coding conventions (vs. MEPS):
  Binary items   : 1 = Yes, 2 = No, 7 = DK, 9 = Refused
  Day-count items: 1-30 = days, 77 = DK, 88 = None (VALID 0), 99 = Refused
  Logical skip   : NaN (e.g., POORHLTH only asked if PHYS/MENT > 0)
 
Module availability notes for 2023:
  - Flu-shot module (_FLSHOT7): NOT run in any NE state; may run in some
    Southern states — recoded for completeness but excluded from indices.
  - Diabetes self-management module (CHKHEMO3, EYEEXAM1): ran in a small
    subset of states only (~24k responses nationally). `preventive_care_eligible`
    flags respondents with non-missing Module 2 data regardless of state.
  - Social Determinants module (Module 29): ran in ~50% of states nationally
    (~225k responses). `sdoh_eligible` flags respondents with full SDOH data.
"""
 
import numpy as np
import pandas as pd
 
 
# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
NEW_ENGLAND_STATES = {
    9: "Connecticut",
    23: "Maine",
    25: "Massachusetts",
    33: "New Hampshire",
    44: "Rhode Island",
    50: "Vermont",
}
 
LOWER_INCOME_STATES = {
    1:  "Alabama",
    5:  "Arkansas",
    22: "Louisiana",
    28: "Mississippi",
    54: "West Virginia",
}
 
# Combined target — all states included in the analytic file
TARGET_STATES = {**NEW_ENGLAND_STATES, **LOWER_INCOME_STATES}
 
REGION_MAP = {
    **{fips: "New England"         for fips in NEW_ENGLAND_STATES},
    **{fips: "Lower-income South"  for fips in LOWER_INCOME_STATES},
}
 
AGE_45_64_CODES = [6, 7, 8, 9]  # _AGEG5YR codes for 45-49, 50-54, 55-59, 60-64
 
BINARY_MISSING_CODES = [7, 9]      # Don't Know, Refused
DAY_COUNT_DK_REFUSED = [77, 99]    # Don't Know, Refused (88 is VALID zero)
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def require_columns(df: pd.DataFrame, columns: list, dataset_name: str = "dataset") -> None:
    """Raise KeyError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {dataset_name}: {missing}")
 
 
def recode_yes_no(series: pd.Series, yes_code: int = 1, no_code: int = 2) -> pd.Series:
    """Map BRFSS binary item to {1=Yes, 0=No, NaN=DK/Refused/Other}."""
    return np.select(
        [series == yes_code, series == no_code],
        [1.0, 0.0],
        default=np.nan,
    )
 
 
def recode_day_count(series: pd.Series) -> pd.Series:
    """Map BRFSS day-count item to valid days [0, 30] or NaN.
 
    Rules: 1-30 -> kept as-is, 88 -> 0 (None), 77/99 -> NaN.
    """
    result = series.copy().astype(float)
    result = result.where(~result.isin(DAY_COUNT_DK_REFUSED), np.nan)
    result = result.where(result != 88, 0)
    # Anything outside [0, 30] after recoding becomes NaN
    result = result.where((result >= 0) & (result <= 30), np.nan)
    return result
 
 
# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------
 
def clean_brfss(brfss: pd.DataFrame) -> pd.DataFrame:
    """Clean BRFSS 2023 data for NE adults 45-64 with diabetes.
 
    Parameters
    ----------
    brfss : pd.DataFrame
        Raw BRFSS LLCP2023 dataframe loaded from LLCP2023.XPT.
 
    Returns
    -------
    pd.DataFrame
        Cleaned, filtered analytic dataset with derived flags and labels.
    """
    df = brfss.copy()
    n_start = len(df)
 
    # -----------------------------------------------------------------------
    # 1. Verify required variables exist
    # -----------------------------------------------------------------------
    required = [
        # Sampling design
        "_STATE", "_LLCPWT", "_STSTR", "_PSU",
        # Demographics
        "_AGEG5YR", "SEXVAR", "_IMPRACE", "EDUCA", "INCOME3",
        # Diabetes
        "DIABETE4", "INSULIN1",
        # Access / outcome
        "MEDCOST1", "_HLTHPL1", "PERSDOC3", "CHECKUP1",
        # Self-rated health
        "GENHLTH", "PHYSHLTH", "MENTHLTH", "POORHLTH",
        # Risk behaviors
        "_SMOKER3", "_BMI5CAT", "_PA150R4",
        # Comorbidities
        "BPHIGH6", "TOLDHI3", "CVDINFR4", "CVDCRHD4", "CVDSTRK3",
        "CHCCOPD3", "CHCKDNY2", "_DRDXAR2",
        # Preventive care
        "CHKHEMO3", "EYEEXAM1", "_FLSHOT7", "_PNEUMO3",
        # Social determinants (Module 29 — high participation, ~50% of states)
        "FOODSTMP", "SDHFOOD1", "SDHBILLS", "SDHUTILS", "SDHTRNSP",
    ]
    require_columns(df, required, "BRFSS LLCP2023")
 
    # -----------------------------------------------------------------------
    # 2. Apply primary filters
    # -----------------------------------------------------------------------
    df = df[df["_STATE"].isin(TARGET_STATES.keys())].copy()
    n_after_state = len(df)
    print(f"  Step 2a — after target state filter: {n_after_state:>6,} (removed {n_start - n_after_state:,})")
 
    df = df[df["_AGEG5YR"].isin(AGE_45_64_CODES)].copy()
    n_after_age = len(df)
    print(f"  Step 2b — after age 45-64 filter:  {n_after_age:>6,} (removed {n_after_state - n_after_age:,})")
 
    df = df[df["DIABETE4"] == 1].copy()
    n_after_diab = len(df)
    print(f"  Step 2c — after diabetes filter:   {n_after_diab:>6,} (removed {n_after_age - n_after_diab:,})")
 
    # -----------------------------------------------------------------------
    # 3. State name lookup
    # -----------------------------------------------------------------------
    df["state_name"] = df["_STATE"].map(TARGET_STATES)
    df["region"]     = df["_STATE"].map(REGION_MAP)
 
    # -----------------------------------------------------------------------
    # 4. Recode access / outcome variables (binary 1/2 -> 1/0)
    # -----------------------------------------------------------------------
    df["missed_care_cost"] = recode_yes_no(df["MEDCOST1"])
    df["has_insurance"]    = recode_yes_no(df["_HLTHPL1"])
    df["uninsured"]        = 1 - df["has_insurance"]  # complement; NaN propagates
 
    # PERSDOC3: 1 or 2 = has personal doctor, 3 = none, 7/9 = NaN
    df["has_personal_doctor"] = np.select(
        [df["PERSDOC3"].isin([1, 2]), df["PERSDOC3"] == 3],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # CHECKUP1: 1 = within past year; 2/3/4/8 = no recent; 7/9 = NaN
    df["recent_checkup"] = np.select(
        [df["CHECKUP1"] == 1, df["CHECKUP1"].isin([2, 3, 4, 8])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # -----------------------------------------------------------------------
    # 5. Self-rated health
    # -----------------------------------------------------------------------
    # GENHLTH: 4 (Fair) or 5 (Poor) = poor health; 1-3 = good+; 7/9 = NaN
    df["poor_health"] = np.select(
        [df["GENHLTH"].isin([4, 5]), df["GENHLTH"].isin([1, 2, 3])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # Day-count variables (88 = valid zero, 77/99 = NaN)
    df["physhlth_days"] = recode_day_count(df["PHYSHLTH"])
    df["menthlth_days"] = recode_day_count(df["MENTHLTH"])
    # POORHLTH: logical NaN when both PHYS/MENT == 0; treat that NaN as 0
    poorhlth_recoded = recode_day_count(df["POORHLTH"])
    logical_zero = (df["physhlth_days"] == 0) & (df["menthlth_days"] == 0)
    df["poorhlth_days"] = poorhlth_recoded.where(~logical_zero, 0)
 
    # -----------------------------------------------------------------------
    # 6. Risk behaviors
    # -----------------------------------------------------------------------
    # _SMOKER3: 1=Daily, 2=Some days, 3=Former, 4=Never, 9=DK/Ref
    df["current_smoker"] = np.select(
        [df["_SMOKER3"].isin([1, 2]), df["_SMOKER3"].isin([3, 4])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # _BMI5CAT: 4 = obese, 1-3 = not obese, NaN = missing height/weight
    df["obese"] = np.select(
        [df["_BMI5CAT"] == 4, df["_BMI5CAT"].isin([1, 2, 3])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # _PA150R4: 1=meets both guidelines, 2=aerobic only, 3=neither, 4=strengthening only
    # Codes 2/3/4 all map to "No" (does not meet full guideline); 9 -> NaN
    df["meets_pa_guideline"] = np.select(
        [df["_PA150R4"] == 1, df["_PA150R4"].isin([2, 3, 4])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # -----------------------------------------------------------------------
    # 7. Comorbidities (1=Yes, 2=No, 7/9 -> NaN)
    # Special handling for BPHIGH6: codes 3 (pregnancy-only) and 4
    # (pre-hypertensive) are treated as "No" since they are not chronic dx
    # -----------------------------------------------------------------------
    df["hbp_flag"] = np.select(
        [df["BPHIGH6"] == 1, df["BPHIGH6"].isin([2, 3, 4])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    for col, new_name in [
        ("TOLDHI3", "high_chol_flag"),
        ("CVDINFR4", "heart_attack_flag"),
        ("CVDCRHD4", "chd_flag"),
        ("CVDSTRK3", "stroke_flag"),
        ("CHCCOPD3", "copd_flag"),
        ("CHCKDNY2", "kidney_flag"),
        ("_DRDXAR2", "arthritis_flag"),
    ]:
        df[new_name] = recode_yes_no(df[col])
 
    # Insulin use
    df["takes_insulin"] = recode_yes_no(df["INSULIN1"])
 
    # Comorbidity count (8 conditions: HBP, high chol, MI, CHD, stroke, COPD, kidney, arthritis)
    comorb_flags = [
        "hbp_flag", "high_chol_flag", "heart_attack_flag", "chd_flag",
        "stroke_flag", "copd_flag", "kidney_flag", "arthritis_flag",
    ]
    df["comorbidity_count"] = df[comorb_flags].sum(axis=1, min_count=1)
 
    # -----------------------------------------------------------------------
    # 8a. Social Determinants of Health (Module 29)
    #     High participation module — runs in ~50% of states nationally.
    #     Variables will be NaN for states that did not run the module.
    #     Use `sdoh_eligible` flag to scope analyses to respondents with data.
    # -----------------------------------------------------------------------
 
    # FOODSTMP: receives food stamps / SNAP (1=Yes, 2=No, 7/9->NaN)
    df["receives_snap"] = recode_yes_no(df["FOODSTMP"])
 
    # SDHFOOD1: worried food would run out before money to buy more
    #   1=Always, 2=Sometimes, 3=Never, 7/9->NaN
    #   Recode: 1 or 2 = food-insecure (1), 3 = food-secure (0)
    df["food_insecure"] = np.select(
        [df["SDHFOOD1"].isin([1, 2]), df["SDHFOOD1"] == 3],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # SDHBILLS: unable to pay mortgage, rent, or utilities in past 12 months
    df["housing_cost_burden"] = recode_yes_no(df["SDHBILLS"])
 
    # SDHUTILS: had utilities (heat/electricity) shut off in past 12 months
    df["utilities_shutoff"] = recode_yes_no(df["SDHUTILS"])
 
    # SDHTRNSP: transportation barrier to medical appointments
    df["transport_barrier"] = recode_yes_no(df["SDHTRNSP"])
 
    # Composite SDOH burden index (0-4): sum of the four hardship flags
    # Requires all four non-missing (min_count=4 not applicable for sum — use manual)
    sdoh_flags = ["food_insecure", "housing_cost_burden", "utilities_shutoff", "transport_barrier"]
    df["sdoh_burden_index"] = df[sdoh_flags].sum(axis=1, min_count=1)
 
    # Eligible flag: respondent was asked the SDOH module (all four items non-missing)
    df["sdoh_eligible"] = df[sdoh_flags].notna().all(axis=1).astype(int)
 
    # -----------------------------------------------------------------------
    # 8b. Preventive care (unchanged)
    # -----------------------------------------------------------------------
    # CHKHEMO3: numeric count of A1C checks in past year.
    #   1-76 = number of checks, 88 = none, 77/99 = DK/Refused, NaN = not asked
    chk = df["CHKHEMO3"]
    df["recent_a1c"] = np.select(
        [(chk >= 1) & (chk <= 76), chk == 88],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # EYEEXAM1: 1=within past year, 2-4/8 = not recent, 7/9 = NaN
    df["recent_eye_exam"] = np.select(
        [df["EYEEXAM1"] == 1, df["EYEEXAM1"].isin([2, 3, 4, 8])],
        [1.0, 0.0],
        default=np.nan,
    )
 
    # Flu shot and pneumonia vaccine — recoded for completeness even though
    # flu_shot will be all-NaN in our NE sample (module wasn't run in 2023)
    df["flu_shot"]    = recode_yes_no(df["_FLSHOT7"])
    df["pneumo_shot"] = recode_yes_no(df["_PNEUMO3"])
 
    # Preventive-care index: A1C + eye exam (0-2 scale).
    # flu_shot dropped: 0 valid responses in NE for 2023.
    # Index is computed only when BOTH items are non-missing (no partial sums).
    # The eligible subgroup is ~680 respondents from Maine and New Hampshire
    # (the only NE states that ran the diabetes self-management module).
    pc_items = ["recent_a1c", "recent_eye_exam"]
    df["preventive_care_index"] = df[pc_items].sum(axis=1, min_count=2)
    df["preventive_care_eligible"] = df[pc_items].notna().all(axis=1).astype(int)
 
    # -----------------------------------------------------------------------
    # 9. Demographic labels
    # -----------------------------------------------------------------------
    df["sex_label"] = df["SEXVAR"].map({1: "Male", 2: "Female"})
 
    df["race_ethnicity"] = df["_IMPRACE"].map({
        1: "NH White",
        2: "NH Black",
        3: "NH Asian",
        4: "AI/AN",
        5: "Hispanic",
        6: "Other/Multiple",
    })
 
    df["age_band"] = df["_AGEG5YR"].map({
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64",
    })
 
    df["age_band_collapsed"] = df["_AGEG5YR"].map({
        6: "45-54", 7: "45-54", 8: "55-64", 9: "55-64",
    })
 
    df["education_label"] = df["EDUCA"].map({
        1: "No school/K only",
        2: "Grades 1-8",
        3: "Grades 9-11",
        4: "HS/GED",
        5: "Some college",
        6: "College grad",
    })
 
    # INCOME3: 1-11 brackets; 77/99/NaN -> NaN
    df["income_valid"] = df["INCOME3"].where(df["INCOME3"].between(1, 11))
    df["income_tier"] = pd.cut(
        df["income_valid"],
        bins=[0, 4, 6, 8, 11],
        labels=["<$25K", "$25-50K", "$50-75K", "$75K+"],
    )
    # Low-income flag (< ~$35K, BRFSS's rough proxy for <200% FPL for a household)
    df["is_low_income"] = (df["income_valid"] <= 5).astype("Int64")
    df.loc[df["income_valid"].isna(), "is_low_income"] = pd.NA
 
    # -----------------------------------------------------------------------
    # 10. Reorder columns
    # -----------------------------------------------------------------------
    final_cols = [
        # IDs and weights
        "_STATE", "state_name", "region", "_LLCPWT", "_STSTR", "_PSU",
        # Demographics
        "_AGEG5YR", "age_band", "age_band_collapsed",
        "sex_label", "race_ethnicity",
        "EDUCA", "education_label",
        "INCOME3", "income_tier", "is_low_income",
        # Outcomes — access
        "missed_care_cost", "has_insurance", "uninsured",
        "has_personal_doctor", "recent_checkup",
        # Self-rated health
        "poor_health", "physhlth_days", "menthlth_days", "poorhlth_days",
        # Risk behaviors
        "current_smoker", "obese", "meets_pa_guideline",
        # Diabetes self-management
        "takes_insulin",
        # Comorbidity flags + count
        "hbp_flag", "high_chol_flag", "heart_attack_flag", "chd_flag",
        "stroke_flag", "copd_flag", "kidney_flag", "arthritis_flag",
        "comorbidity_count",
        # Social determinants (Module 29)
        "receives_snap", "food_insecure", "housing_cost_burden",
        "utilities_shutoff", "transport_barrier",
        "sdoh_burden_index", "sdoh_eligible",
        # Preventive care
        "recent_a1c", "recent_eye_exam", "flu_shot", "pneumo_shot",
        "preventive_care_index", "preventive_care_eligible",
    ]
    final_cols = [c for c in final_cols if c in df.columns]
    df = df[final_cols].copy()
 
    # -----------------------------------------------------------------------
    # 11. Final summary
    # -----------------------------------------------------------------------
    n_eligible = int(df["preventive_care_eligible"].sum())
    pc_mean = df.loc[df["preventive_care_eligible"] == 1, "preventive_care_index"].mean()
    n_sdoh = int(df["sdoh_eligible"].sum())
 
    print(f"\n  Final analytic sample: {len(df):,} people across {df['state_name'].nunique()} states")
    print(f"  Regions: { {r: int(v) for r, v in df['region'].value_counts().items()} }")
    print(f"  Missed-care-cost rate: {df['missed_care_cost'].mean():.1%}")
    print(f"  Uninsured rate:        {df['uninsured'].mean():.1%}")
    print(f"  Mean comorbidity count: {df['comorbidity_count'].mean():.2f}")
    print(f"  SDOH-module eligible:  {n_sdoh:,}")
    print(f"  Preventive-care eligible subgroup: {n_eligible:,} (states that ran Module 2)")
    print(f"    Mean preventive-care index (0-2, eligible only): {pc_mean:.2f}")
    print(f"  Columns: {len(df.columns)}")
 
    return df