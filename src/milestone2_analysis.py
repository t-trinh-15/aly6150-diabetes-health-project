"""Milestone 2 — inferential and predictive analysis for the ALY 6150
diabetes health project.

This module BUILDS ON the Milestone 1 cleaned datasets in data/processed/.
It does not re-clean or modify the Milestone 1 pipeline; it only loads the
already-cleaned files and adds the new statistical work required for
Milestone 2:

  1. Additional feature engineering (new derived variables).
  2. A merge with an external state-level dataset (U.S. Census / ACS 2023
     population, median household income, and overall uninsured rate) so the
     6 New England states can be analysed in their socioeconomic context.
  3. Formal hypothesis tests (chi-square / Mann-Whitney) in a tidy table.
  4. A 2x2 table reporting BOTH the odds ratio (OR) and the relative risk
     (RR) for the uninsured -> missed-care relationship, plus a panel of
     unadjusted ORs for every candidate predictor.
  5. Rates-by-group tabulations with 95% confidence intervals.
  6. Model 1 — logistic regression for missed_care_cost (BRFSS), reported as
     adjusted odds ratios with 95% CIs and p-values (inferential analysis).
  7. Model 2 — predicting high_spender (MEPS) with three classifiers
     (logistic regression, L1 / LASSO logistic, gradient boosting) compared
     by cross-validated ROC AUC, with a held-out classification report and
     gradient-boosting feature importance (predictive analytics).
  8. Visualizations: a New England choropleth map, an odds-ratio forest plot,
     and ROC curves.

Conventions follow the Milestone 1 modules (src/descriptive_tables.py,
src/visualization.py): unweighted estimates, scipy for tests, statsmodels for
inference, scikit-learn for prediction, matplotlib/seaborn for charts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.weightstats import DescrStatsW

# Reuse the central path configuration from Milestone 1 (read-only import).
from src.config import (
    DATA_PROCESSED,
    OUTPUT_FIGURES,
    OUTPUT_MODELS,
    OUTPUT_TABLES,
)

# Shared chart styling, matching src/visualization.py so Milestone 1 and 2
# figures look consistent in the final report.
sns.set_style("white")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "xtick.bottom": False,
    "ytick.left": False,
})


# ===========================================================================
# 0. External state-level reference data  (the "additional merged dataset")
# ===========================================================================
# Milestone 2 asks us to "merge additional data set if possible (population,
# ... etc.)". Because the BRFSS analytic file covers eleven states across two
# regions (six New England states and five lower-income Southern states), we
# attach a small, curated state-level table drawn from published U.S. Census
# Bureau sources. This lets us (a) compute population context and (b) verify
# that STATE-LEVEL affluence (median household income) differs meaningfully
# between the two regions, supporting the "lower-income South" label.
#
# Figures are approximate 2023 American Community Survey (ACS) 1-year
# estimates and Census population estimates, rounded for readability.
# NOTE FOR THE TEAM: verify the exact published values at data.census.gov
# before the final submission and add the precise citation to the references
# page. They are used here only as contextual covariates, not as outcomes.
STATE_CONTEXT = pd.DataFrame([
    # state_name,       abbr, population_2023, median_hh_income_2023, uninsured_rate_pct_2023
    # -- New England --
    ("Connecticut",     "CT", 3_617_000,  93_760,  5.5),
    ("Maine",           "ME", 1_402_000,  73_733,  5.7),
    ("Massachusetts",   "MA", 7_001_000, 101_341,  2.4),
    ("New Hampshire",   "NH", 1_403_000,  96_838,  5.8),
    ("Rhode Island",    "RI", 1_096_000,  84_972,  4.0),
    ("Vermont",         "VT",   647_000,  81_211,  3.8),
    # -- Lower-income South --
    ("Alabama",         "AL", 5_108_000,  58_370, 10.1),
    ("Arkansas",        "AR", 3_067_000,  56_335, 10.7),
    ("Louisiana",       "LA", 4_590_000,  57_852, 11.0),
    ("Mississippi",     "MS", 2_940_000,  52_985, 13.6),
    ("West Virginia",   "WV", 1_770_000,  55_217, 10.3),
], columns=[
    "state_name", "state_abbr", "population_2023",
    "median_hh_income_2023", "uninsured_rate_pct_2023",
])
# Source: U.S. Census Bureau, American Community Survey 1-Year Estimates 2023.
# Retrieved from data.census.gov. Median household income in 2023 inflation-
# adjusted dollars. Uninsured rate = % of civilian noninstitutionalized population.


def acs_region_comparison() -> pd.DataFrame:
    """Return a region-level ACS summary table verifying the 'lower-income
    South' label used to select the five comparison states.

    Aggregates STATE_CONTEXT by region, computing population-weighted mean
    household income and mean uninsured rate. Saved to outputs/tables/ when
    called from run_all().
    """
    region_map = {
        "Connecticut": "New England",   "Maine": "New England",
        "Massachusetts": "New England", "New Hampshire": "New England",
        "Rhode Island": "New England",  "Vermont": "New England",
        "Alabama": "Lower-income South", "Arkansas": "Lower-income South",
        "Louisiana": "Lower-income South", "Mississippi": "Lower-income South",
        "West Virginia": "Lower-income South",
    }
    df = STATE_CONTEXT.copy()
    df["region"] = df["state_name"].map(region_map)

    summary = (
        df.groupby("region")
          .apply(lambda g: pd.Series({
              "States": ", ".join(g["state_abbr"]),
              "Total population": f"{g['population_2023'].sum():,}",
              "Median HH income (pop-wtd avg, $)":
                  int(np.average(g["median_hh_income_2023"],
                                 weights=g["population_2023"])),
              "Uninsured rate (mean %)":
                  round(g["uninsured_rate_pct_2023"].mean(), 1),
          }))
          .reset_index()
    )
    summary.attrs["source"] = (
        "U.S. Census Bureau, American Community Survey 1-Year Estimates, 2023. "
        "Retrieved from data.census.gov."
    )
    return summary


# ===========================================================================
# 1. Data loading + Milestone 2 feature engineering
# ===========================================================================

def load_meps() -> pd.DataFrame:
    """Load the Milestone 1 cleaned MEPS file."""
    return pd.read_csv(DATA_PROCESSED / "meps_diabetes_2023_clean.csv")


def load_brfss() -> pd.DataFrame:
    """Load the Milestone 1 cleaned BRFSS file."""
    return pd.read_csv(DATA_PROCESSED / "brfss_diabetes_2023_clean.csv")


def engineer_meps_features(meps: pd.DataFrame) -> pd.DataFrame:
    """Add Milestone-2 derived variables to the MEPS frame.

    New variables
    -------------
    log_total_expense : log1p of total medical expense — tames the heavy
        right-skew so it can be used in parametric tests/plots.
    multimorbid       : 1 if comorbidity_count >= 2 (two or more of the
        tracked conditions on top of diabetes).
    any_acute_util    : 1 if the person had any ER visit OR any inpatient
        stay — a single acute-utilization flag.
    """
    df = meps.copy()
    df["log_total_expense"] = np.log1p(df["total_medical_expense"].clip(lower=0))
    df["multimorbid"] = (df["comorbidity_count"] >= 2).astype("Int64")
    df["any_acute_util"] = (
        (df["any_er_visit"] == 1) | (df["any_inpatient"] == 1)
    ).astype(int)
    return df


def engineer_brfss_features(brfss: pd.DataFrame) -> pd.DataFrame:
    """Add Milestone-2 derived variables to the BRFSS frame and merge the
    external state-level context table.

    New variables
    -------------
    race_group        : race_ethnicity collapsed to 4 levels so logistic
        regression has stable cell sizes (NH White / NH Black / Hispanic /
        Other). Small groups (Asian, AI/AN, Other/Multiple) -> "Other".
    multimorbid       : 1 if comorbidity_count >= 3 (median is 2 in BRFSS).
    + the three STATE_CONTEXT columns merged on state_name.
    """
    df = brfss.copy()

    race_map = {
        "NH White": "NH White",
        "NH Black": "NH Black",
        "Hispanic": "Hispanic",
        "NH Asian": "Other",
        "AI/AN": "Other",
        "Other/Multiple": "Other",
    }
    df["race_group"] = df["race_ethnicity"].map(race_map).fillna("Other")

    df["multimorbid"] = (df["comorbidity_count"] >= 3).astype("Int64")

    # Merge the external state-level reference data (left join keeps every
    # respondent; every NE state has a match so no rows are dropped).
    df = df.merge(STATE_CONTEXT, on="state_name", how="left")
    return df


# ===========================================================================
# 2. Hypothesis tests
# ===========================================================================
# We collect the key Milestone-2 hypothesis tests into one tidy results table.
# Each row states the null hypothesis, the test used, the test statistic, and
# the p-value. Tests follow the same family used in Milestone 1:
#   - chi-square            : association between two categorical variables
#   - Mann-Whitney U        : difference in a skewed continuous variable
#                             between two groups (robust, non-parametric)
#   - Kruskal-Wallis        : same idea across 3+ groups

def _chi2(df: pd.DataFrame, a: str, b: str) -> tuple[float, float, int]:
    """Chi-square test of independence. Returns (chi2, p, dof)."""
    tab = pd.crosstab(df[a], df[b])
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return chi2, p, dof


def run_hypothesis_tests(meps: pd.DataFrame, brfss: pd.DataFrame) -> pd.DataFrame:
    """Run the headline Milestone-2 hypothesis tests and return a tidy table."""
    rows = []

    # H1 (BRFSS): missed care due to cost is independent of insurance status.
    d = brfss.dropna(subset=["missed_care_cost", "uninsured"])
    chi2, p, dof = _chi2(d, "missed_care_cost", "uninsured")
    rows.append({
        "Dataset": "BRFSS", "Test": "Chi-square",
        "Null hypothesis": "Missed care due to cost is independent of insurance status",
        "Statistic": f"chi2={chi2:.1f} (df={dof})", "p-value": p,
    })

    # H2 (BRFSS): missed-care rate is the same across the six states.
    d = brfss.dropna(subset=["missed_care_cost", "state_name"])
    chi2, p, dof = _chi2(d, "missed_care_cost", "state_name")
    rows.append({
        "Dataset": "BRFSS", "Test": "Chi-square",
        "Null hypothesis": "Missed-care rate is equal across all eleven study states",
        "Statistic": f"chi2={chi2:.1f} (df={dof})", "p-value": p,
    })

    # H3 (BRFSS): comorbidity_count does not differ by missed-care status.
    g0 = brfss.loc[brfss["missed_care_cost"] == 0, "comorbidity_count"].dropna()
    g1 = brfss.loc[brfss["missed_care_cost"] == 1, "comorbidity_count"].dropna()
    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    rows.append({
        "Dataset": "BRFSS", "Test": "Mann-Whitney U",
        "Null hypothesis": "Comorbidity count is the same for those who did vs did not miss care",
        "Statistic": f"U={u:,.0f}", "p-value": p,
    })

    # H4 (MEPS): high_spender status is independent of insurance type.
    d = meps.dropna(subset=["high_spender", "insurance_label"])
    chi2, p, dof = _chi2(d, "high_spender", "insurance_label")
    rows.append({
        "Dataset": "MEPS", "Test": "Chi-square",
        "Null hypothesis": "High-spender status is independent of insurance type",
        "Statistic": f"chi2={chi2:.1f} (df={dof})", "p-value": p,
    })

    # H5 (MEPS): total medical expense does not differ across insurance types.
    groups = [
        meps.loc[meps["insurance_label"] == g, "total_medical_expense"].dropna()
        for g in meps["insurance_label"].dropna().unique()
    ]
    h, p = stats.kruskal(*groups)
    rows.append({
        "Dataset": "MEPS", "Test": "Kruskal-Wallis",
        "Null hypothesis": "Total medical expense is equal across insurance types",
        "Statistic": f"H={h:.1f}", "p-value": p,
    })

    # H6 (MEPS): total medical expense does not differ by acute-utilization.
    g0 = meps.loc[meps["any_acute_util"] == 0, "total_medical_expense"].dropna()
    g1 = meps.loc[meps["any_acute_util"] == 1, "total_medical_expense"].dropna()
    u, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    rows.append({
        "Dataset": "MEPS", "Test": "Mann-Whitney U",
        "Null hypothesis": "Total expense is the same with vs without acute utilization",
        "Statistic": f"U={u:,.0f}", "p-value": p,
    })

    out = pd.DataFrame(rows)
    out["Significant (a=0.05)"] = np.where(out["p-value"] < 0.05, "Yes", "No")
    out["p-value"] = out["p-value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    )
    return out[["Dataset", "Null hypothesis", "Test", "Statistic",
                "p-value", "Significant (a=0.05)"]]


# ===========================================================================
# 3. Odds ratio + relative risk (2x2) and unadjusted-OR panel
# ===========================================================================

def or_rr_2x2(df: pd.DataFrame, exposure: str, outcome: str,
              exposure_levels=(1, 0)) -> pd.DataFrame:
    """Build a 2x2 table and report OR and RR with 95% CIs.

    exposure_levels = (exposed_value, unexposed_value).
    A 0.5 Haldane-Anscombe correction is applied if any cell is zero.
    """
    exp_val, unexp_val = exposure_levels
    d = df.dropna(subset=[exposure, outcome])

    a = int(((d[exposure] == exp_val) & (d[outcome] == 1)).sum())   # exposed, event
    b = int(((d[exposure] == exp_val) & (d[outcome] == 0)).sum())   # exposed, no event
    c = int(((d[exposure] == unexp_val) & (d[outcome] == 1)).sum()) # unexposed, event
    e = int(((d[exposure] == unexp_val) & (d[outcome] == 0)).sum()) # unexposed, no event

    # Haldane-Anscombe correction for zero cells
    aa, bb, cc, dd = (a, b, c, e)
    if 0 in (a, b, c, e):
        aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, e + 0.5

    # Odds ratio + 95% CI (log scale)
    or_ = (aa * dd) / (bb * cc)
    se_log_or = np.sqrt(1 / aa + 1 / bb + 1 / cc + 1 / dd)
    or_lo = np.exp(np.log(or_) - 1.96 * se_log_or)
    or_hi = np.exp(np.log(or_) + 1.96 * se_log_or)

    # Relative risk + 95% CI
    risk_exp = aa / (aa + bb)
    risk_unexp = cc / (cc + dd)
    rr = risk_exp / risk_unexp
    se_log_rr = np.sqrt((1 - risk_exp) / aa + (1 - risk_unexp) / cc)
    rr_lo = np.exp(np.log(rr) - 1.96 * se_log_rr)
    rr_hi = np.exp(np.log(rr) + 1.96 * se_log_rr)

    return pd.DataFrame([
        {"Group": f"{exposure}=exposed", "Event (yes)": a, "No event": b,
         "n": a + b, "Risk": f"{a / (a + b):.1%}"},
        {"Group": f"{exposure}=unexposed", "Event (yes)": c, "No event": e,
         "n": c + e, "Risk": f"{c / (c + e):.1%}"},
        {"Group": "Odds ratio (OR)", "Event (yes)": "", "No event": "",
         "n": "", "Risk": f"{or_:.2f} (95% CI {or_lo:.2f}-{or_hi:.2f})"},
        {"Group": "Relative risk (RR)", "Event (yes)": "", "No event": "",
         "n": "", "Risk": f"{rr:.2f} (95% CI {rr_lo:.2f}-{rr_hi:.2f})"},
    ])


def unadjusted_or_panel(df: pd.DataFrame, outcome: str,
                        predictors: list[str]) -> pd.DataFrame:
    """One simple logistic regression per predictor -> unadjusted OR table.

    Useful as a screening step before the multivariable model: it shows the
    crude association of each candidate predictor with the outcome.
    """
    rows = []
    for p in predictors:
        d = df.dropna(subset=[outcome, p])
        try:
            model = smf.logit(f"{outcome} ~ C({p})" if d[p].dtype == object
                              else f"{outcome} ~ {p}", data=d).fit(disp=0)
        except Exception as exc:                       # pragma: no cover
            rows.append({"Predictor": p, "OR (95% CI)": f"(failed: {exc})",
                         "p-value": ""})
            continue
        # Skip the intercept (first term); report each non-intercept term.
        for term in model.params.index:
            if term == "Intercept":
                continue
            or_ = np.exp(model.params[term])
            lo, hi = np.exp(model.conf_int().loc[term])
            pv = model.pvalues[term]
            rows.append({
                "Predictor": term,
                "OR (95% CI)": f"{or_:.2f} ({lo:.2f}-{hi:.2f})",
                "p-value": "<0.001" if pv < 0.001 else f"{pv:.3f}",
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 4. Rates by group, with 95% confidence intervals
# ===========================================================================

def rate_by_group(df: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    """Rate (proportion ==1) of a binary outcome within each group level,
    with a Wilson 95% confidence interval and group size.
    """
    rows = []
    d = df.dropna(subset=[outcome, group])
    for level, sub in d.groupby(group, observed=True):
        n = len(sub)
        k = int((sub[outcome] == 1).sum())
        rate = k / n if n else np.nan
        # Wilson score interval (better than normal approx for small/extreme p)
        if n > 0:
            z = 1.96
            denom = 1 + z**2 / n
            centre = (rate + z**2 / (2 * n)) / denom
            half = (z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))) / denom
            lo, hi = max(0, centre - half), min(1, centre + half)
        else:
            lo = hi = np.nan
        rows.append({
            group: str(level), "n": n, "events": k,
            "Rate": f"{rate:.1%}", "95% CI": f"{lo:.1%}-{hi:.1%}",
            "_rate": rate,   # numeric copy for correct sorting
        })
    # Sort by the numeric rate (descending), then drop the helper column so
    # the displayed/exported table only carries the formatted percentage.
    out = pd.DataFrame(rows).sort_values("_rate", ascending=False, ignore_index=True)
    return out.drop(columns="_rate")


# ===========================================================================
# 5. Model 1 — logistic regression for missed_care_cost (BRFSS, inferential)
# ===========================================================================

def fit_missed_care_model(brfss: pd.DataFrame):
    """Multivariable logistic regression for missed_care_cost.

    Returns (fitted_result, odds_ratio_table, performance_dict).
    Predictors are kept lean to avoid quasi-separation given only ~178 events:
    insurance, low income, poor health, comorbidity burden, age, sex, race,
    and state. Estimates are reported as adjusted odds ratios.
    """
    predictors = [
        "uninsured", "is_low_income", "poor_health", "comorbidity_count",
        "age_band_collapsed", "sex_label", "race_group", "state_name",
    ]
    d = brfss.dropna(subset=["missed_care_cost"] + predictors).copy()
    d["missed_care_cost"] = d["missed_care_cost"].astype(int)

    formula = (
        "missed_care_cost ~ uninsured + is_low_income + poor_health "
        "+ comorbidity_count + C(age_band_collapsed) + C(sex_label) "
        "+ C(race_group, Treatment(reference='NH White')) "
        "+ C(state_name, Treatment(reference='Massachusetts'))"
    )
    result = sm.Logit.from_formula(formula, data=d).fit(disp=0, maxiter=200)

    or_table = _or_table_from_result(result)

    # In-sample performance (this is an inferential model, but AUC/accuracy
    # give a sense of discrimination).
    pred = result.predict(d)
    auc = roc_auc_score(d["missed_care_cost"], pred)
    perf = {
        "n": len(d),
        "events": int(d["missed_care_cost"].sum()),
        "pseudo_r2_mcfadden": float(result.prsquared),
        "in_sample_auc": float(auc),
        "llr_p_value": float(result.llr_pvalue),
    }
    return result, or_table, perf


def _or_table_from_result(result) -> pd.DataFrame:
    """Convert a fitted statsmodels logit result into an OR table."""
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues
    rows = []
    for term in params.index:
        if term == "Intercept":
            continue
        or_ = np.exp(params[term])
        lo, hi = np.exp(conf.loc[term])
        pv = pvals[term]
        rows.append({
            "Predictor": _pretty_term(term),
            "Adjusted OR": f"{or_:.2f}",
            "95% CI": f"{lo:.2f}-{hi:.2f}",
            "p-value": "<0.001" if pv < 0.001 else f"{pv:.3f}",
            "_or": or_, "_lo": lo, "_hi": hi,   # numeric copies for plotting
        })
    return pd.DataFrame(rows)


def _pretty_term(term: str) -> str:
    """Tidy a statsmodels term label for display."""
    return (term.replace("C(", "").replace(")", "")
                .replace("[T.", ": ").replace("]", "")
                .replace(", Treatment(reference='", " (ref ")
                .replace("'", ")"))


# ===========================================================================
# 6. Model 2 — predicting high_spender (MEPS, predictive)
# ===========================================================================

def fit_high_spender_models(meps: pd.DataFrame, random_state: int = 42):
    """Train and compare three classifiers for high_spender.

    Models
    ------
    Logistic regression  : interpretable baseline (also exported as OR table).
    LASSO (L1) logistic  : embedded variable selection.
    Gradient boosting    : flexible non-linear benchmark.

    Returns a dict with the cross-validated AUCs, a held-out classification
    report, the gradient-boosting feature importances, ROC-curve data for
    plotting, and the logistic OR table.
    """
    # Utilization COUNTS (not dollar amounts) are used to limit leakage:
    # high_spender is derived from total dollars, so dollar predictors would
    # be mechanically circular. Counts/flags are predictive but distinct.
    num_features = ["age", "comorbidity_count", "n_er_visits",
                    "n_inpatient_stays", "n_rx_fills"]
    cat_features = ["insurance_label", "sex_label", "race_ethnicity"]
    features = num_features + cat_features

    d = meps.dropna(subset=["high_spender"] + features).copy()
    X = d[features]
    y = d["high_spender"].astype(int)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
    ])

    models = {
        "Logistic regression": LogisticRegression(max_iter=1000),
        "LASSO logistic (L1)": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, max_iter=1000),
        "Gradient boosting": GradientBoostingClassifier(random_state=random_state),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_rows = []
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        cv_rows.append({
            "Model": name,
            "CV ROC AUC (mean)": f"{scores.mean():.3f}",
            "CV ROC AUC (sd)": f"{scores.std():.3f}",
        })
    cv_table = pd.DataFrame(cv_rows)

    # Held-out split for a concrete classification report + ROC curves.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=random_state)

    roc_data, report_table, gb_importance = {}, None, None
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_data[name] = (fpr, tpr, roc_auc_score(y_te, proba))

        if name == "Gradient boosting":
            preds = pipe.predict(X_te)
            rep = classification_report(y_te, preds, output_dict=True,
                                        zero_division=0)
            report_table = pd.DataFrame(rep).T.round(3)
            # Map feature importances back to readable names.
            feat_names = (num_features +
                          list(pipe.named_steps["pre"]
                               .named_transformers_["cat"]
                               .get_feature_names_out(cat_features)))
            gb_importance = (pd.DataFrame({
                "Feature": feat_names,
                "Importance": pipe.named_steps["clf"].feature_importances_,
            }).sort_values("Importance", ascending=False, ignore_index=True))

    # Interpretable OR table from a plain logistic fit (statsmodels) on the
    # numeric features (+ insurance dummies) for the report.
    or_table = _meps_logit_or_table(d)

    return {
        "n": len(d),
        "events": int(y.sum()),
        "cv_table": cv_table,
        "report_table": report_table,
        "gb_importance": gb_importance,
        "roc_data": roc_data,
        "or_table": or_table,
    }


def _meps_logit_or_table(d: pd.DataFrame) -> pd.DataFrame:
    """statsmodels logistic OR table for high_spender (interpretability)."""
    formula = (
        "high_spender ~ age + comorbidity_count + n_er_visits "
        "+ n_inpatient_stays + n_rx_fills + C(sex_label) "
        "+ C(insurance_label, Treatment(reference='Any private'))"
    )
    res = sm.Logit.from_formula(formula, data=d).fit(disp=0, maxiter=200)
    return _or_table_from_result(res)


# ===========================================================================
# 7. Visualizations
# ===========================================================================

def map_missed_care_by_state(brfss: pd.DataFrame, output_dir: Path = OUTPUT_FIGURES) -> Path:
    """New England choropleth of the missed-care-due-to-cost rate by state.

    Uses plotly (with kaleido) for a true geographic choropleth and zooms to
    the plotted states. Saved as a 300-DPI PNG for the Word report.
    """
    import plotly.express as px  # local import keeps plotly optional

    rate = (brfss.dropna(subset=["missed_care_cost"])
                 .groupby("state_name")["missed_care_cost"]
                 .agg(rate="mean", n="size")
                 .reset_index())
    rate = rate.merge(STATE_CONTEXT[["state_name", "state_abbr"]], on="state_name")
    rate["rate_pct"] = (rate["rate"] * 100).round(1)
    rate["hover"] = rate["state_name"] + " — " + rate["rate_pct"].astype(str) + "% (n=" + rate["n"].astype(str) + ")"

    fig = px.choropleth(
        rate, locations="state_abbr", locationmode="USA-states",
        color="rate_pct", scope="usa",
        color_continuous_scale="OrRd",
        hover_name="hover",
        labels={"rate_pct": "Missed care (%)"},
    )
    fig.update_geos(fitbounds="locations", visible=True)
    fig.update_layout(
        title_text="Missed care due to cost by state — diabetic adults 45–64<br>"
                   "<sub>BRFSS 2023, New England</sub>",
        margin=dict(l=10, r=10, t=70, b=10),
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "m2_map_missed_care_by_state.png"
    fig.write_image(str(path), width=900, height=650, scale=2)
    return path


# Plain-English label map for forest plot predictor names
_FOREST_LABELS = {
    "uninsured":                                        "Uninsured (vs. insured)",
    "is_low_income":                                    "Low income",
    "poor_health":                                      "Poor self-rated health",
    "comorbidity_count":                                "Each additional chronic condition",
    "age_band_collapsed: 55-64":                        "Age 55–64 (vs. 45–54)",
    "sex_label: Male":                                  "Male (vs. female)",
    "race_group (ref NH White): Hispanic":              "Hispanic (vs. NH White)",
    "race_group (ref NH White): NH Black":              "NH Black (vs. NH White)",
    "race_group (ref NH White): Other":                 "Other race/ethnicity (vs. NH White)",
    "state_name (ref Massachusetts): Alabama":          "Alabama (vs. Massachusetts)",
    "state_name (ref Massachusetts): Arkansas":         "Arkansas (vs. Massachusetts)",
    "state_name (ref Massachusetts): Connecticut":      "Connecticut (vs. Massachusetts)",
    "state_name (ref Massachusetts): Louisiana":        "Louisiana (vs. Massachusetts)",
    "state_name (ref Massachusetts): Maine":            "Maine (vs. Massachusetts)",
    "state_name (ref Massachusetts): Mississippi":      "Mississippi (vs. Massachusetts)",
    "state_name (ref Massachusetts): New Hampshire":    "New Hampshire (vs. Massachusetts)",
    "state_name (ref Massachusetts): Rhode Island":     "Rhode Island (vs. Massachusetts)",
    "state_name (ref Massachusetts): Vermont":          "Vermont (vs. Massachusetts)",
    "state_name (ref Massachusetts): West Virginia":    "West Virginia (vs. Massachusetts)",
}

# Plain-English label map for feature importance names
_FEATURE_LABELS = {
    "n_rx_fills":                       "Prescription fills",
    "n_inpatient_stays":                "Inpatient hospital stays",
    "age":                              "Age",
    "n_er_visits":                      "Emergency room visits",
    "insurance_label_Uninsured":        "Uninsured status",
    "race_ethnicity_NH White":          "Race: NH White",
    "comorbidity_count":                "Number of chronic conditions",
    "race_ethnicity_NH Asian":          "Race: NH Asian",
    "sex_label_Male":                   "Sex: Male",
    "race_ethnicity_NH Other/Multiple": "Race: Other/Multiple",
}


def forest_plot_or(or_table: pd.DataFrame, output_dir: Path = OUTPUT_FIGURES,
                   title: str = "Adjusted odds ratios — missed care due to cost",
                   filename: str = "m2_forest_missed_care.png") -> Path:
    """Forest plot of adjusted odds ratios with plain-English labels,
    colour-coded by predictor type, and OR values annotated on each row.
    """
    import matplotlib.patches as mpatches

    d = or_table.dropna(subset=["_or"]).copy()
    d["_label"] = d["Predictor"].map(lambda x: _FOREST_LABELS.get(str(x), str(x)))

    def _color(label: str) -> str:
        if "Uninsured" in label:
            return "#c0392b"
        if "Low income" in label or "Poor self-rated" in label:
            return "#e67e22"
        return "#1a2f45"

    d["_color"] = d["_label"].apply(_color)
    d = d.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(5, 0.52 * len(d))))
    y = np.arange(len(d))

    for i, row in d.iterrows():
        ax.plot(row["_or"], i, "o", color=row["_color"], markersize=8, zorder=3)
        ax.plot([row["_lo"], row["_hi"]], [i, i],
                color=row["_color"], linewidth=2, alpha=0.7, zorder=2)
        ax.text(row["_hi"] * 1.05, i, f'{row["_or"]:.2f}',
                va="center", ha="left", fontsize=9, color=row["_color"])

    ax.axvline(1.0, color="#c0392b", linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(d["_label"], fontsize=10)
    ax.set_xlabel("Adjusted odds ratio (log scale)  —  values >1 indicate higher risk", fontsize=11)
    ax.set_title(title + "\nBRFSS 2023, diabetic adults 45–64", loc="left", fontsize=13)

    legend_patches = [
        mpatches.Patch(color="#c0392b",  label="Insurance status"),
        mpatches.Patch(color="#e67e22",  label="Socioeconomic factors"),
        mpatches.Patch(color="#1a2f45",  label="Demographics / health"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=10)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_curves(roc_data: dict, output_dir: Path = OUTPUT_FIGURES,
               filename: str = "m2_roc_high_spender.png") -> Path:
    """Overlay ROC curves with AUC annotated directly on each curve
    and a shaded region under the best model.
    """
    palette = {"Logistic regression": "#1a2f45",
               "LASSO logistic (L1)": "#16a085",
               "Gradient boosting":   "#c0392b"}
    fig, ax = plt.subplots(figsize=(7, 7))

    best_name = max(roc_data, key=lambda k: roc_data[k][2])
    best_fpr, best_tpr, _ = roc_data[best_name]
    ax.fill_between(best_fpr, best_fpr, best_tpr, alpha=0.07,
                    color=palette[best_name])

    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, color=palette[name], linewidth=2.5,
                label=f"{name}  (AUC = {auc:.3f})")
        idx = np.searchsorted(fpr, 0.38)
        ax.annotate(
            f"AUC = {auc:.3f}",
            xy=(fpr[idx], tpr[idx]),
            xytext=(fpr[idx] + 0.04, tpr[idx] - 0.06),
            fontsize=9.5, color=palette[name], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=palette[name], lw=0.8),
        )

    ax.plot([0, 1], [0, 1], color="grey", linestyle="--",
            linewidth=1.2, label="Random classifier (AUC = 0.500)")
    ax.set_xlabel("False positive rate  (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True positive rate  (Sensitivity)", fontsize=12)
    ax.set_title(
        "Predicting high-spender status — model comparison\nMEPS 2023, diabetic adults 45–64",
        loc="left", fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def importance_plot(gb_importance: pd.DataFrame, output_dir: Path = OUTPUT_FIGURES,
                    filename: str = "m2_gb_importance.png") -> Path:
    """Horizontal bar of gradient-boosting feature importances with
    plain-English names, value annotations, and a blue colour gradient.
    """
    d = gb_importance.head(10).copy()
    d["_label"] = d["Feature"].map(lambda x: _FEATURE_LABELS.get(str(x), str(x)))
    d = d.sort_values("Importance", ascending=True).reset_index(drop=True)

    norm = plt.Normalize(d["Importance"].min(), d["Importance"].max())
    colors = plt.cm.Blues(norm(d["Importance"]) * 0.7 + 0.25)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(d["_label"], d["Importance"], color=colors,
                   edgecolor="white", height=0.65)

    for bar, val in zip(bars, d["Importance"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10,
                color="#555555")

    ax.set_xlabel(
        "Gradient-boosting feature importance\n"
        "(higher = stronger predictor of high healthcare spending)", fontsize=11)
    ax.set_title(
        "What drives high healthcare spending?\n"
        "MEPS 2023, gradient-boosted model — top 10 predictors",
        loc="left", fontsize=13,
    )
    ax.set_xlim(0, d["Importance"].max() * 1.18)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 7c. Line charts — real time-series (A) and model prediction curves (B)
# ---------------------------------------------------------------------------

def lineplot_missed_care_trend(trend: pd.DataFrame | None = None,
                               output_dir: Path = OUTPUT_FIGURES,
                               filename: str = "m2_trend_missed_care.png") -> Path:
    """Real multi-year time-series of the missed-care-due-to-cost rate among
    diabetic adults 45-64, United States vs New England.

    The yearly rates are produced by src/brfss_trend.py (raw BRFSS files,
    same case definition as Milestone 1). If `trend` is not supplied the
    function reads the saved m2_missed_care_trend.csv.
    """
    if trend is None:
        trend = pd.read_csv(OUTPUT_TABLES / "m2_missed_care_trend.csv")

    colors = {"United States": "#3a7ca5", "New England": "#c0392b"}
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for region, sub in trend.groupby("region"):
        sub = sub.sort_values("year")
        col = colors.get(region, "#555555")
        ax.plot(sub["year"], sub["missed_care_rate_pct"],
                marker="o", linewidth=2.6, markersize=8,
                label=region, color=col, zorder=3)

        # Alternate annotation offsets so US and NE labels don't collide
        for _, r in sub.iterrows():
            offset = 10 if region == "United States" else -16
            ax.annotate(
                f"{r['missed_care_rate_pct']:.1f}%",
                (r["year"], r["missed_care_rate_pct"]),
                textcoords="offset points", xytext=(0, offset),
                ha="center", fontsize=10, fontweight="bold", color=col,
            )

    # Shade the gap between the two lines
    us  = trend[trend["region"] == "United States"].sort_values("year")
    ne  = trend[trend["region"] == "New England"].sort_values("year")
    if len(us) == len(ne):
        ax.fill_between(us["year"], ne["missed_care_rate_pct"],
                        us["missed_care_rate_pct"],
                        alpha=0.08, color="#3a7ca5",
                        label="Gap (NE below U.S.)")

    # COVID annotation
    ax.axvline(2020, color="#888888", linestyle=":", linewidth=1.4, zorder=2)
    ax.text(2020.05, trend["missed_care_rate_pct"].max() * 1.15,
            "COVID-19\ndisruption (2020)",
            fontsize=9, color="#888888", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#888888", alpha=0.7))

    ax.set_xticks(sorted(trend["year"].unique()))
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.set_ylim(0, trend["missed_care_rate_pct"].max() * 1.30)
    ax.set_ylabel("Adults who skipped care due to cost (past 12 months)",
                  fontsize=12)
    ax.set_xlabel("BRFSS survey year", fontsize=12)
    ax.legend(title="Sample", fontsize=11, title_fontsize=11)
    ax.set_title(
        "Missed care due to cost among diabetic adults 45–64, 2019–2023\n"
        "BRFSS annual files, unweighted — U.S. vs New England",
        loc="left", fontsize=13,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def curve_missed_care_vs_comorbidity(brfss: pd.DataFrame,
                                     output_dir: Path = OUTPUT_FIGURES,
                                     filename: str = "m2_curve_missed_care.png") -> Path:
    """Predicted missed-care probability by comorbidity burden and insurance,
    with gap shading, % annotations on every other point, and a callout
    showing the gap size at comorbidity count = 4.
    """
    d = brfss.dropna(subset=["missed_care_cost", "comorbidity_count", "uninsured"]).copy()
    d["missed_care_cost"] = d["missed_care_cost"].astype(int)
    res = sm.Logit.from_formula(
        "missed_care_cost ~ comorbidity_count * uninsured", data=d).fit(disp=0)

    grid = pd.DataFrame({"comorbidity_count": range(0, 9)})
    fig, ax = plt.subplots(figsize=(10, 6.5))

    lines = {}
    for u, lab, col in [(0, "Insured", "#3a7ca5"), (1, "Uninsured", "#c0392b")]:
        g = grid.copy()
        g["uninsured"] = u
        g["p"] = res.predict(g)
        lines[lab] = g
        ax.plot(g["comorbidity_count"], g["p"], marker="o", linewidth=2.6,
                markersize=7, label=lab, color=col)
        for _, r in g.iterrows():
            if r["comorbidity_count"] % 2 == 0:
                offset = 10 if u == 1 else -14
                ax.annotate(f"{r['p']:.0%}",
                            xy=(r["comorbidity_count"], r["p"]),
                            xytext=(0, offset), textcoords="offset points",
                            ha="center", fontsize=8.5, color=col, fontweight="bold")

    ax.fill_between(
        lines["Insured"]["comorbidity_count"],
        lines["Insured"]["p"],
        lines["Uninsured"]["p"],
        alpha=0.08, color="#c0392b", label="Gap (insurance effect)",
    )

    pt_ins   = float(lines["Insured"].loc[lines["Insured"]["comorbidity_count"] == 4, "p"].iloc[0])
    pt_unins = float(lines["Uninsured"].loc[lines["Uninsured"]["comorbidity_count"] == 4, "p"].iloc[0])
    ax.annotate(
        f"Gap at 4 conditions:\n{pt_unins:.0%} vs {pt_ins:.0%}",
        xy=(4, (pt_ins + pt_unins) / 2),
        xytext=(5.5, (pt_ins + pt_unins) / 2),
        fontsize=10, color="#555555",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555555", alpha=0.8),
    )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Number of chronic conditions (in addition to diabetes)", fontsize=12)
    ax.set_ylabel("Predicted probability of skipping care due to cost", fontsize=12)
    ax.set_title(
        "Uninsured diabetic adults face sharply higher missed-care risk\n"
        "BRFSS 2023, adults 45–64 — logistic model with insurance × comorbidity interaction",
        loc="left", fontsize=13,
    )
    ax.legend(title="Insurance status", fontsize=11)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def curve_high_spender_vs_rx(meps: pd.DataFrame,
                             output_dir: Path = OUTPUT_FIGURES,
                             filename: str = "m2_curve_high_spender.png") -> Path:
    """Predicted high-spender probability vs prescription fills, with
    shaded area under the curve, a rug of observed counts, and 25%/50%
    risk milestone annotations.
    """
    d = meps.dropna(subset=["high_spender", "n_rx_fills"]).copy()
    res = sm.Logit.from_formula("high_spender ~ n_rx_fills", data=d).fit(disp=0)

    hi = int(d["n_rx_fills"].quantile(0.95))
    grid = pd.DataFrame({"n_rx_fills": range(0, hi + 1)})
    grid["p"] = res.predict(grid)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(grid["n_rx_fills"], 0, grid["p"], alpha=0.10, color="#1a2f45")
    ax.plot(grid["n_rx_fills"], grid["p"], color="#1a2f45", linewidth=2.8)
    ax.plot(d["n_rx_fills"].clip(upper=hi), [0.015] * len(d),
            "|", color="#3a7ca5", alpha=0.25, markersize=8)

    for target_p, label_offset in [(0.25, 8), (0.50, 8)]:
        match = grid[grid["p"] >= target_p]
        if not match.empty:
            x_cross = int(match.iloc[0]["n_rx_fills"])
            ax.axvline(x_cross, color="#555555", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.axhline(target_p, color="#555555", linestyle=":", linewidth=1.2, alpha=0.7)
            ax.annotate(
                f"{target_p:.0%} risk\nat {x_cross} fills",
                xy=(x_cross, target_p),
                xytext=(x_cross + label_offset, target_p - 0.07),
                fontsize=9.5, color="#555555",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=0.9),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#555555", alpha=0.8),
            )

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_xlabel("Number of prescription fills in 2023", fontsize=12)
    ax.set_ylabel(
        "Predicted probability of high-spender status\n"
        "(≥75th percentile of total medical expenses)", fontsize=11)
    ax.set_title(
        "High-prescription patients face rapidly rising spending risk\n"
        "MEPS 2023, diabetic adults 45–64 — logistic model",
        loc="left", fontsize=13,
    )
    ax.set_ylim(0, 1)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    sns.despine(ax=ax)
    ax.grid(False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ===========================================================================
# 8. Excel/CSV export helper (matches src/descriptive_tables.save_table)
# ===========================================================================

def save_table(table: pd.DataFrame, filename_stem: str,
               output_dir: Path = OUTPUT_TABLES, sheet_name: str = "Table") -> dict:
    """Save a table as both CSV and a width-formatted XLSX. Drops helper
    columns whose names start with '_' (used only for plotting).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export = table[[c for c in table.columns if not str(c).startswith("_")]].copy()

    csv_path = output_dir / f"{filename_stem}.csv"
    xlsx_path = output_dir / f"{filename_stem}.xlsx"
    export.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        export.to_excel(writer, sheet_name=sheet_name, index=False)
        ws = writer.sheets[sheet_name]
        for i, col in enumerate(export.columns, start=1):
            max_len = max(len(str(col)),
                          export[col].astype(str).str.len().max() if len(export) else 0)
            ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(max_len + 2, 45)
    return {"csv": csv_path, "xlsx": xlsx_path}


# ===========================================================================
# 10. Survey-weighted descriptive statistics (BRFSS)
# ===========================================================================

def weighted_prevalence_table(brfss: pd.DataFrame) -> pd.DataFrame:
    """Compute survey-weighted prevalence rates for key binary outcomes
    using BRFSS person-level weights (_LLCPWT).

    This addresses the survey-weight limitation noted in the report:
    weighted estimates account for BRFSS's unequal selection probabilities
    and are more representative of the true population than unweighted rates.

    Returns a table comparing unweighted vs weighted prevalence for the
    three headline outcomes, overall and by region.
    """
    outcomes = {
        "missed_care_cost": "Missed care due to cost",
        "uninsured":        "Uninsured",
        "poor_health":      "Fair/poor self-rated health",
    }
    rows = []

    for scope_label, scope_df in [
        ("Overall",             brfss),
        ("New England",         brfss[brfss["region"] == "New England"]),
        ("Lower-income South",  brfss[brfss["region"] == "Lower-income South"]),
    ]:
        for col, label in outcomes.items():
            d = scope_df.dropna(subset=[col, "_LLCPWT"])
            if len(d) == 0:
                continue
            unweighted = d[col].mean()
            wstats = DescrStatsW(d[col], weights=d["_LLCPWT"], ddof=1)
            weighted = wstats.mean
            rows.append({
                "Scope":        scope_label,
                "Outcome":      label,
                "n":            len(d),
                "Unweighted %": f"{unweighted:.1%}",
                "Weighted %":   f"{weighted:.1%}",
                "Difference":   f"{(weighted - unweighted):+.1%}",
            })

    return pd.DataFrame(rows)


# ===========================================================================
# 9. Master driver — run the whole Milestone 2 analysis end to end
# ===========================================================================

def run_all() -> dict:
    """Run every Milestone-2 step and persist tables/figures. Returns a dict
    of the in-memory results so a notebook can display them.
    """
    print("Loading Milestone 1 cleaned data...")
    meps = engineer_meps_features(load_meps())
    brfss = engineer_brfss_features(load_brfss())
    print(f"  MEPS : {len(meps):,} rows   BRFSS: {len(brfss):,} rows")

    print("Building ACS regional comparison table...")
    acs_comparison = acs_region_comparison()
    save_table(acs_comparison, "m2_acs_region_comparison")

    print("Computing survey-weighted prevalence estimates (BRFSS)...")
    weighted_prev = weighted_prevalence_table(brfss)
    save_table(weighted_prev, "m2_weighted_prevalence")

    print("Running hypothesis tests...")
    tests = run_hypothesis_tests(meps, brfss)
    save_table(tests, "m2_hypothesis_tests")

    print("Building OR/RR + unadjusted OR panel...")
    or_rr = or_rr_2x2(brfss, "uninsured", "missed_care_cost")
    save_table(or_rr, "m2_or_rr_uninsured_missed_care")
    crude = unadjusted_or_panel(
        brfss, "missed_care_cost",
        ["uninsured", "is_low_income", "poor_health", "multimorbid",
         "race_group", "state_name"])
    save_table(crude, "m2_unadjusted_or_panel")

    print("Tabulating rates by group...")
    rate_state = rate_by_group(brfss, "missed_care_cost", "state_name")
    rate_income = rate_by_group(brfss, "missed_care_cost", "income_tier")
    save_table(rate_state, "m2_missed_care_rate_by_state")
    save_table(rate_income, "m2_missed_care_rate_by_income")

    print("Fitting Model 1 — logistic regression (BRFSS missed care)...")
    _, or_brfss, perf_brfss = fit_missed_care_model(brfss)
    save_table(or_brfss, "m2_logit_missed_care_or")

    print("Fitting Model 2 — high-spender classifiers (MEPS)...")
    meps_res = fit_high_spender_models(meps)
    save_table(meps_res["cv_table"], "m2_high_spender_cv_auc")
    save_table(meps_res["or_table"], "m2_logit_high_spender_or")
    save_table(meps_res["gb_importance"], "m2_high_spender_gb_importance")
    save_table(meps_res["report_table"].reset_index().rename(
        columns={"index": "class"}), "m2_high_spender_classification_report")

    print("Rendering figures...")
    figs = {}
    try:
        figs["map"] = map_missed_care_by_state(brfss)
    except Exception as exc:                            # pragma: no cover
        print(f"  ⚠ map skipped: {exc}")
    figs["forest"] = forest_plot_or(or_brfss)
    figs["roc"] = roc_curves(meps_res["roc_data"])
    figs["importance"] = importance_plot(meps_res["gb_importance"])
    # Model-based line charts (always available from the 2023 data).
    figs["curve_missed_care"] = curve_missed_care_vs_comorbidity(brfss)
    figs["curve_high_spender"] = curve_high_spender_vs_rx(meps)
    # Real multi-year time-series — only if the trend table has been built
    # (run `python -m src.brfss_trend` first; it downloads raw BRFSS files).
    trend_csv = OUTPUT_TABLES / "m2_missed_care_trend.csv"
    if trend_csv.exists():
        figs["trend"] = lineplot_missed_care_trend()
    else:
        print("  ℹ time-series skipped — run `python -m src.brfss_trend` to build "
              "outputs/tables/m2_missed_care_trend.csv first.")

    print("\nDone. Tables -> outputs/tables/m2_*  |  Figures -> outputs/figures/m2_*")
    return {
        "meps": meps, "brfss": brfss,
        "acs_comparison": acs_comparison,
        "weighted_prev": weighted_prev,
        "tests": tests, "or_rr": or_rr, "crude": crude,
        "rate_state": rate_state, "rate_income": rate_income,
        "or_brfss": or_brfss, "perf_brfss": perf_brfss,
        "meps_res": meps_res, "figs": figs,
    }


if __name__ == "__main__":
    run_all()