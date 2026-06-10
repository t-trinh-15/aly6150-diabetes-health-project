"""Descriptive charts for the MEPS + BRFSS diabetes analysis.

Two key charts:
  Chart 1 (MEPS)  — Cost composition by insurance type
  Chart 2 (BRFSS) — Missed care due to cost by insurance status

Both saved as 300 DPI PNGs to OUTPUT_FIGURES.
"""

from __future__ import annotations
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ===========================================================================
# Global styling
# ===========================================================================

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "Arial",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, output_dir: Path, filename: str) -> Path:
    """Save figure to PNG and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path


# ===========================================================================
# Chart 1 — MEPS: cost composition by insurance type
# ===========================================================================

def chart_meps_cost_composition(meps: pd.DataFrame, output_dir: Path) -> Path:
    """Stacked bar of mean spending by category, faceted by insurance type."""
    expense_cols = {
        "inpatient_expense":    "Inpatient",
        "er_expense":           "ER",
        "office_visit_expense": "Office visits",
        "prescription_expense": "Prescriptions",
    }
    grouped = (meps.groupby("insurance_label")[list(expense_cols.keys())]
                   .mean()
                   .rename(columns=expense_cols))

    cat_order = ["Any private", "Public only", "Uninsured"]
    grouped = grouped.reindex(cat_order)
    n_by_group = meps["insurance_label"].value_counts().reindex(cat_order)

    fig, ax = plt.subplots(figsize=(9, 6))
    # Cool blue palette ordered by care intensity (darkest = most intensive)
    colors = ["#2c3e50", "#5d8aa8", "#85b8d8", "#c5dfee"]
    grouped.plot(kind="bar", stacked=True, ax=ax, color=colors,
                 edgecolor="#34495e", linewidth=0.8)

    ax.set_ylabel("Mean spending per person in 2023 (USD)")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"{cat}\n(n={n_by_group[cat]:,})" for cat in cat_order],
        rotation=0,
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    for i, cat in enumerate(cat_order):
        total = grouped.loc[cat].sum()
        ax.text(i, total + 800, f"${total:,.0f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    # Add headroom so the total labels don't get clipped
    ax.set_ylim(0, grouped.sum(axis=1).max() * 1.12)

    # Push legend outside the plot area to the right
    ax.legend(title="Spending category", loc="upper left",
              bbox_to_anchor=(1.02, 1), framealpha=0.95,
              borderaxespad=0)
    ax.set_title(
        "Cost composition by insurance type — diabetic adults 45–64\n"
        "MEPS 2023, mean per-person spending",
        loc="left",
    )

    return _save(fig, output_dir, "01_meps_cost_composition.png")

# ===========================================================================
# Chart 2 — BRFSS: missed care by insurance status
# ===========================================================================

def chart_brfss_missed_care_by_insurance(brfss: pd.DataFrame, output_dir: Path) -> Path:
    """Side-by-side bar: missed_care_cost rate, insured vs uninsured."""
    g = (brfss.dropna(subset=["uninsured", "missed_care_cost"])
              .groupby("uninsured")
              .agg(n=("missed_care_cost", "size"),
                   rate=("missed_care_cost", "mean"))
              .reset_index())
    g["label"] = g["uninsured"].map({0.0: "Insured", 1.0: "Uninsured"})

    # Build a dynamic subtitle reflecting which regions are in the data
    if "region" in brfss.columns:
        regions = sorted(brfss["region"].dropna().unique())
        region_str = " vs. ".join(regions) if len(regions) > 1 else regions[0]
    else:
        region_str = "New England"

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(g["label"], g["rate"],
                  color=["#5b8b9e", "#c0392b"], edgecolor="white",
                  linewidth=1.5, width=0.6)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, max(g["rate"]) * 1.25)
    ax.set_ylabel("Missed care due to cost (past year)")
    ax.set_xlabel("")
    ax.set_xticklabels(
        [f"{lbl}\n(n={n:,})" for lbl, n in zip(g["label"], g["n"])],
    )

    for bar, rate in zip(bars, g["rate"]):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.01,
                f"{rate:.1%}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")

    if g["rate"].iloc[0] > 0:
        ratio = g["rate"].iloc[1] / g["rate"].iloc[0]
        ax.text(0.5, max(g["rate"]) * 1.10,
                f"Uninsured are {ratio:.1f}× more likely to miss care",
                transform=ax.transAxes, ha="center",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff4ce",
                          edgecolor="#bf8f00"))

    ax.set_title(
        "Missed care due to cost by insurance status\n"
        f"BRFSS 2023, diabetic adults 45–64 — {region_str}",
        loc="left",
    )

    return _save(fig, output_dir, "02_brfss_missed_care_by_insurance.png")


# ===========================================================================
# Chart 3 — BRFSS: missed care by insurance status, New England vs. South
# ===========================================================================

def chart_brfss_missed_care_by_region(brfss: pd.DataFrame, output_dir: Path) -> Path:
    """Grouped bar: missed_care_cost rate by insurance status, split by region.

    Requires a `region` column produced by the expanded clean_brfss().
    If region is absent the chart falls back to the overall rate only.
    """
    if "region" not in brfss.columns:
        raise ValueError("DataFrame has no 'region' column — run expanded clean_brfss() first.")

    plot_df = (
        brfss.dropna(subset=["uninsured", "missed_care_cost", "region"])
             .groupby(["region", "uninsured"])
             .agg(n=("missed_care_cost", "size"),
                  rate=("missed_care_cost", "mean"))
             .reset_index()
    )
    plot_df["insurance_label"] = plot_df["uninsured"].map({0.0: "Insured", 1.0: "Uninsured"})

    regions = sorted(plot_df["region"].unique())
    x = np.arange(len(regions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Insured": "#5b8b9e", "Uninsured": "#c0392b"}

    for i, ins_label in enumerate(["Insured", "Uninsured"]):
        subset = plot_df[plot_df["insurance_label"] == ins_label].set_index("region")
        rates  = [subset.loc[r, "rate"] if r in subset.index else 0 for r in regions]
        ns     = [int(subset.loc[r, "n"])  if r in subset.index else 0 for r in regions]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width,
                      label=ins_label, color=colors[ins_label],
                      edgecolor="white", linewidth=1.2)
        for bar, rate, n in zip(bars, rates, ns):
            if n > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        rate + 0.005,
                        f"{rate:.1%}\n(n={n:,})",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, plot_df["rate"].max() * 1.35)
    ax.set_ylabel("Missed care due to cost (past year)")
    ax.legend(title="Insurance status", framealpha=0.9)
    ax.set_title(
        "Missed care due to cost by insurance status and region\n"
        "BRFSS 2023, diabetic adults 45–64",
        loc="left",
    )

    return _save(fig, output_dir, "03_brfss_missed_care_by_region.png")


# ===========================================================================
# Chart 4 — BRFSS: SDOH burden index by region (Module 29 subset)
# ===========================================================================

def chart_brfss_sdoh_by_region(brfss: pd.DataFrame, output_dir: Path) -> Path:
    """Stacked bar showing SDOH hardship item prevalence by region.

    Scoped to respondents with sdoh_eligible == 1 (Module 29 participants).
    """
    sdoh_flags = {
        "food_insecure":        "Food insecure",
        "housing_cost_burden":  "Housing cost burden",
        "utilities_shutoff":    "Utilities shut off",
        "transport_barrier":    "Transport barrier",
    }

    # Filter to SDOH-eligible
    if "sdoh_eligible" in brfss.columns:
        sub = brfss[brfss["sdoh_eligible"] == 1].copy()
    else:
        sub = brfss.dropna(subset=list(sdoh_flags.keys())).copy()

    if "region" not in sub.columns or sub.empty:
        raise ValueError("Need 'region' column and SDOH-eligible rows.")

    # Mean prevalence of each hardship by region
    grouped = (
        sub.groupby("region")[list(sdoh_flags.keys())]
           .mean()
           .rename(columns=sdoh_flags)
    )
    regions = sorted(grouped.index)
    grouped = grouped.reindex(regions)

    fig, ax = plt.subplots(figsize=(9, 6))
    sdoh_colors = ["#e67e22", "#e74c3c", "#8e44ad", "#2980b9"]
    grouped.T.plot(kind="bar", ax=ax, color=sdoh_colors,
                   edgecolor="white", linewidth=0.8)

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0, grouped.values.max() * 1.25)
    ax.set_xlabel("")
    ax.set_ylabel("Prevalence (% of SDOH-eligible respondents)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Region", loc="upper right", framealpha=0.9)
    n_by_region = sub["region"].value_counts().to_dict()
    n_note = "  |  ".join(f"{r}: n={n_by_region.get(r, 0):,}" for r in regions)
    ax.set_title(
        "Social determinants of health burden by region\n"
        f"BRFSS 2023, diabetic adults 45–64 (SDOH-eligible)  —  {n_note}",
        loc="left",
    )

    return _save(fig, output_dir, "04_brfss_sdoh_by_region.png")


# ===========================================================================
# Master function
# ===========================================================================

def build_all_charts(meps: pd.DataFrame, brfss: pd.DataFrame,
                     output_dir: Path) -> dict:
    """Generate all charts. Charts 3 and 4 require the expanded BRFSS
    dataset with a `region` column and SDOH variables.
    """
    output_dir = Path(output_dir)
    print("Building charts...")

    paths = {
        "chart_1_meps_composition":    chart_meps_cost_composition(meps, output_dir),
        "chart_2_brfss_insurance":     chart_brfss_missed_care_by_insurance(brfss, output_dir),
    }

    # Charts 3 and 4 require the expanded dataset
    if "region" in brfss.columns and brfss["region"].notna().any():
        try:
            paths["chart_3_regional_comparison"] = chart_brfss_missed_care_by_region(brfss, output_dir)
        except Exception as e:
            print(f"  ⚠ Chart 3 skipped: {e}")
        try:
            paths["chart_4_sdoh_burden"] = chart_brfss_sdoh_by_region(brfss, output_dir)
        except Exception as e:
            print(f"  ⚠ Chart 4 skipped: {e}")
    else:
        print("  ℹ Charts 3 & 4 skipped — 'region' column not found (run expanded clean_brfss).")

    for name, path in paths.items():
        print(f"  ✓ {path.name}")

    print(f"\nCharts saved to: {output_dir}")
    return paths