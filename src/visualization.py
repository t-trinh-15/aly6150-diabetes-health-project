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
        "BRFSS 2023, diabetic adults 45–64 in New England",
        loc="left",
    )

    return _save(fig, output_dir, "02_brfss_missed_care_by_insurance.png")


# ===========================================================================
# Master function
# ===========================================================================

def build_all_charts(meps: pd.DataFrame, brfss: pd.DataFrame,
                     output_dir: Path) -> dict:
    """Generate the two required charts."""
    output_dir = Path(output_dir)
    print("Building charts...")

    paths = {
        "chart_1_meps_composition":  chart_meps_cost_composition(meps, output_dir),
        "chart_2_brfss_insurance":   chart_brfss_missed_care_by_insurance(brfss, output_dir),
    }

    for name, path in paths.items():
        print(f"  ✓ {path.name}")

    print(f"\nBoth charts saved to: {output_dir}")
    return paths