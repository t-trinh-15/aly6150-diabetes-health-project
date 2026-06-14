"""Milestone 2 — multi-year BRFSS trend builder (real time-series data).

The Milestone 1 analysis is single-year (2023, cross-sectional), so it has no
time dimension to plot. This module fills that gap HONESTLY: it downloads the
raw BRFSS annual files for several years, applies the SAME case definition used
in Milestone 1 (adults aged 45-64 with diagnosed diabetes), and computes the
yearly "missed care due to cost" rate. The result is a genuine, methodology-
consistent time-series for the national sample and for the six New England
states.

It is a stand-alone helper — it does NOT modify the Milestone 1 cleaning
pipeline. Raw files land in data/raw/brfss_trend/ (gitignored). Because the
raw .XPT files are ~1.1 GB each, every year is downloaded, processed, and then
the large .XPT is deleted, keeping disk use bounded (only the ~85 MB zips are
kept so re-runs skip the download).

Variable notes (names changed across BRFSS waves — handled automatically):
  - Cost barrier : MEDCOST (<=2022) / MEDCOST1 (2023)   1=Yes, 2=No, 7/9=missing
  - Diabetes     : DIABETE3 (<=2020) / DIABETE4 (2021+) 1=Yes (diagnosed)
  - Age band     : _AGEG5YR  categories 6-9 == ages 45-64
  - State (FIPS) : _STATE
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src.config import DATA_RAW, OUTPUT_TABLES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TREND_RAW = DATA_RAW / "brfss_trend"
TREND_RAW.mkdir(parents=True, exist_ok=True)

YEARS = [2019, 2020, 2021, 2022, 2023]

URL_TEMPLATE = "https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"

# Six New England states by FIPS code (matches src/brfss_cleaning.py).
NEW_ENGLAND_FIPS = {9, 23, 25, 33, 44, 50}

# _AGEG5YR categories 6,7,8,9 -> 45-49, 50-54, 55-59, 60-64 (i.e. ages 45-64).
AGE_45_64 = {6, 7, 8, 9}


# ---------------------------------------------------------------------------
# Download + extract
# ---------------------------------------------------------------------------

def _zip_path(year: int) -> Path:
    return TREND_RAW / f"LLCP{year}.zip"


def _xpt_path(year: int) -> Path:
    return TREND_RAW / f"LLCP{year}.XPT"


def _remove_trailing_space_files(year: int) -> None:
    """BRFSS zips store the member as 'LLCP{year}.XPT ' (trailing space).
    On Windows that name needs the \\\\?\\ raw-path prefix to delete. Clean up
    any such stray file left by a previous extraction attempt.
    """
    # TREND_RAW is already absolute; build the path manually because
    # Path.resolve()/abspath() silently strip the trailing space. The \\?\
    # prefix tells Windows not to normalize the name away.
    raw = "\\\\?\\" + str(TREND_RAW) + f"\\LLCP{year}.XPT "  # note trailing space
    try:
        if os.path.exists(raw):
            os.remove(raw)
            print(f"  cleaned stray '{year}.XPT ' (trailing space)")
    except OSError:
        pass


def download_year(year: int) -> Path:
    """Download the BRFSS zip for `year` (skip if already present)."""
    zpath = _zip_path(year)
    if zpath.exists() and zpath.stat().st_size > 1_000_000:
        return zpath
    url = URL_TEMPLATE.format(y=year)
    print(f"  downloading {year} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, zpath)
    print(f"{zpath.stat().st_size/1e6:.0f} MB")
    return zpath


def extract_year(year: int) -> Path:
    """Extract the .XPT to a clean filename using zipfile (handles the
    trailing-space member name and is cross-platform).
    """
    xpath = _xpt_path(year)
    if xpath.exists() and xpath.stat().st_size > 1_000_000:
        return xpath
    zpath = download_year(year)
    with zipfile.ZipFile(zpath) as z:
        member = z.namelist()[0]            # e.g. 'LLCP2022.XPT ' (with space)
        with z.open(member) as src, open(xpath, "wb") as dst:
            shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
    _remove_trailing_space_files(year)
    return xpath


# ---------------------------------------------------------------------------
# Per-year cleaning + rate computation
# ---------------------------------------------------------------------------

def _resolve_cols(columns) -> dict:
    """Pick the correct variable names for this wave."""
    cols = set(columns)
    medcost = "MEDCOST1" if "MEDCOST1" in cols else "MEDCOST"
    diab = "DIABETE4" if "DIABETE4" in cols else "DIABETE3"
    return {"medcost": medcost, "diabetes": diab,
            "age": "_AGEG5YR", "state": "_STATE"}


def compute_year_rates(year: int, chunksize: int = 100_000) -> list[dict]:
    """Stream the .XPT in chunks, keep diabetic adults aged 45-64, and return
    the unweighted missed-care-due-to-cost rate for the national sample and
    for the six New England states.
    """
    xpath = extract_year(year)

    # Peek at the header to resolve wave-specific column names.
    head = next(pd.read_sas(str(xpath), format="xport", chunksize=5))
    names = _resolve_cols(head.columns)
    keep = list(names.values())

    n_nat = k_nat = n_ne = k_ne = 0
    reader = pd.read_sas(str(xpath), format="xport", chunksize=chunksize)
    for chunk in reader:
        c = chunk[keep].copy()
        # Diagnosed diabetes (==1) and ages 45-64.
        c = c[(c[names["diabetes"]] == 1) & (c[names["age"]].isin(AGE_45_64))]
        # Missed care due to cost: 1=Yes -> 1, 2=No -> 0, else missing.
        mc = c[names["medcost"]].map({1: 1, 2: 0})
        valid = mc.notna()
        c, mc = c[valid], mc[valid]

        n_nat += len(c)
        k_nat += int(mc.sum())

        ne_mask = c[names["state"]].isin(NEW_ENGLAND_FIPS)
        n_ne += int(ne_mask.sum())
        k_ne += int(mc[ne_mask].sum())

    # Free the large file once we have the counts.
    try:
        xpath.unlink()
    except OSError:
        pass

    return [
        {"year": year, "region": "United States",
         "n": n_nat, "events": k_nat,
         "missed_care_rate_pct": round(100 * k_nat / n_nat, 2) if n_nat else None},
        {"year": year, "region": "New England",
         "n": n_ne, "events": k_ne,
         "missed_care_rate_pct": round(100 * k_ne / n_ne, 2) if n_ne else None},
    ]


def build_trend(years: list[int] | None = None, save: bool = True) -> pd.DataFrame:
    """Build the full multi-year trend table and (optionally) save it."""
    years = years or YEARS
    rows = []
    for y in years:
        print(f"Processing BRFSS {y}...")
        rows.extend(compute_year_rates(y))
    trend = pd.DataFrame(rows).sort_values(["region", "year"], ignore_index=True)
    if save:
        out = OUTPUT_TABLES / "m2_missed_care_trend.csv"
        trend.to_csv(out, index=False)
        print(f"\nSaved trend table -> {out}")
    return trend


if __name__ == "__main__":
    print(build_trend().to_string(index=False))
