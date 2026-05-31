"""Central path configuration for the ALY 6150 diabetes health project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MEPS_SAS7BDAT = DATA_RAW / "meps_2023" / "h251.sas7bdat"
BRFSS_XPT = DATA_RAW / "brfss_2023" / "LLCP2023.XPT"

OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_FIGURES = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_MODELS = PROJECT_ROOT / "outputs" / "models"
OUTPUT_TABLEAU = PROJECT_ROOT / "outputs" / "tableau"

for folder in [DATA_PROCESSED, OUTPUT_TABLES, OUTPUT_FIGURES, OUTPUT_MODELS, OUTPUT_TABLEAU]:
    folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("MEPS_XPT     :", MEPS_XPT, "exists:", MEPS_XPT.exists())
    print("BRFSS_XPT    :", BRFSS_XPT, "exists:", BRFSS_XPT.exists())
