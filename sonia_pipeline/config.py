"""
Configuration for the SONIA rate data pipeline.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "sonia_rates.db"
LOG_DIR = BASE_DIR / "sonia_pipeline" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Bank of England Data Source ─────────────────────────────────────────────
BOE_OIS_ARCHIVE_URL = (
    "https://www.bankofengland.co.uk/-/media/boe/files/"
    "statistics/yield-curves/oisddata.zip"
)

# The ZIP contains XLSX files with sheets structured as:
#   Sheet "4. spot curve"
#     Row 4: Maturity headers (0.5, 1, 1.5, 2 ... 25 years), columns 2-51
#     Row 5+: Date in col 1, spot rates in cols 2-51
ARCHIVE_FILES = [
    "OIS daily data_2016 to 2024.xlsx",
    "OIS daily data_2025 to present.xlsx",
]
SPOT_CURVE_SHEET = "4. spot curve"
HEADER_ROW = 4          # 1-indexed row containing maturity labels
DATA_START_ROW = 5      # 1-indexed row where date + rates begin

# ─── Tenors of Interest (years) ─────────────────────────────────────────────
TENORS = [1, 2, 3, 4, 5, 6, 7]

# Mapping from tenor (years) to column index (1-indexed) in the XLSX:
#   0.5 → col 2, 1 → col 3, 1.5 → col 4, 2 → col 5, ...
# Formula: col = 2 + (tenor_years - 0.5) / 0.5 = 2 + 2 * tenor_years - 1
#   i.e. col = 1 + 2 * tenor_years
TENOR_TO_COL = {t: 1 + 2 * t for t in TENORS}
# Verification: 1yr→col3, 2yr→col5, 3yr→col7, 4yr→col9, 5yr→col11, 6yr→col13, 7yr→col15

# ─── Historical Window ───────────────────────────────────────────────────────
HISTORY_YEARS = 5       # Fetch the last N years of data on initial load

# ─── HTTP ────────────────────────────────────────────────────────────────────
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT = 120   # seconds
