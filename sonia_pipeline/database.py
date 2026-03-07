"""
Database layer for SONIA rates (SQLite).

Table: sonia_rates
  - date           TEXT  (YYYY-MM-DD, primary key)
  - tenor_1y       REAL  (1-year OIS spot rate, %)
  - tenor_2y       REAL
  - tenor_3y       REAL
  - tenor_4y       REAL
  - tenor_5y       REAL
  - tenor_6y       REAL
  - tenor_7y       REAL
  - fetched_at     TEXT  (ISO timestamp of when the row was inserted/updated)
"""

import sqlite3
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from .config import DB_PATH, TENORS

logger = logging.getLogger(__name__)

TENOR_COLUMNS = [f"tenor_{t}y" for t in TENORS]


def _get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Create the sonia_rates table if it does not exist."""
    cols_ddl = ",\n    ".join(f"{c} REAL" for c in TENOR_COLUMNS)
    ddl = f"""
    CREATE TABLE IF NOT EXISTS sonia_rates (
        date       TEXT PRIMARY KEY,
        {cols_ddl},
        fetched_at TEXT NOT NULL
    );
    """
    with _get_connection() as conn:
        conn.execute(ddl)
        # Index for fast date-range queries
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sonia_date ON sonia_rates(date);"
        )
    logger.info("Database initialised at %s", DB_PATH)


def upsert_rates(df: pd.DataFrame) -> int:
    """
    Insert or update SONIA rates from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: date (datetime/str), tenor_1y … tenor_7y (float).

    Returns
    -------
    int
        Number of rows upserted.
    """
    if df.empty:
        logger.info("Nothing to upsert – DataFrame is empty.")
        return 0

    now = datetime.utcnow().isoformat()
    cols = ["date"] + TENOR_COLUMNS + ["fetched_at"]
    placeholders = ", ".join("?" for _ in cols)
    update_set = ", ".join(f"{c}=excluded.{c}" for c in TENOR_COLUMNS + ["fetched_at"])

    sql = f"""
    INSERT INTO sonia_rates ({', '.join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(date) DO UPDATE SET {update_set};
    """

    records = []
    for _, row in df.iterrows():
        date_val = row["date"]
        # Skip rows with missing dates
        if pd.isna(date_val):
            continue
        date_str = (
            date_val.strftime("%Y-%m-%d")
            if hasattr(date_val, "strftime")
            else str(date_val)
        )
        tenor_vals = []
        for c in TENOR_COLUMNS:
            v = row.get(c)
            # Convert pandas NaN/NaT to Python None for SQLite compatibility
            tenor_vals.append(None if pd.isna(v) else float(v))
        values = [date_str] + tenor_vals + [now]
        records.append(values)

    with _get_connection() as conn:
        conn.executemany(sql, records)

    logger.info("Upserted %d rows into sonia_rates.", len(records))
    return len(records)


def get_latest_date() -> Optional[str]:
    """Return the most recent date string in the DB, or None if empty."""
    with _get_connection() as conn:
        cur = conn.execute("SELECT MAX(date) FROM sonia_rates;")
        result = cur.fetchone()
    return result[0] if result and result[0] else None


def get_row_count() -> int:
    """Return total number of rows in sonia_rates."""
    with _get_connection() as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sonia_rates;")
        return cur.fetchone()[0]


def query_rates(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query SONIA rates, optionally filtered by date range.

    Parameters
    ----------
    start_date : str, optional  (YYYY-MM-DD)
    end_date   : str, optional  (YYYY-MM-DD)

    Returns
    -------
    pd.DataFrame
    """
    sql = "SELECT * FROM sonia_rates"
    params: list = []
    conditions: list[str] = []

    if start_date:
        conditions.append("date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("date <= ?")
        params.append(end_date)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY date;"

    with _get_connection() as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return df
