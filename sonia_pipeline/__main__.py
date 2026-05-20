"""
SONIA Rate Pipeline – CLI entry-point.

Usage
-----
  # Initial historical load (last 5 years):
  py -m sonia_pipeline run --mode historical

  # Daily incremental update (only new dates):
  py -m sonia_pipeline run --mode daily

  # Query the database:
  py -m sonia_pipeline query --start 2024-01-01 --end 2024-12-31

  # Show database summary:
  py -m sonia_pipeline status
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Optional
import pandas as pd
import dlt
from prefect import task, flow, get_run_logger

from .config import DB_PATH, HISTORY_YEARS, LOG_DIR
from .database import get_latest_date, get_row_count, init_db, query_rates, get_latest_fred_date
from .fetcher import fetch_historical, fetch_latest
from .fred_fetcher import fetch_fred_sonia, fetch_fred_latest


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    log_file = LOG_DIR / f"sonia_{datetime.now():%Y%m%d_%H%M%S}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


@task(name="Initialize Database")
def initialize_database() -> None:
    logger = get_run_logger()
    logger.info("Initializing SQLite database tables...")
    init_db()


@task(name="Fetch Rates Data", retries=3, retry_delay_seconds=30)
def fetch_data(mode: str, source: str, latest_date: Optional[str]) -> pd.DataFrame:
    logger = get_run_logger()
    if mode == "historical":
        logger.info(f"Fetching historical data from {source.upper()}...")
        if source == "fred":
            return fetch_fred_sonia()
        else:
            return fetch_historical()
    else:
        logger.info(f"Fetching latest data from {source.upper()} since {latest_date or 'beginning'}...")
        if source == "fred":
            return fetch_fred_latest(latest_date)
        else:
            return fetch_latest(latest_date)


@task(name="Load Data via dlt (Merge)")
def load_data_with_dlt(df: pd.DataFrame, table_name: str) -> int:
    logger = get_run_logger()
    if df.empty:
        logger.info("DataFrame is empty. Nothing to load.")
        return 0

    @dlt.resource(
        name=table_name,
        primary_key="date",
        write_disposition="merge"
    )
    def load_resource():
        df_copy = df.copy()
        df_copy["fetched_at"] = datetime.utcnow().isoformat()
        if "date" in df_copy.columns:
            df_copy["date"] = pd.to_datetime(df_copy["date"]).dt.strftime("%Y-%m-%d")
        yield df_copy.to_dict(orient="records")

    logger.info(f"Running DLT pipeline to load data into '{table_name}' table...")
    pipeline = dlt.pipeline(
        pipeline_name=f"sonia_{table_name}_pipeline",
        destination=dlt.destinations.sqlalchemy(f"sqlite:///{DB_PATH.absolute()}"),
        dataset_name="main"
    )
    load_info = pipeline.run(load_resource())
    logger.info(f"DLT pipeline loaded successfully. Info:\n{load_info}")
    return len(df)


@task(name="Push to Google Sheets", retries=2, retry_delay_seconds=15)
def push_to_sheets_task(df: pd.DataFrame) -> None:
    logger = get_run_logger()
    logger.info("Pushing data to Google Sheets...")
    try:
        from .sheets import push_to_sheets
        push_to_sheets(df)
    except ImportError:
        logger.warning("Google Sheets module not found.")
    except Exception as e:
        logger.error(f"Error pushing to Sheets: {e}")


@flow(name="SONIA Rate Pipeline Flow")
def run_sonia_pipeline_flow(mode: str, source: str) -> None:
    logger = get_run_logger()
    logger.info(f"Starting SONIA Rate Pipeline Flow (Mode: {mode}, Source: {source})")
    
    # 1. Init Database
    initialize_database()
    
    # 2. Get high-water mark for incremental run
    latest_date = None
    if mode == "daily":
        if source == "fred":
            latest_date = get_latest_fred_date()
        else:
            latest_date = get_latest_date()
            
    # 3. Fetch rates data
    df = fetch_data(mode, source, latest_date)
    
    # 4. Load rates data using dlt
    table_name = "sonia_overnight_fred" if source == "fred" else "sonia_rates"
    rows_written = load_data_with_dlt(df, table_name)
    
    # 5. Push to Sheets if daily BOE data loaded successfully
    if rows_written > 0 and mode == "daily" and source == "boe":
        push_to_sheets_task(df)
        
    logger.info("SONIA Rate Pipeline Flow completed successfully.")


def cmd_run(args: argparse.Namespace) -> None:
    """Run the data pipeline flow."""
    run_sonia_pipeline_flow(mode=args.mode, source=getattr(args, "source", "boe"))


def cmd_query(args: argparse.Namespace) -> None:
    """Query the database and print results."""
    init_db()
    df = query_rates(start_date=args.start, end_date=args.end)
    if df.empty:
        print("No data found for the specified range.")
    else:
        print(f"\n{'-' * 90}")
        print(f"  SONIA OIS Spot Rates  ({len(df)} rows)")
        print(f"{'-' * 90}")
        # Format for display
        display = df.copy()
        for col in display.columns:
            if col.startswith("tenor_"):
                display[col] = display[col].apply(
                    lambda x: f"{x:.4f}" if x is not None and not (isinstance(x, float) and x != x) else ""
                )
        print(display.to_string(index=False))
        print()


def cmd_status(args: argparse.Namespace) -> None:
    """Show database summary."""
    init_db()
    count = get_row_count()
    latest = get_latest_date()
    print(f"\n{'-' * 50}")
    print(f"  SONIA Rate Database Status")
    print(f"{'-' * 50}")
    print(f"  Database path : {DB_PATH}")
    print(f"  Total rows    : {count:,}")
    print(f"  Latest date   : {latest or '(none)'}")
    print(f"{'-' * 50}\n")

    if count > 0:
        # Show last 5 entries
        df = query_rates()
        if not df.empty:
            tail = df.tail(5)
            print("  Last 5 entries:")
            for _, row in tail.iterrows():
                rates = "  ".join(
                    f"{t}y={row[f'tenor_{t}y']:.2f}%"
                    for t in [1, 2, 3, 4, 5, 6, 7]
                    if row.get(f"tenor_{t}y") is not None
                    and not (isinstance(row.get(f"tenor_{t}y"), float)
                             and row.get(f"tenor_{t}y") != row.get(f"tenor_{t}y"))
                )
                print(f"    {row['date']}  {rates}")
            print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sonia_pipeline",
        description="SONIA OIS rate data pipeline – Bank of England",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p_run = sub.add_parser("run", help="Run the data pipeline")
    p_run.add_argument(
        "--mode",
        choices=["historical", "daily"],
        default="daily",
        help="'historical' for full 5yr backfill; 'daily' for incremental update",
    )
    p_run.add_argument(
        "--source",
        choices=["boe", "fred"],
        default="boe",
        help="'boe' for OIS spot curve (default); 'fred' for overnight rate only",
    )
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run)

    # ── query ──
    p_query = sub.add_parser("query", help="Query stored rates")
    p_query.add_argument("--start", help="Start date (YYYY-MM-DD)")
    p_query.add_argument("--end", help="End date (YYYY-MM-DD)")
    p_query.add_argument("-v", "--verbose", action="store_true")
    p_query.set_defaults(func=cmd_query)

    # ── status ──
    p_status = sub.add_parser("status", help="Show database status")
    p_status.add_argument("-v", "--verbose", action="store_true")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    _setup_logging(verbose=getattr(args, "verbose", False))
    args.func(args)


if __name__ == "__main__":
    main()
