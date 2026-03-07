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

from .config import DB_PATH, HISTORY_YEARS, LOG_DIR
from .database import get_latest_date, get_row_count, init_db, query_rates, upsert_rates
from .fetcher import fetch_historical, fetch_latest


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    log_file = LOG_DIR / f"sonia_{datetime.now():%Y%m%d_%H%M%S}.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), encoding="utf-8"),
    ]
    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the data pipeline."""
    init_db()

    if args.mode == "historical":
        logging.info("=" * 60)
        logging.info("HISTORICAL LOAD - last %d years", HISTORY_YEARS)
        logging.info("=" * 60)
        df = fetch_historical()
        n = upsert_rates(df)
        logging.info("Historical load complete – %d rows written.", n)

    elif args.mode == "daily":
        logging.info("=" * 60)
        logging.info("DAILY INCREMENTAL UPDATE")
        logging.info("=" * 60)
        latest = get_latest_date()
        logging.info("Latest date in DB: %s", latest or "(empty)")
        df = fetch_latest(latest)
        if df.empty:
            logging.info("No new data available today.")
        else:
            n = upsert_rates(df)
            logging.info("Daily update complete – %d new rows.", n)

    else:
        logging.error("Unknown mode: %s", args.mode)
        sys.exit(1)


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
