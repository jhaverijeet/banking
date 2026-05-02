"""
Alternative Fetcher for SONIA using the FRED API (St. Louis Fed).
Note: FRED only provides the overnight SONIA rate (IUDSOIA), not the full OIS yield curve.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=IUDSOIA"

def fetch_fred_sonia() -> pd.DataFrame:
    """
    Fetches the entire history of the overnight SONIA rate from FRED.
    Returns a DataFrame with columns: ['date', 'rate']
    """
    logger.info("Fetching overnight SONIA rate from FRED: %s", FRED_CSV_URL)
    
    try:
        df = pd.read_csv(FRED_CSV_URL, na_values=['.'])
        df = df.rename(columns={"observation_date": "date", "IUDSOIA": "rate"})
        df['date'] = pd.to_datetime(df['date'])
        
        # Drop any rows where rate is missing
        df = df.dropna(subset=['rate'])
        
        logger.info("Fetched %d rows from FRED.", len(df))
        return df
        
    except Exception as e:
        logger.error("Failed to fetch data from FRED: %e", e)
        return pd.DataFrame()

def fetch_fred_latest(latest_date_in_db: str | None) -> pd.DataFrame:
    """
    Fetches the overnight SONIA rate from FRED and filters for new dates.
    """
    df = fetch_fred_sonia()
    if df.empty or latest_date_in_db is None:
        return df
        
    # Filter for dates > latest_date_in_db
    new_data = df[df['date'] > pd.to_datetime(latest_date_in_db)].copy()
    logger.info("FRED fetch found %d new rows since %s.", len(new_data), latest_date_in_db)
    return new_data
