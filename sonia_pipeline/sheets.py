import json
import logging
import subprocess
from pathlib import Path
import pandas as pd

from .config import BASE_DIR, TENORS

logger = logging.getLogger(__name__)

GWS_PATH = r"c:\Users\jeetj\cli\gws_bin\gws.exe"
SHEET_ID_FILE = BASE_DIR / ".sonia_sheet_id"

def get_or_create_sheet() -> str:
    """Reads the spreadsheet ID from file, or creates a new one using gws."""
    if SHEET_ID_FILE.exists():
        return SHEET_ID_FILE.read_text().strip()

    logger.info("No Google Sheet ID found. Creating a new spreadsheet...")
    
    # Create the sheet
    create_cmd = [
        GWS_PATH, "sheets", "spreadsheets", "create",
        "--json", '{"properties": {"title": "SONIA Spot Curves"}}'
    ]
    
    result = subprocess.run(create_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to create Google Sheet via gws. Make sure gws is authenticated.\n{result.stderr}")
        raise RuntimeError("Google Workspace CLI (gws) error")

    try:
        data = json.loads(result.stdout)
        sheet_id = data["spreadsheetId"]
        SHEET_ID_FILE.write_text(sheet_id)
        logger.info(f"Created new Google Sheet with ID: {sheet_id}")
        
        # Write headers
        headers = ["Date"] + [f"{t}Y" for t in TENORS]
        _append_values(sheet_id, [headers])
        return sheet_id
        
    except Exception as e:
        logger.error("Failed to parse gws output.")
        raise e

def _append_values(sheet_id: str, values: list) -> None:
    """Appends rows to the Sheet."""
    if not values:
        return
        
    # Format according to ValueRange schema
    payload = {
        "values": values
    }
    
    params = json.dumps({
        "spreadsheetId": sheet_id,
        "range": "Sheet1!A1",
        "valueInputOption": "USER_ENTERED"
    })
    
    cmd = [
        GWS_PATH, "sheets", "spreadsheets", "values", "append",
        "--params", params,
        "--json", json.dumps(payload)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to append to Google Sheet via gws.\n{result.stderr}")
        raise RuntimeError("Google Workspace CLI (gws) error")
        
    logger.info(f"Successfully appended {len(values)} rows to Google Sheet.")


def push_to_sheets(df: pd.DataFrame) -> None:
    """Entry point to push a Pandas DataFrame of rates to Google Sheets."""
    if df is None or df.empty:
        logger.info("No data to push to Google Sheets.")
        return
        
    try:
        sheet_id = get_or_create_sheet()
        
        df_clean = df.copy()
        
        # Convert dates to string so JSON can serialize them
        if 'date' in df_clean.columns:
            df_clean['date'] = df_clean['date'].astype(str)

        for t in TENORS:
            col = f"tenor_{t}y"
            if col in df_clean.columns:
                # Replace NaNs with an empty string, otherwise round
                df_clean[col] = df_clean[col].apply(lambda x: "" if pd.isna(x) else round(x, 4))
                
        # Convert df to list of lists structure: [ [date, 1y, 2y, ...], [date, 1y, ... ] ]
        # Ensure we only export tenors we care about
        cols = ["date"] + [f"tenor_{t}y" for t in TENORS]
        
        # Only keep columns that actually exist in the dataframe to prevent KeyError
        existing_cols = [c for c in cols if c in df_clean.columns]
        
        values = df_clean[existing_cols].values.tolist()
        _append_values(sheet_id, values)
        
    except Exception as e:
        logger.error(f"Failed pushing to Google Sheets: {e}")
