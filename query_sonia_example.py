import sqlite3
import pandas as pd
from pathlib import Path

# Path to the database file you created
DB_PATH = Path(r"c:\Users\jeetj\banking\sonia_rates.db")

def query_example_1_pandas():
    """Example 1: The easiest way - query directly into a pandas DataFrame."""
    print("--- Example 1: Loading directly into Pandas ---")
    
    # 1. Create a connection
    conn = sqlite3.connect(DB_PATH)
    
    # 2. Write your SQL query. 
    # Let's get the 1Y and 5Y rates for the first week of 2024
    query = """
        SELECT date, tenor_1y, tenor_5y 
        FROM sonia_rates 
        WHERE date >= '2024-01-01' AND date <= '2024-01-07'
        ORDER BY date ASC
    """
    
    # 3. Execute and load into DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Clean up connection
    conn.close()
    
    print(df)
    print("\n")


def query_example_2_native_sqlite():
    """Example 2: Using native sqlite3 to iterate through rows line-by-line."""
    print("--- Example 2: Iterating with Native sqlite3 ---")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Row dictionary factory makes it easy to access columns by name
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    
    # Let's find the highest ever 1Y swap rate in the database
    query = """
        SELECT date, tenor_1y 
        FROM sonia_rates 
        WHERE tenor_1y IS NOT NULL
        ORDER BY tenor_1y DESC 
        LIMIT 5
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    print("Top 5 highest 1Y SONIA rates:")
    for row in rows:
        print(f"Date: {row['date']} -> Rate: {row['tenor_1y']}%")
        
    conn.close()
    print("\n")


def query_example_3_average_by_year():
    """Example 3: Doing aggregations in SQL instead of Python."""
    print("--- Example 3: SQL Aggregation ---")
    
    conn = sqlite3.connect(DB_PATH)
    
    # Calculate the average 5Y rate grouped by year
    query = """
        SELECT 
            strftime('%Y', date) as year,
            COUNT(*) as trading_days,
            ROUND(AVG(tenor_5y), 3) as avg_5y_rate
        FROM sonia_rates
        GROUP BY year
        ORDER BY year DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(df)


if __name__ == "__main__":
    if not DB_PATH.exists():
        print(f"Error: Could not find database at {DB_PATH}")
        print("Make sure the pipeline has been run first!")
    else:
        query_example_1_pandas()
        query_example_2_native_sqlite()
        query_example_3_average_by_year()
