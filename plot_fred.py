import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Connect to database
db_path = Path("sonia_rates.db")
conn = sqlite3.connect(db_path)

# Query 2026 data
query = "SELECT date, rate FROM sonia_overnight_fred WHERE date LIKE '2026-%' ORDER BY date"
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found for 2026.")
else:
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['rate'], marker='o', linestyle='-', color='indigo', markersize=4)
    plt.title('Overnight SONIA Rate - 2026 (Source: FRED)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Rate (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save image
    output_file = "fred_sonia_2026.png"
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
