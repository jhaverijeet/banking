"""
SONIA Rate Dashboard – Flask backend.

Serves the SQLite data as JSON and hosts the static frontend.

Run:
    py sonia_dashboard/app.py
"""

import sqlite3
import os
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory

# ─── Config ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "sonia_rates.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = Flask(__name__, static_folder=str(STATIC_DIR))


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.route("/api/rates", methods=["GET"])
def api_rates():
    """
    GET /api/rates?start=YYYY-MM-DD&end=YYYY-MM-DD&tenors=1,2,3
    Returns JSON array of rate objects.
    """
    start = request.args.get("start")
    end = request.args.get("end")
    tenors_param = request.args.get("tenors", "1,2,3,4,5,6,7")

    # Parse requested tenors
    try:
        tenors = [int(t.strip()) for t in tenors_param.split(",") if t.strip()]
    except ValueError:
        tenors = [1, 2, 3, 4, 5, 6, 7]

    tenor_cols = [f"tenor_{t}y" for t in tenors]
    select_cols = ", ".join(["date"] + tenor_cols)

    sql = f"SELECT {select_cols} FROM sonia_rates"
    params = []
    conditions = []

    if start:
        conditions.append("date >= ?")
        params.append(start)
    if end:
        conditions.append("date <= ?")
        params.append(end)

    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY date"

    conn = get_db()
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    conn.close()

    data = []
    for row in rows:
        obj = {"date": row["date"]}
        for t in tenors:
            col = f"tenor_{t}y"
            obj[col] = row[col]
        data.append(obj)

    return jsonify(data)


@app.route("/api/summary", methods=["GET"])
def api_summary():
    """
    GET /api/summary
    Returns overall database stats and latest rates.
    """
    conn = get_db()
    cur = conn.execute(
        "SELECT COUNT(*) as cnt, MIN(date) as min_date, MAX(date) as max_date "
        "FROM sonia_rates"
    )
    row = cur.fetchone()

    # Latest row
    cur2 = conn.execute(
        "SELECT * FROM sonia_rates ORDER BY date DESC LIMIT 1"
    )
    latest = cur2.fetchone()
    conn.close()

    summary = {
        "total_rows": row["cnt"],
        "earliest_date": row["min_date"],
        "latest_date": row["max_date"],
    }

    if latest:
        summary["latest_rates"] = {
            f"tenor_{t}y": latest[f"tenor_{t}y"]
            for t in range(1, 8)
        }
        summary["latest_rates"]["date"] = latest["date"]

    return jsonify(summary)


@app.route("/api/date-range", methods=["GET"])
def api_date_range():
    """Return min and max dates available."""
    conn = get_db()
    cur = conn.execute(
        "SELECT MIN(date) as min_date, MAX(date) as max_date FROM sonia_rates"
    )
    row = cur.fetchone()
    conn.close()
    return jsonify({
        "min_date": row["min_date"],
        "max_date": row["max_date"],
    })


# ─── Frontend ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(STATIC_DIR), filename)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  SONIA Rate Dashboard")
    print(f"  DB: {DB_PATH}")
    print(f"  Open: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=True)
