"""
SONIA Rate Dashboard — Streamlit

Run:
    py -m streamlit run sonia_dashboard/streamlit_app.py
"""

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────────────
DB_PATH = Path(__file__).resolve().parent.parent / "sonia_rates.db"

TENORS = [1, 2, 3, 4, 5, 6, 7]
TENOR_LABELS = {t: f"{t}Y" for t in TENORS}

# Purple → orange gradient matching the original dashboard
TENOR_COLORS = {
    1: "#6366f1",
    2: "#8b5cf6",
    3: "#a855f7",
    4: "#d946ef",
    5: "#ec4899",
    6: "#f43f5e",
    7: "#f97316",
}

# ─── Page setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SONIA Rate Dashboard",
    page_icon="📈",
    layout="wide",
)

# Inject custom CSS for a dark premium look
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #f1f5f9;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 16px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.9);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

/* Headers */
h1, h2, h3 { color: #f1f5f9 !important; }

/* Divider */
hr { border-color: rgba(99, 102, 241, 0.2); }

/* Plotly chart background */
.js-plotly-plot { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Data loading ────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)  # refresh cache every 60 s
def load_data(start: str | None = None, end: str | None = None) -> pd.DataFrame:
    conn = sqlite3.connect(str(DB_PATH))
    sql = "SELECT * FROM sonia_rates"
    params: list = []
    conds: list[str] = []
    if start:
        conds.append("date >= ?")
        params.append(start)
    if end:
        conds.append("date <= ?")
        params.append(end)
    if conds:
        sql += " WHERE " + " AND ".join(conds)
    sql += " ORDER BY date;"
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(ttl=60)
def load_date_range() -> tuple[str, str]:
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.execute("SELECT MIN(date), MAX(date) FROM sonia_rates;")
    row = cur.fetchone()
    conn.close()
    return row[0], row[1]


# ─── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    db_min, db_max = load_date_range()
    db_min_d = date.fromisoformat(db_min)
    db_max_d = date.fromisoformat(db_max)

    # Quick range presets
    st.markdown("**Quick Select**")
    preset = st.radio(
        "preset",
        ["1M", "3M", "6M", "1Y", "2Y", "5Y", "ALL"],
        index=5,
        horizontal=True,
        label_visibility="collapsed",
    )

    def start_for_preset(p: str) -> date:
        n = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60}.get(p)
        if n is None:
            return db_min_d
        # subtract n months from db_max_d
        m = db_max_d.month - n
        y = db_max_d.year + m // 12
        m = m % 12 or 12
        if m == 0:
            m = 12
            y -= 1
        try:
            return date(y, m, db_max_d.day)
        except ValueError:
            # handle month-end edge cases
            import calendar
            last = calendar.monthrange(y, m)[1]
            return date(y, m, last)

    default_start = max(start_for_preset(preset), db_min_d)

    st.markdown("**Date Range**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=default_start, min_value=db_min_d, max_value=db_max_d, label_visibility="collapsed")
    with col_e:
        end_date = st.date_input("To", value=db_max_d, min_value=db_min_d, max_value=db_max_d, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**Swap Curves**")
    selected_tenors = []
    for t in TENORS:
        if st.checkbox(f"{t}Y", value=True, key=f"tenor_{t}"):
            selected_tenors.append(t)
    if not selected_tenors:
        selected_tenors = [1]   # always keep at least one

    st.markdown("---")
    show_points = st.toggle("Show data points", value=False)

    st.markdown("---")
    st.caption(f"DB path: `{DB_PATH.name}`")
    st.caption(f"Data: {db_min} → {db_max}")


# ─── Load filtered data ───────────────────────────────────────────────────────
df = load_data(start=str(start_date), end=str(end_date))

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:4px;">
  <svg viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" width="36" height="36">
    <path d="M4 24L8 12L14 20L20 8L28 18" stroke="url(#g)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    <defs><linearGradient id="g" x1="4" y1="24" x2="28" y2="8"><stop stop-color="#6366f1"/><stop offset="1" stop-color="#f97316"/></linearGradient></defs>
  </svg>
  <div>
    <h1 style="margin:0; font-size:1.7rem; background:linear-gradient(90deg,#6366f1,#f97316); -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
      SONIA Rate Dashboard
    </h1>
    <p style="margin:0; color:#64748b; font-size:0.85rem;">Bank of England · OIS Spot Curve</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Summary metrics ─────────────────────────────────────────────────────────
if not df.empty:
    latest = df.iloc[-1]
    first = df.iloc[0]

    # Top stat bar
    total_rows = len(df)
    latest_date_str = latest["date"].strftime("%d %b %Y")

    st.markdown(f"""
    <div style="display:flex; gap:16px; margin:16px 0; flex-wrap:wrap;">
      <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25);
                  border-radius:10px; padding:10px 20px;">
        <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px;">Records</div>
        <div style="color:#f1f5f9; font-size:1.25rem; font-weight:700;">{total_rows:,}</div>
      </div>
      <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.25);
                  border-radius:10px; padding:10px 20px;">
        <div style="color:#94a3b8; font-size:0.72rem; text-transform:uppercase; letter-spacing:1px;">Latest</div>
        <div style="color:#06b6d4; font-size:1.25rem; font-weight:700;">{latest_date_str}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Rate cards
    n_cols = len(selected_tenors)
    rate_cols = st.columns(n_cols)
    for i, t in enumerate(selected_tenors):
        col_name = f"tenor_{t}y"
        current = latest[col_name]
        prev = first[col_name]
        delta_bp = (current - prev) * 100 if pd.notna(current) and pd.notna(prev) else None
        color = TENOR_COLORS[t]
        arrow = "▲" if delta_bp and delta_bp >= 0 else "▼"
        delta_color = "#22c55e" if delta_bp and delta_bp >= 0 else "#f43f5e"
        with rate_cols[i]:
            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6); border:1px solid {color}40;
                        border-top: 2px solid {color}; border-radius:12px; padding:14px 12px; text-align:center;">
              <div style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
                {t}Y Swap
              </div>
              <div style="color:#f1f5f9; font-size:1.6rem; font-weight:700; line-height:1;">
                {current:.2f}<span style="font-size:0.9rem; color:#94a3b8;">%</span>
              </div>
              {"" if delta_bp is None else f'<div style="color:{delta_color}; font-size:0.75rem; margin-top:4px;">{arrow} {abs(delta_bp):.1f}bp</div>'}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Main Chart: Historical Lines ────────────────────────────────────────
    st.markdown("### 📈 SONIA OIS Spot Rate Curves")

    fig_main = go.Figure()
    for t in selected_tenors:
        col = f"tenor_{t}y"
        color = TENOR_COLORS[t]
        fig_main.add_trace(go.Scatter(
            x=df["date"],
            y=df[col],
            name=f"{t} Year",
            line=dict(color=color, width=1.8),
            mode="lines+markers" if show_points else "lines",
            marker=dict(size=3, color=color) if show_points else None,
            hovertemplate=f"<b>{t}Y</b>: %{{y:.4f}}%<br>%{{x|%d %b %Y}}<extra></extra>",
        ))

    fig_main.update_layout(
        plot_bgcolor="rgba(15,23,42,0.0)",
        paper_bgcolor="rgba(15,23,42,0.0)",
        font=dict(family="Inter", color="#94a3b8", size=12),
        xaxis=dict(
            showgrid=True, gridcolor="rgba(99,102,241,0.08)",
            tickcolor="#475569", linecolor="#334155",
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(99,102,241,0.08)",
            tickcolor="#475569", linecolor="#334155",
            tickfont=dict(color="#94a3b8"),
            ticksuffix="%",
            title=dict(text="Rate (%)", font=dict(color="#64748b", size=11)),
        ),
        legend=dict(
            bgcolor="rgba(15,23,42,0.6)", bordercolor="rgba(99,102,241,0.2)",
            borderwidth=1, font=dict(color="#e2e8f0"),
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        ),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    st.plotly_chart(fig_main, use_container_width=True)

    st.caption("Source: Bank of England OIS Yield Curve Archive · Annually compounded zero-coupon spot rates (%)")

    st.markdown("---")

    # ─── Bottom section: Term Structure + Stats ───────────────────────────────
    col_term, col_stats = st.columns([1, 1], gap="large")

    with col_term:
        st.markdown(f"### 📊 Current Term Structure")
        st.caption(f"As of {latest_date_str}")

        labels = [f"{t}Y" for t in selected_tenors]
        values = [latest[f"tenor_{t}y"] for t in selected_tenors]
        colors = [TENOR_COLORS[t] for t in selected_tenors]

        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(color="#a855f7", width=2.5),
            marker=dict(color=colors, size=10, line=dict(color="#0f172a", width=2)),
            fill="tozeroy",
            fillcolor="rgba(168,85,247,0.08)",
            hovertemplate="<b>%{x}</b>: %{y:.4f}%<extra></extra>",
        ))
        fig_term.update_layout(
            plot_bgcolor="rgba(15,23,42,0.0)",
            paper_bgcolor="rgba(15,23,42,0.0)",
            font=dict(family="Inter", color="#94a3b8", size=12),
            xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", size=12, family="JetBrains Mono, monospace")),
            yaxis=dict(
                showgrid=True, gridcolor="rgba(99,102,241,0.08)",
                ticksuffix="%", tickfont=dict(color="#94a3b8"),
            ),
            margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_term, use_container_width=True)

    with col_stats:
        st.markdown("### 📋 Rate Statistics")
        st.caption("For selected period")

        rows = []
        for t in selected_tenors:
            col = f"tenor_{t}y"
            series = df[col].dropna()
            if series.empty:
                continue
            current_val = latest[col]
            first_val   = first[col]
            change_bp   = (current_val - first_val) * 100 if pd.notna(current_val) and pd.notna(first_val) else None
            rows.append({
                "Tenor":   f"{t}Y",
                "Current": f"{current_val:.2f}%",
                "Min":     f"{series.min():.2f}%",
                "Max":     f"{series.max():.2f}%",
                "Avg":     f"{series.mean():.2f}%",
                "Change":  f"{'+' if change_bp and change_bp >= 0 else ''}{change_bp:.0f}bp" if change_bp is not None else "—",
            })

        stats_df = pd.DataFrame(rows)
        st.dataframe(
            stats_df,
            use_container_width=True,
            hide_index=True,
            height=280,
        )

else:
    st.warning("No data found in the database for the selected range. Run the pipeline first.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("SONIA Rate Dashboard · Data sourced from Bank of England · Built with Streamlit & Plotly")
