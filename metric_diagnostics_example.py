"""
Example: Metric Diagnostics on Synthetic Banking Data
======================================================

This script demonstrates the MetricDiagnostics library using synthetic
loan-portfolio data.  We create two populations:

    - Base:    Loan portfolio from "Period 1"
    - Compare: Loan portfolio from "Period 2" with shifted distributions
               and changed default behaviour

We then diagnose *why* the default rate differs between the two periods
and export the results to a PowerPoint presentation.
"""

import sys
import numpy as np
import pandas as pd

# Ensure the parent directory is on the path so the package is importable
sys.path.insert(0, ".")

from metric_diagnostics import MetricDiagnostics


# ---------------------------------------------------------------------------
# 1.  Generate synthetic data
# ---------------------------------------------------------------------------

def make_synthetic_data(n: int, seed: int, period: str) -> pd.DataFrame:
    """Create a synthetic loan portfolio."""
    rng = np.random.default_rng(seed)

    credit_score = rng.normal(680, 50, n).clip(300, 850).astype(int)
    loan_amount = rng.lognormal(mean=10.5, sigma=0.6, size=n).round(0)
    ltv = rng.beta(5, 3, n) * 100       # loan-to-value %
    dti = rng.beta(2, 5, n) * 60        # debt-to-income %
    income = rng.lognormal(mean=11, sigma=0.5, size=n).round(0)
    employment_years = rng.exponential(5, n).clip(0, 40).round(1)
    loan_term = rng.choice([12, 24, 36, 48, 60], n, p=[0.05, 0.10, 0.30, 0.25, 0.30])
    loan_purpose = rng.choice(
        ["home", "auto", "personal", "education"], n,
        p=[0.35, 0.25, 0.25, 0.15],
    )
    region = rng.choice(["North", "South", "East", "West"], n)

    # Default probability driven by features
    log_odds = (
        -4.0
        + 0.02 * (700 - credit_score)
        + 0.3 * (ltv / 100)
        + 0.5 * (dti / 60)
        - 0.01 * (employment_years)
        + 0.000001 * (loan_amount)
        + rng.normal(0, 0.3, n)
    )
    prob_default = 1 / (1 + np.exp(-log_odds))
    default_flag = (rng.uniform(0, 1, n) < prob_default).astype(int)

    return pd.DataFrame({
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "ltv": ltv,
        "dti": dti,
        "income": income,
        "employment_years": employment_years,
        "loan_term": loan_term,
        "loan_purpose": loan_purpose,
        "region": region,
        "default_flag": default_flag,
        "period": period,
    })


# Base population
base_df = make_synthetic_data(5_000, seed=42, period="Period 1")

# Compare population -- shifted distributions to simulate change
compare_df = make_synthetic_data(5_000, seed=99, period="Period 2")

# Introduce realistic shifts in Period 2:
# - Credit scores drop slightly (economy worsens)
compare_df["credit_score"] = (compare_df["credit_score"] - 20).clip(300, 850)
# - LTV increases (housing market stress)
compare_df["ltv"] = (compare_df["ltv"] * 1.10).clip(0, 120)
# - DTI increases
compare_df["dti"] = (compare_df["dti"] * 1.15).clip(0, 60)
# - Re-derive defaults with a *shifted* relationship (rates also change)
rng2 = np.random.default_rng(123)
log_odds_2 = (
    -3.5  # higher intercept -> more defaults
    + 0.025 * (700 - compare_df["credit_score"])
    + 0.35 * (compare_df["ltv"] / 100)
    + 0.55 * (compare_df["dti"] / 60)
    - 0.008 * compare_df["employment_years"]
    + 0.0000012 * compare_df["loan_amount"]
    + rng2.normal(0, 0.3, len(compare_df))
)
prob_default_2 = 1 / (1 + np.exp(-log_odds_2))
compare_df["default_flag"] = (rng2.uniform(0, 1, len(compare_df)) < prob_default_2).astype(int)

# Drop the helper 'period' column before analysis
base_df = base_df.drop(columns=["period"])
compare_df = compare_df.drop(columns=["period"])

print(f"Base default rate:    {base_df['default_flag'].mean():.4f}")
print(f"Compare default rate: {compare_df['default_flag'].mean():.4f}")
print()

# ---------------------------------------------------------------------------
# 2.  Run Diagnostics
# ---------------------------------------------------------------------------

# Using LightGBM (default)
diag = MetricDiagnostics(
    base_df=base_df,
    compare_df=compare_df,
    target_col="default_flag",
    model_type="lightgbm",      # try "linear" or "logistic" too
    n_bins=5,
)
results = diag.run()

# Print text summary to console
diag.summary()

# Per-variable detail for the top driver
top_var = results["variable_summary"].iloc[0]["variable"]
print(f"\n--- Detail for top driver: {top_var} ---")
print(diag.detail(top_var).to_string(index=False))

# ---------------------------------------------------------------------------
# 3.  Export PowerPoint report
# ---------------------------------------------------------------------------

diag.to_pptx("metric_diagnostics_report.pptx", top_n_variables=5)

# ---------------------------------------------------------------------------
# 4.  Export to Excel (optional)
# ---------------------------------------------------------------------------

diag.to_excel("metric_diagnostics_output.xlsx")

print("\n[OK] Example complete.")
