"""
Metric Diagnostics
==================

A Python package for diagnosing what drives the difference in a metric
between two populations (dataframes).

Core decomposition:
    Total Difference = Mix Effect (distribution shift) + Rate Effect (behavioral change)

Quick start::

    from metric_diagnostics import MetricDiagnostics

    diag = MetricDiagnostics(
        base_df=df_base,
        compare_df=df_compare,
        target_col='default_rate',
        model_type='lightgbm',   # or 'linear', 'logistic'
    )
    results = diag.run()
    diag.summary()
    diag.to_pptx("my_report.pptx")
    diag.to_excel("my_data.xlsx")
"""

from .diagnostics import MetricDiagnostics
from .report import generate_report

__all__ = ["MetricDiagnostics", "generate_report"]
__version__ = "1.0.0"
