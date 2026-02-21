"""
Core diagnostic engine for comparing a metric across two populations.

Decomposition approach (Oaxaca-Blinder style):
    Total Difference  = Compare_Metric - Base_Metric
    Mix Effect        = Σ (compare_weight[b] - base_weight[b]) × base_rate[b]
    Rate Effect       = Σ compare_weight[b] × (compare_rate[b] - base_rate[b])
    Total             ≈ Mix Effect + Rate Effect

The model-based decomposition provides an overall view:
    Model Distribution Effect = mean(predicted_compare) - mean(actual_base)
    Model Relationship Effect = mean(actual_compare) - mean(predicted_compare)
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _detect_column_types(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_unique_for_categorical: int = 15,
) -> Tuple[List[str], List[str]]:
    """Classify columns as continuous or categorical."""
    continuous, categorical = [], []
    for col in feature_cols:
        if df[col].dtype.kind in ("O", "b") or df[col].nunique() <= max_unique_for_categorical:
            categorical.append(col)
        else:
            continuous.append(col)
    return continuous, categorical


def _create_pentile_bins(
    base_series: pd.Series,
    compare_series: pd.Series,
    n_bins: int = 5,
) -> Tuple[pd.Categorical, pd.Categorical]:
    """Create pentile (or n-tile) bins based on the base distribution.

    If there are fewer than ``n_bins`` unique values the raw values are used
    as categories instead of quantile bins.
    """
    combined = pd.concat([base_series, compare_series], ignore_index=True).dropna()

    if combined.nunique() <= n_bins:
        # Few unique values – treat as categorical
        cats = sorted(combined.unique())
        base_binned = pd.Categorical(base_series, categories=cats, ordered=True)
        compare_binned = pd.Categorical(compare_series, categories=cats, ordered=True)
    else:
        # Quantile-based binning from the *base* distribution
        try:
            bin_edges = np.nanquantile(
                base_series.dropna(), np.linspace(0, 1, n_bins + 1)
            )
            bin_edges = np.unique(bin_edges)  # deduplicate ties
            if len(bin_edges) < 2:
                bin_edges = np.array([combined.min(), combined.max()])
            # Extend edges slightly so all compare values are captured
            bin_edges[0] = min(bin_edges[0], combined.min()) - 1e-8
            bin_edges[-1] = max(bin_edges[-1], combined.max()) + 1e-8

            labels = [
                f"({bin_edges[i]:.4g}, {bin_edges[i+1]:.4g}]"
                for i in range(len(bin_edges) - 1)
            ]
            base_binned = pd.cut(base_series, bins=bin_edges, labels=labels, include_lowest=True)
            compare_binned = pd.cut(compare_series, bins=bin_edges, labels=labels, include_lowest=True)
        except Exception:
            # Fall back: equal-width bins
            base_binned = pd.cut(base_series, bins=n_bins, include_lowest=True)
            compare_binned = pd.cut(compare_series, bins=n_bins, include_lowest=True)

    return base_binned, compare_binned


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MetricDiagnostics:
    """Diagnose what drives a metric difference between two populations.

    Parameters
    ----------
    base_df : pd.DataFrame
        The *base* (reference / control) population.
    compare_df : pd.DataFrame
        The *compare* (test / treatment) population.
    target_col : str
        Column name of the metric to compare.
    model_type : str, default ``'lightgbm'``
        One of ``'linear'``, ``'logistic'``, or ``'lightgbm'``.
    feature_cols : list[str] | None
        Columns to use as features.  If ``None``, all non-target numeric and
        categorical columns are used.
    n_bins : int, default 5
        Number of quantile bins (pentiles by default) for continuous variables.
    max_unique_for_categorical : int, default 15
        Columns with at most this many unique values are treated as categorical.
    model_params : dict | None
        Additional keyword arguments forwarded to the model constructor.
    """

    VALID_MODELS = ("linear", "logistic", "lightgbm")

    def __init__(
        self,
        base_df: pd.DataFrame,
        compare_df: pd.DataFrame,
        target_col: str,
        model_type: str = "lightgbm",
        feature_cols: Optional[List[str]] = None,
        n_bins: int = 5,
        max_unique_for_categorical: int = 15,
        model_params: Optional[Dict] = None,
    ):
        if model_type not in self.VALID_MODELS:
            raise ValueError(
                f"model_type must be one of {self.VALID_MODELS}, got '{model_type}'"
            )
        if model_type == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is not installed. Install it with `pip install lightgbm` "
                "or choose model_type='linear' / 'logistic'."
            )
        if target_col not in base_df.columns or target_col not in compare_df.columns:
            raise ValueError(f"target_col '{target_col}' must be present in both dataframes.")

        self.base_df = base_df.copy()
        self.compare_df = compare_df.copy()
        self.target_col = target_col
        self.model_type = model_type
        self.n_bins = n_bins
        self.max_unique_for_categorical = max_unique_for_categorical
        self.model_params = model_params or {}

        # Resolve feature columns -------------------------------------------------
        if feature_cols is not None:
            self.feature_cols = list(feature_cols)
        else:
            common_cols = sorted(
                set(base_df.columns) & set(compare_df.columns) - {target_col}
            )
            self.feature_cols = [
                c for c in common_cols
                if base_df[c].dtype.kind in ("i", "u", "f", "O", "b")
            ]

        if not self.feature_cols:
            raise ValueError("No feature columns found. Specify feature_cols explicitly.")

        # Detect column types
        self.continuous_cols, self.categorical_cols = _detect_column_types(
            base_df, self.feature_cols, max_unique_for_categorical
        )

        # Placeholders
        self.model = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self._base_pred: Optional[np.ndarray] = None
        self._compare_pred: Optional[np.ndarray] = None
        self._overall_result: Optional[Dict] = None
        self._variable_results: Optional[Dict[str, pd.DataFrame]] = None

    # ------------------------------------------------------------------
    # Model building & scoring
    # ------------------------------------------------------------------

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Encode categoricals and return a numeric feature matrix."""
        out = df[self.feature_cols].copy()

        for col in self.categorical_cols:
            out[col] = out[col].astype(str)
            if fit:
                le = LabelEncoder()
                out[col] = le.fit_transform(out[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels by mapping to -1
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                out[col] = out[col].map(mapping).fillna(-1).astype(int)

        # Fill remaining NaNs with column median
        out = out.fillna(out.median())
        return out

    def _build_model(self):
        """Fit the chosen model on the base population."""
        X_base = self._prepare_features(self.base_df, fit=True)
        y_base = self.base_df[self.target_col].values

        if self.model_type == "linear":
            self.model = LinearRegression(**self.model_params)
            self.model.fit(X_base, y_base)

        elif self.model_type == "logistic":
            params = {"max_iter": 1000, "solver": "lbfgs"}
            params.update(self.model_params)
            self.model = LogisticRegression(**params)
            self.model.fit(X_base, y_base)

        elif self.model_type == "lightgbm":
            params = {
                "objective": "regression",
                "verbosity": -1,
                "n_estimators": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 5,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }
            # If target looks binary, switch to binary classification
            unique_vals = np.unique(y_base[~np.isnan(y_base)])
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                params["objective"] = "binary"
                params["metric"] = "auc"
            params.update(self.model_params)
            self.model = lgb.LGBMRegressor(**params) if params["objective"] == "regression" else lgb.LGBMClassifier(**params)
            self.model.fit(X_base, y_base)

    def _score(self, df: pd.DataFrame) -> np.ndarray:
        """Score a dataframe using the fitted model."""
        X = self._prepare_features(df, fit=False)
        if self.model_type == "logistic":
            return self.model.predict_proba(X)[:, 1]
        elif self.model_type == "lightgbm" and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

    # ------------------------------------------------------------------
    # Overall model-based decomposition
    # ------------------------------------------------------------------

    def _overall_decomposition(self) -> Dict:
        """Model-based overall decomposition."""
        actual_base = self.base_df[self.target_col].mean()
        actual_compare = self.compare_df[self.target_col].mean()
        predicted_compare = float(np.mean(self._compare_pred))
        predicted_base = float(np.mean(self._base_pred))

        total_diff = actual_compare - actual_base
        distribution_effect = predicted_compare - actual_base  # features changed
        relationship_effect = actual_compare - predicted_compare  # behaviour changed

        return {
            "base_metric": actual_base,
            "compare_metric": actual_compare,
            "total_difference": total_diff,
            "predicted_base_metric": predicted_base,
            "predicted_compare_metric": predicted_compare,
            "distribution_effect": distribution_effect,
            "relationship_effect": relationship_effect,
            "distribution_pct": (distribution_effect / total_diff * 100) if total_diff != 0 else np.nan,
            "relationship_pct": (relationship_effect / total_diff * 100) if total_diff != 0 else np.nan,
            "model_r2_base": self._r2(
                self.base_df[self.target_col].values, self._base_pred
            ),
        }

    @staticmethod
    def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    # ------------------------------------------------------------------
    # Per-variable decomposition
    # ------------------------------------------------------------------

    def _decompose_variable(self, col: str) -> pd.DataFrame:
        """Oaxaca-Blinder decomposition for a single variable.

        For each bucket *b*:
            mix_effect[b]  = (compare_weight[b] - base_weight[b]) × base_rate[b]
            rate_effect[b] = compare_weight[b] × (compare_rate[b] - base_rate[b])

        Returns a DataFrame with one row per bucket and columns:
            bucket, base_count, compare_count, base_weight, compare_weight,
            base_rate, compare_rate, mix_effect, rate_effect, total_effect
        """
        is_continuous = col in self.continuous_cols

        if is_continuous:
            base_binned, compare_binned = _create_pentile_bins(
                self.base_df[col], self.compare_df[col], n_bins=self.n_bins
            )
        else:
            # Categorical: use raw values
            all_cats = sorted(
                set(self.base_df[col].dropna().unique()) | set(self.compare_df[col].dropna().unique()),
                key=str,
            )
            base_binned = pd.Categorical(self.base_df[col], categories=all_cats, ordered=False)
            compare_binned = pd.Categorical(self.compare_df[col], categories=all_cats, ordered=False)

        # Build per-bucket stats
        base_target = self.base_df[self.target_col].values
        compare_target = self.compare_df[self.target_col].values

        n_base = len(self.base_df)
        n_compare = len(self.compare_df)

        records = []
        buckets = base_binned.categories if hasattr(base_binned, "categories") else sorted(
            set(base_binned.dropna().unique()) | set(compare_binned.dropna().unique()), key=str
        )

        for bucket in buckets:
            base_mask = base_binned == bucket
            compare_mask = compare_binned == bucket

            base_count = int(np.sum(base_mask))
            compare_count = int(np.sum(compare_mask))

            base_weight = base_count / n_base if n_base > 0 else 0
            compare_weight = compare_count / n_compare if n_compare > 0 else 0

            base_rate = float(np.nanmean(base_target[base_mask])) if base_count > 0 else 0.0
            compare_rate = float(np.nanmean(compare_target[compare_mask])) if compare_count > 0 else 0.0

            # Predicted rate from model (to separate model-explained vs unexplained)
            pred_compare_rate = float(np.nanmean(self._compare_pred[compare_mask])) if compare_count > 0 else 0.0

            mix_effect = (compare_weight - base_weight) * base_rate
            rate_effect = compare_weight * (compare_rate - base_rate)
            total_effect = mix_effect + rate_effect

            # Further decompose rate effect into model-explained and residual
            model_explained_rate = compare_weight * (pred_compare_rate - base_rate)
            residual_rate = compare_weight * (compare_rate - pred_compare_rate)

            records.append({
                "bucket": str(bucket),
                "base_count": base_count,
                "compare_count": compare_count,
                "base_weight": round(base_weight, 4),
                "compare_weight": round(compare_weight, 4),
                "weight_change": round(compare_weight - base_weight, 4),
                "base_rate": round(base_rate, 6),
                "compare_rate": round(compare_rate, 6),
                "predicted_compare_rate": round(pred_compare_rate, 6),
                "rate_change": round(compare_rate - base_rate, 6),
                "mix_effect": round(mix_effect, 6),
                "rate_effect": round(rate_effect, 6),
                "model_explained_rate_effect": round(model_explained_rate, 6),
                "residual_rate_effect": round(residual_rate, 6),
                "total_effect": round(total_effect, 6),
            })

        result_df = pd.DataFrame(records)

        # Add summary row
        summary = {
            "bucket": "TOTAL",
            "base_count": n_base,
            "compare_count": n_compare,
            "base_weight": 1.0,
            "compare_weight": 1.0,
            "weight_change": 0.0,
            "base_rate": round(float(np.nanmean(base_target)), 6),
            "compare_rate": round(float(np.nanmean(compare_target)), 6),
            "predicted_compare_rate": round(float(np.nanmean(self._compare_pred)), 6),
            "rate_change": round(float(np.nanmean(compare_target) - np.nanmean(base_target)), 6),
            "mix_effect": round(result_df["mix_effect"].sum(), 6),
            "rate_effect": round(result_df["rate_effect"].sum(), 6),
            "model_explained_rate_effect": round(result_df["model_explained_rate_effect"].sum(), 6),
            "residual_rate_effect": round(result_df["residual_rate_effect"].sum(), 6),
            "total_effect": round(result_df["total_effect"].sum(), 6),
        }
        result_df = pd.concat([result_df, pd.DataFrame([summary])], ignore_index=True)
        return result_df

    # ------------------------------------------------------------------
    # Feature importance from the model
    # ------------------------------------------------------------------

    def _feature_importance(self) -> pd.DataFrame:
        """Return feature importances from the fitted model."""
        if self.model_type in ("linear", "logistic"):
            coefs = np.abs(self.model.coef_.ravel())
        elif self.model_type == "lightgbm":
            coefs = self.model.feature_importances_
        else:
            return pd.DataFrame()

        imp_df = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": coefs,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["importance_pct"] = (imp_df["importance"] / imp_df["importance"].sum() * 100).round(2)
        return imp_df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Run the full diagnostic pipeline.

        Returns
        -------
        dict with keys:
            - ``overall``: model-based overall decomposition
            - ``by_variable``: dict of DataFrames, one per feature
            - ``feature_importance``: DataFrame of model feature importances
            - ``variable_summary``: DataFrame ranking variables by total effect
        """
        # 1. Build model on base
        print(">> Building model on base population ...")
        self._build_model()

        # 2. Score both populations
        print(">> Scoring populations ...")
        self._base_pred = self._score(self.base_df)
        self._compare_pred = self._score(self.compare_df)

        # 3. Overall decomposition
        print(">> Computing overall decomposition ...")
        self._overall_result = self._overall_decomposition()

        # 4. Per-variable decomposition
        print(">> Decomposing by variable ...")
        self._variable_results = {}
        for col in self.feature_cols:
            self._variable_results[col] = self._decompose_variable(col)

        # 5. Feature importance
        self._importance_df = self._feature_importance()

        # 6. Variable-level summary (total mix and rate effects)
        summary_records = []
        for col, vdf in self._variable_results.items():
            total_row = vdf[vdf["bucket"] == "TOTAL"].iloc[0]
            summary_records.append({
                "variable": col,
                "mix_effect": total_row["mix_effect"],
                "rate_effect": total_row["rate_effect"],
                "total_effect": total_row["total_effect"],
                "abs_total_effect": abs(total_row["total_effect"]),
            })
        self._variable_summary = (
            pd.DataFrame(summary_records)
            .sort_values("abs_total_effect", ascending=False)
            .reset_index(drop=True)
        )

        print("[OK] Done.\n")

        return {
            "overall": self._overall_result,
            "by_variable": self._variable_results,
            "feature_importance": self._importance_df,
            "variable_summary": self._variable_summary,
        }

    # ------------------------------------------------------------------
    # Pretty-printed summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print and return a human-readable summary of the diagnostic."""
        if self._overall_result is None:
            raise RuntimeError("Call .run() before .summary()")

        o = self._overall_result
        lines = [
            "=" * 70,
            "  METRIC DIAGNOSTIC SUMMARY",
            "=" * 70,
            f"  Target Column       : {self.target_col}",
            f"  Model Type          : {self.model_type}",
            f"  Features Used       : {len(self.feature_cols)}",
            f"  Base Observations   : {len(self.base_df):,}",
            f"  Compare Observations: {len(self.compare_df):,}",
            "-" * 70,
            "  OVERALL METRICS",
            "-" * 70,
            f"  Base Metric (actual)          : {o['base_metric']:.6f}",
            f"  Compare Metric (actual)       : {o['compare_metric']:.6f}",
            f"  Total Difference              : {o['total_difference']:+.6f}",
            "",
            f"  Predicted Compare Metric      : {o['predicted_compare_metric']:.6f}",
            f"  Distribution Effect (mix)     : {o['distribution_effect']:+.6f}  ({o['distribution_pct']:+.1f}%)",
            f"  Relationship Effect (rate)    : {o['relationship_effect']:+.6f}  ({o['relationship_pct']:+.1f}%)",
            "",
            f"  Model R-squared on Base       : {o['model_r2_base']:.4f}",
            "-" * 70,
            "  TOP DRIVERS (by |total_effect|)",
            "-" * 70,
        ]

        for _, row in self._variable_summary.head(10).iterrows():
            lines.append(
                f"  {row['variable']:<30s}  mix={row['mix_effect']:+.6f}  "
                f"rate={row['rate_effect']:+.6f}  total={row['total_effect']:+.6f}"
            )

        lines.append("=" * 70)
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Detailed per-variable view
    # ------------------------------------------------------------------

    def detail(self, variable: str) -> pd.DataFrame:
        """Return the per-bucket decomposition for a single variable."""
        if self._variable_results is None:
            raise RuntimeError("Call .run() before .detail()")
        if variable not in self._variable_results:
            raise KeyError(f"Variable '{variable}' not found in results.")
        return self._variable_results[variable]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_excel(self, path: str = "metric_diagnostics_output.xlsx"):
        """Export all results to an Excel workbook with multiple tabs."""
        if self._overall_result is None:
            raise RuntimeError("Call .run() before .to_excel()")

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Overall summary
            pd.DataFrame([self._overall_result]).to_excel(
                writer, sheet_name="Overall", index=False
            )
            # Variable summary
            self._variable_summary.to_excel(
                writer, sheet_name="Variable Summary", index=False
            )
            # Feature importance
            self._importance_df.to_excel(
                writer, sheet_name="Feature Importance", index=False
            )
            # Per-variable detail tabs
            for col, vdf in self._variable_results.items():
                sheet_name = col[:31]  # Excel 31-char limit
                vdf.to_excel(writer, sheet_name=sheet_name, index=False)

        print(f"[OK] Results exported to {path}")

    def to_pptx(
        self,
        path: str = "metric_diagnostics_report.pptx",
        top_n_variables: int = 5,
    ) -> str:
        """Generate a PowerPoint report with charts and tables.

        Parameters
        ----------
        path : str
            Output file path for the ``.pptx`` file.
        top_n_variables : int
            Number of top-driver variables to create detail slides for.

        Returns
        -------
        str
            The output file path.
        """
        if self._overall_result is None:
            raise RuntimeError("Call .run() before .to_pptx()")

        from .report import generate_report

        results = {
            "overall": self._overall_result,
            "by_variable": self._variable_results,
            "feature_importance": self._importance_df,
            "variable_summary": self._variable_summary,
        }
        output = generate_report(
            results=results,
            target_col=self.target_col,
            model_type=self.model_type,
            n_base=len(self.base_df),
            n_compare=len(self.compare_df),
            n_features=len(self.feature_cols),
            output_path=path,
            top_n_variables=top_n_variables,
        )
        print(f"[OK] PowerPoint report saved to {output}")
        return output
