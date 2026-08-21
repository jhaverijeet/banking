import os
import pickle
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class AccountNPVEngine:
    """Vectorized account-level NPV engine with validated, chunked processing.

    The engine expects trained models in production. The deterministic dummy
    models are available only when ``allow_demo_fallback=True``.
    """

    REQUIRED_CONSTANTS = ("tax_rate", "capital_requirement_rate", "lgd")

    def __init__(
        self,
        num_months: int = 99,
        annual_discount_rate: float = 0.08,
        rdm_model_path: Optional[Any] = None,
        curve_model_paths: Optional[Sequence[Any]] = None,
        constants_csv_path: Optional[os.PathLike] = None,
        chunk_size: int = 10_000,
        allow_demo_fallback: bool = False,
        credit_risk_premium: float = 0.0,
    ):
        if not isinstance(num_months, (int, np.integer)) or num_months <= 0:
            raise ValueError("num_months must be a positive integer")
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if not np.isfinite(annual_discount_rate) or annual_discount_rate <= -1.0:
            raise ValueError("annual_discount_rate must be finite and greater than -1")
        if not np.isfinite(credit_risk_premium) or credit_risk_premium < 0.0:
            raise ValueError("credit_risk_premium must be finite and non-negative")

        self.num_months = int(num_months)
        self.chunk_size = int(chunk_size)
        self.allow_demo_fallback = bool(allow_demo_fallback)
        self.base_annual_discount_rate = np.float32(annual_discount_rate)
        self.credit_risk_premium = np.float32(credit_risk_premium)
        self.months_array = np.arange(1, self.num_months + 1, dtype=np.float32)
        self._demo_curve_coefficients = {}

        self._load_constants(constants_csv_path)

        if rdm_model_path is None:
            if not self.allow_demo_fallback:
                raise ValueError(
                    "rdm_model_path is required; use allow_demo_fallback=True "
                    "only for deterministic demonstrations"
                )
            self.rdm_model = None
        else:
            self.rdm_model = self._load_model(rdm_model_path, "rdm_model")

        if curve_model_paths is None:
            if not self.allow_demo_fallback:
                raise ValueError(
                    "exactly four curve models are required; use "
                    "allow_demo_fallback=True only for deterministic demonstrations"
                )
            self.curve_models = None
        else:
            if len(curve_model_paths) != 4:
                raise ValueError("curve_model_paths must contain exactly four models")
            self.curve_models = [
                self._load_model(model_path, f"curve_model_{index + 1}")
                for index, model_path in enumerate(curve_model_paths)
            ]

        # Fallback RDM coefficients are retained only for explicit demo mode.
        self.rdm_coefs = np.array([0.45, -0.15, 0.22, 0.05, -0.30], dtype=np.float32)
        self.rdm_intercept = np.float32(1.2)

    def _load_model(self, model_or_path: Any, model_name: str) -> Any:
        """Load a model and fail clearly if it is missing or unusable."""
        if hasattr(model_or_path, "predict"):
            model = model_or_path
        else:
            if not isinstance(model_or_path, (str, os.PathLike)):
                raise TypeError(f"{model_name} must be a model object or filesystem path")
            model_path = Path(model_or_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"{model_name} was not found: {model_path}")
            with model_path.open("rb") as model_file:
                model = pickle.load(model_file)

        if not hasattr(model, "predict"):
            raise TypeError(f"{model_name} must expose a predict method")
        return model

    def _load_constants(self, csv_path: Optional[os.PathLike]) -> None:
        """Load and validate the single-row constants configuration."""
        if csv_path is None:
            constants_path = Path(__file__).resolve().with_name("constants.csv")
        else:
            constants_path = Path(csv_path)

        if not constants_path.is_file():
            raise FileNotFoundError(f"Constants CSV was not found: {constants_path}")

        df_const = pd.read_csv(constants_path)
        missing = [name for name in self.REQUIRED_CONSTANTS if name not in df_const.columns]
        if missing:
            raise ValueError(
                f"Constants CSV must have columns {self.REQUIRED_CONSTANTS}; "
                f"missing {missing}: {constants_path}"
            )
        if len(df_const) != 1:
            raise ValueError(
                f"Constants CSV must contain exactly one data row: {constants_path}"
            )

        values = {}
        for name in self.REQUIRED_CONSTANTS:
            try:
                value = float(df_const.loc[0, name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Constant {name!r} must be numeric") from exc
            if not np.isfinite(value):
                raise ValueError(f"Constant {name!r} must be finite")
            values[name] = value

        if not 0.0 <= values["tax_rate"] <= 1.0:
            raise ValueError("tax_rate must be between 0 and 1")
        if not 0.0 < values["capital_requirement_rate"] <= 1.0:
            raise ValueError("capital_requirement_rate must be greater than 0 and at most 1")
        if not 0.0 <= values["lgd"] <= 1.0:
            raise ValueError("lgd must be between 0 and 1")

        self.tax_rate = np.float32(values["tax_rate"])
        self.capital_requirement_rate = np.float32(values["capital_requirement_rate"])
        self.lgd = np.float32(values["lgd"])

    @staticmethod
    def _as_feature_array(values: Any, name: str) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional numeric array")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return array

    def _validate_features(
        self, rdm_features: Any, other_features: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        rdm_array = self._as_feature_array(rdm_features, "rdm_features")
        other_array = self._as_feature_array(other_features, "other_features")
        if rdm_array.shape[1] != 5:
            raise ValueError(f"rdm_features must have exactly 5 columns, got {rdm_array.shape[1]}")
        if rdm_array.shape[0] != other_array.shape[0]:
            raise ValueError("rdm_features and other_features must have the same row count")
        return rdm_array, other_array

    @staticmethod
    def _validate_curve_shapes(curves: Sequence[np.ndarray], expected_shape: Tuple[int, int]) -> None:
        for index, curve in enumerate(curves, start=1):
            if curve.shape != expected_shape:
                raise ValueError(
                    f"curve_{index} must have shape {expected_shape}, got {curve.shape}"
                )
            if not np.isfinite(curve).all():
                raise ValueError(f"curve_{index} contains NaN or infinite values")

    @staticmethod
    def _bound_curves(curves: Sequence[np.ndarray]) -> Tuple[np.ndarray, ...]:
        """Apply domain bounds to model outputs.

        c1 is probability of default, c2 is balance, c3 is prepayment rate,
        and c4 is a fee amount that may be positive or negative.
        """
        c1, c2, c3, c4 = curves
        return (
            np.clip(c1, 0.0, 1.0),
            np.maximum(c2, 0.0),
            np.clip(c3, 0.0, 1.0),
            c4,
        )

    def score_initial_rdm_model(self, rdm_features: Any) -> np.ndarray:
        """Score one RDM value per account."""
        rdm_array = self._as_feature_array(rdm_features, "rdm_features")
        if rdm_array.shape[1] != 5:
            raise ValueError(f"rdm_features must have exactly 5 columns, got {rdm_array.shape[1]}")

        if self.rdm_model is not None:
            predictions = self.rdm_model.predict(rdm_array)
        else:
            predictions = np.dot(rdm_array, self.rdm_coefs) + self.rdm_intercept

        rdm = np.asarray(predictions, dtype=np.float32).reshape(-1)
        if rdm.shape[0] != rdm_array.shape[0]:
            raise ValueError(
                f"rdm model returned {rdm.shape[0]} predictions for {rdm_array.shape[0]} accounts"
            )
        if not np.isfinite(rdm).all():
            raise ValueError("rdm model returned NaN or infinite values")
        return rdm

    def _get_demo_curve_coefficients(self, num_base_features: int) -> Tuple[np.ndarray, ...]:
        if num_base_features not in self._demo_curve_coefficients:
            rng = np.random.default_rng(42)
            self._demo_curve_coefficients[num_base_features] = tuple(
                rng.normal(size=num_base_features + 1).astype(np.float32) * scale
                for scale in (0.01, 0.05, 0.02, 1.5)
            )
        return self._demo_curve_coefficients[num_base_features]

    def score_curve_models(self, rdm: Any, other_features: Any) -> Tuple[np.ndarray, ...]:
        """Score four account-month curves in bounded memory chunks."""
        rdm_array = np.asarray(rdm, dtype=np.float32).reshape(-1)
        other_array = self._as_feature_array(other_features, "other_features")
        if rdm_array.shape[0] != other_array.shape[0]:
            raise ValueError("rdm and other_features must have the same row count")
        if not np.isfinite(rdm_array).all():
            raise ValueError("rdm contains NaN or infinite values")

        num_accounts = rdm_array.shape[0]
        base_features = np.column_stack((rdm_array, other_array)).astype(np.float32, copy=False)
        num_base_features = base_features.shape[1]
        curves = tuple(
            np.empty((num_accounts, self.num_months), dtype=np.float32) for _ in range(4)
        )

        for start in range(0, num_accounts, self.chunk_size):
            end = min(start + self.chunk_size, num_accounts)
            chunk_accounts = end - start
            repeated_features = np.repeat(base_features[start:end], self.num_months, axis=0)
            expanded_features = np.empty(
                (chunk_accounts * self.num_months, num_base_features + 1), dtype=np.float32
            )
            expanded_features[:, :-1] = repeated_features
            expanded_features[:, -1] = np.tile(self.months_array, chunk_accounts)

            if self.curve_models is not None:
                feature_cols = [f"feature_{i}" for i in range(num_base_features)] + [
                    "month_on_book"
                ]
                model_features = pd.DataFrame(expanded_features, columns=feature_cols)
                predictions = [model.predict(model_features) for model in self.curve_models]
            else:
                coefficients = self._get_demo_curve_coefficients(num_base_features)
                predictions = [np.dot(expanded_features, coefficient) for coefficient in coefficients]

            for curve, prediction in zip(curves, predictions):
                flat_prediction = np.asarray(prediction, dtype=np.float32).reshape(-1)
                expected_size = chunk_accounts * self.num_months
                if flat_prediction.size != expected_size:
                    raise ValueError(
                        f"curve model returned {flat_prediction.size} values; expected {expected_size}"
                    )
                if not np.isfinite(flat_prediction).all():
                    raise ValueError("curve model returned NaN or infinite values")
                curve[start:end] = flat_prediction.reshape(chunk_accounts, self.num_months)

        return self._bound_curves(curves)

    def calculate_cashflows(
        self, c1: np.ndarray, c2: np.ndarray, c3: np.ndarray, c4: np.ndarray
    ) -> np.ndarray:
        """Combine bounded curves into after-tax monthly cash flows."""
        curves = tuple(np.asarray(curve, dtype=np.float32) for curve in (c1, c2, c3, c4))
        expected_shape = (curves[0].shape[0], self.num_months)
        self._validate_curve_shapes(curves, expected_shape)
        c1, c2, c3, c4 = self._bound_curves(curves)

        cashflows = np.empty_like(c2, dtype=np.float32)
        work = np.empty_like(c2, dtype=np.float32)
        np.multiply(c2, np.float32(0.02), out=cashflows)
        cashflows += c4
        np.multiply(c1, self.lgd, out=work)
        np.multiply(work, c2, out=work)
        cashflows -= work
        np.multiply(c3, c2, out=work)
        cashflows -= work
        cashflows *= np.float32(1.0 - self.tax_rate)

        if not np.isfinite(cashflows).all():
            raise ValueError("cash-flow calculation returned NaN or infinite values")
        return cashflows

    def calculate_annual_loss_rate(self, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        """Calculate a simple annualized, exposure-weighted loss rate.

        The model curves are interpreted as monthly rates. The result is the
        exposure-weighted average monthly expected loss multiplied by 12 (or
        annualized when fewer than 12 months are available). It is reported as
        a metric only; it is not automatically added to the NPV discount rate.
        """
        c1_array = np.asarray(c1, dtype=np.float32)
        c2_array = np.asarray(c2, dtype=np.float32)
        expected_shape = (c1_array.shape[0], self.num_months)
        self._validate_curve_shapes((c1_array, c2_array), expected_shape)
        c1_array, c2_array = self._bound_curves((c1_array, c2_array, c1_array, c2_array))[:2]

        months = min(12, self.num_months)
        exposure = np.sum(c2_array[:, :months], axis=1, dtype=np.float64)
        losses = np.sum(
            c1_array[:, :months] * self.lgd * c2_array[:, :months], axis=1, dtype=np.float64
        )
        annual_loss_rate = np.divide(
            losses * (12.0 / months),
            exposure,
            out=np.zeros_like(losses),
            where=exposure > 0.0,
        )
        return annual_loss_rate.astype(np.float32)

    def calculate_npv(
        self, cashflows: np.ndarray, credit_risk_premium: Optional[Any] = None
    ) -> np.ndarray:
        """Discount cash flows without re-adding expected losses.

        ``credit_risk_premium`` is an optional, separately calibrated premium;
        expected loss rates must not be passed here because expected losses are
        already included in the cash flows.
        """
        cashflow_array = np.asarray(cashflows, dtype=np.float32)
        if cashflow_array.ndim != 2 or cashflow_array.shape[1] != self.num_months:
            raise ValueError(
                f"cashflows must have shape (num_accounts, {self.num_months})"
            )
        if not np.isfinite(cashflow_array).all():
            raise ValueError("cashflows contains NaN or infinite values")

        if credit_risk_premium is None:
            premium = np.full(
                cashflow_array.shape[0], self.credit_risk_premium, dtype=np.float32
            )
        else:
            premium = np.asarray(credit_risk_premium, dtype=np.float32)
            if premium.ndim == 0:
                premium = np.full(cashflow_array.shape[0], premium, dtype=np.float32)
            elif premium.shape != (cashflow_array.shape[0],):
                raise ValueError("credit_risk_premium must be scalar or one value per account")
        if not np.isfinite(premium).all() or np.any(premium < 0.0):
            raise ValueError("credit_risk_premium must be finite and non-negative")

        annual_rates = self.base_annual_discount_rate + premium
        monthly_rates = annual_rates / np.float32(12.0)
        if np.any(1.0 + monthly_rates <= 0.0):
            raise ValueError("discount rate produces a non-positive monthly discount base")

        npv = np.empty(cashflow_array.shape[0], dtype=np.float64)
        for start in range(0, cashflow_array.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, cashflow_array.shape[0])
            discount_factors = 1.0 / np.power(
                1.0 + monthly_rates[start:end, None], self.months_array[None, :]
            )
            npv[start:end] = np.sum(
                cashflow_array[start:end] * discount_factors, axis=1, dtype=np.float64
            )
        return npv

    def calculate_metrics(
        self, c1: np.ndarray, c2: np.ndarray, cashflows: np.ndarray, npv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate five-year loss, return, and payback metrics."""
        c1_array = np.asarray(c1, dtype=np.float32)
        c2_array = np.asarray(c2, dtype=np.float32)
        cashflow_array = np.asarray(cashflows, dtype=np.float32)
        expected_shape = (c1_array.shape[0], self.num_months)
        self._validate_curve_shapes((c1_array, c2_array), expected_shape)
        if cashflow_array.shape != expected_shape:
            raise ValueError(f"cashflows must have shape {expected_shape}")
        c1_array, c2_array = self._bound_curves((c1_array, c2_array, c1_array, c2_array))[:2]

        months_5yr = min(60, self.num_months)
        c1_5yr = c1_array[:, :months_5yr]
        c2_5yr = c2_array[:, :months_5yr]
        cf_5yr = cashflow_array[:, :months_5yr]
        avg_bal_5yr = np.mean(c2_5yr, axis=1, dtype=np.float64)
        safe_avg_bal = np.where(avg_bal_5yr > 0.0, avg_bal_5yr, np.nan)

        losses_5yr = np.sum((c1_5yr * self.lgd) * c2_5yr, axis=1, dtype=np.float64)
        total_cf_5yr = np.sum(cf_5yr, axis=1, dtype=np.float64)
        net_loss_rate_5yr = np.divide(
            losses_5yr, safe_avg_bal, out=np.zeros_like(losses_5yr), where=np.isfinite(safe_avg_bal)
        )
        roa_5yr = np.divide(
            total_cf_5yr, safe_avg_bal, out=np.zeros_like(total_cf_5yr), where=np.isfinite(safe_avg_bal)
        )
        avg_equity_5yr = safe_avg_bal * self.capital_requirement_rate
        roe_5yr = np.divide(
            total_cf_5yr, avg_equity_5yr, out=np.zeros_like(total_cf_5yr), where=np.isfinite(avg_equity_5yr)
        )

        payback_period = np.full(cashflow_array.shape[0], -1, dtype=np.int32)
        for start in range(0, cashflow_array.shape[0], self.chunk_size):
            end = min(start + self.chunk_size, cashflow_array.shape[0])
            cumulative_cashflows = np.cumsum(cashflow_array[start:end], axis=1, dtype=np.float64)
            positive_mask = cumulative_cashflows > 0.0
            has_paid_back = np.any(positive_mask, axis=1)
            first_positive = np.argmax(positive_mask, axis=1) + 1
            payback_period[start:end] = np.where(has_paid_back, first_positive, -1)

        return (
            net_loss_rate_5yr.astype(np.float32),
            roa_5yr.astype(np.float32),
            roe_5yr.astype(np.float32),
            payback_period,
        )

    def run(self, rdm_features: Any, other_features: Any) -> pd.DataFrame:
        """Run the end-to-end pipeline without materializing all account-month data."""
        rdm_array, other_array = self._validate_features(rdm_features, other_features)

        num_accounts = rdm_array.shape[0]
        result_arrays = {
            "NPV": np.empty(num_accounts, dtype=np.float64),
            "Annual Loss Rate": np.empty(num_accounts, dtype=np.float32),
            "5 Yr Net Loss Rate": np.empty(num_accounts, dtype=np.float32),
            "5 Yr ROA": np.empty(num_accounts, dtype=np.float32),
            "5 Yr ROE": np.empty(num_accounts, dtype=np.float32),
            "Payback Period (Months)": np.empty(num_accounts, dtype=np.int32),
        }

        for start in range(0, num_accounts, self.chunk_size):
            end = min(start + self.chunk_size, num_accounts)
            rdm = self.score_initial_rdm_model(rdm_array[start:end])
            c1, c2, c3, c4 = self.score_curve_models(rdm, other_array[start:end])
            cashflows = self.calculate_cashflows(c1, c2, c3, c4)
            annual_loss_rate = self.calculate_annual_loss_rate(c1, c2)
            npv = self.calculate_npv(cashflows)
            metrics = self.calculate_metrics(c1, c2, cashflows, npv)

            result_arrays["NPV"][start:end] = npv
            result_arrays["Annual Loss Rate"][start:end] = annual_loss_rate
            result_arrays["5 Yr Net Loss Rate"][start:end] = metrics[0]
            result_arrays["5 Yr ROA"][start:end] = metrics[1]
            result_arrays["5 Yr ROE"][start:end] = metrics[2]
            result_arrays["Payback Period (Months)"][start:end] = metrics[3]

        return pd.DataFrame(result_arrays)


if __name__ == "__main__":
    NUM_ACCOUNTS = 1_000_000
    NUM_MONTHS = 99

    print(f"Generating synthetic characteristics for {NUM_ACCOUNTS:,} accounts...")
    synthetic_rdm_features = np.random.default_rng(42).normal(
        size=(NUM_ACCOUNTS, 5)
    ).astype(np.float32)
    synthetic_other_features = np.random.default_rng(43).normal(
        size=(NUM_ACCOUNTS, 3)
    ).astype(np.float32)

    engine = AccountNPVEngine(
        num_months=NUM_MONTHS,
        chunk_size=10_000,
        allow_demo_fallback=True,
    )

    print("Running vectorized NPV engine in chunks...")
    start_time = time.time()
    results_df = engine.run(synthetic_rdm_features, synthetic_other_features)
    end_time = time.time()

    print("-" * 50)
    print(f"Engine completed in: {end_time - start_time:.4f} seconds.")
    print(f"Accounts processed : {NUM_ACCOUNTS:,}")
    print(f"Months on book     : {NUM_MONTHS}")
    print("\nSample account results (first 5):")
    print(results_df.head(5).to_string())
    print("-" * 50)
