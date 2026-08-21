import unittest
from pathlib import Path

import numpy as np

from npv_engine import AccountNPVEngine


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, features):
        return np.full(len(features), self.value, dtype=np.float32)


class AccountNPVEngineTests(unittest.TestCase):
    def setUp(self):
        self.constants_path = Path(__file__).resolve().parents[1] / "constants.csv"

    def make_demo_engine(self, **kwargs):
        return AccountNPVEngine(
            num_months=3,
            chunk_size=2,
            allow_demo_fallback=True,
            **kwargs,
        )

    def test_constants_csv_is_loaded_from_module_configuration(self):
        engine = self.make_demo_engine()
        self.assertAlmostEqual(float(engine.tax_rate), 0.25)
        self.assertAlmostEqual(float(engine.capital_requirement_rate), 0.10)
        self.assertAlmostEqual(float(engine.lgd), 0.60)

    def test_production_mode_requires_models(self):
        with self.assertRaises(ValueError):
            AccountNPVEngine(num_months=3, constants_csv_path=self.constants_path)

    def test_invalid_curve_model_configuration_fails_fast(self):
        with self.assertRaises(ValueError):
            AccountNPVEngine(
                num_months=3,
                constants_csv_path=self.constants_path,
                rdm_model_path=ConstantModel(0.0),
                curve_model_paths=[ConstantModel(0.0)],
            )

    def test_curve_scoring_is_chunked_and_bounded(self):
        engine = AccountNPVEngine(
            num_months=3,
            chunk_size=1,
            constants_csv_path=self.constants_path,
            rdm_model_path=ConstantModel(0.0),
            curve_model_paths=[
                ConstantModel(2.0),
                ConstantModel(-10.0),
                ConstantModel(-0.5),
                ConstantModel(1.0),
            ],
        )
        rdm = engine.score_initial_rdm_model(np.zeros((3, 5), dtype=np.float32))
        curves = engine.score_curve_models(rdm, np.zeros((3, 2), dtype=np.float32))

        np.testing.assert_array_equal(curves[0], 1.0)
        np.testing.assert_array_equal(curves[1], 0.0)
        np.testing.assert_array_equal(curves[2], 0.0)
        np.testing.assert_array_equal(curves[3], 1.0)

    def test_annual_loss_rate_is_exposure_weighted_and_annualized(self):
        engine = self.make_demo_engine()
        c1 = np.full((1, 3), 0.01, dtype=np.float32)
        c2 = np.array([[100.0, 200.0, 100.0]], dtype=np.float32)

        result = engine.calculate_annual_loss_rate(c1, c2)

        # 2.4 units of expected loss / 400 units of exposure, annualized from 3 months.
        self.assertAlmostEqual(float(result[0]), 0.024, places=5)

    def test_npv_does_not_add_expected_loss_to_discount_rate(self):
        engine = self.make_demo_engine(annual_discount_rate=0.12)
        cashflows = np.ones((1, 3), dtype=np.float32)

        result = engine.calculate_npv(cashflows)
        expected = sum(1.0 / (1.0 + 0.12 / 12.0) ** month for month in (1, 2, 3))

        self.assertAlmostEqual(float(result[0]), expected, places=6)

    def test_chunked_pipeline_returns_one_result_per_account(self):
        engine = self.make_demo_engine()
        rdm_features = np.zeros((5, 5), dtype=np.float32)
        other_features = np.zeros((5, 2), dtype=np.float32)

        results = engine.run(rdm_features, other_features)

        self.assertEqual(len(results), 5)
        self.assertTrue(np.isfinite(results["NPV"]).all())
        self.assertIn("Annual Loss Rate", results.columns)


if __name__ == "__main__":
    unittest.main()
