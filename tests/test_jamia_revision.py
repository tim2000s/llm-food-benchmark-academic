"""Tests for the photograph-level analyses added for the revised manuscript."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jamia_revision_analysis as jr


class TestTotals(unittest.TestCase):
    def test_total_carbs_definition(self):
        items = [{"carbs_per_100": 40, "portion_estimate_size": 50},
                 {"carbs_per_100": None, "portion_estimate_size": 100}]
        self.assertAlmostEqual(jr.total_carbs_g(items), 20.0)


class TestPhotographLevelInference(unittest.TestCase):
    def test_bootstrap_interval_covers_the_estimate(self):
        rng = np.random.default_rng(0)
        est, lo, hi = jr.boot_ci([8.0, 9.0, 10.0, 11.0, 12.0], np.mean, rng)
        self.assertAlmostEqual(est, 10.0)
        self.assertLess(lo, est)
        self.assertGreater(hi, est)

    def test_photograph_level_interval_is_wider_than_query_level(self):
        """The point of the revision: repeated queries of one photograph are not
        independent observations, and treating them as such shrinks the interval."""
        rng = np.random.default_rng(1)
        per_photo = [4.0, 6.0, 9.0, 12.0, 15.0]
        _, lo, hi = jr.boot_ci(per_photo, np.mean, rng)
        queries = np.repeat(per_photo, 500)
        qse = queries.std(ddof=1) / np.sqrt(len(queries))
        self.assertGreater(hi - lo, 8 * (2 * 1.96 * qse))

    def test_per_photograph_cv(self):
        pp = jr.per_photograph({"m": {"a.jpg": [90.0, 100.0, 110.0]}})
        self.assertAlmostEqual(pp["m"]["a.jpg"]["cv_pct"], 10.0)
        self.assertEqual(pp["m"]["a.jpg"]["range_g"], 20.0)


class TestThresholdsAndAggregation(unittest.TestCase):
    REF = {"a.jpg": {"total_portion_carbs_g": 40, "reference_quality": 1}}

    def test_overdose_rate_scales_with_the_ratio(self):
        rng = np.random.default_rng(2)
        res = jr.dosing_thresholds({"m": {"a.jpg": [55.0] * 50}}, self.REF, rng)["m"]
        self.assertEqual(res[5]["one_query_2u"], 1.0)     # 15 g over is > 2 U at 1 U per 5 g
        self.assertEqual(res[10]["one_query_2u"], 0.0)
        self.assertEqual(res[20]["one_query_2u"], 0.0)

    def test_aggregation_converges_on_the_models_own_answer(self):
        rng = np.random.default_rng(3)
        vals = list(np.random.default_rng(4).normal(65, 8, 500))   # typical answer 25 g over
        agg = jr.aggregation({"m": {"a.jpg": vals}}, self.REF, rng)["m"]
        self.assertLess(agg[20]["off_own_10g_pct"], agg[1]["off_own_10g_pct"] / 5)
        self.assertLess(agg[20]["width_median_g"], agg[1]["width_median_g"] / 2)
        # A systematic 25 g overestimate becomes a near-certain overdose once averaged.
        self.assertGreater(agg[20]["over_2u_strong_pct"], agg[1]["over_2u_strong_pct"])
        self.assertGreater(agg[20]["over_2u_strong_pct"], 95.0)

    def test_aggregation_clears_overdoses_when_the_typical_answer_is_close(self):
        rng = np.random.default_rng(5)
        vals = list(np.random.default_rng(6).normal(50, 8, 500))   # typical answer 10 g over
        agg = jr.aggregation({"m": {"a.jpg": vals}}, self.REF, rng)["m"]
        self.assertGreater(agg[1]["over_2u_strong_pct"], 5.0)
        self.assertLess(agg[20]["over_2u_strong_pct"], 0.5)


class TestFailures(unittest.TestCase):
    def test_failures_counted_per_photograph(self):
        rows = {"m": [{"image_file": "IMG-20260410-WA0018.jpg", "success": i > 4, "iteration": i,
                       "error": "Expecting ',' delimiter", "food_items": []} for i in range(1, 11)]}
        f = jr.failures(rows)["m"]
        self.assertEqual((f["submitted"], f["parsed"], f["failed"]), (10, 6, 4))
        self.assertEqual(f["per_photograph"]["Eggs benedict"]["failed"], 4)


if __name__ == "__main__":
    unittest.main()
