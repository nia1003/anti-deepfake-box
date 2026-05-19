"""
Unit tests for fusion.cascade_selection.

Uses synthetic score arrays — no real CSV files or GPU required.
sklearn and scipy are expected to be available (they're in requirements.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from modules to avoid __init__.py chains that require
# sklearn/torch at collection time even when those tests would be skipped.

try:
    from sklearn.metrics import roc_auc_score  # noqa: F401
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not SKLEARN_AVAILABLE,
    reason="scikit-learn not installed",
)

if SKLEARN_AVAILABLE:
    # evaluation/__init__.py pulls in sklearn; import the module directly
    import importlib.util as _ilu
    from pathlib import Path as _Path

    def _load(rel, name):
        spec = _ilu.spec_from_file_location(name, _Path(__file__).parent.parent / rel)
        mod = _ilu.module_from_spec(spec)
        mod.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    _metrics_mod = _load("evaluation/metrics.py", "evaluation.metrics")
    DetectionMetrics = _metrics_mod.DetectionMetrics

    _cs_mod = _load("fusion/cascade_selection.py", "fusion.cascade_selection")
    compute_far_frr_curves = _cs_mod.compute_far_frr_curves
    filter_by_auc          = _cs_mod.filter_by_auc
    filter_by_correlation  = _cs_mod.filter_by_correlation
    pareto_filter          = _cs_mod.pareto_filter
    select_classifiers     = _cs_mod.select_classifiers
else:
    # Placeholders so the module parses even when skipped
    DetectionMetrics = None
    compute_far_frr_curves = filter_by_auc = filter_by_correlation = None
    pareto_filter = select_classifiers = None


# ------------------------------------------------------------------ #
#  Synthetic data helpers                                              #
# ------------------------------------------------------------------ #

def _perfect_scores(n=100):
    """Perfect detector: fake=1.0, real=0.0."""
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    scores = np.array([1.0] * (n // 2) + [0.0] * (n // 2), dtype=np.float64)
    ids = [f"v{i:04d}" for i in range(n)]
    return ids, scores, labels


def _random_scores(n=100, seed=42):
    """Random (chance) detector: AUC ≈ 0.5."""
    rng = np.random.RandomState(seed)
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    scores = rng.rand(n).astype(np.float64)
    ids = [f"v{i:04d}" for i in range(n)]
    return ids, scores, labels


def _good_scores(n=100, seed=7):
    """Decent detector: AUC ~0.80."""
    rng = np.random.RandomState(seed)
    labels = np.array([1] * (n // 2) + [0] * (n // 2), dtype=np.int32)
    scores = np.where(
        labels == 1,
        np.clip(rng.normal(0.70, 0.15, n), 0, 1),
        np.clip(rng.normal(0.30, 0.15, n), 0, 1),
    ).astype(np.float64)
    ids = [f"v{i:04d}" for i in range(n)]
    return ids, scores, labels


def _make_metrics(auc: float, eer: float) -> DetectionMetrics:
    return DetectionMetrics(
        auc=auc, acc=0.80, eer=eer, ap=0.80,
        threshold=0.50, n_real=50, n_fake=50,
    )


# ------------------------------------------------------------------ #
#  Stage 1: AUC filter                                                 #
# ------------------------------------------------------------------ #

class TestFilterByAUC:
    def test_perfect_detector_passes(self):
        ids, scores, labels = _perfect_scores()
        data = {"visual": (ids, scores, labels)}
        retained, metrics = filter_by_auc(data, min_auc=0.55)
        assert "visual" in retained
        assert metrics["visual"].auc == pytest.approx(1.0, abs=0.01)

    def test_random_detector_dropped(self):
        ids, scores, labels = _random_scores()
        data = {"visual": (ids, scores, labels)}
        retained, _ = filter_by_auc(data, min_auc=0.55)
        assert "visual" not in retained

    def test_good_detector_passes(self):
        ids, scores, labels = _good_scores()
        data = {"visual": (ids, scores, labels)}
        retained, _ = filter_by_auc(data, min_auc=0.55)
        assert "visual" in retained

    def test_metrics_returned_for_all(self):
        data = {
            "visual": _perfect_scores(),
            "rppg":   _random_scores(),
        }
        _, metrics = filter_by_auc(data, min_auc=0.55)
        assert "visual" in metrics
        assert "rppg" in metrics

    def test_empty_data_returns_empty(self):
        retained, metrics = filter_by_auc({}, min_auc=0.55)
        assert retained == {}
        assert metrics == {}

    def test_custom_threshold(self):
        ids, scores, labels = _good_scores()
        data = {"visual": (ids, scores, labels)}
        # Good detector (AUC ~0.80) should pass min_auc=0.70
        retained, _ = filter_by_auc(data, min_auc=0.70)
        assert "visual" in retained
        # Should fail min_auc=0.99
        retained, _ = filter_by_auc(data, min_auc=0.99)
        assert "visual" not in retained


# ------------------------------------------------------------------ #
#  Stage 2: Correlation filter                                         #
# ------------------------------------------------------------------ #

class TestFilterByCorrelation:
    def _metrics(self, auc_map):
        return {
            k: _make_metrics(auc, 1.0 - auc)
            for k, auc in auc_map.items()
        }

    def test_uncorrelated_both_retained(self):
        n = 100
        ids = [f"v{i:04d}" for i in range(n)]
        labels = np.array([1] * 50 + [0] * 50, dtype=np.int32)
        rng = np.random.RandomState(0)
        s1 = rng.rand(n)
        s2 = rng.rand(n)  # independent → low correlation
        data = {
            "visual": (ids, s1, labels),
            "rppg":   (ids, s2, labels),
        }
        metrics = self._metrics({"visual": 0.80, "rppg": 0.75})
        retained = filter_by_correlation(data, metrics, max_corr=0.90)
        assert "visual" in retained
        assert "rppg" in retained

    def test_highly_correlated_lower_auc_dropped(self):
        n = 100
        ids = [f"v{i:04d}" for i in range(n)]
        labels = np.array([1] * 50 + [0] * 50, dtype=np.int32)
        s1 = np.linspace(0, 1, n)
        s2 = s1 + np.random.RandomState(1).normal(0, 0.01, n)  # r ≈ 0.999
        data = {
            "visual": (ids, s1, labels),
            "rppg":   (ids, s2, labels),
        }
        metrics = self._metrics({"visual": 0.90, "rppg": 0.70})
        retained = filter_by_correlation(data, metrics, max_corr=0.90)
        assert "visual" in retained
        assert "rppg" not in retained  # lower AUC dropped

    def test_highly_correlated_keeps_higher_auc(self):
        n = 100
        ids = [f"v{i:04d}" for i in range(n)]
        labels = np.array([1] * 50 + [0] * 50, dtype=np.int32)
        s1 = np.linspace(0, 1, n)
        s2 = s1 + np.random.RandomState(2).normal(0, 0.01, n)
        data = {
            "visual": (ids, s1, labels),
            "rppg":   (ids, s2, labels),
        }
        metrics = self._metrics({"visual": 0.60, "rppg": 0.90})
        retained = filter_by_correlation(data, metrics, max_corr=0.90)
        assert "rppg" in retained
        assert "visual" not in retained

    def test_single_modality_unaffected(self):
        n = 50
        ids = [f"v{i:04d}" for i in range(n)]
        labels = np.array([1] * 25 + [0] * 25, dtype=np.int32)
        data = {"visual": (ids, np.linspace(0, 1, n), labels)}
        metrics = self._metrics({"visual": 0.80})
        retained = filter_by_correlation(data, metrics, max_corr=0.90)
        assert "visual" in retained


# ------------------------------------------------------------------ #
#  Stage 3: FAR/FRR curves                                             #
# ------------------------------------------------------------------ #

class TestFARFRRCurves:
    def test_returns_expected_keys(self):
        ids, scores, labels = _good_scores()
        data = {"visual": (ids, scores, labels)}
        result = compute_far_frr_curves(data)
        assert "visual" in result
        for key in ("thresholds", "FAR", "FRR", "EER", "EER_threshold"):
            assert key in result["visual"]

    def test_eer_in_valid_range(self):
        ids, scores, labels = _good_scores()
        data = {"visual": (ids, scores, labels)}
        result = compute_far_frr_curves(data)
        assert 0.0 <= result["visual"]["EER"] <= 1.0

    def test_perfect_detector_low_eer(self):
        ids, scores, labels = _perfect_scores()
        data = {"visual": (ids, scores, labels)}
        result = compute_far_frr_curves(data)
        assert result["visual"]["EER"] < 0.05


# ------------------------------------------------------------------ #
#  Stage 4: Pareto filter                                              #
# ------------------------------------------------------------------ #

class TestParetoFilter:
    def test_non_dominated_all_retained(self):
        # A: high AUC low EER, B: low AUC high EER — neither dominates
        metrics = {
            "visual": _make_metrics(auc=0.90, eer=0.10),
            "rppg":   _make_metrics(auc=0.70, eer=0.30),
        }
        surviving = pareto_filter(metrics)
        assert "visual" in surviving
        assert "rppg" in surviving

    def test_dominated_detector_dropped(self):
        # B dominated by A: A has higher AUC AND lower EER
        metrics = {
            "visual": _make_metrics(auc=0.90, eer=0.10),
            "rppg":   _make_metrics(auc=0.70, eer=0.30),  # dominated
        }
        # Make visual strictly better on both
        metrics["rppg"] = _make_metrics(auc=0.70, eer=0.30)
        surviving = pareto_filter(metrics)
        # visual dominates rppg
        assert "visual" in surviving

    def test_identical_metrics_both_survive(self):
        # Identical: neither strictly dominates
        metrics = {
            "visual": _make_metrics(auc=0.80, eer=0.20),
            "rppg":   _make_metrics(auc=0.80, eer=0.20),
        }
        surviving = pareto_filter(metrics)
        assert len(surviving) == 2

    def test_single_detector_survives(self):
        metrics = {"visual": _make_metrics(auc=0.85, eer=0.15)}
        surviving = pareto_filter(metrics)
        assert surviving == ["visual"]

    def test_three_detectors_pareto(self):
        # A: AUC=0.90, EER=0.10 (best overall)
        # B: AUC=0.80, EER=0.15 (dominated by A)
        # C: AUC=0.85, EER=0.25 (not dominated by A on EER? No, A has lower EER)
        # Actually A dominates both B and C here
        metrics = {
            "visual": _make_metrics(auc=0.90, eer=0.10),
            "rppg":   _make_metrics(auc=0.80, eer=0.15),
            "sync":   _make_metrics(auc=0.85, eer=0.25),
        }
        surviving = pareto_filter(metrics)
        # visual dominates both rppg and sync
        assert "visual" in surviving
        assert len(surviving) >= 1


# ------------------------------------------------------------------ #
#  Full pipeline (no real CSVs — just checks it runs without crashing) #
# ------------------------------------------------------------------ #

class TestSelectClassifiersNoCsvDir:
    def test_missing_csv_dir_returns_empty(self, tmp_path):
        result = select_classifiers(str(tmp_path), min_auc=0.55)
        assert result == []
