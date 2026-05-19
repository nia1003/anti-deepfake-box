"""
Unit tests for fusion.serial_cascade.SerialCascade.

All tests are self-contained (no ffmpeg, no torch, no pandas required for
the pure-logic tests). CSV-loader tests require pandas and are skipped when
absent.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import directly from modules to avoid fusion/__init__.py pulling in torch
# (via meta_classifier.py)
import importlib.util as _ilu

def _load_mod(rel_path, name):
    spec = _ilu.spec_from_file_location(name, Path(__file__).parent.parent / rel_path)
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = name.rsplit(".", 1)[0] if "." in name else ""
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_we = _load_mod("fusion/weighted_ensemble.py", "fusion.weighted_ensemble")
FusionResult = _we.FusionResult

_sc = _load_mod("fusion/serial_cascade.py", "fusion.serial_cascade")
CascadeStage = _sc.CascadeStage
SerialCascade = _sc.SerialCascade


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _pandas_available() -> bool:
    try:
        import pandas  # noqa: F401
        return True
    except ImportError:
        return False


def _write_tmp_csv(rows: list[dict]) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    )
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    f.close()
    return f.name


def _write_tmp_json(spec: dict) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    )
    json.dump(spec, f)
    f.close()
    return f.name


# ------------------------------------------------------------------ #
#  CascadeStage                                                        #
# ------------------------------------------------------------------ #

class TestCascadeStage:
    def test_valid_stage(self):
        s = CascadeStage(name="visual", H=0.75, L=0.30)
        assert s.name == "visual"
        assert s.H == pytest.approx(0.75)
        assert s.L == pytest.approx(0.30)

    def test_l_must_be_less_than_h(self):
        with pytest.raises(ValueError, match="must be < H"):
            CascadeStage(name="visual", H=0.30, L=0.75)

    def test_l_equal_h_raises(self):
        with pytest.raises(ValueError):
            CascadeStage(name="visual", H=0.50, L=0.50)


# ------------------------------------------------------------------ #
#  SerialCascade — JSON loader (no external deps)                      #
# ------------------------------------------------------------------ #

@pytest.fixture
def simple_json_cascade(tmp_path):
    """Two-stage JSON cascade: visual(H=0.75,L=0.30) → sync(H=0.65,L=0.20)."""
    spec = {
        "stages": [
            {"name": "visual",  "H": 0.75, "L": 0.30},
            {"name": "av_sync", "H": 0.65, "L": 0.20},
        ],
        "default_threshold": 0.50,
        "fallback_score": 0.50,
    }
    p = tmp_path / "cascade.json"
    p.write_text(json.dumps(spec))
    return str(p)


class TestSerialCascadeJSON:
    def test_from_json_loads_stages(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        assert len(c.stages) == 2
        assert c.stages[0].name == "visual"
        assert c.stages[0].H == pytest.approx(0.75)

    def test_av_sync_alias_json(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        assert c.stages[1].name == "sync"  # av_sync → sync

    def test_fake_exit_stage1(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.90)
        assert result.is_fake
        assert result.fake_score == pytest.approx(0.90)

    def test_real_exit_stage1(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.10)
        assert not result.is_fake
        assert result.fake_score == pytest.approx(0.10)

    def test_uncertain_passes_to_stage2_fake(self, simple_json_cascade):
        # visual=0.50 → uncertain (0.30 < 0.50 < 0.75)
        # sync=0.80 → FAKE (>= 0.65)
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.50, sync_score=0.80)
        assert result.is_fake
        assert result.fake_score == pytest.approx(0.80)

    def test_uncertain_passes_to_stage2_real(self, simple_json_cascade):
        # visual=0.50 → uncertain
        # sync=0.10 → REAL (<= 0.20)
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.50, sync_score=0.10)
        assert not result.is_fake
        assert result.fake_score == pytest.approx(0.10)

    def test_all_uncertain_returns_fallback(self, simple_json_cascade):
        # visual=0.50 uncertain, sync=0.40 uncertain (0.20 < 0.40 < 0.65)
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.50, sync_score=0.40)
        assert result.fake_score == pytest.approx(0.50)  # fallback
        assert not result.is_fake

    def test_all_none_returns_fallback(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse()
        assert result.fake_score == pytest.approx(0.50)
        assert not result.is_fake
        assert result.modalities_used == 0

    def test_none_stage_skipped(self, simple_json_cascade):
        # visual=None → skip; sync=0.80 → FAKE
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=None, sync_score=0.80)
        assert result.is_fake

    def test_returns_fusion_result_type(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.90)
        assert isinstance(result, FusionResult)

    def test_modalities_used_count(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.90, rppg_score=0.60, sync_score=None)
        assert result.modalities_used == 2

    def test_weights_used_contains_stage_names(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.90)
        assert "visual" in result.weights_used
        assert "sync" in result.weights_used

    def test_scores_dict_has_all_modalities(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.90)
        assert set(result.scores.keys()) == {"visual", "rppg", "sync"}

    def test_threshold_boundary_h_inclusive(self, simple_json_cascade):
        # Exactly at H: should be FAKE
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.75)
        assert result.is_fake

    def test_threshold_boundary_l_inclusive(self, simple_json_cascade):
        # Exactly at L: should be REAL
        c = SerialCascade.from_json(simple_json_cascade)
        result = c.fuse(visual_score=0.30)
        assert not result.is_fake

    def test_repr(self, simple_json_cascade):
        c = SerialCascade.from_json(simple_json_cascade)
        r = repr(c)
        assert "SerialCascade" in r
        assert "visual" in r

    def test_json_with_threshold_H_L_keys(self, tmp_path):
        # Legacy: JSON with threshold_H / threshold_L keys instead of H / L
        spec = {
            "stages": [
                {"modality": "visual", "threshold_H": 0.80, "threshold_L": 0.25},
            ],
            "default_threshold": 0.50,
        }
        p = tmp_path / "alt.json"
        p.write_text(json.dumps(spec))
        c = SerialCascade.from_json(str(p))
        assert c.stages[0].H == pytest.approx(0.80)
        assert c.stages[0].L == pytest.approx(0.25)


# ------------------------------------------------------------------ #
#  SerialCascade — CSV loader (requires pandas)                        #
# ------------------------------------------------------------------ #

@pytest.mark.skipif(not _pandas_available(), reason="pandas not installed")
class TestSerialCascadeCSV:
    def _make_csv(self, rows):
        return _write_tmp_csv(rows)

    def test_from_csv_loads_stages(self):
        path = self._make_csv([
            {"stage_order": 1, "modality": "visual",  "threshold_H": 0.75, "threshold_L": 0.30},
            {"stage_order": 2, "modality": "rppg",    "threshold_H": 0.70, "threshold_L": 0.25},
            {"stage_order": 3, "modality": "av_sync", "threshold_H": 0.65, "threshold_L": 0.20},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert len(c.stages) == 3
        finally:
            os.unlink(path)

    def test_av_sync_normalised_to_sync(self):
        path = self._make_csv([
            {"stage_order": 1, "modality": "visual",  "threshold_H": 0.75, "threshold_L": 0.30},
            {"stage_order": 2, "modality": "av_sync", "threshold_H": 0.65, "threshold_L": 0.20},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert c.stages[1].name == "sync"
        finally:
            os.unlink(path)

    def test_stage_order_respected(self):
        # Write rows in reverse order; should still sort by stage_order
        path = self._make_csv([
            {"stage_order": 2, "modality": "rppg",   "threshold_H": 0.70, "threshold_L": 0.25},
            {"stage_order": 1, "modality": "visual",  "threshold_H": 0.75, "threshold_L": 0.30},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert c.stages[0].name == "visual"
            assert c.stages[1].name == "rppg"
        finally:
            os.unlink(path)

    def test_fake_exit(self):
        path = self._make_csv([
            {"stage_order": 1, "modality": "visual", "threshold_H": 0.75, "threshold_L": 0.30},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert c.fuse(visual_score=0.90).is_fake
        finally:
            os.unlink(path)

    def test_real_exit(self):
        path = self._make_csv([
            {"stage_order": 1, "modality": "visual", "threshold_H": 0.75, "threshold_L": 0.30},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert not c.fuse(visual_score=0.10).is_fake
        finally:
            os.unlink(path)

    def test_fallback_all_none(self):
        path = self._make_csv([
            {"stage_order": 1, "modality": "visual", "threshold_H": 0.75, "threshold_L": 0.30},
        ])
        try:
            c = SerialCascade.from_csv(path)
            assert c.fuse().fake_score == pytest.approx(0.50)
        finally:
            os.unlink(path)

    def test_missing_columns_raises(self):
        path = _write_tmp_csv([
            {"stage_order": 1, "modality": "visual", "threshold_H": 0.75},  # missing threshold_L
        ])
        try:
            with pytest.raises(ValueError, match="missing columns"):
                SerialCascade.from_csv(path)
        finally:
            os.unlink(path)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            SerialCascade.from_csv("/nonexistent/path/settings.csv")


# ------------------------------------------------------------------ #
#  SerialCascade — config dict interface                               #
# ------------------------------------------------------------------ #

class TestSerialCascadeConfig:
    def test_missing_cascade_config_raises(self):
        with pytest.raises(ValueError, match="cascade_config"):
            SerialCascade({"fusion": {"threshold": 0.50}})

    def test_empty_cascade_config_raises(self):
        with pytest.raises(ValueError, match="cascade_config"):
            SerialCascade({"fusion": {"cascade_config": "", "threshold": 0.50}})

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            SerialCascade({"fusion": {"cascade_config": "/no/such/file.json"}})
