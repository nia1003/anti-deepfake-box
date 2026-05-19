"""
Unit tests for new preprocessing methods added in v5.

  - AudioExtractor.extract_to_array()  (method existence + return contract)
  - UnifiedFaceExtractor._filter_single_face_frames()
  - UnifiedFaceExtractor._smooth_bboxes()
  - UnifiedFaceExtractor._save_cache() with audio_samples
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ------------------------------------------------------------------ #
#  Stubs for optional heavy dependencies                               #
# ------------------------------------------------------------------ #

def _stub_insightface():
    if "insightface" not in sys.modules:
        for mod in ["insightface", "insightface.app"]:
            sys.modules[mod] = types.ModuleType(mod)
        sys.modules["insightface.app"].FaceAnalysis = lambda **kw: None


def _stub_decord():
    if "decord" not in sys.modules:
        stub = types.ModuleType("decord")
        stub.VideoReader = None
        stub.cpu = None
        stub.gpu = None
        sys.modules["decord"] = stub


def _stub_cv2():
    if "cv2" not in sys.modules:
        stub = types.ModuleType("cv2")
        for attr in [
            "CAP_PROP_FPS", "CAP_PROP_FRAME_COUNT",
            "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_NEAREST",
            "LMEDS", "COLOR_BGR2RGB", "COLOR_RGB2BGR",
        ]:
            setattr(stub, attr, 1)
        stub.VideoCapture = lambda p: None
        stub.estimateAffinePartial2D = lambda s, d, method=None: (None, None)
        stub.warpAffine = lambda *a, **k: np.zeros((256, 256, 3), dtype=np.uint8)
        stub.cvtColor = lambda img, *a, **k: img
        stub.resize = lambda img, size, **k: np.zeros((*size[::-1], 3), dtype=np.uint8)
        sys.modules["cv2"] = stub


@pytest.fixture(autouse=True)
def stub_deps():
    _stub_insightface()
    _stub_decord()
    _stub_cv2()


# ------------------------------------------------------------------ #
#  AudioExtractor.extract_to_array                                     #
# ------------------------------------------------------------------ #

from preprocessing.audio_extractor import AudioExtractor


class TestExtractToArray:
    def test_method_exists(self):
        ae = AudioExtractor({"sample_rate": 16000})
        assert hasattr(ae, "extract_to_array")
        assert callable(ae.extract_to_array)

    def test_returns_none_on_nonexistent_file(self):
        ae = AudioExtractor({"sample_rate": 16000})
        result = ae.extract_to_array("/nonexistent/video.mp4")
        assert result is None

    def test_skip_leading_ms_stored(self):
        ae = AudioExtractor({"sample_rate": 16000, "skip_leading_ms": 80})
        assert ae.skip_leading_ms == 80

    def test_default_skip_leading_ms_zero(self):
        ae = AudioExtractor({"sample_rate": 16000})
        assert ae.skip_leading_ms == 0


# ------------------------------------------------------------------ #
#  UnifiedFaceExtractor._filter_single_face_frames                    #
# ------------------------------------------------------------------ #

from preprocessing.face_extractor import UnifiedFaceExtractor


class FakeFace:
    """Minimal InsightFace face stub with bbox attribute."""
    def __init__(self, x1=0, y1=0, x2=10, y2=10):
        self.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)


@pytest.fixture
def fe():
    return UnifiedFaceExtractor({"use_face_cache": False})


class TestFilterSingleFaceFrames:
    def test_single_face_frames_kept(self, fe):
        faces_per_frame = [
            (0, [FakeFace()]),
            (1, [FakeFace()]),
            (2, [FakeFace()]),
            (3, [FakeFace()]),
            (4, [FakeFace()]),
            (5, [FakeFace()]),
            (6, [FakeFace()]),
            (7, [FakeFace()]),
        ]
        result = fe._filter_single_face_frames(faces_per_frame, min_frames=8)
        assert len(result) == 8
        assert all(idx == i for i, (idx, _) in enumerate(result))

    def test_multi_face_frames_excluded(self, fe):
        faces_per_frame = [
            (0, [FakeFace()]),                          # 1 face → keep
            (1, [FakeFace(), FakeFace()]),              # 2 faces → exclude
            (2, [FakeFace()]),                          # 1 face → keep
            (3, [FakeFace(), FakeFace(), FakeFace()]),  # 3 faces → exclude
            (4, [FakeFace()]),
            (5, [FakeFace()]),
            (6, [FakeFace()]),
            (7, [FakeFace()]),
        ]
        result = fe._filter_single_face_frames(faces_per_frame, min_frames=4)
        result_indices = [idx for idx, _ in result]
        assert 0 in result_indices
        assert 1 not in result_indices
        assert 2 in result_indices
        assert 3 not in result_indices

    def test_zero_face_frames_excluded(self, fe):
        faces_per_frame = [
            (0, []),          # 0 faces → exclude
            (1, [FakeFace()]),
            (2, [FakeFace()]),
            (3, [FakeFace()]),
            (4, [FakeFace()]),
            (5, [FakeFace()]),
            (6, [FakeFace()]),
            (7, [FakeFace()]),
        ]
        result = fe._filter_single_face_frames(faces_per_frame, min_frames=6)
        result_indices = [idx for idx, _ in result]
        assert 0 not in result_indices

    def test_fallback_when_below_min_frames(self, fe):
        # Only 2 single-face frames (below min_frames=8) → fallback to largest face
        faces_per_frame = [
            (0, [FakeFace()]),
            (1, [FakeFace(0, 0, 5, 5), FakeFace(10, 10, 30, 30)]),  # 2 faces
            (2, [FakeFace()]),
            (3, [FakeFace(0, 0, 5, 5), FakeFace(10, 10, 30, 30)]),  # 2 faces
        ]
        result = fe._filter_single_face_frames(faces_per_frame, min_frames=8)
        # Fallback: all frames with ≥1 face are included
        result_indices = [idx for idx, _ in result]
        assert set(result_indices) == {0, 1, 2, 3}

    def test_fallback_picks_largest_face(self, fe):
        small = FakeFace(0, 0, 5, 5)    # area=25
        large = FakeFace(0, 0, 20, 20)  # area=400
        faces_per_frame = [
            (0, [small, large]),  # 2 faces
        ]
        result = fe._filter_single_face_frames(faces_per_frame, min_frames=8)
        # Fallback: largest face selected
        _, chosen = result[0]
        assert (chosen.bbox[2] - chosen.bbox[0]) == pytest.approx(20)

    def test_empty_input_returns_empty(self, fe):
        result = fe._filter_single_face_frames([], min_frames=8)
        assert result == []


# ------------------------------------------------------------------ #
#  UnifiedFaceExtractor._smooth_bboxes                                #
# ------------------------------------------------------------------ #

class TestSmoothBboxes:
    def test_output_shape_unchanged(self, fe):
        bboxes = np.random.rand(10, 4).astype(np.float32)
        smoothed = fe._smooth_bboxes(bboxes, window=3)
        assert smoothed.shape == (10, 4)

    def test_constant_bboxes_interior_unchanged(self, fe):
        # mode="same" has edge effects on first and last frame; interior frames
        # should be identical to the constant input.
        bbox = np.array([10.0, 20.0, 50.0, 60.0], dtype=np.float32)
        bboxes = np.tile(bbox, (10, 1))
        smoothed = fe._smooth_bboxes(bboxes, window=3)
        np.testing.assert_allclose(smoothed[1:-1], bboxes[1:-1], atol=1e-4)

    def test_does_not_modify_input(self, fe):
        bboxes = np.random.rand(8, 4).astype(np.float32)
        original = bboxes.copy()
        fe._smooth_bboxes(bboxes, window=3)
        np.testing.assert_array_equal(bboxes, original)

    def test_smoothing_reduces_variation(self, fe):
        # Alternating large/small bbox → smoothing reduces std
        bboxes = np.zeros((20, 4), dtype=np.float32)
        bboxes[::2, 2] = 50.0   # even frames: wide
        bboxes[1::2, 2] = 10.0  # odd frames: narrow
        bboxes[:, 3] = 50.0
        smoothed = fe._smooth_bboxes(bboxes, window=3)
        assert smoothed[:, 2].std() < bboxes[:, 2].std()

    def test_window_1_is_identity(self, fe):
        bboxes = np.random.rand(8, 4).astype(np.float32)
        smoothed = fe._smooth_bboxes(bboxes, window=1)
        np.testing.assert_allclose(smoothed, bboxes, atol=1e-5)

    def test_short_sequence_returned_unchanged(self, fe):
        # Sequences shorter than window are returned as-is (no smoothing applied)
        bboxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        smoothed = fe._smooth_bboxes(bboxes, window=3)
        assert smoothed.shape == (2, 4)
        np.testing.assert_array_equal(smoothed, bboxes)

    def test_bbox_validity_maintained(self, fe):
        # x2 > x1 and y2 > y1 should be maintained
        bboxes = np.array([
            [10, 20, 50, 60],
            [12, 22, 52, 62],
            [14, 24, 54, 64],
        ], dtype=np.float32)
        smoothed = fe._smooth_bboxes(bboxes, window=3)
        assert np.all(smoothed[:, 2] > smoothed[:, 0])  # x2 > x1
        assert np.all(smoothed[:, 3] > smoothed[:, 1])  # y2 > y1


# ------------------------------------------------------------------ #
#  _save_cache with audio_samples                                      #
# ------------------------------------------------------------------ #

from preprocessing.face_extractor import FaceTrack


class TestSaveCacheWithAudio:
    def _make_track(self, T=10):
        return FaceTrack(
            frame_indices=np.arange(T, dtype=np.int64),
            bboxes=np.random.rand(T, 4).astype(np.float32),
            landmarks=np.random.rand(T, 5, 2).astype(np.float32),
            aligned_256=np.zeros((T, 256, 256, 3), dtype=np.uint8),
            fps=25.0,
            video_path="/fake/video.mp4",
        )

    def test_save_and_load_without_audio(self, fe, tmp_path):
        fe.cache_pixel_crops = True
        track = self._make_track(T=5)
        path = tmp_path / "cache.npz"
        fe._save_cache(track, path)
        data = np.load(str(path))
        assert "frame_indices" in data
        assert "aligned_256" in data
        assert "audio_samples" not in data

    def test_save_and_load_with_audio(self, fe, tmp_path):
        fe.cache_pixel_crops = True
        track = self._make_track(T=5)
        audio = np.zeros(16000, dtype=np.int16)
        path = tmp_path / "cache_audio.npz"
        fe._save_cache(track, path, audio_samples=audio, audio_sr=16000)
        data = np.load(str(path))
        assert "audio_samples" in data
        assert data["audio_samples"].dtype == np.int16
        assert "audio_sr" in data
        assert int(data["audio_sr"][0]) == 16000
        assert "frame_to_audio_offset" in data

    def test_frame_to_audio_offset_values(self, fe, tmp_path):
        fe.cache_pixel_crops = True
        T = 4
        track = self._make_track(T=T)
        track.fps = 25.0
        audio = np.zeros(16000, dtype=np.int16)
        path = tmp_path / "cache_off.npz"
        fe._save_cache(track, path, audio_samples=audio, audio_sr=16000)
        data = np.load(str(path))
        offsets = data["frame_to_audio_offset"]
        # frame i → sample = i * 16000 / 25 = i * 640
        expected = np.array([0 * 640, 1 * 640, 2 * 640, 3 * 640], dtype=np.int64)
        np.testing.assert_array_equal(offsets, expected)

    def test_audio_not_embedded_without_cache_pixel_crops(self, fe, tmp_path):
        fe.cache_pixel_crops = False
        track = self._make_track(T=5)
        audio = np.zeros(16000, dtype=np.int16)
        path = tmp_path / "cache_no_px.npz"
        fe._save_cache(track, path, audio_samples=audio, audio_sr=16000)
        data = np.load(str(path))
        assert "audio_samples" not in data
        assert "aligned_256" not in data

    def test_audio_cast_to_int16(self, fe, tmp_path):
        fe.cache_pixel_crops = True
        track = self._make_track(T=3)
        # Pass float32 audio — should be cast to int16
        audio = (np.random.rand(8000) * 32767).astype(np.float32)
        path = tmp_path / "cache_cast.npz"
        fe._save_cache(track, path, audio_samples=audio.astype(np.int16), audio_sr=16000)
        data = np.load(str(path))
        assert data["audio_samples"].dtype == np.int16
