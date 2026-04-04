"""
Single Source of Truth (SSOT) Face Extraction Pipeline.

One face detection pass using InsightFace; all three detectors receive
pre-aligned crops at their required resolution without redundant inference.
Disk cache (.npz) stores bboxes + landmarks only (not pixel data) to save
storage while still accelerating repeat inference on the same video.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

try:
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


@dataclass
class FaceTrack:
    """
    Unified face tracking result for one video clip.

    All three detectors share the same aligned_256 base and derive their
    required resolution via on-demand resize, avoiding triple detection cost.
    """
    frame_indices: np.ndarray        # (T,) original frame indices
    bboxes: np.ndarray               # (T, 4) xyxy
    landmarks: np.ndarray            # (T, 5, 2) five keypoints
    aligned_256: np.ndarray          # (T, 256, 256, 3) uint8, base resolution
    fps: float = 25.0
    video_path: str = ""

    @property
    def T(self) -> int:
        return len(self.frame_indices)

    def crops_for_resolution(self, size: int) -> np.ndarray:
        """Return (T, size, size, 3) uint8 crops via resize from aligned_256."""
        if size == 256:
            return self.aligned_256
        out = np.empty((self.T, size, size, 3), dtype=np.uint8)
        for i, frame in enumerate(self.aligned_256):
            out[i] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        return out

    @property
    def crops_299(self) -> np.ndarray:
        """(T, 299, 299, 3) for XceptionNet."""
        return self.crops_for_resolution(299)

    @property
    def crops_128(self) -> np.ndarray:
        """(T, 128, 128, 3) for PhysNet."""
        return self.crops_for_resolution(128)

    @property
    def crops_256(self) -> np.ndarray:
        """(T, 256, 256, 3) for SyncNet."""
        return self.aligned_256

    def to_float32_chw(self, size: int = 256,
                       mean: Tuple = (0.5, 0.5, 0.5),
                       std: Tuple = (0.5, 0.5, 0.5)) -> np.ndarray:
        """(T, 3, H, W) float32 normalised tensor-ready array."""
        crops = self.crops_for_resolution(size).astype(np.float32) / 255.0
        crops = (crops - np.array(mean)) / np.array(std)
        return crops.transpose(0, 3, 1, 2)  # (T, 3, H, W)


class UnifiedFaceExtractor:
    """
    Single-pass face extraction using InsightFace with disk caching.

    Usage
    -----
    extractor = UnifiedFaceExtractor(config)
    track = extractor.extract("video.mp4")
    # track.crops_299  → XceptionNet input
    # track.crops_128  → PhysNet input
    # track.crops_256  → SyncNet input
    """

    CACHE_VERSION = "v1"

    def __init__(self, config: dict):
        self.fps_target: float = config.get("fps_target", 25.0)
        self.batch_size: int = config.get("insightface_batch_size", 32)
        self.model_name: str = config.get("insightface_model", "buffalo_sc")
        self.cache_dir: Path = Path(config.get("face_cache_dir", ".face_cache"))
        self.use_cache: bool = config.get("use_face_cache", True)
        self.min_face_size: int = config.get("min_face_size", 30)

        self._app: Optional[FaceAnalysis] = None

    def _init_app(self) -> FaceAnalysis:
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError(
                "insightface is required. Install with: pip install insightface"
            )
        app = FaceAnalysis(
            name=self.model_name,
            allowed_modules=["detection", "landmark_2d_106"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    @property
    def app(self) -> FaceAnalysis:
        if self._app is None:
            self._app = self._init_app()
        return self._app

    # ------------------------------------------------------------------ #
    #  Cache helpers                                                       #
    # ------------------------------------------------------------------ #

    def _cache_key(self, video_path: str) -> str:
        abs_path = str(Path(video_path).resolve())
        mtime = str(Path(video_path).stat().st_mtime) if Path(video_path).exists() else ""
        h = hashlib.sha256(f"{abs_path}:{mtime}:{self.fps_target}:{self.CACHE_VERSION}".encode()).hexdigest()[:16]
        return h

    def _cache_path(self, video_path: str) -> Path:
        return self.cache_dir / f"{self._cache_key(video_path)}.npz"

    def _save_cache(self, track: FaceTrack, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store lightweight metadata only (no pixel data)
        np.savez_compressed(
            str(path),
            frame_indices=track.frame_indices,
            bboxes=track.bboxes,
            landmarks=track.landmarks,
            fps=np.array([track.fps]),
            video_path=np.array([track.video_path]),
        )

    def _load_cache_meta(self, path: Path, video_path: str) -> Optional[dict]:
        """Load cached bboxes/landmarks; pixel data must be re-extracted."""
        try:
            data = np.load(str(path), allow_pickle=True)
            return {
                "frame_indices": data["frame_indices"],
                "bboxes": data["bboxes"],
                "landmarks": data["landmarks"],
                "fps": float(data["fps"][0]),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Video reading                                                       #
    # ------------------------------------------------------------------ #

    def _read_frames(self, video_path: str) -> Tuple[np.ndarray, float, List[int]]:
        """
        Read frames at fps_target, return (frames_bgr, actual_fps, frame_indices).
        frames_bgr: (N, H, W, 3) uint8 BGR
        """
        if DECORD_AVAILABLE:
            return self._read_frames_decord(video_path)
        return self._read_frames_cv2(video_path)

    def _read_frames_decord(self, video_path: str) -> Tuple[np.ndarray, float, List[int]]:
        vr = VideoReader(video_path, ctx=cpu(0))
        native_fps = vr.get_avg_fps()
        total = len(vr)
        step = max(1, round(native_fps / self.fps_target))
        indices = list(range(0, total, step))
        frames_rgb = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) RGB
        frames_bgr = frames_rgb[:, :, :, ::-1].copy()
        return frames_bgr, native_fps, indices

    def _read_frames_cv2(self, video_path: str) -> Tuple[np.ndarray, float, List[int]]:
        cap = cv2.VideoCapture(video_path)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(native_fps / self.fps_target))
        indices = list(range(0, total, step))
        frames = []
        idx_set = set(indices)
        fi = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if fi in idx_set:
                frames.append(frame)
            fi += 1
        cap.release()
        return np.stack(frames) if frames else np.empty((0, 0, 0, 3), np.uint8), native_fps, indices[:len(frames)]

    # ------------------------------------------------------------------ #
    #  Face detection & alignment                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _get_affine_matrix(landmarks_5pt: np.ndarray, output_size: int = 256) -> np.ndarray:
        """
        Compute affine transform mapping 5 facial landmarks to canonical positions.
        Reference template matches InsightFace / ArcFace alignment standard.
        """
        TEMPLATE = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ], dtype=np.float32)
        scale = output_size / 112.0
        dst = TEMPLATE * scale
        src = landmarks_5pt.astype(np.float32)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if M is None:
            M = np.eye(2, 3, dtype=np.float32)
        return M

    def _detect_and_align_batch(
        self, frames_bgr: np.ndarray
    ) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
        """
        Detect faces in batches, return aligned crops + bboxes + landmarks.

        Returns
        -------
        aligned : list of (256,256,3) uint8 or None if no face found
        bboxes  : list of (4,) or None
        lmks    : list of (5,2) or None
        """
        aligned_list: List[Optional[np.ndarray]] = []
        bbox_list: List[Optional[np.ndarray]] = []
        lmk_list: List[Optional[np.ndarray]] = []

        N = len(frames_bgr)
        for start in range(0, N, self.batch_size):
            batch = frames_bgr[start: start + self.batch_size]
            for frame_bgr in batch:
                faces = self.app.get(frame_bgr)
                if not faces:
                    aligned_list.append(None)
                    bbox_list.append(None)
                    lmk_list.append(None)
                    continue

                # Pick the largest face
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                bbox = face.bbox.astype(np.float32)

                # 5-point landmarks from kps attribute
                if hasattr(face, "kps") and face.kps is not None:
                    lmk5 = face.kps.astype(np.float32)
                else:
                    # Fallback: derive from bbox corners
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    lmk5 = np.array([
                        [x1 + w * 0.3, y1 + h * 0.35],
                        [x1 + w * 0.7, y1 + h * 0.35],
                        [x1 + w * 0.5, y1 + h * 0.55],
                        [x1 + w * 0.35, y1 + h * 0.75],
                        [x1 + w * 0.65, y1 + h * 0.75],
                    ], dtype=np.float32)

                M = self._get_affine_matrix(lmk5, output_size=256)
                aligned = cv2.warpAffine(frame_bgr, M, (256, 256), flags=cv2.INTER_LINEAR)
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

                aligned_list.append(aligned_rgb)
                bbox_list.append(bbox)
                lmk_list.append(lmk5)

        return aligned_list, bbox_list, lmk_list

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, video_path: str) -> Optional[FaceTrack]:
        """
        Extract face track from a video.

        Uses disk cache for bbox/landmark metadata when available.
        Always re-extracts pixel crops (not cached) for memory efficiency.

        Returns None if no face is detected in any frame.
        """
        cache_path = self._cache_path(video_path)

        # Read frames (always needed for pixel data)
        frames_bgr, native_fps, raw_indices = self._read_frames(video_path)
        if len(frames_bgr) == 0:
            return None

        # Try loading metadata cache
        meta = None
        if self.use_cache and cache_path.exists():
            meta = self._load_cache_meta(cache_path, video_path)

        if meta is not None:
            # Cache hit: re-align using cached landmarks (fast path)
            cached_indices = meta["frame_indices"]
            cached_lmks = meta["landmarks"]  # (T, 5, 2)
            cached_bboxes = meta["bboxes"]

            # Map cached frame indices to positions in frames_bgr
            idx_map = {v: i for i, v in enumerate(raw_indices)}
            valid_positions = [idx_map[fi] for fi in cached_indices if fi in idx_map]

            if not valid_positions:
                return None

            aligned_list = []
            for pos, lmk5 in zip(valid_positions, cached_lmks):
                M = self._get_affine_matrix(lmk5, output_size=256)
                aligned = cv2.warpAffine(frames_bgr[pos], M, (256, 256), flags=cv2.INTER_LINEAR)
                aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
                aligned_list.append(aligned_rgb)

            aligned_256 = np.stack(aligned_list)
            track = FaceTrack(
                frame_indices=cached_indices[:len(valid_positions)],
                bboxes=cached_bboxes[:len(valid_positions)],
                landmarks=cached_lmks[:len(valid_positions)],
                aligned_256=aligned_256,
                fps=meta["fps"],
                video_path=video_path,
            )
            return track

        # Cache miss: run full InsightFace detection
        aligned_list, bbox_list, lmk_list = self._detect_and_align_batch(frames_bgr)

        # Filter frames where detection succeeded
        valid_mask = [a is not None for a in aligned_list]
        if not any(valid_mask):
            return None

        valid_indices = np.array([raw_indices[i] for i, v in enumerate(valid_mask) if v])
        valid_bboxes = np.stack([bbox_list[i] for i, v in enumerate(valid_mask) if v])
        valid_lmks = np.stack([lmk_list[i] for i, v in enumerate(valid_mask) if v])
        valid_aligned = np.stack([aligned_list[i] for i, v in enumerate(valid_mask) if v])

        track = FaceTrack(
            frame_indices=valid_indices,
            bboxes=valid_bboxes,
            landmarks=valid_lmks,
            aligned_256=valid_aligned,
            fps=self.fps_target,
            video_path=video_path,
        )

        if self.use_cache:
            self._save_cache(track, cache_path)

        return track
