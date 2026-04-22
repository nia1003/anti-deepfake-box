"""
Anti-Deepfake-Box Edge API
FastAPI server with REST + WebSocket endpoints.

Endpoints:
  GET  /health                 — liveness check
  GET  /profile                — hardware info + active config
  POST /detect/frame           — single base64 frame (browser extension)
  POST /detect/video           — full video file path (server-side)
  WS   /stream/{session_id}    — real-time frame stream

Usage:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

Docker:
  docker-compose up
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from edge.hardware_profile import detect_hardware, load_profile
from preprocessing.face_extractor import UnifiedFaceExtractor, FaceTrack
from preprocessing.audio_extractor import AudioExtractor
from detectors import VisualDetector, RPPGDetector, SyncDetector
from detectors.fft_detector import FFTDetector
from fusion.weighted_ensemble import WeightedEnsemble
from api.stream_handler import SessionManager, FrameResult, laplacian_quality

# ── Startup ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Anti-Deepfake-Box Edge API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded lazily on first request
_cfg: Optional[dict] = None
_face_extractor: Optional[UnifiedFaceExtractor] = None
_audio_extractor: Optional[AudioExtractor] = None
_visual: Optional[VisualDetector] = None
_rppg: Optional[RPPGDetector] = None
_sync: Optional[SyncDetector] = None
_fft: Optional[FFTDetector] = None
_fuser: Optional[WeightedEnsemble] = None
_sessions = SessionManager()
_hw = detect_hardware()


def _get_cfg() -> dict:
    global _cfg
    if _cfg is None:
        profile = os.environ.get("ADB_PROFILE")   # override via env var
        _cfg = load_profile(profile)
    return _cfg


def _ensure_loaded() -> None:
    global _face_extractor, _audio_extractor, _visual, _rppg, _sync, _fft, _fuser
    if _face_extractor is not None:
        return

    cfg = _get_cfg()
    _face_extractor = UnifiedFaceExtractor(cfg.get("preprocessing", {}))
    _audio_extractor = AudioExtractor(cfg.get("preprocessing", {}))
    _fft    = FFTDetector(cfg.get("detectors", {}).get("fft", {}))
    _visual = VisualDetector(cfg.get("detectors", {}).get("visual", {}))
    _rppg   = RPPGDetector(cfg.get("detectors", {}).get("rppg", {}))
    _sync   = SyncDetector(cfg.get("detectors", {}).get("sync", {}))
    _fuser  = WeightedEnsemble(cfg.get("fusion", {}))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _face_track_from_frame(frame_bgr: np.ndarray) -> Optional[FaceTrack]:
    """Wrap a single BGR frame into a FaceTrack for inference."""
    _ensure_loaded()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Crop to 256×256 centre if no face found
    h, w = rgb.shape[:2]
    size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    crop = cv2.resize(rgb[y0:y0+size, x0:x0+size], (256, 256))
    aligned = crop[np.newaxis]   # (1, 256, 256, 3)
    return FaceTrack(
        frame_indices=np.array([0]),
        bboxes=np.array([[0, 0, 256, 256]], dtype=np.float32),
        landmarks=np.zeros((1, 5, 2), dtype=np.float32),
        aligned_256=aligned,
        fps=25.0,
    )


async def _infer_frame(frame_bgr: np.ndarray) -> dict:
    """Run all active detectors on a single frame, return score dict."""
    _ensure_loaded()
    t0 = time.time()
    quality = laplacian_quality(frame_bgr)
    face_track = _face_track_from_frame(frame_bgr)

    # Run FFT (fast, always) and visual in parallel
    fft_score    = await _fft.detect_async(face_track)
    visual_score = await _visual.detect_async(face_track)
    rppg_score   = await _rppg.detect_async(face_track)
    # Sync needs audio — skip for single-frame mode
    sync_score: Optional[float] = None

    raw = _fuser.fuse(visual_score, rppg_score, sync_score)
    # Blend FFT score into fusion result (4th path)
    combined = raw.fake_score
    if fft_score is not None:
        fft_w = _get_cfg().get("detectors", {}).get("fft", {}).get("weight", 0.15)
        combined = (1 - fft_w) * combined + fft_w * fft_score

    return {
        "fake_score":  round(combined, 4),
        "is_fake":     combined >= _get_cfg().get("fusion", {}).get("threshold", 0.5),
        "scores": {
            "visual": visual_score,
            "rppg":   rppg_score,
            "fft":    fft_score,
            "sync":   sync_score,
        },
        "quality":      round(quality, 1),
        "face_detected": True,
        "latency_ms":   round((time.time() - t0) * 1000, 1),
    }


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "sessions": _sessions.active_count()}


@app.get("/profile")
def profile():
    _ensure_loaded()
    cfg = _get_cfg()
    return {
        "hardware": {
            "cpu_cores": _hw.cpu_cores,
            "ram_gb":    round(_hw.ram_gb, 1),
            "gpu":       _hw.gpu_name,
            "vram_gb":   round(_hw.vram_gb, 1) if _hw.vram_gb else None,
            "is_jetson": _hw.is_jetson,
        },
        "profile": _hw.profile_name,
        "device":  cfg.get("device", "cpu"),
        "detectors_enabled": {
            "visual": bool(cfg.get("detectors", {}).get("visual", {}).get("enabled", True)),
            "rppg":   bool(cfg.get("detectors", {}).get("rppg", {}).get("enabled", True)),
            "sync":   bool(cfg.get("detectors", {}).get("sync", {}).get("enabled", True)),
            "fft":    bool(cfg.get("detectors", {}).get("fft", {}).get("enabled", True)),
        },
    }


class FrameRequest(BaseModel):
    image: str          # base64-encoded JPEG/PNG
    session_id: str = "default"


@app.post("/detect/frame")
async def detect_frame(req: FrameRequest):
    """Detect deepfake in a single base64-encoded frame."""
    frame_bgr = _base64_to_bgr(req.image)
    if frame_bgr is None:
        raise HTTPException(400, "Invalid image data")

    result_dict = await _infer_frame(frame_bgr)
    state = _sessions.get_or_create(req.session_id)
    state.update(FrameResult(
        timestamp=time.time(),
        fake_score=result_dict["fake_score"],
        scores=result_dict["scores"],
        quality=result_dict["quality"],
        face_detected=result_dict["face_detected"],
        latency_ms=result_dict["latency_ms"],
    ))
    return state.to_response()


class VideoRequest(BaseModel):
    video_path: str     # path accessible on the server
    async_mode: bool = True


@app.post("/detect/video")
async def detect_video_endpoint(req: VideoRequest):
    """Full video detection (server-side path)."""
    _ensure_loaded()
    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    t0 = time.time()
    face_track = _face_extractor.extract(req.video_path)
    audio_path = _audio_extractor.extract_to_temp(req.video_path)

    if req.async_mode:
        visual_s, rppg_s, sync_s, fft_s = await asyncio.gather(
            _visual.detect_async(face_track),
            _rppg.detect_async(face_track),
            _sync.detect_async(face_track, audio_path),
            _fft.detect_async(face_track),
        )
    else:
        visual_s = _visual.detect(face_track)
        rppg_s   = _rppg.detect(face_track)
        sync_s   = _sync.detect(face_track, audio_path)
        fft_s    = _fft.detect(face_track)

    fusion = _fuser.fuse(visual_s, rppg_s, sync_s)
    fft_w = _get_cfg().get("detectors", {}).get("fft", {}).get("weight", 0.15)
    combined = (1 - fft_w) * fusion.fake_score + (fft_w * fft_s if fft_s else 0)

    return {
        "prediction":  "FAKE" if combined >= 0.5 else "REAL",
        "fake_score":  round(combined, 4),
        "scores": {
            "visual": visual_s, "rppg": rppg_s,
            "sync": sync_s, "fft": fft_s,
        },
        "latency_s": round(time.time() - t0, 2),
    }


# ── WebSocket Stream ──────────────────────────────────────────────────────────

@app.websocket("/stream/{session_id}")
async def websocket_stream(websocket: WebSocket, session_id: str):
    """
    Real-time frame stream.
    Client sends: JSON {"image": "<base64>"} at desired FPS
    Server replies: JSON detection result with smoothed_score
    """
    await websocket.accept()
    state = _sessions.get_or_create(session_id)
    try:
        while True:
            data = await websocket.receive_json()
            frame_bgr = _base64_to_bgr(data.get("image", ""))
            if frame_bgr is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            result_dict = await _infer_frame(frame_bgr)
            state.update(FrameResult(
                timestamp=time.time(),
                fake_score=result_dict["fake_score"],
                scores=result_dict["scores"],
                quality=result_dict["quality"],
                face_detected=result_dict["face_detected"],
                latency_ms=result_dict["latency_ms"],
            ))
            await websocket.send_json(state.to_response())

    except WebSocketDisconnect:
        _sessions.delete(session_id)
