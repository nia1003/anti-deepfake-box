"""
Anti-Deepfake-Box Edge API
FastAPI server with REST + WebSocket endpoints.

Endpoints:
  GET  /health                 — liveness check
  GET  /profile                — hardware info + active config
  GET  /modes                  — list available detection modes with metadata
  POST /mode                   — set the global default detection mode
  GET  /sessions               — all active session states (for dashboard polling)
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
import copy
import io
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Serve the web dashboard from /web directory
_web_dir = ROOT / "web"
if _web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(_web_dir)), name="web")

# Loaded lazily on first request
_cfg: Optional[dict] = None
_face_extractor: Optional[UnifiedFaceExtractor] = None
_audio_extractor: Optional[AudioExtractor] = None
_visual: Optional[VisualDetector] = None
_rppg: Optional[RPPGDetector] = None
_sync: Optional[SyncDetector] = None
_fft: Optional[FFTDetector] = None

# Mode-specific fusers (created in _ensure_loaded)
_fusers: Dict[str, WeightedEnsemble] = {}

# Mode configs loaded from configs/modes/*.yaml
_mode_configs: Dict[str, dict] = {}

# Global default mode (can be overridden per-request or via POST /mode)
_default_mode: str = "online"

_sessions = SessionManager()
_hw = detect_hardware()


def _get_cfg() -> dict:
    global _cfg
    if _cfg is None:
        profile = os.environ.get("ADB_PROFILE")
        _cfg = load_profile(profile)
    return _cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_mode_configs() -> None:
    """Read configs/modes/*.yaml and build per-mode merged configs + fusers."""
    base_cfg = _get_cfg()
    modes_dir = ROOT / "configs" / "modes"
    for path in sorted(modes_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        mode_name = raw.get("mode_name", path.stem)
        merged = _deep_merge(base_cfg, raw)
        _mode_configs[mode_name] = merged
        # Build a WeightedEnsemble for this mode using the merged fusion config
        _fusers[mode_name] = WeightedEnsemble(merged)

    # Always ensure both canonical modes exist (fall back to base config)
    for fallback in ("offline", "online"):
        if fallback not in _fusers:
            _fusers[fallback] = WeightedEnsemble(base_cfg)
            _mode_configs[fallback] = base_cfg


def _ensure_loaded() -> None:
    global _face_extractor, _audio_extractor, _visual, _rppg, _sync, _fft

    if _face_extractor is not None:
        return

    cfg = _get_cfg()
    _load_mode_configs()

    _face_extractor = UnifiedFaceExtractor(cfg.get("preprocessing", {}))
    _audio_extractor = AudioExtractor(cfg.get("preprocessing", {}))
    _fft    = FFTDetector(cfg.get("detectors", {}).get("fft", {}))
    _visual = VisualDetector(cfg.get("detectors", {}).get("visual", {}))
    _rppg   = RPPGDetector(cfg.get("detectors", {}).get("rppg", {}))
    _sync   = SyncDetector(cfg.get("detectors", {}).get("sync", {}))


def _get_fuser(mode: str) -> WeightedEnsemble:
    """Return the WeightedEnsemble for the given mode, defaulting to online."""
    return _fusers.get(mode, _fusers.get("online", WeightedEnsemble({})))


def _get_mode_threshold(mode: str) -> float:
    mcfg = _mode_configs.get(mode, {})
    return float(mcfg.get("fusion", {}).get("threshold", 0.50))


def _sync_enabled_for_mode(mode: str) -> bool:
    mcfg = _mode_configs.get(mode, {})
    return bool(mcfg.get("detectors", {}).get("sync", {}).get("enabled", True))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base64_to_bgr(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _face_track_from_frame(frame_bgr: np.ndarray) -> Optional[FaceTrack]:
    """Wrap a single BGR frame into a FaceTrack for inference."""
    _ensure_loaded()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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


async def _infer_frame(frame_bgr: np.ndarray, mode: str = "online") -> dict:
    """Run active detectors on a single frame and return a score dict."""
    _ensure_loaded()
    t0 = time.time()
    quality = laplacian_quality(frame_bgr)
    face_track = _face_track_from_frame(frame_bgr)

    fft_score    = await _fft.detect_async(face_track)
    visual_score = await _visual.detect_async(face_track)
    rppg_score   = await _rppg.detect_async(face_track)
    # Sync requires audio — unavailable in single-frame mode
    sync_score: Optional[float] = None

    fuser = _get_fuser(mode)
    threshold = _get_mode_threshold(mode)
    raw = fuser.fuse(visual_score, rppg_score, sync_score)

    fft_w = _mode_configs.get(mode, _get_cfg()).get("detectors", {}).get("fft", {}).get("weight", 0.15)
    combined = raw.fake_score
    if fft_score is not None:
        combined = (1 - fft_w) * combined + fft_w * fft_score

    return {
        "fake_score":    round(combined, 4),
        "is_fake":       combined >= threshold,
        "threshold":     threshold,
        "mode":          mode,
        "scores": {
            "visual": visual_score,
            "rppg":   rppg_score,
            "fft":    fft_score,
            "sync":   sync_score,
        },
        "quality":       round(quality, 1),
        "face_detected": True,
        "latency_ms":    round((time.time() - t0) * 1000, 1),
    }


# ── REST Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = _web_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Anti-Deepfake-Box API", "docs": "/docs"}


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
        "profile":          _hw.profile_name,
        "device":           cfg.get("device", "cpu"),
        "default_mode":     _default_mode,
        "detectors_enabled": {
            "visual": bool(cfg.get("detectors", {}).get("visual", {}).get("enabled", True)),
            "rppg":   bool(cfg.get("detectors", {}).get("rppg", {}).get("enabled", True)),
            "sync":   bool(cfg.get("detectors", {}).get("sync", {}).get("enabled", True)),
            "fft":    bool(cfg.get("detectors", {}).get("fft", {}).get("enabled", True)),
        },
    }


@app.get("/modes")
def list_modes():
    """Return metadata for all available detection modes."""
    _ensure_loaded()
    result = {}
    for name, mcfg in _mode_configs.items():
        fusion_cfg = mcfg.get("fusion", {})
        det_cfg = mcfg.get("detectors", {})
        result[name] = {
            "label":       mcfg.get("mode_label", name.capitalize()),
            "description": mcfg.get("mode_description", ""),
            "threshold":   fusion_cfg.get("threshold", 0.50),
            "weights":     fusion_cfg.get("weights", {}),
            "detectors": {
                k: bool(det_cfg.get(k, {}).get("enabled", True))
                for k in ("visual", "rppg", "sync", "fft")
            },
        }
    return result


class ModeRequest(BaseModel):
    mode: str


@app.post("/mode")
def set_mode(req: ModeRequest):
    """Set the global default detection mode (online | offline)."""
    global _default_mode
    _ensure_loaded()
    if req.mode not in _mode_configs:
        raise HTTPException(400, f"Unknown mode '{req.mode}'. Available: {list(_mode_configs)}")
    _default_mode = req.mode
    return {"mode": _default_mode}


@app.get("/sessions")
def list_sessions():
    """Return all active session states for live dashboard polling."""
    return _sessions.all_states()


class FrameRequest(BaseModel):
    image: str           # base64-encoded JPEG/PNG
    session_id: str = "default"
    mode: str = ""       # empty → use global default


@app.post("/detect/frame")
async def detect_frame(req: FrameRequest):
    """Detect deepfake in a single base64-encoded frame."""
    frame_bgr = _base64_to_bgr(req.image)
    if frame_bgr is None:
        raise HTTPException(400, "Invalid image data")

    mode = req.mode if req.mode else _default_mode
    result_dict = await _infer_frame(frame_bgr, mode=mode)
    state = _sessions.get_or_create(req.session_id)
    state.update(FrameResult(
        timestamp=time.time(),
        fake_score=result_dict["fake_score"],
        scores=result_dict["scores"],
        quality=result_dict["quality"],
        face_detected=result_dict["face_detected"],
        latency_ms=result_dict["latency_ms"],
    ))
    resp = state.to_response()
    resp["mode"] = mode
    resp["threshold"] = result_dict["threshold"]
    return resp


class VideoRequest(BaseModel):
    video_path: str      # path accessible on the server
    async_mode: bool = True
    mode: str = ""       # empty → use global default


@app.post("/detect/video")
async def detect_video_endpoint(req: VideoRequest):
    """Full video detection (server-side path)."""
    _ensure_loaded()
    if not Path(req.video_path).exists():
        raise HTTPException(404, f"Video not found: {req.video_path}")

    mode = req.mode if req.mode else _default_mode
    t0 = time.time()
    face_track = _face_extractor.extract(req.video_path)
    audio_path = _audio_extractor.extract_to_temp(req.video_path)

    run_sync = _sync_enabled_for_mode(mode)

    if req.async_mode:
        tasks = [
            _visual.detect_async(face_track),
            _rppg.detect_async(face_track),
            _sync.detect_async(face_track, audio_path) if run_sync else asyncio.coroutine(lambda: None)(),
            _fft.detect_async(face_track),
        ]
        visual_s, rppg_s, sync_s, fft_s = await asyncio.gather(*tasks)
    else:
        visual_s = _visual.detect(face_track)
        rppg_s   = _rppg.detect(face_track)
        sync_s   = _sync.detect(face_track, audio_path) if run_sync else None
        fft_s    = _fft.detect(face_track)

    fuser = _get_fuser(mode)
    threshold = _get_mode_threshold(mode)
    fusion = fuser.fuse(visual_s, rppg_s, sync_s)

    fft_w = _mode_configs.get(mode, _get_cfg()).get("detectors", {}).get("fft", {}).get("weight", 0.15)
    combined = (1 - fft_w) * fusion.fake_score + (fft_w * fft_s if fft_s is not None else 0)

    return {
        "prediction": "FAKE" if combined >= threshold else "REAL",
        "fake_score": round(combined, 4),
        "threshold":  threshold,
        "mode":       mode,
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
    Client sends: JSON {"image": "<base64>", "mode": "online"}
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

            mode = data.get("mode", _default_mode)
            result_dict = await _infer_frame(frame_bgr, mode=mode)
            state.update(FrameResult(
                timestamp=time.time(),
                fake_score=result_dict["fake_score"],
                scores=result_dict["scores"],
                quality=result_dict["quality"],
                face_detected=result_dict["face_detected"],
                latency_ms=result_dict["latency_ms"],
            ))
            resp = state.to_response()
            resp["mode"] = mode
            resp["threshold"] = result_dict["threshold"]
            await websocket.send_json(resp)

    except WebSocketDisconnect:
        _sessions.delete(session_id)
