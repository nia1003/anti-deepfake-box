# Anti-Deepfake-Box
### Multimodal Real-Time Deepfake Detection System

> **Role:** Research Engineer — ML Pipeline, Benchmark Integration, Edge Deployment  
> **Period:** Feb. 2026 – Present  
> **Stack:** Python · PyTorch · FastAPI · Chrome Extension · DeepfakeBench · LatentSync

---

## Overview

Deepfake video calls are now the primary vector for financial fraud and identity impersonation. **Anti-Deepfake-Box (ADB)** is an end-to-end deepfake detection system targeting the real-time, low-latency scenario of live video calls, while remaining rigorous enough for forensic-grade batch analysis.

The system fuses three independent detection modalities — visual face artifacts, physiological rPPG signals, and audio-visual lip-sync coherence — through a **BMMA-GPT-inspired serial cascade** that selects the cheapest sufficient classifier for each input, avoiding unnecessary computation when early confidence is high.

---

## System Architecture

```
Input Video / Live Stream
        │
        ▼
┌──────────────────────────────────────────┐
│         SSOT Preprocessing               │
│  InsightFace face track  +  FFmpeg WAV   │
│  (forensic: buffalo_l, 128 frames, NPZ)  │
│  (realtime: buffalo_sc,  32 frames)      │
└──────────┬───────────────┬───────────────┘
           │               │
    ┌──────▼──────┐  ┌─────▼──────┐  ┌────────────┐
    │   Visual    │  │    rPPG    │  │    Sync    │
    │ XceptionNet │  │  POS algo  │  │  SyncNet   │
    │  (FF++ c23) │  │(no ckpt)   │  │ + Whisper  │
    └──────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
              ┌────────────────────────┐
              │   Serial Cascade /     │
              │   Weighted Ensemble /  │
              │   Meta-Classifier      │
              └────────────┬───────────┘
                           ▼
                    REAL / FAKE + score
```

---

## Key Components

### 1. Preprocessing Pipeline

Single Source of Truth (SSOT) design: face detection runs **once per video**, and all three detectors share the same aligned face crops without redundant inference.

| Parameter | Forensic Mode | Realtime Mode |
|-----------|--------------|---------------|
| Face model | `buffalo_l` | `buffalo_sc` |
| Frame count | 128 | 32 |
| Pixel crop cache | ✅ NPZ (aligned_256) | ❌ |
| Leading silence skip | 80 ms | 0 ms |
| Whisper model | small | tiny |

Switched via a single CLI flag: `--mode forensic` or `--mode realtime`.

The **80 ms leading-silence skip** was added after identifying that FakeAVCeleb fake audio contains 25–30 ms of leading silence — a known shortcut bias documented in DeepfakeBench-MM §B.1 that causes classifiers to overfit to silence rather than actual artifacts.

---

### 2. Detection Modalities

#### Visual — XceptionNet
- Pre-trained on FaceForensics++ (c23 compression)
- Detects spatial face-swap artifacts and GAN fingerprints
- Operates on per-frame crops at 299×299

#### rPPG — POS Algorithm (Wang 2017)
- **No checkpoint required** — pure NumPy signal processing
- Extracts micro blood-flow signals from facial RGB fluctuations
- Deepfakes suppress physiological variation → detectable by low rPPG SNR
- Calibrated per-dataset via `scripts/calibrate_snr.py`

#### Audio-Visual Sync — LatentSync SyncNet
- Measures temporal coherence between lip movements and speech
- Desynchronisation (e.g., dubbed deepfakes, voice cloning) → high fake score
- Whisper mel spectrogram as audio representation
- Falls back to mouth-region motion heuristic when SyncNet weights unavailable

---

### 3. Fusion Strategy

Three modes available:

**Weighted Ensemble** (default)
```
final_score = α·visual + β·rppg + γ·sync
```
Weights determined empirically from per-modality AUC on benchmark datasets.

**BMMA-GPT Serial Cascade** *(in development)*
- Each stage applies independent high threshold H (→ FAKE early exit) and low threshold L (→ REAL early exit)
- Samples only reach the next stage if score ∈ (L, H)
- Stage ordering and thresholds selected via **Pareto-front search** over FAR/FRR space
- Reduces average inference cost by skipping downstream modalities on confident early exits

**Meta-Classifier**
- MLP trained on fused score vectors
- Supports `scores`, `features`, or `both` input modes

---

### 4. DeepfakeBench Integration

Extended [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) with three drop-in detector adapters conforming to its `AbstractDetector` interface (7 required methods):

| Adapter | Registry key | Notes |
|---------|-------------|-------|
| `adb_visual_detector.py` | `adb_visual` | XceptionNet via ADB |
| `adb_rppg_detector.py` | `adb_rppg` | POS rPPG via ADB |
| `adb_sync_detector.py` | `adb_sync` | SyncNet, on-the-fly audio extraction |

This enables ADB detectors to be evaluated on DFB's standardised cross-dataset AUC protocol alongside 30+ existing detectors.

A `DummyDetector` was also contributed as a minimal integration verification template for future contributors.

---

### 5. Weak Classifier Selection Pipeline

Before committing to a cascade configuration, each candidate weak classifier is evaluated independently:

```
Per-classifier CSV (video_id, fake_score, label)
        ↓
① AUC filter  (drop if AUC < 0.55)
② Correlation matrix  (drop redundant classifiers, corr > 0.9)
③ FAR/FRR curve comparison  (keep complementary operating points)
④ Pareto dominance check  (remove dominated configurations)
        ↓
Final weak classifier set → cascade design
```

The **sync-score CSV bridge** (`scripts/sync_score_csv.py`) produces the input CSV for this pipeline, running SyncDetector on a video directory without requiring the full DFB dataset loader.

---

### 6. Real-Time Edge Deployment

**Chrome Extension**
- Content script detects the largest `<video>` element on any webpage
- Captures frames at 1 FPS via Canvas API, sends base64 JPEG to local server
- Overlays real/fake verdict with per-modality score breakdown

**FastAPI Edge Server**
- Async three-modal inference pipeline (FFT + visual + rPPG)
- Session-aware score smoothing with exponential moving average
- Hardware-adaptive profiling: auto-selects `cloud / jetson_orin / jetson_nano / cpu_only / browser` profile
- WebSocket endpoint for sub-second streaming

---

## Research Context

This project implements and extends the **BMMA-GPT framework** (IEEE TDSC 2025, DOI: 10.1109/TDSC.2025.3620382), which proposes serial cascades of biometric detectors for efficient deepfake screening.

Key extensions beyond the paper:
- Added audio-visual sync as a third modality (paper covers visual + rPPG)
- Forensic/realtime preprocessing switch addressing DFB-MM's leading-silence finding
- Standalone CSV evaluation bridge decoupled from any benchmark framework

---

## Current Status

| Component | Status |
|-----------|--------|
| Visual + rPPG detectors | ✅ Complete |
| Audio extraction + mel features | ✅ Complete |
| SyncNet integration | ✅ Complete (weights optional) |
| Forensic/realtime mode switch | ✅ Complete |
| DeepfakeBench adapters | ✅ Complete |
| Chrome extension + FastAPI | ✅ Complete |
| Weak classifier AUC benchmarking | 🔄 In progress (awaiting dataset) |
| Serial cascade Pareto search | 🔄 Pending classifier scores |
| DFB dataset loader `video_path` fix (P9) | 🔄 Pending team member |
| Fusion weight optimisation | 🔄 Pending AUC results |

---

## Repository Structure

```
anti-deepfake-box/
├── preprocessing/        SSOT face + audio pipeline
├── detectors/            visual / rppg / sync / fft
├── fusion/               weighted ensemble, meta-classifier, serial cascade
├── deepfakebench_adapters/  DFB-compatible wrappers
├── api/                  FastAPI edge server
├── extension/            Chrome extension
├── scripts/              inference, evaluate, sync_score_csv, train_fusion
├── configs/              default, mode_forensic, mode_realtime
└── tests/                audio pipeline unit tests (21 cases)
```
