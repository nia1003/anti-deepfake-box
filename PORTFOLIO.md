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

### 4. DeepfakeBench Framework Integration

Extended [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) with drop-in detector adapters conforming to its `AbstractDetector` interface.

#### Adapter Architecture

Every DFB detector must implement 7 abstract methods and register via decorator:

```python
@DETECTOR.register_module(module_name='adb_visual')
class ADBVisualDetector(AbstractDetector):
    def build_backbone(self, config)  → nn.Module   # model architecture
    def build_loss(self, config)      → nn.Module   # loss function
    def features(self, data_dict)     → Tensor      # backbone forward pass
    def classifier(self, features)    → Tensor      # classification head
    def forward(self, data_dict, inference=False) → dict  # {cls, prob, feat}
    def get_losses(self, data_dict, pred_dict)    → dict  # {overall, cls, ...}
    def get_train_metrics(self, data_dict, pred_dict) → dict  # {acc, auc, eer, ap}
```

Plus a YAML config in `training/config/detector/` specifying model name, frame count, loss function, and test datasets.

| Adapter | Registry key | Notes |
|---------|-------------|-------|
| `adb_visual_detector.py` | `adb_visual` | Wraps XceptionNet; converts DFB image batch to FaceTrack |
| `adb_rppg_detector.py` | `adb_rppg` | Wraps POS algorithm; no GPU required |
| `adb_sync_detector.py` | `adb_sync` | Wraps SyncNet; extracts audio on-the-fly from video_path |
| `dummy_detector.py` | `dummy` | Minimal template for future contributors; always predicts 0.5 |

This enables ADB detectors to run on DFB's standardised cross-dataset AUC benchmarking protocol alongside 30+ existing detectors.

#### Known Blocker — `video_path` in `data_dict` (P9)

DFB's standard `data_dict = {image, label}` carries no video path. `adb_sync_detector` calls `_get_audio_path(data_dict)` which returns `None` → sync score always 0.5. Fix requires modifying DFB's dataset loader to inject `video_path`; currently blocked pending a team member's dataset PR.

**Workaround for testing**: pass a mock `data_dict` with `video_path` injected directly, or use `scripts/sync_score_csv.py` which bypasses DFB entirely.

```python
# P9 fix location: DeepfakeBench/training/dataset/abstract_dataset.py
# Add to __getitem__ return dict:
#   "video_path": self.video_list[index]
```

---

### 5. Weak Classifier Selection Pipeline

The cascade design requires empirical evidence that each modality is genuinely informative and non-redundant before committing to a configuration. The selection pipeline runs entirely on per-classifier CSV scores — no architectural changes needed.

#### Stage 0 — Score Generation

Each candidate detector generates its own score CSV via the ADB evaluation pipeline or the standalone CSV bridge:

```bash
# Visual / rPPG scores — via DFB evaluate:
python training/train.py --detector_path config/detector/adb_visual.yaml --phase test

# Sync scores — via ADB standalone (no DFB loader dependency):
python scripts/sync_score_csv.py \
    --input_dir /data/FakeAVCeleb/test \
    --label_csv labels.csv \
    --output results/sync_scores.csv \
    --mode forensic          # 80ms leading-silence skip, whisper=small
```

Output CSV format (matches 曉蓮's GP-solver input):

```
video_id, fake_score, label, inference_ms
RealVideo_001, 0.142310, 0, 38.4
FakeVideo_002, 0.871204, 1, 41.1
```

#### Stage 1–4 — Selection Pipeline

```
Per-classifier CSV (video_id, fake_score, label)
        ↓
① AUC filter         drop if AUC < 0.55  (better than random)
② Correlation matrix  drop if corr > 0.9  (redundant classifiers add no information)
③ FAR/FRR curves     keep complementary operating points
                      (a low-FAR specialist pairs well with a low-FRR generalist)
④ Pareto dominance   remove dominated configurations
                      (a classifier worse at both FAR and FRR than another is excluded)
        ↓
Final weak classifier set → cascade stage ordering
```

The **correlation check** is critical: two detectors that both detect GAN frequency artifacts will be highly correlated and contribute little in combination. An rPPG detector (physiological signal) and a visual detector (spatial artifacts) are structurally independent, making them strong cascade partners.

---

### 6. Serial Cascade & GP-Solver Interface

The BMMA-GPT serial cascade applies per-stage dual thresholds; the optimal thresholds and stage ordering are found by a Pareto-front search over FAR/FRR space.

#### Cascade Decision Logic

```
Stage k receives sample s with score p_k(s):
  p_k(s) ≥ H_k  →  early exit: FAKE    (high confidence fake)
  p_k(s) ≤ L_k  →  early exit: REAL    (high confidence real)
  L_k < p_k(s) < H_k  →  pass to stage k+1
```

The cascade terminates at the last stage with a hard decision regardless of confidence.

#### Pareto-Front Search

For a given ordered set of weak classifiers {C₁, C₂, …, Cₙ} and thresholds {(H₁,L₁), …, (Hₙ,Lₙ)}, the search minimises:

- **FAR** (False Accept Rate) — fake accepted as real  
- **FRR** (False Reject Rate) — real rejected as fake  
- **Average inference cost** — fraction of samples reaching each stage

The Pareto front is the set of threshold configurations where no configuration is strictly better on all three objectives simultaneously. In practice FAR and FRR are weighted by application cost (fraud scenario: FAR >> FRR penalty).

#### Interface with GP Solver (曉蓮)

The CSV bridge (`sync_score_csv.py`) produces the exact input format expected by the team's GP-solver component. The solver takes one CSV per candidate classifier, and outputs:

1. Optimal stage ordering (permutation of classifiers)
2. Per-stage `(H, L)` threshold pair
3. Expected FAR / FRR / cost at the Pareto-optimal point

```
ADB scripts/          →  per-classifier CSVs   →  GP solver  →  cascade config
sync_score_csv.py          {video_id, fake_score,                 {stage_order,
evaluate.py                 label, inference_ms}                   thresholds,
                                                                    pareto_point}
```

The cascade config is then loaded by `fusion/serial_cascade.py` (planned; depends on GP-solver output).

#### Thresholds vs. Weights

The cascade is architecturally distinct from the weighted ensemble:

| | Weighted Ensemble | Serial Cascade |
|---|---|---|
| Computation | All modalities always run | Early exit skips downstream |
| Fusion | α·visual + β·rppg + γ·sync | Per-stage dual-threshold gate |
| Optimisation | Weight grid search on AUC | Pareto search on FAR/FRR/cost |
| Use case | Batch analysis, max accuracy | Real-time, latency-constrained |

---

### 7. Real-Time Edge Deployment

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
