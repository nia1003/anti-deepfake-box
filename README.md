# Anti-Deepfake-Box

**三路多模態 Deepfake 偵測框架**：整合視覺紋理 (FaceForensics++ XceptionNet)、生理訊號 (rPPG-Toolbox PhysNet + SNR) 與音視訊同步 (LatentSync SyncNet) 三路互補偵測訊號，透過可設定融合策略輸出統一 fake score，並以 DeepfakeBench 進行跨方法對齊分析驗證。

---

## 框架架構

```
video.mp4
    │
    ▼ [PASS 1 — 唯一一次人臉偵測]
preprocessing/face_extractor.py
  └─ InsightFace (buffalo_sc)  ← SSOT 快取 (bboxes + landmarks .npz)
       └─ FaceTrack (aligned_256)
              │
              ├──[resize 299×299]──▶ detectors/visual_detector.py  → visual_score
              │                      XceptionNet (FaceForensics++ c23 pretrained)
              │
              ├──[resize 128×128]──▶ detectors/rppg_detector.py    → rppg_score
              │                      PhysNet → PPG waveform → SNR → fake score
              │                      CHROM fallback (T < 30 frames)
              │
              └──[256×256]──────▶  detectors/sync_detector.py     → sync_score
                                    + audio (FFmpeg → Whisper → mel)
                                    LatentSync SyncNet → 1 − sync_confidence
                                    returns None if no audio track

    ▼ [非同步並行管線]
    asyncio.gather(face_extraction, audio_extraction)   # 完全並行
    asyncio.gather(visual_det, rppg_det, sync_det)      # 三路並行

    ▼ fusion/weighted_ensemble.py  (or meta_classifier.py)
      fake_score = Σ wᵢ·scoreᵢ / Σ wᵢ
      (None → 自動排除，weights 重歸一化)
      Prediction: FAKE / REAL
```

---

## 三路偵測原理

| 模態 | 來源 | 偵測原理 | 優勢 | 侷限 |
|------|------|----------|------|------|
| **視覺/紋理** | FaceForensics++ XceptionNet | 空間偽影、GAN/擴散模型壓縮痕跡 | 高 AUC on FF++ baseline | 高壓縮易失效 |
| **生理/PPG** | rPPG-Toolbox PhysNet+SNR | 真人有週期 PPG 訊號，deepfake 無 | 模態獨立，不受視覺欺騙 | 高品質 deepfake 可能繞過 |
| **音視訊同步** | LatentSync SyncNet | lip-sync 偵測音視訊時序不一致 | 針對 lip-sync deepfake 有效 | 無音訊時不可用，graceful degradation |

---

## 快速開始

### 安裝

```bash
git clone https://github.com/nia1003/anti-deepfake-box.git
cd anti-deepfake-box
git checkout claude/integrate-deepfake-detection-frameworks-Uvq1K

pip install -r requirements.txt

# 設定第三方子模組
git submodule update --init --recursive
# 或：
pip install -e third_party/rppg-toolbox
```

### 準備模型權重

```
checkpoints/
├── xception_ff_c23.pth        # FF++ XceptionNet (nia1003/faceforensics)
├── physnet_ubfc.pth           # PhysNet (nia1003/rppg-toolbox)
└── latentsync_syncnet.pth     # SyncNet (nia1003/latentsync)
```

更新 `configs/default.yaml` 中各 `pretrained` / `syncnet_path` 欄位。

### 單影片推理

```bash
# 同步推理（適合單次測試）
python scripts/inference.py --video sample.mp4 --config configs/default.yaml

# 非同步推理（音訊+人臉並行提取，更快）
python scripts/inference.py --video sample.mp4 --async_mode

# 跳過無音訊的 sync 偵測器
python scripts/inference.py --video sample.mp4 --skip sync

# 使用已訓練的 meta-classifier
python scripts/inference.py --video sample.mp4 --fusion_mode meta
```

預期輸出：
```
Analysing: sample.mp4
============================================================
  Face track: 75 frames @ 25.0 fps (detection: 1.2s)
  Audio extracted: /tmp/tmpXXXX.wav
  [visual  ] score=0.8471  (0.8s)
  [rppg    ] score=0.6343  (1.1s)
  [sync    ] score=0.7213  (2.3s)

Total async inference time: 2.5s
============================================================
Prediction : FAKE
Fake Score : 0.7399 (threshold=0.50)
  visual      : score=0.8471  weight=0.500
  rppg        : score=0.6343  weight=0.250
  sync        : score=0.7213  weight=0.250
```

---

## 資料集驗證流程

### Phase 1：rPPG SNR 閾值校準（必做）

```bash
python scripts/calibrate_snr.py \
    --data_root /data/FF++ \
    --config configs/default.yaml \
    --split val \
    --update_config   # 自動寫入最佳 threshold
```

輸出示例：
```
Optimal SNR threshold : 1.8432 dB
Youden's J statistic  : 0.6721
Mean SNR (real)       : 4.2134 dB
Mean SNR (fake)       : 0.3891 dB
```

### Phase 2：FF++ 全量評估（DeepfakeBench 基準對齊）

```bash
python scripts/evaluate.py \
    --dataset ff++ \
    --data_root /data/FF++ \
    --compression c23 \
    --config configs/ff_eval.yaml \
    --split test \
    --output results/ff_test.json
```

預期指標（FF++ c23，test split，140影片）：

| 偵測器 | Frame AUC | ACC | 說明 |
|--------|-----------|-----|------|
| Xception (DFB baseline) | ~99.7% | ~99.2% | 參考基準 |
| ADB-Visual | ~99.5% | ~98.8% | XceptionNet wrap |
| ADB-rPPG | ~85-90% | ~82-87% | SNR-based，輔助訊號 |
| ADB-Sync | ~88-93% | ~85-90% | 依音訊可用性 |
| **ADB-Ensemble** | **~99.6%** | **~99.0%** | 三路融合 |

### Phase 3：跨資料集泛化（Train FF++ → Test Celeb-DF v2）

在 DeepfakeBench 中使用標準 cross-dataset protocol，驗證泛化能力。

**驗證閘門**（進入訓練框架的條件）：
- ADB-Ensemble FF++ AUC > 0.95
- ADB-Ensemble Celeb-DF v2 AUC > 0.85

---

## DeepfakeBench 對齊分析

### 安裝適配器至 DeepfakeBench

```bash
cp deepfakebench_adapters/adb_*_detector.py /path/to/deepfakebench/training/detectors/
cp deepfakebench_adapters/configs/adb_*.yaml /path/to/deepfakebench/training/config/detector/
```

在 `/path/to/deepfakebench/training/detectors/__init__.py` 末尾新增：
```python
from .adb_visual_detector   import ADBVisualDetector
from .adb_rppg_detector     import ADBRPPGDetector
from .adb_sync_detector     import ADBSyncDetector
from .adb_ensemble_detector import ADBEnsembleDetector
```

### 執行對齊評估

```bash
cd /path/to/deepfakebench

# 各路評估
python training/train.py \
    --detector_path training/config/detector/adb_visual.yaml --phase test
python training/train.py \
    --detector_path training/config/detector/adb_rppg.yaml --phase test
python training/train.py \
    --detector_path training/config/detector/adb_sync.yaml --phase test

# 集成（主要指標）
python training/train.py \
    --detector_path training/config/detector/adb_ensemble.yaml --phase test
```

---

## 訓練框架（驗證通過後啟用）

### Stage 1：融合權重調優（~30 分鐘）

凍結所有 backbone，只優化 3 個權重參數，在 FF++ val 上 grid search：

```bash
python scripts/train_fusion.py \
    --config configs/training.yaml \
    --stage weighted \
    --scores_cache cache/scores.npz
```

### Stage 2：Meta-Classifier 訓練（~2-4 小時）

2 層 MLP (3→512→256→2)，訓練在 FF++ + DFDC 混合資料集：

```bash
python scripts/train_fusion.py \
    --config configs/training.yaml \
    --stage meta \
    --scores_cache cache/scores.npz
```

### 雙軌制資料集策略

| Track | 資料集 | 目的 |
|-------|--------|------|
| **Track A**（基準對齊）| FF++ c23 only | 功能驗證 + SNR 校準 |
| **Track B**（泛化訓練）| FF++ + DFDC + Celeb-DF v2 + ForgeryNet | 防止過擬合 FF++ 舊壓縮痕跡 |

> Track B 的必要性：FF++ 涵蓋的偽造技術（2019年）已被現代擴散模型超越。
> Meta-Classifier 必須在多樣化資料集上訓練才能應對真實世界高品質 deepfake。

---

## 設定說明

```yaml
# configs/default.yaml 關鍵參數
preprocessing:
  insightface_model: "buffalo_sc"  # "buffalo_l" 提升精準度但更慢
  use_face_cache: true             # 重複推理加速 2-3x
  fps_target: 25.0

detectors:
  rppg:
    snr_threshold: 1.5   # 執行 calibrate_snr.py 後更新
    snr_scale: 1.0

  sync:
    whisper_device: "cpu"  # CPU 避免 VRAM 競爭

fusion:
  mode: "weighted"    # "meta" 啟用訓練後的 MLP 融合
  weights:
    visual: 0.50
    rppg:   0.25
    sync:   0.25
```

---

## 引用

```bibtex
@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nießner, Matthias},
  booktitle={ICCV}, year={2019}
}
@inproceedings{liu2023rppgtoolbox,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Sengupta, Roni and Patel, Shwetak and Wang, Yingcong and McDuff, Daniel},
  booktitle={NeurIPS}, year={2023}
}
@inproceedings{peng2025latentsync,
  title={LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync},
  author={Peng, Chunyu and Xu, Yaofang and Ren, Fangyuan and Zhao, Shaobo and Zhang, Ying and others},
  booktitle={CVPR}, year={2025}
}
@inproceedings{yan2023deepfakebench,
  title={DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Fan, Xinhang and Wu, Baoyuan},
  booktitle={NeurIPS}, year={2023}
}
```

---

## 授權

MIT License
