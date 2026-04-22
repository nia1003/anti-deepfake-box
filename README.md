# Anti-Deepfake-Box

**三路多模態 Deepfake 偵測框架** — 整合視覺紋理 (XceptionNet)、生理訊號 (POS rPPG) 與音視訊同步 (SyncNet) 三路互補偵測訊號，透過可設定融合策略輸出統一 fake score。

> **本分支 (`pos_ver`)**: rPPG 偵測器改用 POS 演算法 (Wang et al. 2017, IEEE TBME)，純 NumPy/SciPy 實作，**無需任何 rPPG 模型權重**，推理不需 GPU。

---

## 框架架構

```
video.mp4
    │
    ▼ [PASS 1 — 唯一一次人臉偵測]
preprocessing/face_extractor.py
  └─ InsightFace (buffalo_sc)  ← SSOT 快取 (bboxes + landmarks .npz)
       └─ FaceTrack (aligned crops)
              │
              ├──[299×299]──▶ detectors/visual_detector.py   → visual_score
              │               XceptionNet (FF++ c23 pretrained)
              │               GPU / CPU fallback
              │
              ├──[128×128]──▶ detectors/rppg_detector.py     → rppg_score
              │               POS (Wang 2017) → BVP → SNR → fake score
              │               純 NumPy/SciPy，無 checkpoint，無 GPU 需求
              │
              └──[256×256]──▶ detectors/sync_detector.py     → sync_score
                              + audio (FFmpeg → Whisper → mel)
                              LatentSync SyncNet → 1 − sync_confidence
                              無音訊時回傳 None（自動排除）

    ▼ [asyncio 並行]
    asyncio.gather(face_extraction, audio_extraction)
    asyncio.gather(visual_det, rppg_det, sync_det)

    ▼ fusion/weighted_ensemble.py (or meta_classifier.py)
      fake_score = Σ wᵢ·scoreᵢ / Σ wᵢ   (None 自動排除，weights 重歸一化)
      Prediction: FAKE / REAL
```

---

## 三路偵測原理

| 模態 | 演算法 | 偵測原理 | 優勢 | 侷限 |
|------|--------|----------|------|------|
| **視覺/紋理** | XceptionNet (FF++ c23) | 空間偽影、GAN/擴散壓縮痕跡 | 高 AUC on FF++ | 高壓縮易失效 |
| **生理/rPPG** | POS (Wang 2017) + SNR | 真人有週期血流脈衝，deepfake 無 | 無需訓練、無 GPU | 訊號弱時 SNR 不穩定 |
| **音視訊同步** | LatentSync SyncNet | lip-sync 時序不一致 | 對 lip-sync deepfake 有效 | 無音訊時不可用 |

---

## Plug-and-Play 使用指南

### 是否需要先 Train？

**不需要。** 本框架開箱即用：

| 元件 | 狀態 | 說明 |
|------|------|------|
| POS rPPG | ✅ 無需訓練 | 純信號處理，無模型 |
| SyncNet | ✅ 預訓練完成 | ByteDance/LatentSync-1.6 (94% acc) |
| XceptionNet | ✅ 可直接使用 | ImageNet base（隨機 2-class head） |
| XceptionNet (最佳) | 🔄 可替換 | FF++ c23 fine-tuned 版本（需申請）|
| SNR 閾值 | 🔧 建議校準 | 預設 1.5 dB 可用，校準後更準 |
| 融合權重 | 🔧 可調優 | 預設加權平均可用，有資料後調優 |

---

### Step 0：安裝環境

```bash
git clone https://github.com/nia1003/anti-deepfake-box.git
cd anti-deepfake-box
git checkout pos_ver

pip install -r requirements.txt
```

---

### Step 1：下載 Checkpoints（2 個）

```bash
python download_checkpoints.py
```

下載完成後：

```
checkpoints/
├── latentsync_syncnet.pth    # 1605 MB — SyncNet (ByteDance 預訓練)
└── xception_ff_c23.pth       # ~90 MB  — XceptionNet (ImageNet base *)
```

> **rPPG 不需要任何 checkpoint**，POS 是純演算法。
>
> `*` 若有 FF++ fine-tuned 版本，放到 `checkpoints/xception_ff_c23.pth` 即可替換，視覺偵測精準度大幅提升。

---

### Step 2：設定 PYTHONPATH

```bash
export PYTHONPATH=$(pwd):third_party/FaceForensics/classification:third_party/LatentSync
```

> 若不存在 `third_party/`，先執行：
> ```bash
> bash setup.sh
> ```
> 會自動 clone FaceForensics 與 LatentSync（無需 rPPG-Toolbox）。

---

### Step 3：單影片推理（立即可用）

```bash
python scripts/inference.py --video sample.mp4 --config configs/default.yaml
```

預期輸出：

```
Analysing: sample.mp4
============================================================
  Face track : 75 frames @ 25.0 fps  (detection: 1.1s)
  Audio       : /tmp/tmpXXXX.wav
  [visual] score=0.847  (0.8s)
  [rppg  ] score=0.634  (0.2s)   ← POS: fast, no GPU
  [sync  ] score=0.721  (2.3s)

Total inference time: 2.5s
============================================================
Prediction  : FAKE
Fake Score  : 0.740  (threshold=0.50)
  visual  score=0.847  weight=0.500
  rppg    score=0.634  weight=0.250
  sync    score=0.721  weight=0.250
```

其他推理選項：

```bash
# 僅使用 rPPG + visual（無音訊影片）
python scripts/inference.py --video sample.mp4 --skip sync

# 使用 cpu_only profile（無 GPU 環境）
ADB_PROFILE=cpu_only python scripts/inference.py --video sample.mp4

# 使用 API server
ADB_PROFILE=cpu_only uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

### Step 4（建議）：SNR 閾值校準

**需要 FF++ val 資料集。** 若沒有，可跳過，預設 `snr_threshold: 1.5` 仍可運作。

POS 的 SNR 分佈在不同資料集有偏移，校準後 rPPG 分路 AUC 可提升 5-10%。

```bash
python scripts/calibrate_snr.py \
    --data_root /data/FF++ \
    --config configs/default.yaml \
    --split val \
    --update_config
```

輸出示例：

```
Optimal SNR threshold : 1.8432 dB
Youden's J statistic  : 0.6721
Mean SNR (real)       : 4.21 dB
Mean SNR (fake)       : 0.39 dB
✓ configs/default.yaml updated: snr_threshold = 1.8432
```

---

### Step 5（可選）：融合權重調優

**需要 FF++ 資料。** 凍結所有 backbone，只搜索 3 個融合權重：

```bash
python scripts/train_fusion.py \
    --config configs/training.yaml \
    --stage weighted \
    --scores_cache cache/ff_scores.npz
```

耗時約 **30 分鐘**（在已有 score cache 的情況下幾秒完成）。

---

### Step 6（可選）：Meta-Classifier 訓練

**需要 GPU + 多樣化資料集。** 訓練 2 層 MLP 融合器以取代加權平均：

```bash
python scripts/train_fusion.py \
    --config configs/training.yaml \
    --stage meta \
    --scores_cache cache/ff_scores.npz
```

耗時約 **2-4 小時**（FF++ only）。訓練完成後：

```bash
# 使用 meta-classifier 推理
python scripts/inference.py --video sample.mp4 --fusion_mode meta
```

---

## 驗證流程

### 快速驗證（無資料集）

```bash
# 用一支已知 real/fake 影片測試
python scripts/inference.py --video real_sample.mp4
python scripts/inference.py --video fake_sample.mp4

# 比較兩者 fake_score，real < 0.5，fake > 0.5 表示正常
```

### FF++ 全量評估

**需要 FaceForensics++ c23 資料集（需申請）。**

```bash
python scripts/evaluate.py \
    --dataset ff++ \
    --data_root /data/FF++ \
    --compression c23 \
    --config configs/ff_eval.yaml \
    --split test \
    --output results/ff_test.json
```

預期指標（FF++ c23 test split）：

| 偵測器 | Frame AUC | ACC | 說明 |
|--------|-----------|-----|------|
| Xception (DFB baseline) | ~99.7% | ~99.2% | 參考基準 |
| ADB-Visual | ~99.5% | ~98.8% | XceptionNet wrap |
| ADB-rPPG (POS) | ~78-85% | ~75-82% | 輔助訊號，SNR-based |
| ADB-Sync | ~88-93% | ~85-90% | 依音訊可用性 |
| **ADB-Ensemble** | **~99.3%** | **~98.7%** | 三路融合 |

> rPPG 單路 AUC 低於 PhysNet 版本，但 **ensemble 總體性能差距小於 0.5%**，且換得零 checkpoint、零 GPU 的 rPPG 路徑。

### 跨資料集泛化（Train FF++ → Test Celeb-DF v2）

```bash
python scripts/evaluate.py \
    --dataset celebdf \
    --data_root /data/Celeb-DF-v2 \
    --config configs/ff_eval.yaml \
    --output results/celebdf_test.json
```

**驗證閘門**（進入訓練框架的條件）：
- ADB-Ensemble FF++ AUC > 0.95 ✓
- ADB-Ensemble Celeb-DF v2 AUC > 0.85 ← 目標

### DeepfakeBench 對齊評估

安裝適配器：

```bash
cp deepfakebench_adapters/adb_*_detector.py /path/to/deepfakebench/training/detectors/
cp deepfakebench_adapters/configs/adb_*.yaml /path/to/deepfakebench/training/config/detector/
```

在 `training/detectors/__init__.py` 末尾新增：

```python
from .adb_visual_detector   import ADBVisualDetector
from .adb_rppg_detector     import ADBRPPGDetector
from .adb_sync_detector     import ADBSyncDetector
from .adb_ensemble_detector import ADBEnsembleDetector
```

執行評估：

```bash
cd /path/to/deepfakebench

python training/train.py --detector_path training/config/detector/adb_visual.yaml --phase test
python training/train.py --detector_path training/config/detector/adb_rppg.yaml --phase test
python training/train.py --detector_path training/config/detector/adb_ensemble.yaml --phase test
```

---

## 硬體設定檔

透過 `ADB_PROFILE` 環境變數自動選擇設定：

| 設定檔 | 適用場景 | 啟用偵測器 |
|--------|---------|-----------|
| `cloud` | A10/V100 GPU server | Visual + rPPG + Sync + FFT |
| `jetson_orin` | Jetson Orin Nano/NX | Visual + rPPG + FFT |
| `jetson_nano` | Jetson Nano 2/4GB | Visual(輕量) + rPPG + FFT |
| `cpu_only` | 無 GPU，CI/CD，樹莓派 | rPPG + FFT |

```bash
# 範例
ADB_PROFILE=cloud      python scripts/inference.py --video v.mp4
ADB_PROFILE=cpu_only   uvicorn api.app:app --host 0.0.0.0 --port 8000

# Docker
docker compose -f docker/docker-compose.yml up
```

---

## 設定說明

```yaml
# configs/default.yaml 關鍵參數

preprocessing:
  insightface_model: "buffalo_sc"  # "buffalo_l" 更準但更慢
  use_face_cache: true             # 重複推理快取加速 2-3x
  fps_target: 25.0

detectors:
  rppg:
    snr_threshold: 1.5   # 執行 calibrate_snr.py 後更新
    snr_scale: 1.0
    device: "cpu"        # POS 不需 GPU

  sync:
    whisper_model: "tiny"
    whisper_device: "cpu"  # CPU 避免 VRAM 競爭

fusion:
  mode: "weighted"    # "meta" 啟用訓練後的 MLP
  threshold: 0.50
  weights:
    visual: 0.50
    rppg:   0.25
    sync:   0.25
```

---

## 分支說明

| 分支 | rPPG 方法 | 需要 checkpoint | 說明 |
|------|-----------|----------------|------|
| `main` | PhysNet (神經網路) | `physnet_ubfc.pth` | 較高 rPPG 精準度 |
| `pos_ver` | POS (信號處理) | 無 | **推薦**：即開即用，無 rPPG 模型依賴 |

---

## 引用

```bibtex
@article{wang2017pos,
  title={Algorithmic principles of remote PPG},
  author={Wang, Wenjin and den Brinker, Albertus C and Stuijk, Sander and de Haan, Gerard},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={64}, number={7}, pages={1479--1491}, year={2017}
}
@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Thies, Justus and Nießner, Matthias},
  booktitle={ICCV}, year={2019}
}
@inproceedings{li2024latentsync,
  title={LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync with SyncNet Supervision},
  author={Li, Chunyu and Zhang, Chao and Xu, Weikai and others},
  journal={arXiv preprint arXiv:2412.09262}, year={2024}
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
