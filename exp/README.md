# Experiment: XceptionNet / TS-CAN / Sync on FaceForensics++ & DeepFakeBench

比較三個偵測器在兩個資料集家族上的表現。

---

## 目錄結構

```
exp/
├── configs/                    # 每個 (detector × dataset) 的 YAML 配置
│   ├── base.yaml               # 共用預設值（device、preprocessing、fusion）
│   ├── ff_xception.yaml        # XceptionNet on FF++
│   ├── ff_tscan.yaml           # TS-CAN   on FF++
│   ├── ff_sync.yaml            # Sync      on FF++
│   ├── dfb_xception.yaml       # XceptionNet on DeepFakeBench 套件
│   ├── dfb_tscan.yaml          # TS-CAN   on DeepFakeBench 套件
│   └── dfb_sync.yaml           # Sync      on DeepFakeBench 套件
├── detectors/
│   └── tscan_detector.py       # TS-CAN 神經網路 rPPG 偵測器
├── datasets/
│   └── celebdf_dataset.py      # Celeb-DF v1/v2 資料集讀取器
├── run_exp.py                  # 主要實驗執行器
├── report.py                   # 結果彙整 → 表格
└── results/                    # 執行後產生的 JSON（已 .gitignore）
```

---

## 三個偵測器說明

| 偵測器 | 程式位置 | 原理 | Checkpoint |
|--------|----------|------|------------|
| **XceptionNet** | `detectors/visual_detector.py` | 辨識視覺偽造痕跡（GAN artifacts）；在 FF++ c23 訓練 | `checkpoints/xception_ff_c23.pth` |
| **TS-CAN** | `exp/detectors/tscan_detector.py` | 神經網路 rPPG 估算（motion + appearance branch + temporal attention）；SNR 低 → 疑似 deepfake | `checkpoints/tscan_ubfc.pth`（選用；缺少時自動 fallback 到 POS 演算法） |
| **Sync** | `detectors/sync_detector.py` | 音視訊唇型同步分析（LatentSync SyncNet + Whisper mel）；sync confidence 低 → 疑似 deepfake | `checkpoints/latentsync_syncnet.pth` |

---

## 兩個資料集家族

### FaceForensics++ (`--dataset ff`)
| 欄位 | 預設值 |
|------|--------|
| Compression | c23 |
| Split | test |
| Manipulation types | Deepfakes, Face2Face, FaceSwap, NeuralTextures |
| 資料根目錄 | `--ff_root` |

目錄結構：
```
<ff_root>/
├── original_sequences/youtube/c23/videos/
└── manipulated_sequences/{Deepfakes,Face2Face,...}/c23/videos/
```

### DeepFakeBench 套件 (`--dataset dfb`)
同時測試以下三個子資料集（有提供根目錄的才會執行）：

| 子資料集 | CLI 參數 | 格式 |
|----------|----------|------|
| FF++ c23 test | `--ff_root` | 同上 |
| Celeb-DF v2 test | `--celebdf_root` | 需有 `List_of_testing_videos.txt` |
| DFDC (train subset) | `--dfdc_root` | 需有 `metadata.json` per chunk |

Celeb-DF 目錄結構：
```
<celebdf_root>/
├── Celeb-real/
├── Celeb-synthesis/
├── YouTube-real/
└── List_of_testing_videos.txt
```

DFDC 目錄結構：
```
<dfdc_root>/
├── dfdc_train_part_0/
│   ├── metadata.json
│   └── *.mp4
├── dfdc_train_part_1/
│   └── ...
```

---

## 評估指標

| 指標 | 說明 |
|------|------|
| **AUC** | Area Under ROC Curve；越高越好 |
| **ACC** | Accuracy at EER threshold；越高越好 |
| **EER** | Equal Error Rate；越低越好 |
| **AP** | Average Precision；越高越好 |

---

## 快速開始

### 1. 準備 checkpoints（選用）

```bash
# XceptionNet（FF++ 官方訓練權重）
# 放到 checkpoints/xception_ff_c23.pth

# TS-CAN（rppg-toolbox UBFC 訓練權重）
# 放到 checkpoints/tscan_ubfc.pth
# 若缺少，TS-CAN 自動使用 POS 演算法替代

# LatentSync SyncNet
# 放到 checkpoints/latentsync_syncnet.pth
```

### 2. 執行單一實驗

```bash
# XceptionNet on FF++ (c23 test, 最多 200 支影片)
python exp/run_exp.py \
    --detector xception \
    --dataset ff \
    --ff_root /data/FF++ \
    --compression c23 \
    --split test \
    --max_videos 200

# TS-CAN on FF++
python exp/run_exp.py \
    --detector tscan \
    --dataset ff \
    --ff_root /data/FF++

# Sync on FF++
python exp/run_exp.py \
    --detector sync \
    --dataset ff \
    --ff_root /data/FF++
```

### 3. 執行全部 6 個組合

```bash
python exp/run_exp.py \
    --detector all \
    --dataset all \
    --ff_root /data/FF++ \
    --celebdf_root /data/Celeb-DF-v2 \
    --dfdc_root /data/DFDC \
    --max_videos 200 \
    --device cuda        # 或 cpu
```

每個 `(detector, dataset)` 組合的結果自動存到 `exp/results/<detector>_<dataset>.json`。
若 JSON 已存在則跳過，方便斷點續跑。

### 4. 查看結果表格

```bash
# ASCII 表格（預設）
python exp/report.py

# Markdown 格式（方便貼到 PR / issue）
python exp/report.py --fmt markdown

# 存成 CSV
python exp/report.py --fmt csv --out exp/results/summary.csv
```

輸出範例：
```
======================================================================
  DETECTOR     DATASET                            AUC    ACC    EER     AP
----------------------------------------------------------------------
  xception     ff_c23_test                     0.9820 0.9310 0.0720 0.9780
  tscan        ff_c23_test                     0.7450 0.6800 0.2850 0.7200
  sync         ff_c23_test                     0.6120 0.5900 0.3800 0.6050
----------------------------------------------------------------------
  xception     celebdf_v2_test                 0.7230 0.6700 0.3150 0.7010
  tscan        celebdf_v2_test                 0.6480 0.6100 0.3700 0.6200
  sync         celebdf_v2_test                 0.5830 0.5500 0.4200 0.5710
======================================================================
```

---

## 進階：使用 config 檔覆寫設定

```bash
# 用 exp/configs/ff_xception.yaml 覆寫
python exp/run_exp.py \
    --detector xception \
    --dataset ff \
    --ff_root /data/FF++ \
    --config exp/configs/ff_xception.yaml
```

也可直接編輯 `exp/configs/base.yaml` 調整 `snr_threshold`、`visual_max_frames` 等參數。

---

## TS-CAN 技術細節

TS-CAN（Liu et al., NeurIPS 2020）是一個用於接觸式生命體徵估測的時序卷積注意力網路，在這裡被用來從臉部影片估算 rPPG 信號：

```
face crops (T, H, W, 3)
    │
    ├─ motion branch   (frame diff / mean brightness → ConvBN×2 → AvgPool)
    │                                                                   ↓
    └─ appearance branch (raw frames → ConvBN×2 → AvgPool) → attention gate
                                                                        ↓
                                                                  merged ConvBN×2
                                                                        ↓
                                                                  FC → rPPG signal (T,)
                                                                        ↓
                                                                   SNR 計算
                                                                        ↓
                                                                  fake score ∈ [0,1]
```

若 `checkpoints/tscan_ubfc.pth` 不存在，自動使用 **POS 演算法**（Wang 2017）作為 fallback，不影響其他偵測器的執行。
