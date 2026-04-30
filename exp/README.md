# Experiment: XceptionNet / TS-CAN / Sync on FaceForensics++ & DeepFakeBench

比較三個偵測器在兩個資料集家族上的表現。

---

## 目錄結構

```
exp/
├── configs/                    # 每個 (detector × dataset) 的 YAML 配置
│   ├── base.yaml               # 共用預設值（device: auto、preprocessing、fusion）
│   ├── ff_xception.yaml        # XceptionNet on FF++
│   ├── ff_tscan.yaml           # TS-CAN   on FF++
│   ├── ff_sync.yaml            # Sync      on FF++
│   ├── dfb_xception.yaml       # XceptionNet on DeepFakeBench 套件
│   ├── dfb_tscan.yaml          # TS-CAN   on DeepFakeBench 套件
│   └── dfb_sync.yaml           # Sync      on DeepFakeBench 套件
├── detectors/
│   └── tscan_detector.py       # TS-CAN 神經網路 rPPG 偵測器（MPS / MLX / CPU）
├── datasets/
│   └── celebdf_dataset.py      # Celeb-DF v1/v2 資料集讀取器
├── utils/
│   ├── device.py               # 自動偵測 cuda → mps → cpu
│   └── mlx_pos.py              # MLX 加速 POS 演算法（Apple Silicon Metal GPU）
├── run_exp.py                  # 主要實驗執行器
├── report.py                   # 結果彙整 → 表格
└── results/                    # 執行後產生的 JSON（已 .gitignore）
```

---

## 三個偵測器說明

| 偵測器 | 程式位置 | 原理 | Checkpoint |
|--------|----------|------|------------|
| **XceptionNet** | `detectors/visual_detector.py` | 辨識視覺偽造痕跡（GAN artifacts）；在 FF++ c23 訓練 | `checkpoints/xception_ff_c23.pth` |
| **TS-CAN** | `exp/detectors/tscan_detector.py` | 神經網路 rPPG 估算（motion + appearance branch + temporal attention）；SNR 低 → 疑似 deepfake | `checkpoints/tscan_ubfc.pth`（選用；缺少時自動 fallback 到 POS / MLX） |
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

---

## 評估指標

| 指標 | 說明 |
|------|------|
| **AUC** | Area Under ROC Curve；越高越好 |
| **ACC** | Accuracy at EER threshold；越高越好 |
| **EER** | Equal Error Rate；越低越好 |
| **AP** | Average Precision；越高越好 |

---

## Mac / Apple Silicon 設定（MLX + MPS）

沒有 CUDA 的 Mac 環境使用兩層加速：

| 操作 | 後端 | 說明 |
|------|------|------|
| PyTorch 模型推論（XceptionNet、TS-CAN、SyncNet） | **MPS**（Metal Performance Shaders，PyTorch 內建） | `device="mps"` |
| rPPG 訊號處理（POS 演算法） | **MLX**（Apple 的 Metal 陣列框架） | `exp/utils/mlx_pos.py` |
| Whisper 語音轉錄 | **CPU** | MPS 支援有限，保持 CPU |

### 安裝

```bash
# 1. 安裝 Mac 專用依賴（含 mlx）
pip install -r requirements_mac.txt

# 2. 確認 MPS 可用
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# 3. 確認 MLX 可用
python -c "import mlx.core; print('MLX: OK')"
```

### Device 自動偵測

`--device auto`（預設）：

```
CUDA available? → use CUDA
MPS  available? → use MPS   ← Mac M1/M2/M3
otherwise       → use CPU
```

啟動時會印出偵測結果：
```
── Compute backends ──────────────────────
CUDA  : no
MPS   : yes
MLX   : yes
Active: mps
──────────────────────────────────────────
```

### 手動指定 device

```bash
# 明確使用 MPS（Apple Silicon Metal）
python exp/run_exp.py --detector tscan --dataset ff --ff_root /data/FF++ --device mps

# 純 CPU（無 GPU 環境）
python exp/run_exp.py --detector xception --dataset ff --ff_root /data/FF++ --device cpu
```

---

## 快速開始

### 1. 準備 checkpoints（選用）

```bash
# XceptionNet（FF++ 官方訓練權重）
# 放到 checkpoints/xception_ff_c23.pth

# TS-CAN（rppg-toolbox UBFC 訓練權重）
# 放到 checkpoints/tscan_ubfc.pth
# 若缺少，TS-CAN 自動使用 POS+MLX 替代

# LatentSync SyncNet
# 放到 checkpoints/latentsync_syncnet.pth
```

### 2. 執行單一實驗

```bash
# XceptionNet on FF++ (Mac，自動使用 MPS)
python exp/run_exp.py \
    --detector xception \
    --dataset ff \
    --ff_root /data/FF++ \
    --compression c23 \
    --split test \
    --max_videos 200

# TS-CAN on FF++（POS 訊號走 MLX，若有 checkpoint 神經網路走 MPS）
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
    --max_videos 200
    # --device auto  (預設，Mac 上自動選 mps)
```

每個 `(detector, dataset)` 組合的結果自動存到 `exp/results/<detector>_<dataset>.json`。
若 JSON 已存在則跳過，方便斷點續跑。

### 4. 查看結果表格

```bash
python exp/report.py                          # ASCII 表格
python exp/report.py --fmt markdown           # Markdown（貼 PR/issue）
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
```

---

## TS-CAN 技術細節

TS-CAN（Liu et al., NeurIPS 2020）：時序卷積注意力網路，在此用來從臉部影片估算 rPPG 信號。

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

**Fallback 鏈：**
```
TS-CAN checkpoint 存在？
  └─ Yes → 神經網路推論（MPS / CUDA / CPU）
  └─ No  → POS 演算法
              └─ MLX 可用（Apple Silicon）→ MLX POS（Metal GPU）
              └─ 否則 → NumPy POS（CPU）
```

---

## MLX POS 技術細節

`exp/utils/mlx_pos.py` 將 Wang (2017) POS 演算法的內層矩陣運算移到 MLX（Metal GPU）：

```python
rgb = mx.array(...)             # 空間平均 RGB → Metal 記憶體
Cn  = (chunk / mean_c).T       # 正規化（Metal）
S   = P @ Cn                   # 投影矩陣乘法（Metal）
mx.eval(S)                     # 刷新 Metal command buffer
# 累積 H、bandpass filter 仍在 NumPy/SciPy
```

MLX 的惰性求值（lazy evaluation）讓多個 Metal 操作批次執行後才同步，減少 CPU↔GPU 往返開銷。
