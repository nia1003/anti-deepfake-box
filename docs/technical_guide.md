# Anti-Deepfake-Box 技術指南

> 版本：v5（detector registry 版）  
> 最後更新：2026-05-19  
> 對應 commit：`anti-deepfake-box@3738724`

---

## 目錄

1. [環境設置](#1-環境設置)
2. [快速開始（5 分鐘）](#2-快速開始)
3. [偵測器選擇](#3-偵測器選擇)
4. [批次評分：collect_scores.py](#4-批次評分)
5. [弱分類器篩選：cascade_selection.py](#5-弱分類器篩選)
6. [串列級聯推論：serial_cascade.py](#6-串列級聯推論)
7. [單影片推論：inference.py](#7-單影片推論)
8. [DeepfakeBench 整合](#8-deepfakebench-整合)
9. [設定檔與模式切換](#9-設定檔與模式切換)
10. [Testing 指南](#10-testing-指南)
11. [常見問題排除](#11-常見問題排除)

---

## 1. 環境設置

### 1.1 系統需求

| 項目 | 最低需求 | 建議 |
|------|---------|------|
| Python | 3.10 | 3.11 |
| CUDA | — | 12.1（推論用） |
| RAM | 8 GB | 16 GB |
| 磁碟 | 5 GB（checkpoints + cache） | 20 GB |
| ffmpeg | 4.x | 6.x |

### 1.2 安裝

```bash
git clone https://github.com/nia1003/anti-deepfake-box.git
cd anti-deepfake-box

# 基礎依賴（所有功能）
pip install -r requirements.txt

# ffmpeg（音訊抽取 + SyncNet 必須）
# Ubuntu/Debian:
sudo apt-get install -y ffmpeg
# macOS:
brew install ffmpeg
# Windows:
# 下載 https://ffmpeg.org/download.html 並加入 PATH

# 確認 ffmpeg 可用
ffmpeg -version | head -1
```

### 1.3 Checkpoint 下載

| 偵測器 | 檔案 | 用途 |
|--------|------|------|
| Xception (FF++ c23) | `checkpoints/xception_ff_c23.pth` | 預設 visual detector |
| LatentSync SyncNet | `checkpoints/latentsync_syncnet.pth` | sync detector |
| DFB visual (UCF/SBI/...) | 自行指定路徑 | 需搭配 `--dfb_pretrained` |

```bash
# Colab / 有 GPU 的環境使用 download 腳本
python download_checkpoints.py

# 或手動放置
mkdir -p checkpoints
# xception: 下載後放到 checkpoints/xception_ff_c23.pth
# syncnet:  下載後放到 checkpoints/latentsync_syncnet.pth
```

### 1.4 DeepfakeBench（DFB）設置（可選）

僅需要跑 UCF / SBI / F3Net / SPSL / SRM 等 DFB 偵測器時才需設定。

```bash
# 方式 A：設定環境變數（推薦）
export DFB_PATH=/home/user/DeepfakeBench
echo 'export DFB_PATH=/home/user/DeepfakeBench' >> ~/.bashrc

# 方式 B：兄弟目錄（自動偵測，不需設定）
# 確保 anti-deepfake-box/ 和 DeepfakeBench/ 在同一個父目錄下
ls /home/user/
# anti-deepfake-box  DeepfakeBench   ← 自動偵測成功

# 確認路徑有效
python3 -c "
from detectors.dfb_visual_wrapper import _find_dfb_training_path
p = _find_dfb_training_path()
print('DFB found:', p or 'NOT FOUND')
"
```

### 1.5 驗證安裝

```bash
# 語法驗證（無需 GPU / 資料集）
python3 -m py_compile detectors/registry.py && echo "registry OK"
python3 -m py_compile scripts/collect_scores.py && echo "collect_scores OK"
python3 -m py_compile fusion/serial_cascade.py && echo "serial_cascade OK"

# 偵測器 registry 完整性
python3 -c "
from detectors.registry import list_detectors
for mod, entries in list_detectors().items():
    available = [k for k, v in entries.items() if v['status'] == 'available']
    print(f'{mod}: available={available}')
"
# 期望輸出：
# visual: available=['xception']
# rppg:   available=['pos']
# sync:   available=['syncnet']
```

---

## 2. 快速開始

### 2.1 單影片推論（最快路徑）

```bash
cd /home/user/anti-deepfake-box

# 需要：checkpoints/xception_ff_c23.pth
python scripts/inference.py \
    --video /path/to/video.mp4 \
    --config configs/default.yaml

# 輸出範例：
# [visual]  score=0.847  (FAKE)
# [rppg]    score=0.612
# [sync]    score=0.731  (audio found)
# ─────────────────────────────────
# FAKE  confidence=0.763  threshold=0.50
```

### 2.2 批次評分（GP 訓練用）

```bash
# 跑一個資料集，產出三份 CSV
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --label_csv /data/FakeAVCeleb/labels.csv \
    --output_dir results/xception \
    --dataset FakeAVCeleb \
    --mode forensic

# 結果：
# results/xception/visual_xception_scores.csv
# results/xception/rppg_pos_scores.csv
# results/xception/av_sync_syncnet_scores.csv
```

### 2.3 不需要 GPU 的最小測試

```bash
# POS 是純 NumPy，不需要任何 checkpoint 或 GPU
python scripts/collect_scores.py \
    --input_dir /path/to/any_videos \
    --output_dir /tmp/test_results \
    --device cpu \
    --skip visual sync   # 只跑 rPPG（無需 checkpoint）

# 確認輸出
cat /tmp/test_results/rppg_pos_scores.csv | head -3
```

---

## 3. 偵測器選擇

### 3.1 Registry 查詢

```bash
# 列出所有已登錄偵測器
python3 -c "
from detectors.registry import list_detectors
import json
for mod, entries in list_detectors().items():
    print(f'\n=== {mod} ===')
    for key, info in entries.items():
        print(f'  {key:20s}  [{info[\"status\"]:15s}]  {info[\"detector_name\"]}')
        if info.get('notes'):
            print(f'                        → {info[\"notes\"]}')
"
```

輸出：
```
=== visual ===
  xception              [available      ]  Xception
                        → ImageNet + FF++ c23 fine-tuned；域內穩健基準
  ucf                   [dfb_required   ]  UCF
                        → SOTA；強域內 + 較好跨域韌性
  sbi                   [dfb_required   ]  SBI
                        → 跨域泛化最強，OOD 表現第一
  ...
=== rppg ===
  pos                   [available      ]  POS
  tscan                 [rppg_toolbox   ]  TS-CAN
  physmamba             [planned        ]  PhysMamba
=== sync ===
  syncnet               [available      ]  SyncNet
  mrdf                  [planned        ]  MRDF
  ...
```

### 3.2 在 collect_scores.py 切換偵測器

```bash
# visual: xception（預設，ADB 原生）
python scripts/collect_scores.py --visual_detector xception ...

# visual: UCF（需 DFB + checkpoint）
export DFB_PATH=/home/user/DeepfakeBench
python scripts/collect_scores.py \
    --visual_detector ucf \
    --dfb_pretrained /ckpts/ucf_ff_c23.pth \
    ...

# visual: SBI（跨域最強）
python scripts/collect_scores.py \
    --visual_detector sbi \
    --dfb_pretrained /ckpts/sbi_ff.pth \
    ...

# rppg: pos（預設），sync: syncnet（預設）
python scripts/collect_scores.py \
    --rppg_detector pos \
    --sync_detector syncnet \
    ...
```

### 3.3 在 inference.py 切換偵測器

在 `configs/default.yaml` 修改 `detector` 鍵：

```yaml
detectors:
  visual:
    detector: "sbi"              # 改成 SBI
    dfb_pretrained: "/ckpts/sbi.pth"
  rppg:
    detector: "pos"              # POS（唯一 available）
  sync:
    detector: "syncnet"          # SyncNet
```

```bash
python scripts/inference.py \
    --video sample.mp4 \
    --config configs/default.yaml
```

### 3.4 多偵測器全批跑法（GP 訓練用）

```bash
#!/bin/bash
# 對同一資料集跑所有已測偵測器，產出可比較的 CSV

DATASET=/data/FakeAVCeleb/test
LABELS=/data/FakeAVCeleb/labels.csv
OUT=results/compare
DFB=/home/user/DeepfakeBench
CKPTS=/home/user/checkpoints

export DFB_PATH=$DFB

# Visual detectors
python scripts/collect_scores.py \
    --visual_detector xception \
    --visual_pretrained $CKPTS/xception_ff_c23.pth \
    --input_dir $DATASET --label_csv $LABELS \
    --output_dir $OUT --dataset FakeAVCeleb --mode forensic --skip rppg sync

for det in ucf sbi f3net spsl srm; do
    python scripts/collect_scores.py \
        --visual_detector $det \
        --dfb_pretrained $CKPTS/${det}_ff.pth \
        --input_dir $DATASET --label_csv $LABELS \
        --output_dir $OUT --dataset FakeAVCeleb --mode forensic --skip rppg sync
done

# rPPG and sync（各只有一個 available）
python scripts/collect_scores.py \
    --skip visual \
    --input_dir $DATASET --label_csv $LABELS \
    --output_dir $OUT --dataset FakeAVCeleb --mode forensic

echo "CSV files in $OUT:"
ls $OUT/*.csv
```

---

## 4. 批次評分

### 4.1 collect_scores.py 完整參數

```
用法：python scripts/collect_scores.py [--input_dir DIR | --video_list FILE] [選項]

必填（擇一）：
  --input_dir DIR       掃描目錄下所有影片（遞迴，.mp4/.avi/.mov/.mkv/.webm）
  --video_list FILE     文字檔，每行一個影片路徑

輸出：
  --output_dir DIR      輸出目錄（預設：results/）
                        輸出檔：{modality}_{detector_key}_scores.csv

標籤與資料集：
  --label_csv FILE      CSV 格式：sample_id（或 video_id/filename）, label
  --dataset STR         寫入 CSV 的資料集名稱（FakeAVCeleb / FF++ / DFDC...）

模態控制：
  --skip {visual,rppg,sync}  跳過指定模態（可多個，例：--skip sync rppg）
  --mode {realtime,forensic} realtime=快速（預設），forensic=高精度（skip_leading_ms=80）

偵測器選擇：
  --visual_detector KEY  choices: xception ucf sbi f3net spsl srm
                                  efficientnet_b4 facexray lsda（預設：xception）
  --rppg_detector KEY    choices: pos tscan physmamba（預設：pos）
  --sync_detector KEY    choices: syncnet mrdf latentsync avad（預設：syncnet）
  --dfb_pretrained PATH  DFB checkpoint 路徑（ucf/sbi/f3net/spsl/srm/... 必填）
  --visual_pretrained PATH  Xception checkpoint（--visual_detector xception 時用）

硬體：
  --device {cuda,cpu}   torch device（預設：cuda）
  --syncnet_path PATH   SyncNet checkpoint（不設則用 motion heuristic fallback）
  --whisper_model SIZE  tiny/base/small/medium/large（預設：tiny；forensic 模式自動升 small）
```

### 4.2 輸出格式（12 欄位 GP data contract）

```csv
sample_id,dataset,label,detector_name,modality,fake_score,score_type,inference_time_ms,window_start_sec,window_end_sec,status,error_message
video_001,FakeAVCeleb,1,Xception,visual,0.847123,probability,45.3,N/A,N/A,ok,
video_002,FakeAVCeleb,0,Xception,visual,0.231045,probability,44.1,N/A,N/A,ok,
video_003,FakeAVCeleb,1,Xception,visual,,probability,0.0,N/A,N/A,failed,face_not_detected
```

**關鍵規則**：
- `status="failed"` 的行**不會被刪除**（GP validator 依賴完整行數對齊）
- `fake_score` 失敗時為空字串 `""`（非 NaN），GP validator 再轉換為 NaN
- `window_start_sec` / `window_end_sec` 全影片偵測器固定填 `"N/A"`

### 4.3 驗證 CSV 輸出

```bash
# 確認行數對齊（三份 CSV 應該行數相同）
python3 -c "
import csv, glob
files = glob.glob('results/**/*_scores.csv', recursive=True)
for f in sorted(files):
    rows = list(csv.DictReader(open(f)))
    ok = sum(1 for r in rows if r['status']=='ok')
    failed = len(rows) - ok
    print(f'{f}:  {len(rows)} rows  ({ok} ok, {failed} failed)')
"

# 確認 12 欄位都存在
python3 -c "
import csv
REQUIRED = {'sample_id','dataset','label','detector_name','modality','fake_score',
            'score_type','inference_time_ms','window_start_sec','window_end_sec',
            'status','error_message'}
for fname in ['results/visual_xception_scores.csv',
              'results/rppg_pos_scores.csv',
              'results/av_sync_syncnet_scores.csv']:
    try:
        cols = set(next(csv.DictReader(open(fname))))
        missing = REQUIRED - cols
        print(f'{fname}: {\"OK\" if not missing else \"MISSING: \" + str(missing)}')
    except FileNotFoundError:
        print(f'{fname}: NOT FOUND')
"
```

---

## 5. 弱分類器篩選

### 5.1 基本用法

```python
from fusion.cascade_selection import select_classifiers

# csv_dir 內應有多個 *_scores.csv 檔（每個偵測器的結果）
selected = select_classifiers(
    csv_dir="results/compare/",
    label_csv="data/labels.csv",    # 必須，含 sample_id, label 欄
    min_auc=0.55,
    max_corr=0.90,
    output_json="results/selected_classifiers.json",
)
print("Selected:", selected)
# e.g. → ["visual_sbi", "av_sync_syncnet"]
```

### 5.2 分步執行

```python
from fusion.cascade_selection import (
    filter_by_auc, filter_by_correlation,
    compute_far_frr_curves, pareto_filter
)

# 先載入所有 CSV，格式：{name: (sample_ids, scores, labels)}
# 例：data["visual_xception"] = (ids_array, scores_array, labels_array)

# Stage 1：AUC 篩選
retained, metrics = filter_by_auc(data, min_auc=0.55)
print("After AUC filter:", list(retained))

# Stage 2：相關性篩選
retained, _ = filter_by_correlation(retained, max_corr=0.90)
print("After correlation filter:", list(retained))

# Stage 3：FAR/FRR 分析（純資訊，不淘汰）
curves = compute_far_frr_curves(retained, labels)
for name, curve in curves.items():
    thresholds, far, frr = curve
    # 找 EER 點
    eer_idx = (far - frr).abs().argmin()
    print(f"{name}: EER≈{far[eer_idx]:.3f} at threshold={thresholds[eer_idx]:.3f}")

# Stage 4：Pareto 支配篩選
selected_names = pareto_filter(metrics)
print("Pareto survivors:", selected_names)
```

### 5.3 輸出結構（selected_classifiers.json）

```json
{
  "selected": ["visual_sbi", "av_sync_syncnet"],
  "all_metrics": {
    "visual_xception": {"auc": 0.892, "eer": 0.142, "acc": 0.851},
    "visual_ucf":      {"auc": 0.921, "eer": 0.112, "acc": 0.876},
    "visual_sbi":      {"auc": 0.935, "eer": 0.098, "acc": 0.889},
    "visual_spsl":     {"auc": 0.841, "eer": 0.201, "acc": 0.772},
    "visual_srm":      {"auc": 0.788, "eer": 0.245, "acc": 0.731},
    "rppg_pos":        {"auc": 0.712, "eer": 0.298, "acc": 0.681},
    "av_sync_syncnet": {"auc": 0.867, "eer": 0.165, "acc": 0.831}
  },
  "filter_params": {"min_auc": 0.55, "max_corr": 0.90},
  "dropped_stages": {
    "auc_filter": ["rppg_pos"],
    "correlation": [],
    "pareto": ["visual_xception", "visual_ucf", "visual_spsl", "visual_srm"]
  }
}
```

---

## 6. 串列級聯推論

### 6.1 從 GP Solver CSV 建立

GP Solver（`fusion_solver_prod_v1.ipynb`）輸出 `real_data_vip_settings.csv`：

```csv
stage_order,modality,threshold_H,threshold_L,pareto_far,pareto_frr
1,visual,0.75,0.30,0.04,0.08
2,rppg,0.70,0.25,0.04,0.08
3,av_sync,0.65,0.20,0.04,0.08
```

```python
from fusion.serial_cascade import SerialCascade

cascade = SerialCascade.from_csv("results/real_data_vip_settings.csv")
print(cascade)
# SerialCascade(stages=[visual(H=0.75,L=0.30), rppg(H=0.70,L=0.25), sync(H=0.65,L=0.20)])
```

### 6.2 推論

```python
# 輸入：三個模態的 fake score（None 表示該模態不可用）
result = cascade.fuse(
    visual_score=0.847,
    rppg_score=0.612,
    sync_score=0.731,   # 無音訊時傳 None
)

print(result.fake_score)    # 0.847（提前退出分數）
print(result.is_fake)       # True
print(result.scores)        # {"visual": 0.847, "rppg": 0.612, "sync": 0.731}
print(result.weights_used)  # {"visual": 0.75, "rppg": 0.70, "sync": 0.65}（H 值）
```

### 6.3 決策邏輯說明

```
對每個 stage（依 stage_order 順序）：
  score >= H → FAKE 提前退出（fake_score = score）
  score <= L → REAL 提前退出（fake_score = score）
  L < score < H → 不確定，傳至下一 stage
  score = None → 跳過（模態不可用）

所有 stage 跑完仍無決策：
  fake_score = 0.5, is_fake = False（不確定 → 保守判定為 REAL）
```

```python
# 情境一：visual 高分，在 stage 1 提前判 FAKE
result = cascade.fuse(visual_score=0.9)
# stage 1: 0.9 >= 0.75 → FAKE exit
assert result.is_fake and result.fake_score == 0.9

# 情境二：visual 低分，在 stage 1 提前判 REAL
result = cascade.fuse(visual_score=0.1)
# stage 1: 0.1 <= 0.30 → REAL exit
assert not result.is_fake and result.fake_score == 0.1

# 情境三：全部不確定，fallback
result = cascade.fuse(visual_score=0.5, rppg_score=0.5)
assert result.fake_score == 0.5 and not result.is_fake

# 情境四：無音訊，sync 被跳過
result = cascade.fuse(visual_score=0.55, rppg_score=0.55, sync_score=None)
# 三個 stage 各自不確定或跳過 → fallback
assert result.fake_score == 0.5
```

### 6.4 手動測試 JSON 建立（不需 GP Solver）

```python
import json, tempfile
from fusion.serial_cascade import SerialCascade

# 手動設定閾值（適合實驗）
config_data = {
    "stages": [
        {"name": "visual",  "threshold_H": 0.75, "threshold_L": 0.30},
        {"name": "rppg",    "threshold_H": 0.70, "threshold_L": 0.25},
        {"name": "sync",    "threshold_H": 0.65, "threshold_L": 0.20},
    ]
}
with open("/tmp/cascade_test.json", "w") as f:
    json.dump(config_data, f)

cascade = SerialCascade.from_json("/tmp/cascade_test.json")
```

---

## 7. 單影片推論

### 7.1 基本用法

```bash
python scripts/inference.py \
    --video /path/to/video.mp4 \
    --config configs/default.yaml
```

### 7.2 完整參數

```
必填：
  --video FILE              輸入影片路徑

選填：
  --config FILE             設定檔（預設：configs/default.yaml）
  --mode {forensic,realtime}  覆蓋 config 的 mode
  --fusion_mode {weighted,meta,cascade}  覆蓋 fusion mode
  --skip {visual,rppg,sync}  跳過指定模態（可多個）
  --async_mode              啟用非同步平行推論（較快，適合長影片）
```

### 7.3 常用情境

```bash
# 1. 最快速（只跑 visual，無需 ffmpeg/SyncNet）
python scripts/inference.py \
    --video sample.mp4 \
    --skip rppg sync

# 2. 完整三模態（需要 ffmpeg + checkpoints）
python scripts/inference.py \
    --video sample.mp4 \
    --config configs/mode_forensic.yaml

# 3. 非同步平行推論（GPU 多工）
python scripts/inference.py \
    --video sample.mp4 \
    --async_mode

# 4. 串列級聯推論（需先有 GP Solver CSV）
python scripts/inference.py \
    --video sample.mp4 \
    --fusion_mode cascade \
    --config configs/default.yaml
# configs/default.yaml 需含：
#   fusion:
#     cascade_config: "results/real_data_vip_settings.csv"

# 5. CPU 模式（無 GPU）
python scripts/inference.py \
    --video sample.mp4 \
    --config configs/profiles/cpu_only.yaml
```

### 7.4 輸出解讀

```
[  inference.py  ─────────────────────────────── ]
  video   : sample.mp4
  mode    : forensic
  detector: visual=sbi  rppg=pos  sync=syncnet
─────────────────────────────────────────────────
  [visual ]  SBI         score=0.931   FAKE
  [rppg   ]  POS         score=0.623
  [sync   ]  SyncNet     score=0.714
─────────────────────────────────────────────────
  FUSION (cascade):
    Stage 1  visual  0.931 >= H=0.75  → FAKE EXIT
  ─────────────────────────────────────────────
  RESULT:  FAKE   fake_score=0.931   (threshold=0.50)
  Time:    2.3 s
```

---

## 8. DeepfakeBench 整合

### 8.1 環境設置

```bash
export PYTHONPATH=/home/user/anti-deepfake-box:$PYTHONPATH
export DFB_PATH=/home/user/DeepfakeBench

# 確認 DFB 可找到 ADB
cd /home/user/DeepfakeBench
python3 -c "
import sys; sys.path.insert(0, 'training')
from detectors import ADBVisualDetector, ADBRPPGDetector, ADBSyncDetector
print('ADB adapters loaded OK')
"
```

### 8.2 三個 ADB Adapter 配置

| Adapter | 模型名稱 | Config YAML |
|---------|---------|------------|
| `ADBVisualDetector` | `adb_visual` | `training/config/detector/adb_visual.yaml` |
| `ADBRPPGDetector` | `adb_rppg` | `training/config/detector/adb_rppg.yaml` |
| `ADBSyncDetector` | `adb_sync` | `training/config/detector/adb_sync.yaml` |

### 8.3 用 DFB 測試 ADB Sync Detector

```bash
cd /home/user/DeepfakeBench

# adb_sync：需要影片有音訊
python training/train.py \
    --detector_path training/config/detector/adb_sync.yaml \
    --phase test

# 注意：data_dict 預設無 video_path，SyncDetector 會回傳 score=0.5
# 若要啟用音訊，設定 with_audio: true 並確保 npz_path 已設定
```

### 8.4 NPZ 音訊管線（with_audio=True 路徑）

**步驟 1：執行 ADB 前處理（建立 NPZ cache）**

```bash
cd /home/user/anti-deepfake-box
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --output_dir results/ \
    --mode forensic \
    --dataset FakeAVCeleb
# forensic 模式 + cache_pixel_crops=True → NPZ 內嵌 audio_samples
```

**步驟 2：設定 npz_path 映射**

```python
# 掃描 .face_cache/ 建立 video_name → npz_path 的映射
import json
from pathlib import Path

face_cache = Path(".face_cache/forensic")
mapping = {}
for npz in face_cache.glob("**/*.npz"):
    video_stem = npz.stem
    mapping[video_stem] = str(npz)

with open("data/npz_path_map.json", "w") as f:
    json.dump(mapping, f)
```

**步驟 3：注入 npz_path 到 DFB dataset**

```python
# 在 DFB 訓練腳本中，建立 dataset 後設定 npz_path
import json
with open("data/npz_path_map.json") as f:
    npz_map = json.load(f)

# dataset.data_dict['npz_path'] 是與 video_name 對應的路徑列表
for i, vname in enumerate(dataset.data_dict['video_name']):
    stem = Path(vname).stem
    dataset.data_dict['npz_path'][i] = npz_map.get(stem)
```

**步驟 4：啟用 with_audio 並訓練/評估**

```yaml
# training/config/train_config.yaml
with_audio: true   # 開啟 NPZ 音訊載入
```

```bash
cd /home/user/DeepfakeBench
python training/train.py \
    --detector_path training/config/detector/adb_sync.yaml \
    --phase test
# 現在 data_dict['audio'] 包含 int16 ndarray，SyncDetector 可正常推論
```

---

## 9. 設定檔與模式切換

### 9.1 設定檔清單

| 檔案 | 用途 | 適合情境 |
|------|------|---------|
| `configs/default.yaml` | 預設設定 | 一般使用 |
| `configs/mode_forensic.yaml` | 高精度模式 | 批次研究、FakeAVCeleb |
| `configs/mode_realtime.yaml` | 即時模式 | 詐騙電話偵測、API |
| `configs/ff_eval.yaml` | FF++ 評估 | FaceForensics++ 資料集 |
| `configs/profiles/cpu_only.yaml` | 無 GPU | CI/CD、低配環境 |
| `configs/profiles/cloud.yaml` | 雲端推論 | AWS/GCP 部署 |
| `configs/profiles/jetson_nano.yaml` | 邊緣裝置 | NVIDIA Jetson Nano |
| `configs/profiles/jetson_orin.yaml` | 邊緣裝置 | NVIDIA Jetson Orin |

### 9.2 重要設定鍵說明

```yaml
# configs/default.yaml

device: "cuda"   # 全域 device，各偵測器可個別覆蓋

preprocessing:
  fps_target: 25.0             # 目標幀率
  insightface_model: "buffalo_sc"   # buffalo_sc（快）或 buffalo_l（準）
  use_face_cache: true         # 快取 face extraction 結果
  cache_pixel_crops: false     # 是否快取 aligned_256（forensic 模式才開）
  skip_leading_ms: 0           # 前導靜音跳過（forensic: 80）

detectors:
  visual:
    detector: "xception"       # ← 切換偵測器的關鍵設定
    dfb_pretrained: ""         # DFB checkpoint 路徑
  rppg:
    detector: "pos"
  sync:
    detector: "syncnet"

fusion:
  mode: "weighted"             # weighted | meta | cascade
  threshold: 0.50
  cascade_config: ""           # cascade 模式需填 GP Solver CSV 路徑
```

### 9.3 模式覆蓋優先順序

```
CLI --mode forensic  >  config YAML  >  default.yaml
CLI --fusion_mode cascade  >  config fusion.mode
CLI --skip sync  >  （每次執行時指定）
```

---

## 10. Testing 指南

### 10.1 目前測試狀態（2026-05-19）

```bash
cd /home/user/anti-deepfake-box
python3 -m pytest tests/ -v
```

| 測試檔案 | 總數 | 通過 | Skip | 原因 |
|---------|------|------|------|------|
| `test_preprocessing_new.py` | 19 | **19** | 0 | — |
| `test_serial_cascade.py` | 31 | **22** | 9 | pandas（CSV tests）|
| `test_audio_features.py` | 12 | **6** | 6 | librosa |
| `test_audio_extractor.py` | 9 | **1** | 8 | ffmpeg |
| `test_cascade_selection.py` | 16 | 0 | **16** | sklearn + scipy |
| **合計** | **87** | **52** | **41** | — |

> 無 FAILED。所有 skip 都是因為可選依賴未安裝，邏輯本身正確。

### 10.2 分層測試

#### 層 0：不需要任何依賴（語法 + registry）

```bash
# 語法檢查（約 1 秒）
for f in detectors/registry.py detectors/dfb_visual_wrapper.py \
         scripts/collect_scores.py fusion/serial_cascade.py \
         fusion/cascade_selection.py preprocessing/audio_extractor.py \
         preprocessing/face_extractor.py scripts/inference.py; do
    python3 -m py_compile "$f" && echo "OK: $f"
done

# Registry 完整性（約 1 秒）
python3 -c "
import types, sys, importlib.util
for stub in ('detectors.visual_detector','detectors.rppg_detector',
             'detectors.sync_detector','detectors.dfb_visual_wrapper',
             'detectors.rppg_tscan_detector','detectors.rppg_physmamba_detector',
             'detectors.base_detector','preprocessing.face_extractor'):
    sys.modules[stub] = types.ModuleType(stub)

spec = importlib.util.spec_from_file_location('detectors.registry','detectors/registry.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert set(mod.VISUAL_REGISTRY) == {'xception','ucf','sbi','f3net','spsl','srm',
                                    'efficientnet_b4','facexray','lsda'}
assert set(mod.RPPG_REGISTRY) == {'pos','tscan','physmamba'}
assert set(mod.SYNC_REGISTRY) == {'syncnet','mrdf','latentsync','avad'}

try: mod.build_detector('sync','mrdf',{})
except NotImplementedError: pass

print('Registry: ALL CHECKS PASSED')
"
```

期望：全部輸出 `OK`

#### 層 1：需要 numpy + scipy（預處理 + cascade 核心）

```bash
python3 -m pytest tests/test_preprocessing_new.py tests/test_serial_cascade.py -v
```

| 測試類別 | 測試項目 | 期望 |
|---------|---------|------|
| `TestExtractToArray` | `extract_to_array` 方法存在、None 回傳 | 4 passed |
| `TestFilterSingleFaceFrames` | 單臉保留、多臉排除、fallback | 6 passed |
| `TestSmoothBboxes` | shape 不變、interior 平滑、不修改輸入、window=1 恆等 | 7 passed |
| `TestSaveCacheWithAudio` | NPZ 結構、frame_to_audio_offset 計算、int16 cast | 5 passed |
| `TestCascadeStage` | L<H 驗證、L=H 丟例外 | 3 passed |
| `TestSerialCascadeJSON` | FAKE/REAL/fallback/None/boundary | 16 passed |
| `TestSerialCascadeConfig` | cascade_config 路徑驗證 | 3 passed |

全部通過後輸出：
```
======================== 52 passed, 8 skipped in 0.21s ========================
```
（8 skipped = CSV tests，需要 pandas）

#### 層 2：需要 pandas（cascade CSV tests）

```bash
pip install pandas

python3 -m pytest tests/test_serial_cascade.py -v
# 期望：31 passed, 0 skipped
```

| 測試類別 | 測試項目 |
|---------|---------|
| `TestSerialCascadeCSV::test_from_csv_loads_stages` | pandas 讀取 GP Solver CSV |
| `TestSerialCascadeCSV::test_av_sync_normalised_to_sync` | av_sync → sync 名稱正規化 |
| `TestSerialCascadeCSV::test_stage_order_respected` | stage_order 排序正確 |
| `TestSerialCascadeCSV::test_fake_exit` | H 閾值觸發 FAKE |
| `TestSerialCascadeCSV::test_real_exit` | L 閾值觸發 REAL |
| `TestSerialCascadeCSV::test_fallback_all_none` | 全 None → score=0.5 |
| `TestSerialCascadeCSV::test_missing_columns_raises` | 缺少欄位的 CSV 丟例外 |
| `TestSerialCascadeCSV::test_file_not_found_raises` | 不存在的路徑丟例外 |

#### 層 3：需要 sklearn + scipy（cascade_selection）

```bash
pip install scikit-learn scipy

python3 -m pytest tests/test_cascade_selection.py -v
# 期望：16 passed, 0 skipped
```

| 測試類別 | 測試項目 |
|---------|---------|
| `TestFilterByAUC` | 完美偵測器通過、隨機偵測器被淘汰、自訂閾值 |
| `TestFilterByCorrelation` | 不相關雙保留、高相關淘汰低 AUC 者 |
| `TestFARFRRCurves` | keys 正確、EER 在合理範圍、完美偵測器 EER≈0 |
| `TestParetoFilter` | 被支配者被淘汰、非支配者全保留 |

#### 層 4：需要 ffmpeg（audio_extractor）

```bash
sudo apt-get install -y ffmpeg   # 或 brew install ffmpeg

python3 -m pytest tests/test_audio_extractor.py -v
# 期望：9 passed, 0 skipped
```

| 測試項目 | 說明 |
|---------|------|
| `test_has_audio_true/false/nonexistent` | has_audio() 偵測 |
| `test_extract_returns_wav` | 抽出有效 WAV |
| `test_extract_returns_none_for_silent` | 靜音影片回傳 None |
| `test_extract_to_temp_*` | 暫存檔管理 |
| `test_custom_sample_rate` | 取樣率設定 |
| `test_instantiation_defaults` | 預設值（不需 ffmpeg，現在已通過） |

#### 層 5：需要 librosa（audio_features）

```bash
pip install librosa

python3 -m pytest tests/test_audio_features.py -v
# 期望：12 passed, 0 skipped
```

#### 全層測試（最終驗證）

```bash
pip install pandas scikit-learn scipy librosa
sudo apt-get install -y ffmpeg

python3 -m pytest tests/ -v
# 目標：87 passed, 0 skipped, 0 failed
```

### 10.3 各模組手動冒煙測試

#### Registry smoke test

```python
# tests/manual/test_registry_smoke.py
from detectors.registry import build_detector, DEFAULTS

# 1. xception（不需 checkpoint，只測 build 不 crash）
det, meta = build_detector("visual", "xception", {"device": "cpu"})
assert meta["detector_name"] == "Xception"
assert meta["modality"] == "visual"

# 2. pos
det, meta = build_detector("rppg", "pos", {"device": "cpu"})
assert meta["detector_name"] == "POS"

# 3. 規劃中的偵測器
try:
    build_detector("sync", "mrdf", {})
    assert False, "應該丟出 NotImplementedError"
except NotImplementedError as e:
    assert "mrdf" in str(e)

# 4. 不存在的 key
try:
    build_detector("visual", "nonexistent", {})
except KeyError as e:
    assert "nonexistent" in str(e)

print("Registry smoke test: PASSED")
```

#### Serial cascade smoke test（不需 pandas）

```python
import json, tempfile
from fusion.serial_cascade import SerialCascade

cfg = {"stages": [
    {"name": "visual",  "threshold_H": 0.75, "threshold_L": 0.30},
    {"name": "av_sync", "threshold_H": 0.65, "threshold_L": 0.20},
]}
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump(cfg, f); fname = f.name

c = SerialCascade.from_json(fname)

# 基本決策路徑
assert c.fuse(visual_score=0.9).is_fake                    # FAKE exit
assert not c.fuse(visual_score=0.1).is_fake                # REAL exit
assert c.fuse(visual_score=0.5).fake_score == 0.5          # uncertain → fallback
assert c.fuse().fake_score == 0.5 and not c.fuse().is_fake # all None → fallback is_fake=False

# av_sync → sync 正規化
assert c.stages[1].name == "sync"

print("Serial cascade smoke test: PASSED")
```

#### collect_scores.py dry-run（不需 GPU）

```bash
# 建立假影片（最小測試，只跑 rPPG）
python3 -c "
import cv2, numpy as np
out = cv2.VideoWriter('/tmp/test_video.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'), 25, (128,128))
for _ in range(50):
    out.write(np.random.randint(0, 255, (128,128,3), dtype=np.uint8))
out.release()
print('Created /tmp/test_video.mp4')
"

python scripts/collect_scores.py \
    --input_dir /tmp \
    --output_dir /tmp/test_results \
    --device cpu \
    --skip visual sync   # 只跑 POS（純 NumPy，無需 checkpoint）

# 驗證輸出
python3 -c "
import csv
rows = list(csv.DictReader(open('/tmp/test_results/rppg_pos_scores.csv')))
print('Rows:', len(rows))
print('Cols:', list(rows[0].keys()))
print('Status:', rows[0]['status'])
"
```

#### Cascade CSV smoke test（需要 pandas）

```python
import csv, tempfile
from fusion.serial_cascade import SerialCascade

rows = [
    {"stage_order":1,"modality":"visual","threshold_H":0.75,"threshold_L":0.30,
     "pareto_far":0.04,"pareto_frr":0.08},
    {"stage_order":2,"modality":"av_sync","threshold_H":0.65,"threshold_L":0.20,
     "pareto_far":0.04,"pareto_frr":0.08},
]
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows); fname = f.name

c = SerialCascade.from_csv(fname)
assert c.fuse(visual_score=0.9).is_fake
assert not c.fuse(visual_score=0.1).is_fake
assert c.stages[1].name == "sync"   # av_sync 正規化
assert c.fuse().fake_score == 0.5

print("Cascade CSV smoke test: PASSED")
```

### 10.4 效能基準（參考值）

| 情境 | 硬體 | 處理時間 |
|------|------|---------|
| Visual only（Xception, 32 幀）| RTX 3090 | ≈ 0.8 s / video |
| rPPG only（POS, 100 幀）| CPU | ≈ 0.2 s / video |
| Sync（SyncNet + Whisper tiny）| RTX 3090 + CPU | ≈ 3.5 s / video |
| 三模態完整（forensic 模式）| RTX 3090 | ≈ 5 s / video |
| 三模態完整（async）| RTX 3090 | ≈ 3 s / video |

---

## 11. 常見問題排除

### Q1：`ModuleNotFoundError: No module named 'torch'`

```bash
# 確認 torch 是否安裝
pip show torch

# 若未安裝
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 或 CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Q2：`ImportError: DeepfakeBench not found`

執行 DFB-backed 偵測器（UCF/SBI/...）時出現。

```bash
# 方法 1：設定環境變數
export DFB_PATH=/home/user/DeepfakeBench

# 方法 2：確認兄弟目錄結構
ls /home/user/
# 應該看到：anti-deepfake-box  DeepfakeBench

# 確認 DFB training 目錄存在
ls $DFB_PATH/training/detectors/ | grep ucf
# 期望：ucf_detector.py
```

### Q3：SyncDetector 回傳 `score=0.5`（無 checkpoint 時）

這是正常的 motion heuristic fallback，不是錯誤。

```bash
# 確認 checkpoint 是否存在
ls checkpoints/latentsync_syncnet.pth

# 若不存在，跑 download 腳本或手動下載
python download_checkpoints.py

# 或明確指定路徑
python scripts/inference.py \
    --video sample.mp4 \
    --config configs/default.yaml
# 在 default.yaml 確認：sync.syncnet_path: "checkpoints/latentsync_syncnet.pth"
```

### Q4：`ffmpeg: command not found`

```bash
# Ubuntu/Debian
sudo apt-get install -y ffmpeg

# macOS
brew install ffmpeg

# 確認
ffmpeg -version | head -1
# 期望：ffmpeg version 6.x ...
```

### Q5：face_not_detected（人臉偵測失敗）

```bash
# 確認 insightface 安裝
pip show insightface

# 確認 buffalo_sc 模型已下載
python3 -c "
import insightface
app = insightface.app.FaceAnalysis(name='buffalo_sc')
app.prepare(ctx_id=0, det_size=(640,640))
print('InsightFace buffalo_sc: OK')
"

# 若顯示下載進度，等待完成

# 降低偵測門檻（預設影像太小時）
# 在 configs/default.yaml 加入：
# preprocessing:
#   insightface_det_thresh: 0.3   （預設 0.5，降低可偵測更小的臉）
```

### Q6：`pandas` 相關的 skip 或 ImportError

```bash
# 安裝
pip install pandas

# 驗證
python3 -c "import pandas; print(pandas.__version__)"

# 重跑 skip 的測試
python3 -m pytest tests/test_serial_cascade.py::TestSerialCascadeCSV -v
```

### Q7：`sklearn` 相關的 skip

```bash
pip install scikit-learn scipy

python3 -m pytest tests/test_cascade_selection.py -v
```

### Q8：`insightface` 在 Python 3.12+ 安裝失敗

```bash
# 改用指定版本
pip install insightface==0.7.3 --no-build-isolation

# macOS ARM 已知問題：需要 Rosetta
ARCHFLAGS="-arch arm64" pip install insightface
```

### Q9：DFB sys.path 衝突（`detectors` 套件錯誤）

ADB 和 DFB 都有 `detectors/` 套件，import 可能混用。

```bash
# 確認是否為混用問題
python3 -c "
import sys
sys.path.insert(0, '/home/user/anti-deepfake-box')
from detectors import VisualDetector  # 應該是 ADB 的
print('VisualDetector module:', VisualDetector.__module__)
# 期望：detectors.visual_detector（非 DFB 的）
"

# 若發生混用，確認 PYTHONPATH 設定
echo $PYTHONPATH
# anti-deepfake-box 應該在前，或不要把 DFB training/ 加入 PYTHONPATH
```

`DFBVisualDetector` 的 `_dfb_path_ctx` context manager 已自動處理此問題，正常情況下不需要手動干預。

### Q10：`test_smooth_bboxes` 相關測試失敗

應該不會發生（已有 short-sequence guard）。若失敗，確認是否在 main 分支：

```bash
git log --oneline -3
# 應該看到：8a6a951 feat: add per-modality detector registry...
#           895ac45 fix: remove decord from hard deps...

python3 -m pytest tests/test_preprocessing_new.py::TestSmoothBboxes -v
# 期望：7 passed
```

---

*最後更新：2026-05-19。如發現文件錯誤，請參考 `docs/v5_implementation_report.md` 中的技術細節。*
