# v5 Paper-Aligned & GP-Ready 實作詳細報告

> 撰寫日期：2026-05-19（更新：detector registry 新增）
> 對應 commits：`anti-deepfake-box@8a6a951`、`DeepfakeBench@59ed713`

---

## 目錄

1. [系統全貌](#1-系統全貌)
2. [設計動機與版本演進](#2-設計動機與版本演進)
3. [檔案依賴關係圖](#3-檔案依賴關係圖)
4. [Component 0：偵測器登錄機制（registry.py）](#4-component-0偵測器登錄機制)
5. [Component 1：統一評分收集（collect_scores.py）](#5-component-1統一評分收集)
6. [Component 2：弱分類器篩選（cascade_selection.py）](#6-component-2弱分類器篩選)
7. [Component 3：串列級聯融合（serial_cascade.py）](#7-component-3串列級聯融合)
8. [Component 4：前處理升級（face_extractor / audio_extractor）](#8-component-4前處理升級)
9. [Component 5：DeepfakeBench 音訊管線整合](#9-component-5deepfakebench-音訊管線整合)
10. [資料流端對端追蹤](#10-資料流端對端追蹤)
11. [關鍵設計決策與原因](#11-關鍵設計決策與原因)
12. [驗證方法](#12-驗證方法)
13. [已知限制與後續工作](#13-已知限制與後續工作)

---

## 1. 系統全貌

Anti-Deepfake-Box（ADB）是針對**詐騙電話情境**設計的多模態 deepfake 偵測系統。三個偵測**模態**各自可搭配多個**演算法（偵測器）**，由 `detectors/registry.py` 統一管理切換。

### 三模態架構

| 模態 | 偵測原理 | 輸入 |
|------|---------|------|
| Visual | 影像紋理鑑真偽（GAN 痕跡） | 人臉裁切 256×256（DFB）或 299×299（Xception） |
| rPPG | 遠端心率訊號真實性（SNR） | 人臉裁切 128×128 |
| AV Sync | 音視覺嘴型同步誤差 | 人臉 256×256 + 16kHz WAV |

### 各模態偵測器清單（A 組週報對應）

#### Visual Detectors

| 偵測器 | 狀態 | 說明 |
|--------|------|------|
| **Xception** | ✅ 已測完（Score CSV 已產出） | ImageNet + FF++ c23 fine-tuned；域內穩健基準 |
| **UCF** | ✅ 已測完 | SOTA；強域內 + 較好跨域韌性 |
| **SBI** | ✅ 已測完 | 跨域泛化最強，OOD 表現第一 |
| **F3Net** | ✅ 已測完 | 域內極強；閾值窗口寬廣 |
| **SPSL** | ✅ 已測完 | 系統性高分偏移問題，閾值需特別校正 |
| **SRM** | ✅ 已測完 | 閾值懸崖效應嚴重，可用窗口極窄 |
| **EfficientNet-B4** | 🔲 規劃實作 | 原與 Xception 並列首選；checkpoint 待取得 |
| **Face X-ray** | 🔲 規劃實作 | 找到權重才跑 |
| **LSDA** | 🔲 規劃實作 | 本週僅撰寫 narrative，不產出 CSV |

#### rPPG Detectors

| 偵測器 | 狀態 | 說明 |
|--------|------|------|
| **POS** | ✅ 已測完 | Wang 2017；無 checkpoint；已在 FF++ 校正 SNR 閾值 |
| **TS-CAN** | 🔲 備用（確認權重後跑） | rPPG-Toolbox；確認 checkpoint 後交出 sweep CSV |
| **PhysMamba** | 🔲 規劃實作 | GP validator 已納入考量 |

#### AV Sync Detectors

| 偵測器 | 狀態 | 說明 |
|--------|------|------|
| **SyncNet (LatentSync-MDS)** | ✅ 已測完 | DFDC、DF-TIMIT 已確認；FakeAVCeleb 待資料集申請 |
| **MRDF** | 🔲 規劃實作 | 需自行訓練，本週不產出 CSV |
| **LatentSync** | 🔲 調查中 | 備選模型 |
| **AVAD** | 🔲 調查中 | 備選模型 |

### v5 外部管線總覽

```
影片目錄
    ↓ collect_scores.py          ← 統一批次評分（支援偵測器切換）
    ↓
  results/
    visual_xception_scores.csv
    visual_ucf_scores.csv        ← 同一目錄存多個偵測器的結果
    visual_sbi_scores.csv
    rppg_pos_scores.csv
    sync_syncnet_scores.csv
    ↓
  cascade_selection.py           ← 4 階段弱分類器篩選（從多個 CSV 選最佳子集）
    ↓
  real_data_vip_settings.csv     ← GP solver 輸出（fusion_solver_prod_v1.ipynb）
    ↓
  serial_cascade.py              ← BMMA-GPT 雙閾值串列級聯
    ↓
  FusionResult（FAKE / REAL）
```

---

## 2. 設計動機與版本演進

### v1–v3（遭拒原因）

| 版本 | 問題 |
|------|------|
| v1 | 忽略 GP solver 實際輸出格式（假設 JSON，實際為 CSV） |
| v2 | CSV schema 不符 Week 11 GP data contract（欄位名稱錯誤） |
| v3 | 無 NPZ 音訊嵌入、無臉部篩選、無時序平滑；每模態硬編碼單一偵測器 |

### v4（過渡版）

修正 GP 接口（CSV→cascade）、補齊 12 欄位 schema，但前處理仍為 patch 方式，偵測器仍硬編碼。

### v5（當前版本）

三項核心升級：

**1. 偵測器可切換（detector registry）**  
不再硬編碼 Xception/POS/SyncNet，改用 `detectors/registry.py` 登錄所有已測與規劃偵測器，`collect_scores.py` 透過 CLI flags 選擇。

**2. 對齊 DeepfakeBench-MM 論文**（CVPR 2023）

| 論文要求 | v5 實作 |
|---------|---------|
| 前導靜音移除 80 ms（§B.1） | `AudioExtractor.skip_leading_ms=80`（forensic 模式） |
| NPZ 格式預解碼音訊（§3.1） | `_save_cache()` 嵌入 `audio_samples` + `frame_to_audio_offset` |
| 單臉幀篩選（§3.2） | `_filter_single_face_frames(min_frames=8)` |
| 時序平滑 window=3（§3.2） | `_smooth_bboxes(window=3)` causal MA |

**3. 多偵測器並存輸出**  
CSV 命名從 `visual_scores.csv` 改為 `visual_xception_scores.csv`，讓多次 run 結果共存於同一 `output_dir`，供 `cascade_selection.py` 比較選優。

---

## 3. 檔案依賴關係圖

```
anti-deepfake-box/
├── detectors/
│   ├── __init__.py               ★ 修改（export registry）
│   ├── base_detector.py          既有
│   ├── registry.py               ★ 新增（所有偵測器的登錄表）
│   ├── visual_detector.py        既有（Xception；ADB 原生）
│   ├── dfb_visual_wrapper.py     ★ 新增（通用 DFB 偵測器包裝器）
│   ├── rppg_detector.py          既有（POS；ADB 原生）
│   ├── rppg_tscan_detector.py    ★ 新增（TS-CAN stub）
│   ├── rppg_physmamba_detector.py ★ 新增（PhysMamba stub）
│   ├── sync_detector.py          既有（SyncNet；ADB 原生）
│   └── fft_detector.py           既有
│
├── scripts/
│   ├── collect_scores.py         ★ 修改（registry 切換、CLI 旗標、CSV 命名）
│   │   ├── uses: detectors/registry.py     (build_detector, DEFAULTS)
│   │   ├── uses: preprocessing/face_extractor.py  (UnifiedFaceExtractor)
│   │   ├── uses: preprocessing/audio_extractor.py (AudioExtractor)
│   │   └── uses: 透過 registry 實例化各偵測器
│   └── inference.py              ★ 修改（build_detectors 使用 registry）
│
├── fusion/
│   ├── __init__.py               ★ 修改（export SerialCascade）
│   ├── weighted_ensemble.py      既有（FusionResult dataclass）
│   ├── meta_classifier.py        既有
│   ├── serial_cascade.py         ★ 新增（BMMA-GPT 雙閾值）
│   └── cascade_selection.py      ★ 新增（4 階段弱分類器篩選）
│
├── preprocessing/
│   ├── audio_extractor.py        ★ 修改（新增 extract_to_array）
│   └── face_extractor.py         ★ 修改（_filter_single_face_frames,
│                                           _smooth_bboxes, NPZ 音訊嵌入）
│
├── evaluation/
│   └── metrics.py                既有（compute_metrics, compute_eer, DetectionMetrics）
│
└── configs/
    └── default.yaml              ★ 修改（detector 鍵、cascade_config 鍵）

DeepfakeBench/training/
├── config/
│   └── train_config.yaml         ★ 修改（with_audio: false）
├── dataset/
│   └── abstract_dataset.py       ★ 修改（5 處改動）
└── detectors/
    └── adb_sync_detector.py      ★ 修改（_get_audio_path 優先 NPZ）
```

---

## 4. Component 0：偵測器登錄機制

### 檔案：`detectors/registry.py`（新增）

#### 設計動機

原始架構將 Xception / POS / SyncNet 硬編碼在 `collect_scores.py` 和 `inference.py` 中：

```python
# 修改前（錯誤）
active["visual"] = VisualDetector({"device": device})      # 永遠 Xception
active["rppg"]   = RPPGDetector({"device": "cpu"})         # 永遠 POS
active["sync"]   = SyncDetector({...})                     # 永遠 SyncNet
```

A 組實際測試了 6 個 visual 偵測器、2 個 rPPG 偵測器，GP cascade_selection 需要比較這些偵測器的 CSV 才能選出最佳子集。單一硬編碼架構無法支援這個工作流程。

#### Registry 結構

```python
# detectors/registry.py

VISUAL_REGISTRY = {
    "xception": {"factory": _vis_xception, "detector_name": "Xception",
                 "modality": "visual", "score_type": "probability", "status": "available"},
    "ucf":      {"factory": _vis_dfb("ucf"), "detector_name": "UCF",
                 "modality": "visual", "score_type": "probability", "status": "dfb_required"},
    "sbi":      {"factory": _vis_dfb("sbi"), "detector_name": "SBI", ...},
    "f3net":    {"factory": _vis_dfb("f3net"), ...},
    "spsl":     {"factory": _vis_dfb("spsl"), ...},
    "srm":      {"factory": _vis_dfb("srm"), ...},
    # 規劃：efficientnet_b4, facexray, lsda
}

RPPG_REGISTRY = {
    "pos":       {"factory": _rppg_pos, "detector_name": "POS", "status": "available"},
    "tscan":     {"factory": _rppg_tscan, "detector_name": "TS-CAN", "status": "rppg_toolbox"},
    "physmamba": {"factory": _rppg_physmamba, "detector_name": "PhysMamba", "status": "planned"},
}

SYNC_REGISTRY = {
    "syncnet":   {"factory": _sync_syncnet, "detector_name": "SyncNet", "status": "available"},
    "mrdf":      {"factory": _sync_stub("mrdf"), ...},      # raises NotImplementedError
    "latentsync":{"factory": _sync_stub("latentsync"), ...},
    "avad":      {"factory": _sync_stub("avad"), ...},
}
```

#### Status 標籤語義

| 標籤 | 意義 |
|------|------|
| `available` | 已實作且測試，可直接使用 |
| `dfb_required` | 包裝 DFB 偵測器；需要 `export DFB_PATH=/path/to/DeepfakeBench` |
| `rppg_toolbox` | 包裝 rPPG-Toolbox 偵測器；需確認 checkpoint 後實作 |
| `planned` | 介面已定義；呼叫 factory 會丟出 `NotImplementedError` 並附說明 |

#### 統一建立介面

```python
from detectors.registry import build_detector

detector, meta = build_detector("visual", "ucf", {
    "device": "cuda",
    "dfb_pretrained": "/ckpts/ucf_ff.pth",
})
# meta == {"detector_name": "UCF", "modality": "visual",
#          "score_type": "probability", "status": "dfb_required", ...}
```

`build_detector` 回傳 `(detector_instance, metadata_dict)`，其中 metadata 的 `detector_name` 欄位直接用於寫入 CSV 的 `detector_name` 欄。

---

### 檔案：`detectors/dfb_visual_wrapper.py`（新增）

DFB visual 偵測器（UCF / SBI / F3Net / SPSL / SRM）存在於 DeepfakeBench 的訓練目錄中。`DFBVisualDetector` 是一個通用包裝器，讓 ADB 的 SSOT face extraction pipeline 可以直接呼叫任意 DFB 視覺偵測器。

#### 輸入轉換

DFB 偵測器期望 `data_dict["image"]`：`(B, 3, H, W)` float tensor，正規化至 `[-1, 1]`。  
FaceTrack 提供 `aligned_256`：`(T, 256, 256, 3)` uint8 RGB。

```python
def _to_dfb_batch(self, face_track):
    crops = face_track.aligned_256                        # (T, 256, 256, 3) uint8
    x = torch.from_numpy(crops.astype(np.float32) / 127.5 - 1.0)
    x = x.permute(0, 3, 1, 2)                           # (T, 3, 256, 256)
    return {"image": x.to(self.device), "label": zeros, ...}
```

#### sys.path 衝突解法

ADB 有自己的 `detectors/` 套件，DFB 也有一個同名的 `detectors/`（含 `DETECTOR` 登錄表）。直接 `import detectors` 會拿到 ADB 的版本。

解法：在 `load()` 時用 context manager 暫時交換：

```python
@contextlib.contextmanager
def _dfb_path_ctx(dfb_training: str):
    # 1. 暫存 ADB 的 sys.modules["detectors*"] 條目
    adb_mods = {k: v for k, v in sys.modules.items()
                if k == "detectors" or k.startswith("detectors.")}
    for k in adb_mods:
        del sys.modules[k]

    # 2. DFB 路徑插到最前
    sys.path.insert(0, dfb_training)
    try:
        yield   # 在此區間 import detectors → DFB 的版本
    finally:
        # 3. 移除 DFB 的 detectors 條目，還原 ADB 的
        for k in [k for k in sys.modules if k == "detectors" or k.startswith("detectors.")]:
            del sys.modules[k]
        sys.path.remove(dfb_training)
        sys.modules.update(adb_mods)
```

這樣 `DFBVisualDetector` 的 `_dfb_det.forward()` 在 context 結束後仍可正常執行，因為 DFB 偵測器類別的 globals 在匯入時已綁定，不依賴 `sys.modules` 的即時狀態。

#### DFB 路徑探測順序

1. `$DFB_PATH` 環境變數：`export DFB_PATH=/home/user/DeepfakeBench`
2. 兄弟目錄：`../DeepfakeBench/training` 或 `../deepfakebench/training`

---

## 5. Component 1：統一評分收集

### 檔案：`scripts/collect_scores.py`（修改）

#### 與前版的差異

| 面向 | v4（原始） | v5（當前） |
|------|-----------|-----------|
| 偵測器選擇 | 硬編碼 Xception / POS / SyncNet | `build_detector()` 從 registry 實例化 |
| DETECTOR_META | 靜態 dict，只寫 Xception/POS/SyncNet | 從 registry metadata 動態取得 |
| CSV 輸出命名 | `visual_scores.csv` | `visual_xception_scores.csv` |
| CLI 偵測器旗標 | 無 | `--visual_detector`, `--rppg_detector`, `--sync_detector` |
| DFB 偵測器支援 | 無 | 透過 `DFBVisualDetector` 包裝 |

#### 核心設計：12 欄位 GP Data Contract

```python
FIELDNAMES = [
    "sample_id",          # 影片 stem（GP 用此鍵對齊三份 CSV）
    "dataset",            # FakeAVCeleb / FF++ / ...
    "label",              # 0=real, 1=fake, ""=未知
    "detector_name",      # 從 registry meta 取得（Xception / UCF / SBI / POS / SyncNet...）
    "modality",           # visual / rppg / av_sync
    "fake_score",         # float 或 ""（失敗時）
    "score_type",         # probability / snr / sync_error
    "inference_time_ms",  # 浮點數
    "window_start_sec",   # "N/A"（全影片偵測器）
    "window_end_sec",     # "N/A"
    "status",             # "ok" 或 "failed"
    "error_message",      # "" 或錯誤描述（最長 120 字）
]
```

#### 偵測器建立（使用 registry）

```python
active_det, active_meta, active_key = {}, {}, {}

if "visual" not in skip:
    det, meta = build_detector("visual", visual_detector, vis_cfg)
    active_det["visual"]  = det
    active_meta["visual"] = meta        # {"detector_name": "UCF", "modality": "visual", ...}
    active_key["visual"]  = visual_detector   # "ucf"

# CSV key = "{modality}_{detector_key}"
row_key_for = {m: f"{active_meta[m]['modality']}_{active_key[m]}" for m in active_det}
# e.g. "visual_ucf", "rppg_pos", "av_sync_syncnet"
```

#### 關鍵原則：永不丟棄失敗樣本

```python
try:
    score = det.detect(face_track, ...)
    if score is None:
        status, err = "failed", "detector_returned_none"
except Exception as exc:
    status, err = "failed", str(exc)[:120]

rows[row_key].append({
    "fake_score": f"{score:.6f}" if score is not None else "",  # 空字串，非 NaN
    "status": status,
    "error_message": err,
})
```

**原因**：GP solver 的 `data_validator_v1` 以 `sample_id` 對齊三份 CSV。若丟棄失敗行，三份 CSV 行數不同，`sample_id` 對齊錯位。

#### CSV 輸出命名設計

輸出檔名格式：`{modality}_{detector_key}_scores.csv`

```bash
results/
  visual_xception_scores.csv     ← --visual_detector xception（第一次 run）
  visual_ucf_scores.csv          ← --visual_detector ucf（第二次 run）
  visual_sbi_scores.csv          ← --visual_detector sbi（第三次 run）
  rppg_pos_scores.csv
  av_sync_syncnet_scores.csv
```

所有 CSV 共存於同一目錄，`cascade_selection.py` 透過 `glob("*_scores.csv")` 一次性讀取所有結果，自動選出最佳偵測器子集。

#### CLI 介面

```bash
# 預設（Xception + POS + SyncNet）
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --label_csv labels.csv \
    --output_dir results/ \
    --dataset FakeAVCeleb \
    --mode forensic

# 改用 UCF（需要 DFB）
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --output_dir results/ \
    --visual_detector ucf \
    --dfb_pretrained /ckpts/ucf_ff.pth

# 跳過 sync（FF++ 無音訊）
python scripts/collect_scores.py \
    --input_dir /data/FF++ \
    --output_dir results/ff \
    --dataset FF++ \
    --skip sync

# 列出所有可選偵測器
python scripts/collect_scores.py --help
```

`--visual_detector` choices：`xception ucf sbi f3net spsl srm efficientnet_b4 facexray lsda`  
`--rppg_detector` choices：`pos tscan physmamba`  
`--sync_detector` choices：`syncnet mrdf latentsync avad`

---

## 6. Component 2：弱分類器篩選

### 檔案：`fusion/cascade_selection.py`（新增，328 行）

#### 使用的既有檔案

- `evaluation/metrics.py`：`compute_metrics()` → `DetectionMetrics`（含 AUC、EER、ACC、AP）
- `evaluation/metrics.py`：`compute_eer()` → EER 與對應閾值
- `scipy.stats.pearsonr`、`sklearn.metrics.roc_curve`

#### 4 階段篩選流程

```
Stage 1: AUC 篩選           drop if AUC < 0.55
Stage 2: 相關性篩選          drop 較低 AUC 的一方（|r| > 0.90）
Stage 3: FAR/FRR 分析        純資訊（不淘汰）
Stage 4: Pareto 支配篩選     drop 在 AUC 和 EER 雙指標均被支配的分類器
```

**Stage 1 細節**：
```python
def filter_by_auc(data, min_auc=0.55):
    for name, (ids, scores, labels) in data.items():
        m = compute_metrics(labels, scores, dataset="stage1_auc", detector=name)
        if m.auc >= min_auc:
            retained[name] = (ids, scores, labels)
    return retained, metrics
```

**Stage 2 細節**：對齊 `sample_id`（用 `NaN` 填補缺失樣本），再計算 Pearson r：
```python
all_ids = sorted({sid for ids, _, _ in data.values() for sid in ids})
scores_aligned[name] = np.array([id_to_score.get(sid, float("nan")) for sid in all_ids])
```
當 `|r| > 0.90`：淘汰 AUC 較低的那個。

**Stage 4 Pareto 細節**：
```python
# Y 支配 X：AUC 不差且 EER 不差，且至少一項嚴格更好
if m2.auc >= m1.auc and m2.eer <= m1.eer:
    if m2.auc > m1.auc or m2.eer < m1.eer:
        dominated.add(n1)
```

#### 輸出

```python
retained = select_classifiers(
    csv_dir="results/",   # 掃描目錄內所有 *_scores.csv
    min_auc=0.55,
    max_corr=0.90,
    output_json="results/selected_classifiers.json",
)
# → e.g. ["visual_sbi", "sync_syncnet"]
```

`output_json` 包含：selected 名單、所有分類器的 AUC/EER/ACC 指標、篩選參數。

---

## 7. Component 3：串列級聯融合

### 檔案：`fusion/serial_cascade.py`（新增，252 行）

#### 使用的既有檔案

- `fusion/weighted_ensemble.py`：`FusionResult` dataclass（drop-in 相容，不修改其定義）

#### GP Solver 接口

GP solver（`fusion_solver_prod_v1.ipynb`）輸出 `real_data_vip_settings.csv`：

```
stage_order, modality, threshold_H, threshold_L, pareto_far, pareto_frr
1,           visual,   0.75,        0.30,        0.04,       0.08
2,           rppg,     0.70,        0.25,        0.04,       0.08
3,           av_sync,  0.65,        0.20,        0.04,       0.08
```

`SerialCascade` 用 pandas 讀取，按 `stage_order` 排序，並正規化 modality 名稱：

```python
MODALITY_ALIASES = {"av_sync": "sync"}  # GP solver → 內部名稱

def _load_stages_csv(self, path):
    df = pd.read_csv(path).sort_values("stage_order").reset_index(drop=True)
    stages = []
    for _, row in df.iterrows():
        raw_name = str(row["modality"]).strip()
        name = self.MODALITY_ALIASES.get(raw_name, raw_name)
        stages.append(CascadeStage(name=name,
                                   H=float(row["threshold_H"]),
                                   L=float(row["threshold_L"])))
    return stages
```

#### BMMA-GPT 雙閾值決策邏輯

每個 Stage k 拿到分數 p_k：

```
p_k >= H_k  →  Early exit: FAKE（fake_score = p_k）
p_k <= L_k  →  Early exit: REAL（fake_score = p_k）
L_k < p_k < H_k  →  Uncertain，傳遞至 Stage k+1
score = None  →  跳過此 stage（模態不可用）
全部跑完仍無決策  →  Fallback: fake_score=0.5, is_fake=False
```

```python
def fuse(self, visual_score=None, rppg_score=None, sync_score=None) -> FusionResult:
    score_map = {"visual": visual_score, "rppg": rppg_score, "sync": sync_score}
    final_score = self.fallback_score  # 0.5
    exit_stage = None

    for stage in self.stages:
        s = score_map.get(stage.name)
        if s is None:
            continue
        if s >= stage.H:
            final_score = s; exit_stage = stage.name; break
        if s <= stage.L:
            final_score = s; exit_stage = stage.name; break

    if exit_stage is None:
        is_fake = False          # Fallback → not fake（與 WeightedEnsemble 一致）
    else:
        is_fake = final_score >= self.default_threshold

    return FusionResult(fake_score=float(final_score), is_fake=is_fake, ...)
```

**Bug fix**：原先 `is_fake = (0.5 >= 0.5) = True`，與 `WeightedEnsemble` 全 None 時回傳 `is_fake=False` 不一致。修正為 `exit_stage is None` 時強制 `is_fake=False`。

#### 三種建立方式

```python
# 方式 1：直接從 GP solver CSV（主要方式）
cascade = SerialCascade.from_csv("results/real_data_vip_settings.csv")

# 方式 2：從 config dict（inference.py 呼叫）
cascade = SerialCascade({
    "fusion": {"cascade_config": "results/real_data_vip_settings.csv", "threshold": 0.50}
})

# 方式 3：從 JSON（單元測試 / 手動設定）
cascade = SerialCascade.from_json("tests/cascade_config.json")
```

---

## 8. Component 4：前處理升級

### 8.1 `preprocessing/audio_extractor.py`：新增 `extract_to_array()`

#### 新增方法

```python
def extract_to_array(self, video_path: str) -> tuple[np.ndarray, int] | None:
    cmd = ["ffmpeg", "-y"]
    if self.skip_leading_ms > 0:
        cmd += ["-ss", f"{self.skip_leading_ms / 1000:.3f}"]
    cmd += ["-i", str(video_path), "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "1",
            "-f", "s16le",
            "pipe:1"]              # stdout pipe，不寫磁碟
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        return None
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    return samples, self.sample_rate
```

**設計原因**：不需要 temp file → 省 I/O；`skip_leading_ms=80` 自動套用；傳回 `np.int16` ndarray 可直接塞入 NPZ，與 DeepfakeBench-MM §3.1 格式一致。

**對齊常數**（硬編碼，論文規格）：
```
16000 Hz / 25 fps = 640 samples / frame     (Feng et al., CVPR 2023 §3.1)
STFT hop = 160 samples (10 ms) → 4 STFT frames / video frame
Leading silence skip = 80 ms = 1280 samples  (DFB-MM §B.1)
```

---

### 8.2 `preprocessing/face_extractor.py`：三項新增

#### (a) `_filter_single_face_frames()`

```python
def _filter_single_face_frames(self, faces_per_frame, min_frames=8):
    single = [(idx, faces[0]) for idx, faces in faces_per_frame if len(faces) == 1]
    if len(single) >= min_frames:
        return single
    return [
        (idx, max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        for idx, faces in faces_per_frame if len(faces) >= 1
    ]
```

**原因**：多張臉的幀無法確定哪張是說話者，混入會污染 AV sync 的嘴型對齊。門檻 8 幀（≈ 0.32 秒 @25fps）是論文的最低可靠窗口。

#### (b) `_smooth_bboxes()`

```python
def _smooth_bboxes(self, bboxes: np.ndarray, window: int = 3) -> np.ndarray:
    bboxes = bboxes.copy().astype(np.float32)
    if len(bboxes) < window:
        return bboxes    # Bug fix：短序列直接回傳，避免 np.convolve shape 錯位
    cx = (bboxes[:,0] + bboxes[:,2]) / 2.0
    cy = (bboxes[:,1] + bboxes[:,3]) / 2.0
    w  =  bboxes[:,2] - bboxes[:,0]
    h  =  bboxes[:,3] - bboxes[:,1]
    k = np.ones(window, dtype=np.float32) / window
    for arr in (cx, cy, w, h):
        arr[:] = np.convolve(arr, k, mode="same")
    return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1).astype(bboxes.dtype)
```

呼叫點：`extract()` cache miss 路徑，`FaceTrack` 建立之前：
```python
if len(valid_bboxes) >= 3:
    valid_bboxes = self._smooth_bboxes(valid_bboxes)
```

**原因**：InsightFace 每幀獨立偵測，bbox 抖動會造成裁切區域閃爍，在 rPPG 和 SyncNet 管線中製造假訊號。MA window=3 是論文最小有效平滑值。

#### (c) `_save_cache()` 嵌入音訊

```python
def _save_cache(self, track, path, audio_samples=None, audio_sr=16000):
    arrays = {
        "frame_indices": track.frame_indices,
        "bboxes":        track.bboxes,
        "landmarks":     track.landmarks,
        "fps":           np.array([track.fps]),
        "video_path":    np.array([track.video_path]),
    }
    if self.cache_pixel_crops:
        arrays["aligned_256"] = track.aligned_256
        if audio_samples is not None:
            arrays["audio_samples"] = audio_samples.astype(np.int16)
            arrays["audio_sr"] = np.array([audio_sr])
            arrays["frame_to_audio_offset"] = (
                track.frame_indices * audio_sr / track.fps
            ).astype(np.int64)
    np.savez_compressed(str(path), **arrays)
```

**NPZ 檔案結構（forensic 模式）**：

| 陣列 | dtype | shape | 說明 |
|------|-------|-------|------|
| `frame_indices` | int64 | (T,) | 原始幀號 |
| `bboxes` | float32 | (T, 4) | xyxy，已平滑 |
| `landmarks` | float32 | (T, 5, 2) | 5 點關鍵點 |
| `fps` | float64 | (1,) | 目標幀率 |
| `aligned_256` | uint8 | (T, 256, 256, 3) | RGB 裁切圖 |
| `audio_samples` | int16 | (N,) | PCM，已移除前 80 ms |
| `audio_sr` | int64 | (1,) | 取樣率（16000） |
| `frame_to_audio_offset` | int64 | (T,) | 每幀對應的 audio sample index |

---

## 9. Component 5：DeepfakeBench 音訊管線整合

### 9.1 `training/dataset/abstract_dataset.py`（5 處改動）

#### 改動 1：train 模式收集 `name_list`

```python
# 修改前：tmp_name 被丟棄
image_list, label_list = [], []
for one_data in dataset_list:
    tmp_image, tmp_label, tmp_name = ...
    image_list.extend(tmp_image); label_list.extend(tmp_label)

# 修改後：保留 name_list
image_list, label_list, name_list = [], [], []
for one_data in dataset_list:
    tmp_image, tmp_label, tmp_name = ...
    image_list.extend(tmp_image); label_list.extend(tmp_label); name_list.extend(tmp_name)
```

#### 改動 2：擴充 `data_dict`

```python
self.data_dict = {
    'image':      self.image_list,
    'label':      self.label_list,
    'video_name': list(name_list),         # 新增
    'npz_path':   [None] * len(name_list), # 新增（由前處理設定後填入）
}
```

#### 改動 3：`__getitem__` 載入 NPZ 音訊，回傳 6-tuple

```python
audio_samples = None
if self.config.get('with_audio', False):
    npz_path = self.data_dict.get('npz_path', [None])[index]
    if npz_path and os.path.exists(str(npz_path)):
        npz = np.load(str(npz_path), allow_pickle=False)
        if 'audio_samples' in npz:
            audio_samples = npz['audio_samples']  # int16 ndarray

video_name = self.data_dict.get('video_name', [None])[index]
return image_tensors, label, landmark_tensors, mask_tensors, video_name, audio_samples
```

#### 改動 4：`collate_fn` 向後相容 tuple 長度守衛

```python
n = len(batch[0])
if n == 6:
    images, labels, landmarks, masks, video_names, audio_list = zip(*batch)
else:
    images, labels, landmarks, masks = zip(*batch)   # 子類別仍回傳 4-tuple
    video_names = None; audio_list = None

data_dict['video_path'] = list(video_names) if video_names is not None else None
data_dict['audio'] = (
    list(audio_list) if audio_list and any(a is not None for a in audio_list) else None
)
```

#### 改動 5：`train_config.yaml`

```yaml
with_audio: false   # set true to load pre-decoded PCM from NPZ cache
```

---

### 9.2 `training/detectors/adb_sync_detector.py`：`_get_audio_path()` 升級

```python
def _get_audio_path(self, data_dict):
    # Priority 1：NPZ 預解碼 PCM（with_audio=True 時由 dataset 載入）
    audio = data_dict.get("audio")
    if audio is not None:
        if isinstance(audio, list): audio = audio[0]
        if audio is not None: return audio  # int16 ndarray

    # Priority 2：即時從 video_path 抽音（向後相容）
    video_path = data_dict.get("video_path")
    if video_path is None: return None
    if isinstance(video_path, list): video_path = video_path[0]
    if video_path and Path(str(video_path)).exists():
        return self.audio_extractor.extract_to_temp(str(video_path))
    return None
```

---

## 10. 資料流端對端追蹤

### GP 訓練期：多偵測器評分 → 選優

```
# 步驟 1：針對每個 visual 偵測器分別跑一次 collect_scores
python scripts/collect_scores.py --visual_detector xception --output_dir results/ ...
python scripts/collect_scores.py --visual_detector ucf --dfb_pretrained /ckpts/ucf.pth --output_dir results/ ...
python scripts/collect_scores.py --visual_detector sbi --dfb_pretrained /ckpts/sbi.pth --output_dir results/ ...
# 以此類推：f3net, spsl, srm

# 步驟 2：弱分類器篩選（比較所有 *_scores.csv）
from fusion.cascade_selection import select_classifiers
selected = select_classifiers("results/", min_auc=0.55, max_corr=0.90)
# e.g. → ["visual_sbi", "sync_syncnet"]  （SPSL/SRM 因偏移/懸崖被淘汰）

# 步驟 3：GP Solver（Notebook）
fusion_solver_prod_v1.ipynb
  load_real_data(["results/visual_sbi_scores.csv", ..., "results/av_sync_syncnet_scores.csv"])
  → 最佳化雙閾值 {H_k, L_k}
  → 輸出 real_data_vip_settings.csv
```

### 推論期：單視訊

```
python scripts/inference.py \
    --video sample.mp4 \
    --config configs/default.yaml \
    --fusion_mode cascade
```

configs/default.yaml 的 `detectors.visual.detector: "sbi"` 決定哪個偵測器被呼叫。

### DeepfakeBench 訓練流程（with_audio=True）

```
abstract_dataset.__init__
  → image_list, label_list, name_list（train 模式現在也保留 name_list）
  → data_dict = {image, label, video_name, npz_path}

abstract_dataset.__getitem__(index)
  → npz = np.load(data_dict['npz_path'][index])
  → audio_samples = npz['audio_samples']   # int16 PCM
  → return (image_tensors, label, landmarks, masks, video_name, audio_samples)

abstract_dataset.collate_fn(batch)
  → len(batch[0]) == 6  → 解包 6 值
  → data_dict['audio'] = list(audio_list)

ADBSyncDetector._get_audio_path(data_dict)
  → Priority 1: data_dict['audio'][0]          # int16 ndarray（快）
  → Priority 2: extract_to_temp(video_path)    # 即時抽取（慢，向後相容）
```

---

## 11. 關鍵設計決策與原因

### 決策 1：偵測器登錄機制（registry）而非直接 import

**問題**：collect_scores 和 inference 都硬編碼偵測器類別，A 組需要能切換 6+ 種 visual 偵測器。

**決策**：中央 registry（`detectors/registry.py`）一次定義全部；所有呼叫端透過 `build_detector(modality, key, config)` 取得實例。

**原因**：新增偵測器只需改一個檔案（registry.py）；呼叫端完全不知道底層用哪個類別；未實作的偵測器有清楚的 `NotImplementedError` 而非神秘的 import 失敗。

### 決策 2：CSV 命名改為 `{modality}_{key}_scores.csv`

**問題**：原本 `visual_scores.csv` 表示「所有 visual 偵測器」，但 cascade_selection 需要比較多個 visual 偵測器。

**決策**：每次 run 產生一份帶偵測器名稱的 CSV（`visual_xception_scores.csv`、`visual_ucf_scores.csv`...）。

**代價**：舊的 GP solver notebook 若直接 hardcode `visual_scores.csv` 需要更新路徑。  
**緩解**：`cascade_selection` 設計為掃描 `glob("*_scores.csv")`，不需要固定檔名。

### 決策 3：GP solver 輸出格式選 CSV 非 JSON

**決策**：`SerialCascade` 直接讀 GP solver 輸出的 `real_data_vip_settings.csv`，而非期待一個 JSON 設定檔。

**原因**：修改「接收端」（SerialCascade）比修改「發送端」（GP solver notebook）成本低。GP solver 已在多人協作環境中穩定使用。

### 決策 4：`av_sync` vs `sync` modality 名稱

**問題**：GP solver 用 `"av_sync"`；fuse() 接受 `sync_score=`；registry 用 `"sync"` 作為 key。

**解法**：`MODALITY_ALIASES = {"av_sync": "sync"}` 在 `_load_stages_csv` 一次正規化；三個系統各保持自己的命名不動。

### 決策 5：DFB sys.path context manager 設計

**問題**：ADB 和 DFB 都有名為 `detectors` 的 Python 套件，直接 `import detectors` 會發生命名衝突。

**決策**：在 `DFBVisualDetector.load()` 中用 context manager 暫時交換 sys.modules 條目，在 context 內完成 DFB import，context 結束後還原 ADB 的 `detectors`。

**原因**：DFB 偵測器類別的方法 globals 在匯入時已綁定，不依賴 sys.modules 的即時狀態，所以 context 外部呼叫 forward() 仍然正確。

### 決策 6：永不丟棄失敗樣本

**原因**：GP data_validator_v1 以 `sample_id` 為鍵做 CSV left join，三份 CSV 行數必須一致。丟棄失敗行會造成對齊錯位。

### 決策 7：`collate_fn` tuple 長度守衛

**原因**：DFB 有多個子類（`FFDataset`、`CelebDFDataset` 等）可能覆寫 `__getitem__` 仍返回 4-tuple。長度守衛讓父類新功能不影響子類。

---

## 12. 驗證方法

### 語法驗證（無需外部依賴）

```bash
cd /home/user/anti-deepfake-box
python3 -m py_compile detectors/registry.py
python3 -m py_compile detectors/dfb_visual_wrapper.py
python3 -m py_compile scripts/collect_scores.py
python3 -m py_compile fusion/cascade_selection.py
python3 -m py_compile fusion/serial_cascade.py
python3 -m py_compile preprocessing/audio_extractor.py
python3 -m py_compile preprocessing/face_extractor.py
python3 -m py_compile scripts/inference.py
```

### Registry 完整性驗證

```bash
python3 -c "
import importlib.util, types, sys

# 繞過 torch 依賴直接載入 registry
for stub in ('detectors.visual_detector','detectors.rppg_detector','detectors.sync_detector',
             'detectors.dfb_visual_wrapper','detectors.rppg_tscan_detector',
             'detectors.rppg_physmamba_detector','detectors.base_detector',
             'preprocessing.face_extractor'):
    sys.modules[stub] = types.ModuleType(stub)

spec = importlib.util.spec_from_file_location('detectors.registry', 'detectors/registry.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

assert set(mod.REGISTRY) == {'visual','rppg','sync'}
assert {'xception','ucf','sbi','f3net','spsl','srm','efficientnet_b4','facexray','lsda'} == set(mod.VISUAL_REGISTRY)
assert {'pos','tscan','physmamba'} == set(mod.RPPG_REGISTRY)
assert {'syncnet','mrdf','latentsync','avad'} == set(mod.SYNC_REGISTRY)

# 確認規劃中的偵測器正確丟出 NotImplementedError
try: mod.build_detector('sync','mrdf',{})
except NotImplementedError as e: assert 'mrdf' in str(e)

print('Registry: ALL CHECKS PASSED')
for m,entries in mod.list_detectors().items():
    for k,info in entries.items():
        print(f'  {m}/{k:20s} {info[\"status\"]:15s} {info[\"detector_name\"]}')
"
```

### 單元測試（已通過）

```bash
python3 -m pytest tests/test_serial_cascade.py tests/test_preprocessing_new.py -q
# 期望：45 passed, 8 skipped（pandas/sklearn 不存在時 skip）
```

### CSV Schema 驗證

```bash
python3 -c "
import csv, sys
sys.path.insert(0, '.')
# FIELDNAMES 是模組層級常數，用 ast 解析不需要執行 import
import ast
src = open('scripts/collect_scores.py').read()
for node in ast.walk(ast.parse(src)):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == 'FIELDNAMES':
                vals = [e.s for e in node.value.elts]
                expected = {'sample_id','dataset','label','detector_name','modality',
                            'fake_score','score_type','inference_time_ms',
                            'window_start_sec','window_end_sec','status','error_message'}
                assert set(vals) == expected, f'Schema mismatch: {set(vals)^expected}'
                print(f'CSV schema ({len(vals)} cols): OK')
"
```

### Cascade 功能驗證（需要 pandas）

```bash
python3 -c "
import tempfile, os, csv
from fusion.serial_cascade import SerialCascade

rows = [
    {'stage_order':1,'modality':'visual','threshold_H':0.75,'threshold_L':0.30},
    {'stage_order':2,'modality':'av_sync','threshold_H':0.65,'threshold_L':0.20},
]
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
    fname = f.name

c = SerialCascade.from_csv(fname)
assert c.fuse(visual_score=0.9).is_fake      # stage 1 FAKE exit
assert not c.fuse(visual_score=0.1).is_fake  # stage 1 REAL exit
assert c.stages[1].name == 'sync'            # av_sync 正規化
assert c.fuse().fake_score == 0.5            # 全 None fallback
assert not c.fuse().is_fake                  # fallback is_fake=False（bug fix 驗證）
os.unlink(fname)
print('cascade: OK')
"
```

---

## 13. 已知限制與後續工作

### 目前阻塞（Blockers）

| 項目 | 說明 | 解除條件 |
|------|------|---------|
| **DFB 偵測器 checkpoint** | UCF/SBI/F3Net/SPSL/SRM 需要各自的預訓練權重 | 下載 DFB pretrained checkpoints |
| **P9 blocker** | `data_dict['npz_path']` 全為 `None`，NPZ 音訊載入無效 | 前處理完成後由外部腳本設定路徑映射 |
| **FakeAVCeleb 資料集** | `collect_scores.py` 需要影片目錄 | 資料集掛載 |
| **pandas 未安裝** | `SerialCascade.from_csv()` 需要 pandas | `pip install pandas` |
| **GP solver 尚未執行** | `real_data_vip_settings.csv` 尚不存在 | 執行 `fusion_solver_prod_v1.ipynb` |
| **TS-CAN / PhysMamba** | `rppg_tscan_detector.py` 為 stub | 確認 rPPG-Toolbox checkpoint 後實作 `_detect_impl()` |

### 後續工作

1. **多 visual 偵測器全批評分**：
   ```bash
   for det in xception ucf sbi f3net spsl srm; do
       python scripts/collect_scores.py --visual_detector $det \
           --dfb_pretrained /ckpts/${det}.pth --output_dir results/
   done
   python -c "from fusion.cascade_selection import select_classifiers; print(select_classifiers('results/'))"
   ```

2. **`_detect_and_align_batch()` 整合 `_filter_single_face_frames()`**：目前 filter 方法存在但 `extract()` cache miss 路徑尚未呼叫它（只呼叫了 `_smooth_bboxes`）

3. **`npz_path` 設定腳本**：掃描 `.face_cache/` 目錄，將 NPZ 路徑寫入 DFB 的 dataset JSON，以啟用 `with_audio=True` 路徑

4. **消融實驗**：Visual / Visual+rPPG / Visual+Sync / 全三模態 AUC 比較（FakeAVCeleb）

5. **即時推論模式**：`extract_to_array()` 改為 chunked subprocess 串流，用於瀏覽器 extension

---

*報告更新：2026-05-19。對應 anti-deepfake-box commit `8a6a951`，DeepfakeBench commit `59ed713`。*
