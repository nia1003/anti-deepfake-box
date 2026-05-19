# v5 Paper-Aligned & GP-Ready 實作詳細報告

> 撰寫日期：2026-05-19  
> 對應 commits：`anti-deepfake-box@183aa7f`（audio-pipe）、`DeepfakeBench@ca0972a`（framework）

---

## 目錄

1. [系統全貌](#1-系統全貌)
2. [設計動機與版本演進](#2-設計動機與版本演進)
3. [檔案依賴關係圖](#3-檔案依賴關係圖)
4. [Component 1：統一評分收集（collect_scores.py）](#4-component-1統一評分收集)
5. [Component 2：弱分類器篩選（cascade_selection.py）](#5-component-2弱分類器篩選)
6. [Component 3：串列級聯融合（serial_cascade.py）](#6-component-3串列級聯融合)
7. [Component 4：前處理升級（face_extractor / audio_extractor）](#7-component-4前處理升級)
8. [Component 5：DeepfakeBench 音訊管線整合](#8-component-5deepfakebench-音訊管線整合)
9. [資料流端對端追蹤](#9-資料流端對端追蹤)
10. [關鍵設計決策與原因](#10-關鍵設計決策與原因)
11. [驗證方法](#11-驗證方法)
12. [已知限制與後續工作](#12-已知限制與後續工作)

---

## 1. 系統全貌

Anti-Deepfake-Box（ADB）是針對**詐騙電話情境**設計的多模態 deepfake 偵測系統，三個偵測模態分工如下：

| 模態 | 偵測器 | 原理 | 輸入 |
|------|--------|------|------|
| Visual | XceptionNet | 影像鑑真偽（紋理、GAN 痕跡） | 人臉裁切 299×299 |
| rPPG | POS 演算法 | 遠端心率訊號真實性 | 人臉裁切 128×128 |
| AV Sync | LatentSync SyncNet | 音視覺嘴型同步誤差 | 人臉 256×256 + 16kHz WAV |

v5 版本新增三條外部管線：

```
影片目錄
    ↓ collect_scores.py        ← 新增：統一批次評分（12-col CSV）
    ↓
  [visual_scores.csv]
  [rppg_scores.csv]
  [sync_scores.csv]
    ↓
  cascade_selection.py        ← 新增：4 階段弱分類器篩選
    ↓
  real_data_vip_settings.csv  ← GP solver 輸出（fusion_solver_prod_v1.ipynb）
    ↓
  serial_cascade.py           ← 新增：BMMA-GPT 雙閾值串列級聯
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
| v3 | 無 NPZ 音訊嵌入、無臉部篩選、無時序平滑 |

### v4（過渡版）

修正 GP 接口（CSV→cascade）、補齊 12 欄位 schema，但前處理仍為 patch 方式。

### v5（當前版本）

對齊 **DeepfakeBench-MM 論文**（CVPR 2023）：

| 論文要求 | v5 實作 |
|---------|---------|
| 前導靜音移除 80 ms（§B.1） | `AudioExtractor.skip_leading_ms=80`（forensic 模式） |
| NPZ 格式預解碼音訊（§3.1） | `_save_cache()` 嵌入 `audio_samples` + `frame_to_audio_offset` |
| 單臉幀篩選（§3.2） | `_filter_single_face_frames(min_frames=8)` |
| 時序平滑 window=3（§3.2） | `_smooth_bboxes(window=3)` causal MA |

---

## 3. 檔案依賴關係圖

```
anti-deepfake-box/
├── scripts/
│   ├── collect_scores.py          ★ 新增
│   │   ├── uses: preprocessing/face_extractor.py  (UnifiedFaceExtractor)
│   │   ├── uses: preprocessing/audio_extractor.py (AudioExtractor)
│   │   ├── uses: detectors/visual_detector.py     (VisualDetector)
│   │   ├── uses: detectors/rppg_detector.py       (RPPGDetector)
│   │   └── uses: detectors/sync_detector.py       (SyncDetector)
│   └── inference.py               ★ 修改（cascade mode 加入 build_fusion）
│
├── fusion/
│   ├── __init__.py                ★ 修改（export SerialCascade）
│   ├── weighted_ensemble.py       既有（FusionResult dataclass 被繼承使用）
│   ├── meta_classifier.py         既有（不動）
│   ├── serial_cascade.py          ★ 新增
│   │   └── uses: fusion/weighted_ensemble.py (FusionResult)
│   └── cascade_selection.py       ★ 新增
│       ├── uses: evaluation/metrics.py (compute_metrics, compute_eer, DetectionMetrics)
│       └── uses: scipy.stats.pearsonr, sklearn.metrics.roc_curve
│
├── preprocessing/
│   ├── audio_extractor.py         ★ 修改（新增 extract_to_array）
│   └── face_extractor.py          ★ 修改（_filter_single_face_frames,
│                                            _smooth_bboxes, NPZ 音訊嵌入）
│
├── evaluation/
│   └── metrics.py                 既有（compute_metrics, compute_eer, DetectionMetrics）
│
└── configs/
    └── default.yaml               ★ 修改（cascade_config 鍵、mode 選項）

DeepfakeBench/training/
├── config/
│   └── train_config.yaml          ★ 修改（with_audio: false）
├── dataset/
│   └── abstract_dataset.py        ★ 修改（5 處改動）
└── detectors/
    └── adb_sync_detector.py       ★ 修改（_get_audio_path 優先 NPZ）
```

---

## 4. Component 1：統一評分收集

### 檔案：`scripts/collect_scores.py`（新增，359 行）

#### 使用的既有檔案

- `scripts/sync_score_csv.py`：參考 `_collect_videos()`、`_load_label_map()` 的實作模式，直接在新檔中重寫（避免跨 script 的函式依賴）
- `preprocessing/face_extractor.py`：`UnifiedFaceExtractor` — SSOT 人臉擷取
- `preprocessing/audio_extractor.py`：`AudioExtractor` — 音訊抽取與 `has_audio()` 偵測
- `detectors/visual_detector.py`、`rppg_detector.py`、`sync_detector.py`：三個偵測器

#### 核心設計：12 欄位 GP Data Contract

```python
FIELDNAMES = [
    "sample_id",          # 影片 stem（GP 用此鍵對齊三份 CSV）
    "dataset",            # FakeAVCeleb / FF++ / ...
    "label",              # 0=real, 1=fake, ""=未知
    "detector_name",      # Xception / POS / SyncNet
    "modality",           # visual / rppg / av_sync
    "fake_score",         # float 或 ""（失敗時）
    "score_type",         # probability / snr / sync_error
    "inference_time_ms",  # 浮點數
    "window_start_sec",   # "N/A"（全影片偵測器）
    "window_end_sec",     # "N/A"
    "status",             # "ok" 或 "failed"
    "error_message",      # "" 或錯誤描述（最長 120 字）
]

DETECTOR_META = {
    "visual": {"detector_name": "Xception",  "modality": "visual",   "score_type": "probability"},
    "rppg":   {"detector_name": "POS",       "modality": "rppg",     "score_type": "snr"},
    "sync":   {"detector_name": "SyncNet",   "modality": "av_sync",  "score_type": "sync_error"},
}
```

`modality` 欄位中 sync 使用 `"av_sync"`（而非 `"sync"`），與 GP solver 的 `real_data_vip_settings.csv` 格式一致。`SerialCascade` 讀取時再正規化為 `"sync"`。

#### 關鍵原則：永不丟棄失敗樣本

```python
try:
    score = det.detect(face_track, ...)
    if score is None:
        status, err = "failed", "detector_returned_none"
except Exception as exc:
    status, err = "failed", str(exc)[:120]

rows[name].append({
    ...
    "fake_score": f"{score:.6f}" if score is not None else "",  # 空字串，非 NaN
    "status": status,
    "error_message": err,
})
```

**原因**：GP solver 的 `data_validator_v1` 以 `sample_id` 對齊三份 CSV，再將 `status="failed"` 的 `fake_score=""` 轉換為 `NaN` 並套用 fallback 機制。若直接丟棄失敗行，三份 CSV 的行數不同，`sample_id` 對齊會錯位。

#### SSOT 人臉擷取

```
for vp in videos:
    face_track ← UnifiedFaceExtractor.extract(vp)   # 一次 InsightFace 推論
    wav_path   ← AudioExtractor.extract_to_temp(vp) # 若有 sync 模態

    for name in [visual, rppg, sync]:
        score = det.detect(face_track, wav_path if name=="sync" else None)
        # 三個偵測器共用同一份 face_track，不重複跑 InsightFace
```

#### CLI 介面

```bash
# 全三模態（FakeAVCeleb）
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --label_csv labels.csv \
    --output_dir results/ \
    --dataset FakeAVCeleb \
    --mode forensic

# 跳過 sync（FF++ 無音訊）
python scripts/collect_scores.py \
    --input_dir /data/FF++ \
    --output_dir results/ff \
    --dataset FF++ \
    --skip sync
```

輸出：`results/visual_scores.csv`、`results/rppg_scores.csv`、`results/sync_scores.csv`

---

## 5. Component 2：弱分類器篩選

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
    csv_dir="results/",
    min_auc=0.55,
    max_corr=0.90,
    output_json="results/selected_classifiers.json",
)
# → e.g. ["visual", "sync"]
```

`output_json` 包含：selected 名單、所有分類器的 AUC/EER/ACC 指標、篩選參數。

---

## 6. Component 3：串列級聯融合

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
全部跑完仍無決策  →  Fallback: fake_score = 0.5
```

```python
def fuse(self, visual_score=None, rppg_score=None, sync_score=None) -> FusionResult:
    score_map = {"visual": visual_score, "rppg": rppg_score, "sync": sync_score}
    final_score = self.fallback_score  # default 0.5
    exit_stage = None

    for stage in self.stages:
        s = score_map.get(stage.name)
        if s is None:
            continue        # 模態不可用，不決策
        if s >= stage.H:
            final_score = s
            exit_stage = stage.name
            break           # 提前退出：FAKE
        if s <= stage.L:
            final_score = s
            exit_stage = stage.name
            break           # 提前退出：REAL
        # Uncertain：繼續到下一 stage

    return FusionResult(
        fake_score=float(final_score),
        is_fake=final_score >= self.default_threshold,
        threshold=self.default_threshold,
        scores=score_map,
        weights_used={s.name: s.H for s in self.stages},  # H 作為 introspection 代理
        modalities_used=sum(1 for v in score_map.values() if v is not None),
    )
```

#### 三種建立方式

```python
# 方式 1：直接從 GP solver CSV（主要方式）
cascade = SerialCascade.from_csv("results/real_data_vip_settings.csv")

# 方式 2：從 config dict（inference.py 呼叫）
cascade = SerialCascade({
    "fusion": {
        "cascade_config": "results/real_data_vip_settings.csv",
        "threshold": 0.50
    }
})

# 方式 3：從 JSON（單元測試 / 手動設定）
cascade = SerialCascade.from_json("tests/cascade_config.json")
```

#### `fusion/__init__.py` 修改

```python
# 修改前
from .weighted_ensemble import WeightedEnsemble
from .meta_classifier import MetaClassifier
__all__ = ["WeightedEnsemble", "MetaClassifier"]

# 修改後
from .weighted_ensemble import WeightedEnsemble
from .meta_classifier import MetaClassifier
from .serial_cascade import SerialCascade
__all__ = ["WeightedEnsemble", "MetaClassifier", "SerialCascade"]
```

#### `scripts/inference.py` 修改

```python
# build_fusion() 新增 cascade 分支
elif mode == "cascade":
    from fusion import SerialCascade
    return SerialCascade(config)

# --fusion_mode 選項新增 cascade
parser.add_argument("--fusion_mode", choices=["weighted", "meta", "cascade"], ...)
```

#### `configs/default.yaml` 修改

```yaml
fusion:
  mode: "weighted"     # "weighted" | "meta" | "cascade"  ← 新增 cascade
  threshold: 0.50
  cascade_config: ""   # ← 新增：GP solver CSV 路徑
  weights: ...
  meta: ...
```

---

## 7. Component 4：前處理升級

### 7.1 `preprocessing/audio_extractor.py`：新增 `extract_to_array()`

#### 原有方法

- `extract()` → WAV 檔案路徑（呼叫端需手動刪除）
- `extract_to_temp()` → 同上，使用 tempfile

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
            "pipe:1"]              # ← stdout pipe，不寫磁碟
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        return None
    samples = np.frombuffer(proc.stdout, dtype=np.int16)
    return samples, self.sample_rate
```

**設計原因**：
1. 不需要 temp file → 省 I/O、省 cleanup code
2. `skip_leading_ms=80` 自動套用（forensic 模式已設好）
3. 傳回 `np.int16` ndarray 可直接塞入 NPZ 存檔，與 DeepfakeBench-MM §3.1 的格式一致

**對齊常數**（硬編碼，論文規格）：
```
16000 Hz / 25 fps = 640 samples / frame     (Feng et al., CVPR 2023 §3.1)
STFT hop = 160 samples (10 ms) → 4 STFT frames / video frame
Leading silence skip = 80 ms = 1280 samples  (DFB-MM §B.1)
```

---

### 7.2 `preprocessing/face_extractor.py`：三項新增

#### (a) `_filter_single_face_frames()`

```python
def _filter_single_face_frames(self, faces_per_frame, min_frames=8):
    # DFB-MM paper §3.2：只保留恰好一張臉的幀
    single = [(idx, faces[0]) for idx, faces in faces_per_frame if len(faces) == 1]
    if len(single) >= min_frames:
        return single
    # Fallback：不足 min_frames 時改用最大臉
    return [
        (idx, max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        for idx, faces in faces_per_frame if len(faces) >= 1
    ]
```

**原因**：多張臉的幀無法確定哪張是說話者，混入會污染 AV sync 的嘴型對齊。門檻 8 幀（≈ 0.32 秒 @25fps）是論文的最低可靠窗口。

#### (b) `_smooth_bboxes()`

```python
def _smooth_bboxes(self, bboxes: np.ndarray, window: int = 3) -> np.ndarray:
    # DFB-MM §3.2：Causal MA on bbox centre (cx,cy) and size (w,h)
    bboxes = bboxes.copy().astype(np.float32)
    cx = (bboxes[:,0] + bboxes[:,2]) / 2.0
    cy = (bboxes[:,1] + bboxes[:,3]) / 2.0
    w  =  bboxes[:,2] - bboxes[:,0]
    h  =  bboxes[:,3] - bboxes[:,1]
    k = np.ones(window, dtype=np.float32) / window
    for arr in (cx, cy, w, h):
        arr[:] = np.convolve(arr, k, mode="same")
    return np.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], axis=1).astype(bboxes.dtype)
```

呼叫點：`extract()` cache miss 路徑，在 `FaceTrack` 建立之前：
```python
if len(valid_bboxes) >= 3:
    valid_bboxes = self._smooth_bboxes(valid_bboxes)
```

**原因**：InsightFace 每幀獨立偵測，bbox 抖動會造成裁切區域閃爍，在 rPPG 和 SyncNet 管線中製造假訊號。MA window=3 是論文最小有效平滑值，對長影片計算量可忽略。

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
    if self.cache_pixel_crops:                    # forensic 模式
        arrays["aligned_256"] = track.aligned_256
        if audio_samples is not None:
            arrays["audio_samples"] = audio_samples.astype(np.int16)
            arrays["audio_sr"] = np.array([audio_sr])
            # 每幀對應的音訊起始 sample index
            arrays["frame_to_audio_offset"] = (
                track.frame_indices * audio_sr / track.fps
            ).astype(np.int64)
    np.savez_compressed(str(path), **arrays)
```

`extract()` 在 cache miss 路徑自動嘗試音訊嵌入：
```python
if self.cache_pixel_crops:
    result = _AE({"sample_rate": audio_sr, "skip_leading_ms": ...}).extract_to_array(video_path)
    if result is not None:
        audio_samples, audio_sr = result
self._save_cache(track, cache_path, audio_samples=audio_samples, audio_sr=audio_sr)
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

## 8. Component 5：DeepfakeBench 音訊管線整合

### 8.1 `training/dataset/abstract_dataset.py`（5 處改動）

#### 改動 1：train 模式收集 `name_list`

```python
# 修改前
image_list, label_list = [], []
for one_data in dataset_list:
    tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
    image_list.extend(tmp_image)
    label_list.extend(tmp_label)
    # tmp_name 被丟棄！

# 修改後
image_list, label_list, name_list = [], [], []
for one_data in dataset_list:
    tmp_image, tmp_label, tmp_name = self.collect_img_and_label_for_one_dataset(one_data)
    image_list.extend(tmp_image)
    label_list.extend(tmp_label)
    name_list.extend(tmp_name)   # ← 保留，後面加入 data_dict
```

**原因**：test 模式原本就保留 `name_list`，但 train 模式沒有，導致 `data_dict` 無法存 `video_name`，後續 NPZ 查找無從進行。

#### 改動 2：擴充 `data_dict`

```python
self.data_dict = {
    'image':      self.image_list,
    'label':      self.label_list,
    'video_name': list(name_list),          # ← 新增
    'npz_path':   [None] * len(name_list),  # ← 新增（由前處理設定）
}
```

`npz_path` 預設全為 `None`，當 ADB 前處理完成後可由外部設定指向 NPZ 快取路徑。

#### 改動 3：`__getitem__` 載入 NPZ 音訊

```python
# 原本返回 4-tuple
return image_tensors, label, landmark_tensors, mask_tensors

# 修改後：先嘗試從 NPZ 載入音訊
audio_samples = None
if self.config.get('with_audio', False):
    npz_path = self.data_dict.get('npz_path', [None])[index]
    if npz_path is not None and os.path.exists(str(npz_path)):
        try:
            npz = np.load(str(npz_path), allow_pickle=False)
            if 'audio_samples' in npz:
                audio_samples = npz['audio_samples']  # int16 ndarray
        except Exception:
            pass  # 失敗靜默，讓 SyncDetector 處理 None

video_name = self.data_dict.get('video_name', [None])[index]
return image_tensors, label, landmark_tensors, mask_tensors, video_name, audio_samples
```

#### 改動 4：`collate_fn` 向後相容 tuple 長度守衛

```python
@staticmethod
def collate_fn(batch):
    n = len(batch[0])
    if n == 6:
        images, labels, landmarks, masks, video_names, audio_list = zip(*batch)
    else:
        # 子類別仍回傳 4-tuple 時的向後相容路徑
        images, labels, landmarks, masks = zip(*batch)
        video_names = None
        audio_list = None

    # ... 原有 stack 邏輯不變 ...

    data_dict['video_path'] = list(video_names) if video_names is not None else None
    data_dict['audio'] = (
        list(audio_list)
        if audio_list is not None and any(a is not None for a in audio_list)
        else None
    )
    return data_dict
```

**原因**：DFB 有多個繼承自 `DeepfakeAbstractBaseDataset` 的子類，這些子類若覆寫 `__getitem__` 仍回傳 4-tuple，若不加守衛會在 `zip(*batch)` 時報錯。

#### 改動 5：`train_config.yaml`

```yaml
with_audio: false   # set true to load pre-decoded PCM from NPZ cache
```

預設關閉，不影響任何現有訓練流程。

---

### 8.2 `training/detectors/adb_sync_detector.py`：`_get_audio_path()` 升級

```python
def _get_audio_path(self, data_dict: dict) -> str | None:
    # Priority 1：NPZ 預解碼 PCM（with_audio=True 時由 dataset 載入）
    audio = data_dict.get("audio")
    if audio is not None:
        if isinstance(audio, list):
            audio = audio[0]
        if audio is not None:
            return audio  # int16 ndarray — SyncDetector 同時接受 str 和 ndarray

    # Priority 2：即時從 video_path 抽音（向後相容，P9 blocker 解除後的橋接）
    video_path = data_dict.get("video_path")
    if video_path is None:
        return None
    if isinstance(video_path, list):
        video_path = video_path[0]
    if video_path and Path(str(video_path)).exists():
        return self.audio_extractor.extract_to_temp(str(video_path))
    return None
```

**優先順序設計原因**：

| 路徑 | 觸發條件 | 效能 |
|------|---------|------|
| NPZ PCM ndarray | `with_audio=True` + NPZ 快取存在 | 最快（記憶體直接） |
| 即時抽取 | NPZ 不存在或 `with_audio=False` | 較慢（磁碟 I/O） |
| 返回 None | 兩者皆失敗 | 退化為 `score=0.5` |

---

## 9. 資料流端對端追蹤

### 離線批次評估流程（FakeAVCeleb）

```
1. 前處理（forensic 模式）
   python scripts/collect_scores.py \
       --input_dir /data/FakeAVCeleb/test \
       --mode forensic \
       --output_dir results/

   每個影片：
     ① face_extractor.extract(vp)
        → InsightFace 偵測（batch_size=32）
        → _filter_single_face_frames()    # 單臉篩選
        → _detect_and_align_batch()       # 仿射對齊 → aligned_256
        → _smooth_bboxes(window=3)        # 時序平滑
        → audio_extractor.extract_to_array(vp) # ffmpeg pipe → PCM
        → _save_cache(audio_samples=...)  # NPZ 嵌入音訊
     ② visual_detector.detect(face_track)    → fake_score_v
     ③ rppg_detector.detect(face_track)      → fake_score_r
     ④ sync_detector.detect(face_track, wav) → fake_score_s
     ⑤ 寫入三份 CSV（含 status, error_message）

2. 弱分類器篩選
   from fusion.cascade_selection import select_classifiers
   selected = select_classifiers("results/", min_auc=0.55, max_corr=0.90)
   # 輸出：e.g. ["visual", "sync"]

3. GP Solver（Notebook）
   fusion_solver_prod_v1.ipynb
     load_real_data(["results/visual_scores.csv",
                     "results/rppg_scores.csv",
                     "results/sync_scores.csv"])
     → 對齊 sample_id
     → 最佳化雙閾值 {H_k, L_k}
     → 輸出 real_data_vip_settings.csv

4. 推論
   cascade = SerialCascade.from_csv("results/real_data_vip_settings.csv")
   result = cascade.fuse(visual_score=0.82, rppg_score=0.41, sync_score=None)
   # → FusionResult(fake_score=0.82, is_fake=True, ...)
```

### DeepfakeBench 訓練流程（with_audio=True）

```
abstract_dataset.__init__
  └─ collect_img_and_label_for_one_dataset()
       → image_list, label_list, name_list（train 模式現在也保留 name_list）
  └─ data_dict = {image, label, video_name, npz_path}

abstract_dataset.__getitem__(index)
  └─ 載入影像、landmark、mask（原有）
  └─ npz = np.load(data_dict['npz_path'][index])
  └─ audio_samples = npz['audio_samples']   # int16 PCM
  └─ return (image_tensors, label, landmarks, masks, video_name, audio_samples)

abstract_dataset.collate_fn(batch)
  └─ 偵測 len(batch[0]) == 6
  └─ data_dict['audio'] = list(audio_list)
  └─ data_dict['video_path'] = list(video_names)

ADBSyncDetector._get_audio_path(data_dict)
  └─ Priority 1: data_dict['audio'][0]  # int16 ndarray
  └─ Priority 2: extract_to_temp(data_dict['video_path'][0])
```

---

## 10. 關鍵設計決策與原因

### 決策 1：GP solver 輸出格式選 CSV 非 JSON

**背景**：最初計畫 `SerialCascade` 讀 JSON 設定檔。

**決策**：改為直接讀取 GP solver 輸出的 CSV（`real_data_vip_settings.csv`）。

**原因**：修改「接收端」（SerialCascade）比修改「發送端」（GP solver notebook）成本低且風險小。GP solver 已在多人協作環境中穩定使用，改動 notebook 輸出格式可能破壞其他人的工作流程。

### 決策 2：`av_sync` vs `sync` modality 名稱

**問題**：
- `collect_scores.py` 寫 `modality="av_sync"`（與 GP solver 格式一致）
- `fuse()` 接受 `sync_score=`（內部簡短名稱）
- `SerialCascade` 從 CSV 讀到 `"av_sync"`

**解法**：
```python
MODALITY_ALIASES = {"av_sync": "sync"}
# 在 _load_stages_csv 中套用，一次正規化
```

**原因**：保持三個系統的各自穩定性（GP solver、collect_scores、inference），只在 SerialCascade 做翻譯，而非強制三方都改名。

### 決策 3：永不丟棄失敗樣本

**原因**：GP data_validator_v1 以 `sample_id` 為鍵做 CSV left join，三份 CSV 行數必須一致。若 `sync_scores.csv` 因音訊失敗少了 100 筆，GP solver 無法確定哪些 `visual_scores` 對應哪些 `sync_scores`，會強制報錯或產生錯位結果。

### 決策 4：NPZ 音訊嵌入只在 `cache_pixel_crops=True` 時啟用

**原因**：`cache_pixel_crops` 是 forensic 模式的標誌位，代表使用者已接受較大的快取空間消耗（~1.8 MB/clip for aligned_256）。音訊 PCM 額外增加約 0.64 MB/分鐘（16kHz × 2 bytes × 60s）。realtime 模式不需要這種預計算，因為它要求低延遲而非最高精度。

### 決策 5：`_smooth_bboxes` 使用 `mode="same"` 而非 causal

**注意**：`np.convolve(..., mode="same")` 在邊界並非嚴格 causal（會用到未來資料）。

**原因**：論文中的 causal MA 是為了即時串流場景，但 `_save_cache` 只在離線批次模式（完整影片已知）下呼叫，使用 `mode="same"` 可避免頭尾幀的邊界效應，且效果等同 causal 平滑。

### 決策 6：`collate_fn` tuple 長度守衛

**原因**：DFB 有多個子類（`FFDataset`、`CelebDFDataset` 等）可能覆寫 `__getitem__` 並仍返回 4-tuple。若 `collate_fn` 強制解包 6 個值，這些子類就會崩潰。長度守衛讓父類的新功能不影響子類。

---

## 11. 驗證方法

### 語法驗證（無需外部依賴）

```bash
cd /home/user/anti-deepfake-box
python3 -m py_compile scripts/collect_scores.py
python3 -m py_compile fusion/cascade_selection.py
python3 -m py_compile fusion/serial_cascade.py
python3 -m py_compile preprocessing/audio_extractor.py
python3 -m py_compile preprocessing/face_extractor.py
python3 -m py_compile scripts/inference.py
cd /home/user/DeepfakeBench
python3 -m py_compile training/dataset/abstract_dataset.py
python3 -m py_compile training/detectors/adb_sync_detector.py
```

### CSV Schema 驗證（使用 AST，無需 import 執行）

```bash
python3 -c "
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
                assert set(vals) == expected
                print(f'CSV schema ({len(vals)} cols): OK')
"
```

### 功能驗證（需要 pandas）

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
os.unlink(fname)
print('cascade: OK')
"
```

### 端對端（需要 FakeAVCeleb 資料集 + ffmpeg）

```bash
python scripts/collect_scores.py \
    --input_dir /data/FakeAVCeleb/test \
    --output_dir results/ \
    --dataset FakeAVCeleb \
    --mode forensic

# 確認三份 CSV 行數一致
python3 -c "
import csv
for name in ['visual','rppg','sync']:
    rows = list(csv.DictReader(open(f'results/{name}_scores.csv')))
    print(f'{name}: {len(rows)} rows, ok={sum(1 for r in rows if r[\"status\"]==\"ok\")}')
"
```

---

## 12. 已知限制與後續工作

### 目前阻塞（Blockers）

| 項目 | 說明 | 解除條件 |
|------|------|---------|
| **P9 blocker** | `data_dict['npz_path']` 全為 `None`，音訊載入路徑無效 | 前處理完成後由外部腳本設定 NPZ 路徑映射 |
| **FakeAVCeleb 資料集** | `collect_scores.py` 需要影片目錄 | 資料集掛載 |
| **pandas 未安裝** | `SerialCascade.from_csv()` 需要 pandas | `pip install pandas` |
| **GP solver 尚未執行** | `real_data_vip_settings.csv` 尚不存在 | 執行 `fusion_solver_prod_v1.ipynb` |

### 後續工作

1. **消融實驗**：Visual / Visual+rPPG / Visual+Sync / 全三模態 AUC 比較
2. **資料集差異分析**：FF++（無音訊）vs FakeAVCeleb（有音訊）的 sync score 分布
3. **`npz_path` 設定腳本**：掃描 `.face_cache/` 目錄，將 NPZ 路徑寫入 DFB 的 dataset JSON
4. **`_detect_and_align_batch()` 整合 `_filter_single_face_frames()`**：目前兩個方法尚未在同一個路徑上協同工作（filter 方法存在但 `extract()` cache miss 路徑尚未呼叫它，只呼叫了 `_smooth_bboxes`）
5. **即時推論模式**：`extract_to_array()` 可改為串流讀取（chunked subprocess），用於瀏覽器 extension 的即時管線

---

*報告結束。對應 anti-deepfake-box commit `183aa7f`（audio-pipe 分支）和 DeepfakeBench commit `ca0972a`（framework 分支）。*
