# v5 + 可解釋性：實作與論文規劃

## 專案訴求

| 訴求 | 說明 |
|---|---|
| **Sandbox** | 任意投入影片 → 結構化分析報告，元件可換、決策可觀測 |
| **Edge 裝置** | Cascade fail-fast，輕量模態先跑，不依賴 GPU server |
| **可解釋性** | 不只 FAKE/REAL，要有多層次理由，供非專業使用者理解 |

---

## 現況（v5 已完成）

```
三模態 pipeline      visual (XceptionNet) + rPPG (POS) + sync (SyncNet)
DFB wrapper         DFBVisualDetector，6 個視覺偵測器可切換
Detector registry   build_detector()，VISUAL/RPPG/SYNC registry
SerialCascade       BMMA-GPT dual-threshold，from_csv() 讀 GP solver 輸出
collect_scores.py   12 欄 CSV，never-drop，適配 GP solver
cascade_selection   AUC filter → correlation filter → Pareto filter
前處理              UnifiedFaceExtractor (InsightFace) + AudioExtractor
測試                52 passed，41 skipped，文件完整
```

**缺口**：FusionResult 無 trigger/evidence；各偵測器只回傳 float；無 edge profile；無實驗 AUC 數字。

---

## 可解釋性架構

參考論文：DF-P2E（ACM MM 2025）——Grad-CAM → 圖像描述（BLIP）→ LLM 敘事

ADB 的多模態解釋比 DF-P2E 更豐富：

```
視覺   →  Grad-CAM heatmap  →  「眼角和下巴邊界有混合偽影」
rPPG   →  BVP 波形 + SNR    →  「SNR=0.42dB < 門檻 1.5dB，心率訊號異常」
Sync   →  Window timeline   →  「1.0–2.0s 嘴型同步信心掉至 0.21」
Cascade→  觸發 stage         →  「Stage 1 visual score=0.83 ≥ H=0.75，FAKE 早退出」
LLM    →  統一敘事           →  用戶可讀的完整說明
```

### 三層解釋輸出

| 層次 | 內容 | Edge 可用 |
|---|---|---|
| Layer 1：數值 | score、觸發原因、執行了哪些 stage | ✅ 免費 |
| Layer 2：訊號 | Grad-CAM、BVP 波形、Sync timeline | ✅ 輕量 |
| Layer 3：敘事 | LLM 生成自然語言說明 | ⚠️ Edge 用模板，Server 用 Phi-3-mini |

---

## 實作規劃

### Phase 1：讓系統說得出理由（Priority：最高）

#### 1a. 擴充 `FusionResult`

**檔案**：`fusion/weighted_ensemble.py`

```python
@dataclass
class FusionResult:
    fake_score: float
    is_fake: bool
    threshold: float
    scores: Dict[str, Optional[float]] = field(default_factory=dict)
    weights_used: Dict[str, float] = field(default_factory=dict)
    modalities_used: int = 0
    # 新增
    trigger: Optional[str] = None          # "visual" | "rppg" | "sync" | None
    trigger_reason: str = ""               # "score=0.83 ≥ H=0.75 → early exit FAKE"
    stages_run: List[str] = field(default_factory=list)   # 實際跑了哪些
    stages_skipped: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    # evidence 結構：
    # {
    #   "visual": {"score": 0.83, "cam_mean": ndarray, "text": "..."},
    #   "rppg":   {"score": 0.41, "snr_db": 0.42, "ppg": ndarray, "text": "..."},
    #   "sync":   {"score": 0.78, "window_scores": [...], "text": "..."},
    # }
```

`__str__()` 擴充為輸出 trigger + stages_run + evidence text。

#### 1b. `SerialCascade.fuse()` 存入 trigger

**檔案**：`fusion/serial_cascade.py`

- `exit_stage` 變數已存在，目前未存進 FusionResult
- 加入：`stages_run`（每個 stage 的名稱）、`trigger_reason`（score 與 H/L 的比較）
- `weights_used` 改存實際 threshold 語意，不再假裝是 weight

#### 1c. `BaseDetector` 新增 `detect_with_evidence()`

**檔案**：`detectors/base_detector.py`

```python
class BaseDetector(ABC):
    def detect(self, face_track, audio_path=None) -> Optional[float]:
        """現有介面，不變。"""

    def detect_with_evidence(self, face_track, audio_path=None) -> dict:
        """
        回傳 score + 中間產物。
        預設實作：呼叫 detect()，包成 {"score": ..., "text": ""} dict。
        子類別覆寫以提供豐富 evidence。
        """
        score = self.detect(face_track, audio_path)
        return {"score": score, "text": ""}
```

#### 1d. `RPPGDetector.detect_with_evidence()`

**檔案**：`detectors/rppg_detector.py`

`get_ppg_and_snr()` 已存在，直接接出：

```python
def detect_with_evidence(self, face_track, audio_path=None):
    if not self._loaded:
        self.load(); self._loaded = True
    bvp = _pos_wang(face_track.crops_128, face_track.fps)
    snr = compute_ppg_snr(bvp, face_track.fps)
    score = snr_to_fake_score(snr, self.snr_threshold, self.snr_scale)
    verdict = "anomaly" if snr < self.snr_threshold else "normal"
    return {
        "score": score,
        "snr_db": round(snr, 3),
        "threshold_db": self.snr_threshold,
        "ppg_waveform": bvp,         # (T,) ndarray，供視覺化
        "verdict": verdict,
        "text": f"rPPG SNR={snr:.2f}dB（門檻={self.snr_threshold}dB）：{verdict}",
    }
```

#### 1e. `SyncDetector.detect_with_evidence()`

**檔案**：`detectors/sync_detector.py`

`_sliding_window_scores()` 改回傳 list，不提前 mean：

```python
def detect_with_evidence(self, face_track, audio_path=None):
    ...
    window_scores = self._sliding_window_scores_detailed(...)
    # [(start_sec, end_sec, sync_confidence), ...]
    mean_conf = float(np.mean([s[2] for s in window_scores])) if window_scores else 0.5
    fake_score = 1.0 - mean_conf
    worst = min(window_scores, key=lambda x: x[2]) if window_scores else None
    return {
        "score": fake_score,
        "mean_sync_confidence": mean_conf,
        "window_scores": window_scores,
        "worst_window": worst,
        "text": (f"最低同步信心={worst[2]:.2f}（{worst[0]:.1f}–{worst[1]:.1f}s）"
                 if worst else "sync N/A"),
    }
```

#### 1f. `VisualDetector.detect_with_evidence()` — Grad-CAM

**檔案**：`detectors/visual_detector.py`

```python
def detect_with_evidence(self, face_track, audio_path=None):
    if not self._loaded:
        self.load(); self._loaded = True
    sampled = self._sample_frames(face_track.crops_299)
    tensor = self._preprocess(sampled).to(self.device)

    # 對最後 conv layer 掛 hook
    cams = []
    def _hook(module, grad_in, grad_out):
        cams.append(grad_out.detach().cpu())

    handle = self.model.conv4.register_forward_hook(_hook)  # 依骨幹調整
    with torch.enable_grad():
        logits = self.model(tensor)
        prob = F.softmax(logits, dim=1)[:, 1]
        prob.mean().backward()
    handle.remove()

    cam_np = F.relu(torch.stack(cams).mean(0)).numpy()  # (T, H, W)
    return {
        "score": float(prob.mean().detach().item()),
        "cam_per_frame": cam_np,
        "cam_mean": cam_np.mean(0),    # (H, W)，疊加在臉上
        "text": "視覺偵測器在臉部偵測到操作痕跡",
    }
```

**Phase 1 驗收**：`python scripts/inference.py --video sample.mp4 --explain` 輸出包含 trigger、stages_run、各模態 text 的結構化報告。

---

### Phase 2：跑出實驗數字（與 Phase 1 並行）

#### 2a. 取得資料集

| 資料集 | 用途 | 取得方式 |
|---|---|---|
| FF++ c23（DFB 預處理版） | 視覺 baseline | DFB GitHub release → Baidu/Google Drive |
| FakeAVCeleb | 多模態主要實驗 | 原論文申請 Google Drive |

FakeAVCeleb 結構：真實影片 + RealFake（換臉）+ FakeVideo-RealAudio + RealVideo-FakeAudio

#### 2b. 消融實驗腳本

```bash
# 建立 results/ 目錄結構
for split in visual_only rppg_only sync_only v_r v_s all_fixed all_cascade; do
    mkdir -p results/$split
done

# 各組合
python scripts/collect_scores.py --dataset FakeAVCeleb \
    --skip rppg sync --output_dir results/visual_only/

python scripts/collect_scores.py --dataset FakeAVCeleb \
    --skip visual sync --output_dir results/rppg_only/

python scripts/collect_scores.py --dataset FakeAVCeleb \
    --skip visual rppg --output_dir results/sync_only/

# 全三模態
python scripts/collect_scores.py --dataset FakeAVCeleb \
    --output_dir results/all_fixed/

# 不同視覺偵測器
for det in xception sbi ucf f3net spsl srm; do
    python scripts/collect_scores.py --dataset FakeAVCeleb \
        --visual_detector $det \
        --dfb_pretrained checkpoints/${det}_best.pth \
        --output_dir results/vis_${det}/
done
```

#### 2c. 壓縮降解實驗

```bash
# 對 FakeAVCeleb test set 施加不同 CRF 壓縮
for crf in 23 28 34 40 51; do
    python scripts/apply_compression.py \
        --input_dir data/FakeAVCeleb/test \
        --output_dir data/FakeAVCeleb/test_crf${crf} \
        --crf $crf

    python scripts/collect_scores.py \
        --input_dir data/FakeAVCeleb/test_crf${crf} \
        --dataset FakeAVCeleb_crf${crf} \
        --output_dir results/compression/crf${crf}/
done
# 畫出 AUC vs CRF 曲線：視覺偵測器 vs rPPG（預期 rPPG 不受影響）
```

**新檔案**：`scripts/apply_compression.py`（約 30 行，呼叫 ffmpeg）

**Phase 2 驗收**：消融表（7 行）+ 壓縮曲線，guideline 附錄的效能數據來源。

---

### Phase 3：Edge Profile（Guideline 完稿後）

#### 3a. Registry 加 edge_profile

**檔案**：`detectors/registry.py`

```python
VISUAL_REGISTRY["xception"]["edge_profile"] = {
    "model_size_mb": 92,
    "latency_cpu_ms": None,   # benchmark 後填入
    "latency_gpu_ms": None,
    "requires_gpu": False,
    "supports_int8": True,
}
RPPG_REGISTRY["pos"]["edge_profile"] = {
    "model_size_mb": 0,
    "latency_cpu_ms": 3,
    "requires_gpu": False,
    "supports_int8": True,   # 無模型，純 numpy
}
SYNC_REGISTRY["syncnet"]["edge_profile"] = {
    "model_size_mb": 30,
    "latency_cpu_ms": None,
    "requires_gpu": False,
    "supports_int8": True,
}
```

#### 3b. Benchmark 腳本

**新檔案**：`scripts/benchmark_latency.py`

```bash
python scripts/benchmark_latency.py --device cpu --runs 100
# 輸出每個偵測器的 latency，填回 edge_profile
```

#### 3c. LLM 敘事模組（可選，論文加分）

**新檔案**：`detectors/explanation/llm_narrator.py`

```python
class LLMNarrator:
    """
    三個模態的 evidence dict → 統一自然語言敘事。

    模式：
      "template" — 純模板，無模型，edge 可用
      "phi3"     — Phi-3-mini 3.8B 4-bit，~2GB RAM，CPU 可跑
      "llama"    — LLaMA-3.2-11B（同 DF-P2E），需 GPU
    """
    def __init__(self, mode="template", model_path=""):
        self.mode = mode

    def narrate(self, fusion_result: FusionResult, user_type="general") -> str:
        if self.mode == "template":
            return self._template_narrate(fusion_result)
        elif self.mode == "phi3":
            return self._llm_narrate(fusion_result, user_type)
```

模板模式優先實作，確保 edge 可用；LLM 模式作為 server 增強。

---

## Guideline 產出規劃

### 目的與適用範圍

**文件定位**：Anti-Deepfake-Box 詐騙電話偵測系統 — 操作人員與部署指引

**適用對象**：
- 電信詐騙防制中心操作人員
- 金融機構遠端身份驗證安全團隊
- 企業資安部門（視訊會議詐騙防護）
- 系統整合商（edge 裝置部署）

**本 guideline 的範圍**：系統輸出解讀、判定流程、部署規範、結果處置。不涵蓋模型訓練與 GP solver 設定（見 `technical_guide.md`）。

---

### Guideline 章節結構

```
§1 系統功能概述          1 頁   三模態 sandbox 架構、輸入輸出、兩種部署模式
§2 多模態證據解讀準則    2 頁   三種證據類型的閾值意義與可信度說明
§3 決策門檻與判定流程    1 頁   Cascade 路徑圖、各 stage 觸發條件
§4 可解釋性報告格式      1 頁   Layer 1–3 輸出欄位定義與範例
§5 Edge 裝置部署規範     1 頁   硬體需求、延遲預算、模態組合建議
§6 處置建議與升級程序    1 頁   偵測到偽造時的標準作業程序
§7 效能驗證方法          1 頁   消融實驗指引、壓縮降解測試、定期校準
§8 限制與注意事項        0.5頁  已知邊界條件、不適用情境
```

### 各章節核心內容

**§2 多模態證據解讀準則**

| 證據類型 | 輸出值 | 正常範圍 | 警戒值 | 可信度說明 |
|---|---|---|---|---|
| Visual（Grad-CAM） | fake_score 0–1 | < 0.30 | > 0.75 | 換臉/換頭偽造最靈敏，重度壓縮（CRF>40）時降解 |
| rPPG SNR（POS） | snr_db；fake_score | SNR > 8 dB | SNR < 2 dB | 生理訊號，不受視覺壓縮影響；光線不足時不穩 |
| AV Sync（SyncNet） | sync_error；fake_score | error < 0.3 | error > 0.65 | 聲音克隆最靈敏；無音訊時自動略過此 stage |

**§3 決策門檻與判定流程**

```
輸入影片
  ↓
[Stage 1: 輕量模態（rPPG POS, ~3ms）]
  score ≥ H → 立即判 FAKE（附 rPPG 波形異常說明）
  score ≤ L → 立即判 REAL（附正常波形確認）
  H > score > L → 進入下一 stage
  ↓
[Stage 2: 中量模態（SyncNet, ~40ms）]
  同上邏輯，附 sync timeline
  ↓
[Stage 3: 重量模態（XceptionNet, ~180ms）]
  同上邏輯，附 Grad-CAM 熱圖
  ↓
[全程不確定] → fake_score = 0.5，標記「需人工複查」
```

**§4 可解釋性報告格式**

```json
{
  "verdict":       "FAKE",
  "fake_score":    0.83,
  "confidence":    "high",
  "trigger":       "visual",
  "trigger_reason":"Visual score 0.83 ≥ threshold H=0.75 (Stage 3)",
  "evidence": {
    "visual":  {"fake_score": 0.83, "gradcam_path": "..."},
    "rppg":    {"fake_score": 0.62, "snr_db": 3.1, "ppg_path": "..."},
    "sync":    {"fake_score": 0.58, "sync_error": 0.44}
  },
  "narrative":     "偵測到臉部區域明顯偽造特徵（眼周 Grad-CAM 熱區）..."
}
```

**§5 Edge 裝置部署規範**

| 部署情境 | 啟用模態 | 記憶體需求 | 總延遲 | 適用場景 |
|---|---|---|---|---|
| 超輕量 | rPPG only | < 50 MB | ~3ms | 行動裝置、IoT |
| 標準 edge | rPPG + Sync | ~130 MB | ~45ms | 嵌入式（Raspberry Pi 5） |
| 完整 edge | rPPG + Sync + Visual | ~220 MB | ~225ms | 中端 ARM（Apple M1） |
| Server 增強 | 全部 + LLM 敘事 | ~2.2 GB | ~700ms | 雲端 / 伺服器 |

**§7 效能驗證方法**

消融實驗基準（部署前驗證）：

| 方法 | AUC 基準 | 說明 |
|---|---|---|
| Visual only (Xception) | — | guideline 附錄的效能數據來源 |
| rPPG only (POS) | — | — |
| Sync only (SyncNet) | — | — |
| Visual + rPPG | — | — |
| Visual + Sync | — | — |
| 全三模態（fixed weight） | — | — |
| 全三模態（GP cascade） | — | 部署目標 |

壓縮降解測試（定期校準）：

| 壓縮等級 | CRF=23 | CRF=28 | CRF=34 | CRF=40 | CRF=51 |
|---|---|---|---|---|---|
| Visual AUC | — | — | — | — | — |
| rPPG（預期不變） | — | — | — | — | — |
| GP Cascade | — | — | — | — | — |

警示：Cascade AUC 低於 0.60 於任意壓縮等級時，應重新調整 GP solver 閾值。

---

## 里程碑

| 週次 | 目標 | 產出 |
|---|---|---|
| Week 1–2 | Phase 1 完成 | `--explain` 模式可用，FusionResult 有 trigger/evidence |
| Week 3 | 取得資料集 | FF++ + FakeAVCeleb 可跑 |
| Week 4–5 | Phase 2 消融實驗 | guideline §7 效能基準表填入數字 |
| Week 6 | 壓縮降解實驗 | guideline §7 壓縮降解表 + Grad-CAM 視覺化範例 |
| Week 7 | Guideline §1–§5 初稿 | 系統說明 + 證據解讀 + 報告格式完稿 |
| Week 8 | Guideline §6–§8 + 審閱 | 處置建議 + 限制條款；內部 review |
| Week 9 | Guideline 完稿 | guideline PDF + 配套操作範例 |
| Week 10+ | Phase 3（Edge profile + LLM） | guideline 附錄：Phi-3-mini 敘事模組 |

---

## 關鍵設計決策

### Cascade 是結構性解釋，不是 post-hoc

DF-P2E 的解釋是 post-hoc（模型跑完再加解釋）。ADB 的 cascade 本身就是透明的決策路徑：

```
為什麼判定 FAKE？
→ 因為 rPPG Stage 1 score=0.15 ≤ L=0.20，立刻判 REAL（不是 FAKE）
→ 接著 Visual Stage 2 score=0.83 ≥ H=0.75，立刻判 FAKE，結束

這條路徑本身就是完整的解釋，不需要事後重建。
```

### Edge 解釋策略

```
輕量 cascade（邊緣）：rPPG → sync → visual
  Stage 早退出 → 只輸出 Layer 1（數值 + 觸發原因）+ Layer 2（波形/timeline）
  不跑 LLM

Server 補強：
  Stage 不確定（全部通過，回傳 0.5）→ 呼叫 LLM 生成敘事
  或使用者主動請求詳細說明時才跑
```

---

## 依賴套件（新增）

```bash
# Phase 1（Grad-CAM）
# 無額外套件，使用 PyTorch hooks

# Phase 3（LLM）
pip install transformers accelerate bitsandbytes   # Phi-3-mini / LLaMA
pip install torch torchvision                       # 已有

# 實驗（壓縮）
apt-get install -y ffmpeg   # 已有

# 視覺化（guideline 附錄圖表 + Grad-CAM 輸出）
pip install matplotlib seaborn
```
