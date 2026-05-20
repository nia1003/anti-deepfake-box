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

**Phase 2 驗收**：消融表（7 行）+ 壓縮曲線，論文 Table 1 和 Figure 的資料來源。

---

### Phase 3：Edge Profile（論文投出後）

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

## 論文規劃

### 定位

**標題**：Multi-Modal Explainable Deepfake Detection for Edge-Deployed Fraud Call Scenarios

**vs DF-P2E 的差異化**：

| | DF-P2E（ACM MM 2025） | ADB（本論文） |
|---|---|---|
| 模態 | 視覺 only | 視覺 + rPPG + AV Sync |
| 輸入 | 靜態圖片 | 影片（含時序） |
| 解釋 | Grad-CAM + caption + LLM | +BVP 波形 + sync timeline + cascade 路徑 |
| Edge | 無（CLIP-large + LLaMA-11B） | Cascade fail-fast，輕量模態先跑 |
| 場景 | 通用 deepfake | 詐騙電話（音視覺偽造） |

### 論文結構

```
§1 Introduction          1.5 頁   詐騙電話情境 + 現有方法兩個缺口
§2 Related Work          1.0 頁   視覺偵測 + rPPG + AV Sync + XAI
§3 Method                2.5 頁   架構 + 三模態 + cascade + 解釋層次
§4 Experiments           3.0 頁   消融表 + 壓縮曲線 + edge latency
§5 Explainability        1.0 頁   解釋範例 + vs DF-P2E 對比
§6 Discussion            0.5 頁   Cascade 是結構性解釋 + Limitations
```

### 實驗產出目標

**Table 1：消融實驗（FakeAVCeleb）**

| 方法 | AUC | FAR | FRR |
|---|---|---|---|
| Visual only (Xception) | - | - | - |
| rPPG only (POS) | - | - | - |
| Sync only (SyncNet) | - | - | - |
| Visual + rPPG | - | - | - |
| Visual + Sync | - | - | - |
| All modalities (fixed weight) | - | - | - |
| All modalities (GP cascade) | **-** | **-** | **-** |

**Table 2：壓縮降解（AUC vs CRF）**

| 偵測器 | CRF=23 | CRF=28 | CRF=34 | CRF=40 | CRF=51 |
|---|---|---|---|---|---|
| Xception | - | - | - | - | - |
| SBI | - | - | - | - | - |（預期最穩）
| rPPG POS | - | - | - | - | - |（預期不變）
| GP Cascade | - | - | - | - | - |

**Table 3：Edge Latency Profile**

| 偵測器 | 模型大小 | CPU latency | 需 GPU |
|---|---|---|---|
| rPPG POS | 0 MB | ~3ms | No |
| SyncNet | ~30 MB | ~40ms | No |
| XceptionNet | ~92 MB | ~180ms | No |
| Phi-3-mini (敘事) | ~2 GB | ~500ms | No |

**Figure：三模態解釋視覺化**
- 詐騙電話截圖 + 三欄：Grad-CAM 熱圖 / BVP 波形 + SNR / Sync timeline
- 對比 DF-P2E：只有左欄（視覺）vs ADB：三欄全有

### 投稿目標

| 選項 | 截稿 | 說明 |
|---|---|---|
| **IEEE TIFS**（期刊） | Rolling | 多模態鑑識最對口，無截稿壓力，優先考慮 |
| ACM MM 2026 | ~2026/03 | DF-P2E 在這裡發，follow-up 最自然 |
| Media Forensics Workshop | ~2026/03 | 4 頁短論文，快速發表中間成果 |

---

## 里程碑

| 週次 | 目標 | 產出 |
|---|---|---|
| Week 1–2 | Phase 1 完成 | `--explain` 模式可用，FusionResult 有 trigger/evidence |
| Week 3 | 取得資料集 | FF++ + FakeAVCeleb 可跑 |
| Week 4–5 | Phase 2 消融實驗 | Table 1 數字 |
| Week 6 | 壓縮降解實驗 | Table 2 + Figure |
| Week 7 | Grad-CAM 視覺化 | Figure 解釋視覺化 |
| Week 8 | 論文初稿 | §1–§4 完成 |
| Week 9 | DF-P2E 對比 + §5 | 完稿 |
| Week 10+ | Phase 3（Edge profile + LLM） | Table 3 + 加分項 |

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

### 與 DF-P2E 的引用關係

DF-P2E 是最直接的 related work（ACM MM 2025，Grad-CAM + caption + LLM 框架的先行者）。  
ADB 不是競爭，而是在其基礎上擴展到：多模態影片 + 生理訊號 + 時序解釋 + edge 部署。  
論文中 cite DF-P2E，說明「我們將其擴展至多模態音視覺場景，並針對邊緣裝置的延遲限制重新設計解釋架構」。

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

# 視覺化（optional，for paper figures）
pip install matplotlib seaborn
```
