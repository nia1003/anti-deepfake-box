# Anti-Deepfake-Box

**三路多模態 Deepfake 偵測框架與即時防護系統**：本專案整合視覺紋理 (FaceForensics++ XceptionNet)、生理訊號 (rPPG-Toolbox PhysNet + SNR) 與音視訊同步 (LatentSync SyncNet) 三路互補偵測訊號。系統具備硬體資源自適應能力，並提供雙情境模式（線上即時 / 離線鑑識），透過瀏覽器擴充功能與網頁儀表板，提供無縫的防偽防詐體驗。

---

## 🌟 雙情境運行模式 (Use-Case Modes)

系統根據實務情境區分設計目標，並由 `configs/modes/` 下的設定檔動態載入：

| 特性 / 參數 | ⚡ 線上即時模式 (Real-time) | 🔬 離線鑑識模式 (Forensic) |
| :--- | :--- | :--- |
| **目標場景** | 一般大眾（通訊軟體即時防詐） | 執法、鑑識單位（事後分析） |
| **設計核心** | 速度與輕量化優先（Time Critical） | 判定精準度絕對優先（不計算力代價） |
| **判定門檻** | **50%**（Threshold 0.50） | **40%**（更嚴格，更容易標記為偽造） |
| **音視頻同步** | 停用（節省算力） | 啟用（LatentSync + Whisper） |
| **視覺抽幀數** | 8 幀 | 32 幀 |
| **人臉辨識模型**| `buffalo_sc` (快速輕量) | `buffalo_l` (高精度) |

---

## 🏗️ 系統架構

```text
瀏覽器播放影片 (YouTube, 視訊通話等)
   │
   ▼ [擴充功能 Extension] 
 content.js 擷取畫面與音訊 ──(帶有 mode 參數的 POST)──▶ FastAPI 後端 (api/app.py)
   │                                                       │
   ▼                                                       ▼ [非同步並行管線]
 [Web Dashboard]                                     1. 視覺紋理 (XceptionNet)
 localhost:8000/                                     2. 生理訊號 (rPPG SNR)
 輪詢 GET /sessions 即時顯示進度與分數                 3. 音視訊同步 (SyncNet) *僅 Forensic
   │                                                       │
   └─ 動態條狀圖 (Animated Bars)                           ▼ [融合決策 WeightedEnsemble]
   └─ 依據閥值顯示 FAKE / REAL ◀───────────────────────── 輸出最終 Fake Score
```

### 三路偵測原理

| 模態 | 核心技術 | 偵測原理 | 優勢 | 侷限 |
|------|------|----------|------|------|
| **視覺紋理** | XceptionNet (FF++ 預訓練) | 辨識空間偽影、GAN/擴散模型壓縮痕跡 | 基礎準確率高 | 遇高強度影片壓縮易失效 |
| **生理訊號** | PhysNet + SNR | 真人具備微血管搏動的週期性 PPG 訊號 | 物理特徵，不受視覺欺騙影響 | 極高品質的生成模型可能繞過 |
| **音視同步** | LatentSync (SyncNet) | 偵測 lip-sync 時序不一致（唇音誤差） | 針對語音替換詐騙極為有效 | 無音訊時不可用 |

---

## 🚀 快速開始

### 1. 安裝與準備

```bash
git clone https://github.com/nia1003/anti-deepfake-box.git
cd anti-deepfake-box

# 安裝依賴
pip install -r requirements.txt

# 載入第三方子模組 (rPPG-Toolbox 等)
git submodule update --init --recursive
```

確認您的模型權重已放置於 `checkpoints/` 目錄下：
* `xception_ff_c23.pth`
* `physnet_ubfc.pth`
* `latentsync_syncnet.pth`

### 2. 啟動後端伺服器與 Web Dashboard

```bash
# 啟動 FastAPI 伺服器
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
啟動後，請打開瀏覽器前往 `http://localhost:8000/`，您將看到 OpenAI 風格的即時監控儀表板。

### 3. 載入瀏覽器擴充功能
1. 打開 Chrome 並前往 `chrome://extensions/`。
2. 開啟右上角的「開發人員模式」。
3. 點擊「載入未封裝項目」，選擇本專案內的 `extension/` 資料夾。
4. 在擴充功能選單中，您可以自由切換 **Real-time** 或 **Forensic** 模式。當網頁播放影片時，系統即會自動開始分析。

---

## 📊 參數調校與 Pareto 最佳化分析

本專案採用雙門檻與平行融合策略。透過網格搜索（Grid Search）尋找最佳的融合權重與閥值。

### Phase 1：rPPG SNR 閾值校準

在融合前，必須先找出區分真人與假人的最佳 SNR 閥值：
```bash
python scripts/calibrate_snr.py \
    --data_root /data/FF++ \
    --config configs/default.yaml \
    --split val \
    --update_config
```

### Phase 2：Pareto 引擎網格搜索

窮舉測試不同模組版本的排列組合與權重（α, β, γ），並繪製 FAR (誤接受率) 與 FRR (誤拒絕率) 的 Pareto 最優前沿（Pareto Front）：

```bash
# 收集各模組獨立分數
python eval/score_collector.py --data_root /data/FF++ --output eval/module_scores.json

# 執行網格搜索與 Pareto 分析
python eval/engine_search.py
```
*系統將自動輸出 `pareto_configs.csv` 與 `pareto_plot.png`，並剔除無效的權重組合，保留能在安全性與便利性之間取得最佳平衡的參數，供 `configs/modes/` 引用。*

---

## 📚 引用與致謝

```bibtex
@inproceedings{rossler2019faceforensics,
  title={FaceForensics++: Learning to Detect Manipulated Facial Images},
  author={Rössler, Andreas et al.},
  booktitle={ICCV}, year={2019}
}
@inproceedings{liu2023rppgtoolbox,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin et al.},
  booktitle={NeurIPS}, year={2023}
}
@inproceedings{peng2025latentsync,
  title={LatentSync: Audio Conditioned Latent Diffusion Models for Lip Sync},
  author={Peng, Chunyu et al.},
  booktitle={CVPR}, year={2025}
}
```

## ⚖️ 授權協議

MIT License
