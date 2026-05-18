# Anti-Deepfake-Box — 專案記憶

## 專案目標

多模態 deepfake 偵測系統，針對**詐騙電話**情境設計：
- 視覺（XceptionNet）+ rPPG（POS）+ 音視覺同步（SyncNet）三模態融合
- 支援即時串流（瀏覽器 extension）與離線批次評估
- DeepfakeBench 整合：adb_visual / adb_rppg / adb_sync 三個 adapter

---

## 音訊 Pipeline（audio-pipe branch）

### 已完成的檔案

| 檔案 | 功能 |
|------|------|
| `preprocessing/audio_extractor.py` | FFmpeg 抽音訊 WAV，graceful fallback（FF++ 無音訊回 None） |
| `preprocessing/audio_features.py` | librosa mel spectrogram：`extract_mel`, `align_to_frames`, `segment_mel` |
| `detectors/sync_detector.py` | Whisper mel + SyncNet 滑窗，無權重時降級為 motion heuristic |
| `deepfakebench_adapters/adb_sync_detector.py` | DFB adapter，從 `data_dict["video_path"]` 抽音訊 |
| `scripts/sync_score_csv.py` | 批次推論 → CSV（video_id, fake_score, label, inference_ms） |
| `tests/test_audio_extractor.py` | 9 個 pytest，需要 ffmpeg |
| `tests/test_audio_features.py` | 12 個 pytest，6 個純 numpy 可直接跑 |

### 驗證指令

```bash
# 層 1：現在就能跑
python3 -m pytest tests/ -v
# 期望：7 passed, 14 skipped

# 層 2：裝 ffmpeg 後
apt-get install -y ffmpeg
python3 -m pytest tests/test_audio_extractor.py -v
# 期望：9 passed

# 層 3：裝 librosa 後
pip install librosa
python3 -m pytest tests/test_audio_features.py -v
# 期望：12 passed

# 層 4：端對端（需要 FakeAVCeleb 影片）
python3 scripts/sync_score_csv.py \
    --input_dir /path/to/FakeAVCeleb/test \
    --output results/sync_scores.csv \
    --device cpu
```

### 關鍵限制

- **DFB 的 `data_dict` 預設沒有 `video_path`**：adb_sync_detector 的音訊永遠是 None（score=0.5），需等 P9（dataset loader 修改）完成
- FF++ 無音訊，只有 FakeAVCeleb 能跑 SyncDetector
- SyncNet 權重（`syncnet_path`）可選，無權重時用 motion heuristic fallback

### 阻塞依賴（等待其他人）

| 項目 | 說明 |
|------|------|
| P9 `training/dataset/audio_dataset.py` | DFB dataset loader 補 video_path，等其他人完成 |
| FakeAVCeleb 資料集部署 | 需等資料集掛載 |
| A 組 score CSV | 等 A 組完成後才需 `logs_to_csv.py` |

---

## DeepfakeBench 整合（framework branch）

### 已完成
- `training/detectors/dummy_detector.py` — B-1 最小驗證
- `training/detectors/adb_visual/rppg/sync_detector.py` — B-2 三個 adapter
- `training/config/detector/adb_*.yaml` — 三個 YAML
- `docs/deepfakebench_integration.md` — 技術文件

### 用 DFB 跑 adb_sync 測試
```bash
cd /home/user/DeepfakeBench
export PYTHONPATH=/home/user/anti-deepfake-box:$PYTHONPATH
python training/train.py \
    --detector_path training/config/detector/adb_sync.yaml \
    --phase test
# 注意：目前 data_dict 無 video_path，音訊不會真的跑，score=0.5
```

---

## 音訊開通後的實驗計畫

1. **消融實驗**：Visual / Visual+rPPG / Visual+Sync / 全三模態 AUC 比較
2. **資料集差異**：FF++（無音訊）vs FakeAVCeleb（有音訊）的 sync score 分布
3. **詐騙電話情境**：聲音克隆 vs 臉部換臉 的偵測率差異
4. **Pareto 優化**：三模態 grid search（α·visual + β·rppg + γ·sync）

---

## 分支規則

| Repo | Branch | 用途 |
|------|--------|------|
| anti-deepfake-box | `main` | 穩定版 |
| anti-deepfake-box | `audio-pipe` | 音訊 pipeline 開發 |
| DeepfakeBench | `framework` | DFB 整合 |

## 依賴套件

```bash
pip install librosa openai-whisper
apt-get install -y ffmpeg
```
