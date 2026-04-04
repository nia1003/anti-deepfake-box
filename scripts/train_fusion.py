#!/usr/bin/env python3
"""
Train the fusion module (Stage 1: weighted calibration, Stage 2: meta-classifier).

Activate only after validation passes:
  - FF++ AUC-ensemble > 0.95
  - Celeb-DF v2 AUC-ensemble > 0.85

Stage 1 (weighted, ~30 min):
    python scripts/train_fusion.py \
        --config configs/training.yaml \
        --stage weighted

Stage 2 (meta-classifier, ~2-4hr):
    python scripts/train_fusion.py \
        --config configs/training.yaml \
        --stage meta \
        --freeze_backbones

Stage 3 (end-to-end fine-tuning, optional):
    python scripts/train_fusion.py \
        --config configs/training.yaml \
        --stage meta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing import UnifiedFaceExtractor, AudioExtractor
from detectors import VisualDetector, RPPGDetector, SyncDetector
from fusion import WeightedEnsemble, MetaClassifier
from datasets.ff_dataset import FaceForensicsDataset
from datasets.dfdc_dataset import DFDCDataset
from evaluation.metrics import compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ #
#  Score collection (shared by both stages)                           #
# ------------------------------------------------------------------ #

def collect_scores(
    dataset,
    config: dict,
    freeze_backbones: bool = True,
    max_videos: Optional[int] = None,
) -> Dict:
    """
    Run all three detectors over the dataset; return collected scores + labels.
    """
    face_extractor = UnifiedFaceExtractor(config.get("preprocessing", config))
    audio_extractor = AudioExtractor(config.get("preprocessing", config))

    visual_det = VisualDetector(config.get("detectors", {}).get("visual", config))
    rppg_det   = RPPGDetector(config.get("detectors", {}).get("rppg", config))
    sync_det   = SyncDetector(config.get("detectors", {}).get("sync", config))

    if freeze_backbones:
        for det in [visual_det, rppg_det, sync_det]:
            if hasattr(det, "model") and det.model is not None:
                for p in det.model.parameters():
                    p.requires_grad = False

    all_scores, all_labels = [], []
    n = min(len(dataset), max_videos) if max_videos else len(dataset)

    for i, sample in enumerate(list(dataset)[:n]):
        print(f"  [{i+1}/{n}] {Path(sample.video_path).name}", end="  ")
        face_track = face_extractor.extract(sample.video_path)
        audio_path = audio_extractor.extract_to_temp(sample.video_path)

        s_v = visual_det.detect(face_track)
        s_r = rppg_det.detect(face_track)
        s_s = sync_det.detect(face_track, audio_path)

        if audio_path and Path(audio_path).exists():
            try:
                os.unlink(audio_path)
            except OSError:
                pass

        score_vec = [
            s_v if s_v is not None else 0.5,
            s_r if s_r is not None else 0.5,
            s_s if s_s is not None else 0.5,
        ]
        all_scores.append(score_vec)
        all_labels.append(sample.label)
        print(f"v={s_v:.3f} r={s_r:.3f} s={'N/A' if s_s is None else f'{s_s:.3f}'}")

    return {
        "scores": np.array(all_scores, dtype=np.float32),  # (N, 3)
        "labels": np.array(all_labels, dtype=np.int64),
    }


# ------------------------------------------------------------------ #
#  Stage 1: Optimise weighted ensemble weights via grid search        #
# ------------------------------------------------------------------ #

def train_weighted_stage(config: dict, train_data: dict, val_data: dict) -> Dict:
    """Grid-search optimal weights over train split, validate on val split."""
    best_auc, best_weights = 0.0, {"visual": 0.5, "rppg": 0.25, "sync": 0.25}
    grid_step = 0.1

    vals = np.arange(0, 1 + grid_step, grid_step).round(1)
    evaluated = 0

    for w_v in vals:
        for w_r in vals:
            w_s = round(1.0 - w_v - w_r, 1)
            if w_s < 0 or abs(w_v + w_r + w_s - 1.0) > 0.01:
                continue

            weights = {"visual": w_v, "rppg": w_r, "sync": w_s}
            ensemble_cfg = {"fusion": {"weights": weights, "threshold": 0.5}}
            fuser = WeightedEnsemble({**config, **ensemble_cfg})

            scores = []
            for row in train_data["scores"]:
                r = fuser.fuse(row[0], row[1], row[2] if row[2] != 0.5 else None)
                scores.append(r.fake_score)

            m = compute_metrics(train_data["labels"].tolist(), scores)
            evaluated += 1

            if m.auc > best_auc:
                best_auc = m.auc
                best_weights = weights.copy()

    print(f"  Searched {evaluated} weight combinations")
    print(f"  Best weights: {best_weights}  Train AUC={best_auc:.4f}")

    # Validate
    fuser = WeightedEnsemble({**config, "fusion": {"weights": best_weights}})
    val_scores = [fuser.fuse(r[0], r[1], r[2] if r[2] != 0.5 else None).fake_score
                  for r in val_data["scores"]]
    val_m = compute_metrics(val_data["labels"].tolist(), val_scores)
    print(f"  Val metrics: {val_m}")
    return best_weights


# ------------------------------------------------------------------ #
#  Stage 2: Train meta-classifier MLP                                 #
# ------------------------------------------------------------------ #

def train_meta_stage(config: dict, train_data: dict, val_data: dict) -> str:
    """Train 2-layer MLP on collected scores."""
    meta = MetaClassifier(config)
    meta.build(input_dim=3)  # 3 scores

    X_train = torch.from_numpy(train_data["scores"])
    y_train = torch.from_numpy(train_data["labels"])
    X_val   = torch.from_numpy(val_data["scores"])
    y_val   = torch.from_numpy(val_data["labels"])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

    training_cfg = config.get("training", {})
    optimizer = torch.optim.AdamW(
        meta.model.parameters(),
        lr=float(training_cfg.get("lr", 1e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training_cfg.get("epochs", 100))
    )
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    save_dir = Path(training_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(save_dir / "meta_classifier_best.pth")

    epochs = int(training_cfg.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        train_loss = meta.train_epoch(train_loader, optimizer, criterion)
        scheduler.step()

        # Validation
        meta.model.eval()
        with torch.no_grad():
            logits = meta.model(X_val.to(meta.device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        val_m = compute_metrics(y_val.numpy().tolist(), probs.tolist())

        if val_m.auc > best_val_auc:
            best_val_auc = val_m.auc
            meta.save(best_path)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                  f"val_AUC={val_m.auc:.4f}  (best={best_val_auc:.4f})")

    print(f"\nMeta-classifier training complete. Best val AUC={best_val_auc:.4f}")
    print(f"Model saved to {best_path}")
    return best_path


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Train Anti-Deepfake-Box fusion")
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--stage", choices=["weighted", "meta"], default="weighted")
    parser.add_argument("--freeze_backbones", action="store_true", default=True)
    parser.add_argument("--max_videos_train", type=int, default=None)
    parser.add_argument("--max_videos_val",   type=int, default=None)
    parser.add_argument("--scores_cache", default="",
                        help="Path to pre-collected scores .npz (skip re-extraction)")
    args = parser.parse_args()

    config = load_config(args.config)
    training_cfg = config.get("training", {})

    # Load datasets
    data_root = training_cfg.get("ff_data_root", "")
    compression = training_cfg.get("compression", "c23")

    train_ds = FaceForensicsDataset(data_root, split="train", compression=compression)
    val_ds   = FaceForensicsDataset(data_root, split="val",   compression=compression)

    # Optionally add DFDC
    dfdc_root = training_cfg.get("dfdc_data_root", "")
    if dfdc_root and Path(dfdc_root).exists():
        dfdc_train = DFDCDataset(dfdc_root, split="train")
        print(f"  Adding DFDC: {len(dfdc_train)} samples")
        from torch.utils.data import ConcatDataset
        train_ds = ConcatDataset([train_ds, dfdc_train])

    # Collect or load scores
    if args.scores_cache and Path(args.scores_cache).exists():
        print(f"Loading pre-collected scores from {args.scores_cache}")
        cached = np.load(args.scores_cache)
        train_data = {"scores": cached["train_scores"], "labels": cached["train_labels"]}
        val_data   = {"scores": cached["val_scores"],   "labels": cached["val_labels"]}
    else:
        print(f"\nCollecting scores on {len(train_ds)} training samples...")
        train_data = collect_scores(train_ds, config, args.freeze_backbones, args.max_videos_train)
        print(f"\nCollecting scores on {len(val_ds)} validation samples...")
        val_data   = collect_scores(val_ds, config, args.freeze_backbones, args.max_videos_val)

        if args.scores_cache:
            np.savez(args.scores_cache,
                     train_scores=train_data["scores"], train_labels=train_data["labels"],
                     val_scores=val_data["scores"],   val_labels=val_data["labels"])
            print(f"Scores cached to {args.scores_cache}")

    # Train
    if args.stage == "weighted":
        print("\n--- Stage 1: Weighted Ensemble Calibration ---")
        best_weights = train_weighted_stage(config, train_data, val_data)
        save_path = Path(training_cfg.get("save_dir", "checkpoints")) / "best_weights.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(best_weights, f, indent=2)
        print(f"Best weights saved to {save_path}")
    else:
        print("\n--- Stage 2: Meta-Classifier Training ---")
        model_path = train_meta_stage(config, train_data, val_data)
        print(f"Update fusion.meta.model_path={model_path} in your config to use it.")


if __name__ == "__main__":
    main()
