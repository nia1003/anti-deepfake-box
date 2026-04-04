"""
DFDC (Deepfake Detection Challenge) Dataset Loader.

Supports DFDC Preview and DFDC Challenge formats.
Labels come from metadata.json per-chunk directory.

Expected structure (DFDC Preview):
    data_root/
    ├── dfdc_train_part_0/
    │   ├── metadata.json     {"video.mp4": {"label": "FAKE", "split": "train", ...}}
    │   └── *.mp4
    ├── dfdc_train_part_1/
    │   └── ...
    └── labels.csv  (Challenge format, optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset

from .ff_dataset import VideoSample


class DFDCDataset(Dataset):
    """
    DFDC dataset loader compatible with VideoSample interface.

    Automatically discovers all parts under data_root and loads metadata.json.
    """

    LABEL_MAP = {"REAL": 0, "FAKE": 1}

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        max_videos: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.max_videos = max_videos
        self.samples: List[VideoSample] = self._build_sample_list()

    def _build_sample_list(self) -> List[VideoSample]:
        samples: List[VideoSample] = []

        # Discover all part directories
        parts = sorted(
            [p for p in self.data_root.iterdir() if p.is_dir() and "part" in p.name.lower()]
        )
        if not parts:
            parts = [self.data_root]  # Single-directory layout

        for part_dir in parts:
            meta_path = part_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                metadata: Dict = json.load(f)

            for fname, info in metadata.items():
                label_str = info.get("label", "").upper()
                if label_str not in self.LABEL_MAP:
                    continue
                label = self.LABEL_MAP[label_str]

                video_split = info.get("split", "train").lower()
                if video_split != self.split:
                    continue

                video_path = part_dir / fname
                if not video_path.exists():
                    continue

                samples.append(VideoSample(
                    video_path=str(video_path),
                    label=label,
                    manipulation_type="dfdc" if label == 1 else "original",
                    compression="native",
                    split=self.split,
                ))

                if self.max_videos and len(samples) >= self.max_videos:
                    return samples

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]
