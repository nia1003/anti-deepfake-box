"""
Generic video dataset for unlabelled or custom data.
Supports scanning a directory tree for .mp4/.avi/.mov files.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset

from .ff_dataset import VideoSample

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoDataset(Dataset):
    """
    Simple directory-scan dataset.
    label=0 → real directory, label=1 → fake directory.
    """

    def __init__(
        self,
        real_dir: Optional[str] = None,
        fake_dir: Optional[str] = None,
        video_list: Optional[List[str]] = None,
        label: int = -1,
    ):
        self.samples: List[VideoSample] = []

        if real_dir:
            self._scan_dir(real_dir, label=0, manip="original")
        if fake_dir:
            self._scan_dir(fake_dir, label=1, manip="unknown")
        if video_list:
            for p in video_list:
                self.samples.append(VideoSample(
                    video_path=str(p),
                    label=label,
                    manipulation_type="unknown",
                    compression="native",
                    split="test",
                ))

    def _scan_dir(self, directory: str, label: int, manip: str) -> None:
        for p in sorted(Path(directory).rglob("*")):
            if p.suffix.lower() in VIDEO_EXTENSIONS:
                self.samples.append(VideoSample(
                    video_path=str(p),
                    label=label,
                    manipulation_type=manip,
                    compression="native",
                    split="test",
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]
