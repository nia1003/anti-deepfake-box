"""
Celeb-DF (v1 / v2) Dataset Loader.

Supported directory layouts
---------------------------
Celeb-DF-v2 (official):
    data_root/
    ├── Celeb-real/       *.mp4  (real videos)
    ├── Celeb-synthesis/  *.mp4  (fake videos generated from Celeb-real)
    ├── YouTube-real/     *.mp4  (YouTube real clips)
    └── List_of_testing_videos.txt
             Format: "<label> <rel/path/to/video.mp4>"  (0=real, 1=fake)

Celeb-DF-v1:
    data_root/
    ├── real/
    ├── fake/
    └── List_of_testing_videos.txt  (same format)

If List_of_testing_videos.txt is absent the loader falls back to scanning
all *.mp4 files under real*/ and fake*/ / synthesis*/ directories.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datasets.ff_dataset import VideoSample


class CelebDFDataset(Dataset):
    """
    Celeb-DF v1 / v2 dataset loader compatible with the VideoSample interface.

    Parameters
    ----------
    data_root : str
        Root directory of the Celeb-DF release.
    version   : "v1" | "v2"
        Dataset version (affects subdirectory names).
    split     : "test" | "all"
        "test" uses List_of_testing_videos.txt; "all" scans all videos.
    max_videos : int | None
        Truncate dataset to at most this many videos.
    """

    _REAL_DIRS_V2 = ["Celeb-real", "YouTube-real"]
    _FAKE_DIRS_V2 = ["Celeb-synthesis"]
    _REAL_DIRS_V1 = ["real"]
    _FAKE_DIRS_V1 = ["fake"]
    _TEST_LIST = "List_of_testing_videos.txt"

    def __init__(
        self,
        data_root: str,
        version: str = "v2",
        split: str = "test",
        max_videos: Optional[int] = None,
    ):
        self.data_root = Path(data_root)
        self.version = version.lower()
        self.split = split
        self.max_videos = max_videos
        self.samples: List[VideoSample] = self._build_sample_list()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _real_dirs(self) -> List[Path]:
        dirs = self._REAL_DIRS_V2 if self.version == "v2" else self._REAL_DIRS_V1
        return [self.data_root / d for d in dirs]

    def _fake_dirs(self) -> List[Path]:
        dirs = self._FAKE_DIRS_V2 if self.version == "v2" else self._FAKE_DIRS_V1
        return [self.data_root / d for d in dirs]

    def _load_from_test_list(self) -> Optional[List[VideoSample]]:
        """Parse List_of_testing_videos.txt → VideoSample list."""
        txt = self.data_root / self._TEST_LIST
        if not txt.exists():
            return None

        samples: List[VideoSample] = []
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            label_str, rel_path = parts
            try:
                label = int(label_str)
            except ValueError:
                continue

            video_path = self.data_root / rel_path
            if not video_path.exists():
                continue

            manip = "celebdf_fake" if label == 1 else "original"
            samples.append(VideoSample(
                video_path=str(video_path),
                label=label,
                manipulation_type=manip,
                compression="native",
                split="test",
            ))
            if self.max_videos and len(samples) >= self.max_videos:
                break
        return samples

    def _scan_directories(self) -> List[VideoSample]:
        """Fallback: scan real/fake directories for all *.mp4 files."""
        samples: List[VideoSample] = []

        for d in self._real_dirs():
            if not d.exists():
                continue
            for p in sorted(d.glob("*.mp4")):
                samples.append(VideoSample(
                    video_path=str(p), label=0,
                    manipulation_type="original",
                    compression="native", split="all",
                ))
                if self.max_videos and len(samples) >= self.max_videos:
                    return samples

        for d in self._fake_dirs():
            if not d.exists():
                continue
            for p in sorted(d.glob("*.mp4")):
                samples.append(VideoSample(
                    video_path=str(p), label=1,
                    manipulation_type="celebdf_fake",
                    compression="native", split="all",
                ))
                if self.max_videos and len(samples) >= self.max_videos:
                    return samples

        return samples

    def _build_sample_list(self) -> List[VideoSample]:
        if self.split == "test":
            from_list = self._load_from_test_list()
            if from_list is not None:
                return from_list
        return self._scan_directories()

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]

    def stats(self) -> dict:
        real = sum(1 for s in self.samples if s.label == 0)
        return {"real": real, "fake": len(self.samples) - real}
