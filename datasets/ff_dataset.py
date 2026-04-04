"""
FaceForensics++ Dataset Loader.

Supports c0 / c23 / c40 compressions and all manipulation types:
Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter, DeepFakeDetection.

Expected directory layout (default DeepfakeBench preprocessing output):
    data_root/
    ├── original_sequences/
    │   └── youtube/
    │       ├── c23/videos/  (or frames/)
    │       └── c40/videos/
    └── manipulated_sequences/
        ├── Deepfakes/c23/videos/
        ├── Face2Face/c23/videos/
        ├── FaceSwap/c23/videos/
        ├── NeuralTextures/c23/videos/
        └── FaceShifter/c23/videos/   (optional)

Splits JSON: dataset/splits/{train,val,test}.json — list of video base names.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from torch.utils.data import Dataset


MANIPULATION_TYPES = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
    "FaceShifter",
    "DeepFakeDetection",
]

COMPRESSION_LEVELS = ["c0", "c23", "c40"]


class VideoSample:
    """Lightweight descriptor for a single video sample."""
    __slots__ = ("video_path", "label", "manipulation_type", "compression", "split")

    def __init__(self, video_path: str, label: int, manipulation_type: str,
                 compression: str, split: str):
        self.video_path = video_path
        self.label = label                    # 0 = real, 1 = fake
        self.manipulation_type = manipulation_type
        self.compression = compression
        self.split = split


class FaceForensicsDataset(Dataset):
    """
    Iterate over FF++ videos returning VideoSample descriptors.

    Labels: 0 = original/real, 1 = manipulated/fake.
    Videos are not decoded here; the caller passes video_path to the
    detection pipeline (UnifiedFaceExtractor + detectors).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        compression: str = "c23",
        manipulation_types: Optional[List[str]] = None,
        splits_dir: Optional[str] = None,
        return_path_only: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.compression = compression
        self.manipulation_types = manipulation_types or MANIPULATION_TYPES
        self.splits_dir = Path(splits_dir) if splits_dir else self.data_root / "splits"
        self.return_path_only = return_path_only

        self.samples: List[VideoSample] = self._build_sample_list()

    def _load_split_names(self) -> List[str]:
        """Load video base names for the current split from JSON or directory scan."""
        json_path = self.splits_dir / f"{self.split}.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            # FF++ splits.json format: list of [name, name] pairs or list of names
            names = []
            for item in data:
                if isinstance(item, list):
                    names.extend(item)
                else:
                    names.append(str(item))
            return list(set(names))
        # Fallback: enumerate original sequence directory
        orig_dir = self._orig_video_dir()
        if orig_dir.exists():
            return [p.stem for p in sorted(orig_dir.glob("*.mp4"))]
        return []

    def _orig_video_dir(self) -> Path:
        return self.data_root / "original_sequences" / "youtube" / self.compression / "videos"

    def _fake_video_dir(self, manip: str) -> Path:
        return self.data_root / "manipulated_sequences" / manip / self.compression / "videos"

    def _build_sample_list(self) -> List[VideoSample]:
        names = self._load_split_names()
        samples: List[VideoSample] = []

        orig_dir = self._orig_video_dir()
        for name in names:
            candidate = orig_dir / f"{name}.mp4"
            if not candidate.exists():
                # Try frames directory
                frames_candidate = orig_dir.parent / "frames" / name
                if frames_candidate.exists():
                    candidate = orig_dir / f"{name}.mp4"  # Still reference by path
                else:
                    continue
            if candidate.exists():
                samples.append(VideoSample(
                    video_path=str(candidate),
                    label=0,
                    manipulation_type="original",
                    compression=self.compression,
                    split=self.split,
                ))

        for manip in self.manipulation_types:
            fake_dir = self._fake_video_dir(manip)
            if not fake_dir.exists():
                continue
            for name in names:
                # FF++ fake naming: <target>_<source>.mp4
                matches = list(fake_dir.glob(f"{name}_*.mp4")) + \
                          list(fake_dir.glob(f"*_{name}.mp4")) + \
                          [fake_dir / f"{name}.mp4"]
                for path in matches:
                    if path.exists():
                        samples.append(VideoSample(
                            video_path=str(path),
                            label=1,
                            manipulation_type=manip,
                            compression=self.compression,
                            split=self.split,
                        ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VideoSample:
        return self.samples[idx]

    def stats(self) -> Dict[str, int]:
        """Return label distribution and per-manipulation counts."""
        counts: Dict[str, int] = {"real": 0, "fake": 0}
        for s in self.samples:
            if s.label == 0:
                counts["real"] += 1
            else:
                counts["fake"] += 1
                counts[s.manipulation_type] = counts.get(s.manipulation_type, 0) + 1
        return counts

    @classmethod
    def all_splits(
        cls,
        data_root: str,
        compression: str = "c23",
        manipulation_types: Optional[List[str]] = None,
    ) -> Dict[str, "FaceForensicsDataset"]:
        """Convenience: return train/val/test datasets at once."""
        return {
            split: cls(data_root, split=split, compression=compression,
                       manipulation_types=manipulation_types)
            for split in ("train", "val", "test")
        }
