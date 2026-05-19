"""
BMMA-GPT Serial Cascade Fusion.

Implements the dual-threshold cascade from the BMMA-GPT paper (IEEE TDSC 2025).
Each stage applies independent high/low thresholds; uncertain samples pass to
the next stage rather than triggering an immediate decision.

This module reads the GP solver's optimised configuration directly from
real_data_vip_settings.csv (Week 12 output of fusion_solver_prod_v1.ipynb).

Interface is drop-in compatible with WeightedEnsemble.fuse().

Usage
-----
# Load from GP solver CSV:
from fusion.serial_cascade import SerialCascade
cascade = SerialCascade.from_csv("results/real_data_vip_settings.csv")
result = cascade.fuse(visual_score=0.82, rppg_score=0.41, sync_score=None)
print(result)   # same FusionResult as WeightedEnsemble

# Load from config dict (cascade_config points to CSV):
cascade = SerialCascade({"fusion": {"cascade_config": "results/real_data_vip_settings.csv",
                                    "threshold": 0.50}})

GP Solver CSV schema (real_data_vip_settings.csv):
    stage_order, modality, threshold_H, threshold_L, pareto_far, pareto_frr
    1,           visual,   0.75,        0.30,        0.04,       0.08
    2,           rppg,     0.70,        0.25,        0.04,       0.08
    3,           av_sync,  0.65,        0.20,        0.04,       0.08

Note: GP solver uses "av_sync" for the sync modality (matching collect_scores.py
DETECTOR_META). SerialCascade normalises "av_sync" → "sync" internally so that
fuse(sync_score=...) works transparently.

Legacy JSON format (for manual testing / unit tests):
    {"stages": [{"name": "visual", "H": 0.75, "L": 0.30}, ...],
     "default_threshold": 0.50, "fallback_score": 0.50}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .weighted_ensemble import FusionResult


# ------------------------------------------------------------------ #
#  Data structures                                                      #
# ------------------------------------------------------------------ #

@dataclass
class CascadeStage:
    """One stage in the BMMA-GPT cascade."""
    name: str     # "visual" | "rppg" | "sync"
    H: float      # score >= H → early exit as FAKE
    L: float      # score <= L → early exit as REAL

    def __post_init__(self):
        if self.L >= self.H:
            raise ValueError(
                f"CascadeStage {self.name!r}: L ({self.L}) must be < H ({self.H})"
            )


# ------------------------------------------------------------------ #
#  SerialCascade                                                        #
# ------------------------------------------------------------------ #

class SerialCascade:
    """
    BMMA-GPT dual-threshold serial cascade.

    Decision logic for each stage k with score p_k:
      p_k >= H_k  →  early exit: FAKE
      p_k <= L_k  →  early exit: REAL
      L_k < p_k < H_k  →  uncertain: pass to stage k+1

    After the last stage, returns fallback_score (default 0.5).

    Modalities unavailable (score=None) are skipped without decision.
    If all stages are skipped (no modalities available), returns fallback_score.
    """

    MODALITY_ALIASES = {"av_sync": "sync"}  # GP solver → internal name

    def __init__(self, config: dict):
        fusion_cfg = config.get("fusion", {})
        cascade_path = fusion_cfg.get("cascade_config", "")
        self.default_threshold: float = float(fusion_cfg.get("threshold", 0.50))
        self.fallback_score: float = float(fusion_cfg.get("fallback_score", 0.50))

        if not cascade_path:
            raise ValueError(
                "SerialCascade requires fusion.cascade_config in config. "
                "Set it to the path of real_data_vip_settings.csv."
            )
        p = Path(cascade_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Cascade config not found: {cascade_path}\n"
                f"Generate it by running the GP solver (fusion_solver_prod_v1.ipynb)."
            )

        if p.suffix.lower() == ".json":
            self.stages = self._load_stages_json(str(p))
        else:
            self.stages = self._load_stages_csv(str(p))

    # ---------------------------------------------------------------- #
    #  Loaders                                                           #
    # ---------------------------------------------------------------- #

    @classmethod
    def from_csv(cls, path: str, default_threshold: float = 0.50) -> "SerialCascade":
        """
        Load directly from GP solver's real_data_vip_settings.csv.

        Columns used: stage_order, modality, threshold_H, threshold_L
        """
        config = {
            "fusion": {
                "cascade_config": path,
                "threshold": default_threshold,
            }
        }
        return cls(config)

    @classmethod
    def from_json(cls, path: str, default_threshold: float = 0.50) -> "SerialCascade":
        """Legacy JSON loader — useful for unit tests and manual config."""
        config = {
            "fusion": {
                "cascade_config": path,
                "threshold": default_threshold,
            }
        }
        return cls(config)

    def _load_stages_csv(self, path: str) -> List[CascadeStage]:
        """Parse GP solver CSV output into ordered CascadeStage list."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to load cascade config from CSV. "
                "Install with: pip install pandas"
            )
        df = pd.read_csv(path)
        required = {"stage_order", "modality", "threshold_H", "threshold_L"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"real_data_vip_settings.csv is missing columns: {missing}\n"
                f"Got: {list(df.columns)}"
            )
        df = df.sort_values("stage_order").reset_index(drop=True)
        stages = []
        for _, row in df.iterrows():
            raw_name = str(row["modality"]).strip()
            name = self.MODALITY_ALIASES.get(raw_name, raw_name)
            stages.append(CascadeStage(
                name=name,
                H=float(row["threshold_H"]),
                L=float(row["threshold_L"]),
            ))
        return stages

    def _load_stages_json(self, path: str) -> List[CascadeStage]:
        """Parse legacy JSON format."""
        with open(path) as f:
            spec = json.load(f)
        if "stages" not in spec:
            raise ValueError(f"JSON cascade config must have a 'stages' key: {path}")
        stages = []
        for s in spec["stages"]:
            raw_name = str(s.get("name", s.get("modality", ""))).strip()
            name = self.MODALITY_ALIASES.get(raw_name, raw_name)
            stages.append(CascadeStage(
                name=name,
                H=float(s.get("H", s.get("threshold_H", 0.75))),
                L=float(s.get("L", s.get("threshold_L", 0.25))),
            ))
        self.default_threshold = float(spec.get("default_threshold", self.default_threshold))
        self.fallback_score = float(spec.get("fallback_score", self.fallback_score))
        return stages

    # ---------------------------------------------------------------- #
    #  Inference                                                          #
    # ---------------------------------------------------------------- #

    def fuse(
        self,
        visual_score: Optional[float] = None,
        rppg_score: Optional[float] = None,
        sync_score: Optional[float] = None,
    ) -> FusionResult:
        """
        Run the serial cascade.

        For each stage in GP-optimised order:
          - If the modality score is None (unavailable), skip without deciding.
          - If score >= H: exit immediately as FAKE.
          - If score <= L: exit immediately as REAL.
          - Otherwise (uncertain): pass to next stage.
        After all stages: return fallback_score (default 0.5).

        Returns FusionResult (drop-in compatible with WeightedEnsemble.fuse()).
        """
        raw_scores: Dict[str, Optional[float]] = {
            "visual": visual_score,
            "rppg":   rppg_score,
            "sync":   sync_score,
        }

        score_map = dict(raw_scores)
        final_score: float = self.fallback_score
        exit_stage: Optional[str] = None

        for stage in self.stages:
            s = score_map.get(stage.name)
            if s is None:
                continue  # modality unavailable — skip stage, do not decide
            if s >= stage.H:
                final_score = s
                exit_stage = stage.name
                break
            if s <= stage.L:
                final_score = s
                exit_stage = stage.name
                break
            # Uncertain: continue to next stage

        modalities_used = sum(1 for v in raw_scores.values() if v is not None)

        # Fallback (no stage made a decision): treat as uncertain → is_fake=False,
        # consistent with WeightedEnsemble all-None behaviour.
        if exit_stage is None:
            is_fake = False
        else:
            is_fake = final_score >= self.default_threshold

        return FusionResult(
            fake_score=float(final_score),
            is_fake=is_fake,
            threshold=self.default_threshold,
            scores=raw_scores,
            weights_used={s.name: s.H for s in self.stages},
            modalities_used=modalities_used,
        )

    def __repr__(self) -> str:
        stage_str = " → ".join(
            f"{s.name}(H={s.H:.2f},L={s.L:.2f})" for s in self.stages
        )
        return f"SerialCascade([{stage_str}], thr={self.default_threshold})"
