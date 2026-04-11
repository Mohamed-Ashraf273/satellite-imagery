from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ROOT = Path('.')
    DATA_DIR = ROOT / 'data'
    RANDOM_STATE = 42
    MIN_CONFIDENCE = 20
    CONFIDENCE_FLOOR = 0.25
    DARK_PIXEL_MEAN_THRESHOLD = 0.03
    DARK_PIXEL_MAX_THRESHOLD = 0.08
    BRIGHT_PIXEL_MEAN_THRESHOLD = 0.85
    SATURATED_BAND_THRESHOLD = 0.98
    MAX_SATURATED_BANDS = 6
    WATER_WEIGHT_MULTIPLIER = 3.0
    CEMENT_WEIGHT_MULTIPLIER = 2.0

    BAND_MEANS: np.ndarray | None = field(default=None, init=False)
    BAND_STDS: np.ndarray | None = field(default=None, init=False)

    TRAIN_CAPS_CLASSIC = {
        1: 400_000,   # Greenery
        2: 400_000,   # Sand
        3: None,      # Water -> keep all
        4: None,      # Cement -> keep all
    }

    TRAIN_CAPS_DEEP = {
        1: 800_000,   # Greenery
        2: 800_000,   # Sand
        3: 800_000,   # Water -> keep all
        4: 800_000,   # Cement -> keep all
    }

    VAL_TEST_CAPS = {
        1: None,   # Greenery
        2: None,   # Sand
        3: None,      # Water -> keep all
        4: None,      # Cement -> keep all
    }

    CLASS_NAMES = {
        0: 'Unknown',
        1: 'Greenery',
        2: 'Sand',
        3: 'Water',
        4: 'Cement',
    }

    REFINE_KWARGS = {
        "low_conf_threshold": 40,
        "max_component_size": 128,
        "min_neighbor_pixels": 8,
        "iterations": 2,
        "connectivity": 2,
        "target_classes": [0],
        "confidence_cap": 80,
    }

    LABEL_TO_XGB = {1: 0, 2: 1, 3: 2, 4: 3}
    XGB_TO_LABEL = {v: k for k, v in LABEL_TO_XGB.items()}

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LR = 3e-4
    NUM_WORKERS = 2

    NUM_CLASSES = 5
    IGNORE_INDEX = 0

    def set_band_stats_once(self, means, stds):
        if self.BAND_MEANS is not None or self.BAND_STDS is not None:
            raise RuntimeError("Band stats were already assigned.")

        object.__setattr__(self, "BAND_MEANS", np.asarray(means, dtype=np.float32))
        object.__setattr__(self, "BAND_STDS", np.asarray(stds, dtype=np.float32))
    

config = Config()
