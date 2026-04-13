from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ROOT = Path('.')
    DATA_DIR = ROOT / 'data'
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LR = 3e-4
    NUM_WORKERS = 2
    NUM_CLASSES = 5
    IGNORE_INDEX = 0
    RANDOM_STATE = 42
    MIN_CONFIDENCE = 20
    CONFIDENCE_FLOOR = 0.25
    DARK_PIXEL_MEAN_THRESHOLD = 0.03
    DARK_PIXEL_MAX_THRESHOLD = 0.08
    BRIGHT_PIXEL_MEAN_THRESHOLD = 0.85
    SATURATED_BAND_THRESHOLD = 0.98
    MAX_SATURATED_BANDS = 6
    MAX_NAN_PIXEL_FRACTION = 0.95
    OUTLIER_WINDOW_SIZE = 3
    OUTLIER_MEAN_DIFF_THRESHOLD = 0.10
    OUTLIER_MAX_DIFF_THRESHOLD = 0.25
    OUTLIER_MIN_BAND_COUNT = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    TRAIN_CAPS_CLASSIC = {
        1: 500_000,   # Greenery
        2: 550_000,   # Sand
        3: 500_000,      # Water -> keep all
        4: 600_000,      # Cement -> keep all
    }

    CLASS_NAMES = {
        0: 'Unknown',
        1: 'Greenery',
        2: 'Sand',
        3: 'Water',
        4: 'Cement',
    }

    REFINE_KWARGS = {
        "max_component_size": 128,
        "min_neighbor_pixels": 8,
        "iterations": 2,
        "connectivity": 2,
        "target_classes": [0],
    }

    LABEL_TO_XGB = {1: 0, 2: 1, 3: 2, 4: 3}
    XGB_TO_LABEL = {v: k for k, v in LABEL_TO_XGB.items()}

    def set_band_stats_once(self, means, stds):
        if self.BAND_MEANS is not None or self.BAND_STDS is not None:
            raise RuntimeError("Band stats were already assigned.")

        object.__setattr__(self, "BAND_MEANS", np.asarray(means, dtype=np.float32))
        object.__setattr__(self, "BAND_STDS", np.asarray(stds, dtype=np.float32))
    

config = Config()
