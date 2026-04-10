from __future__ import annotations

import numpy as np
import torch

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    ROOT = Path('.')
    DATA_DIR = ROOT / 'data'
    RANDOM_STATE = 42
    MIN_CONFIDENCE = 15
    CONFIDENCE_FLOOR = 0.20
    WATER_WEIGHT_MULTIPLIER = 3.0
    CEMENT_WEIGHT_MULTIPLIER = 2.0

    BAND_MEANS = None
    BAND_STDS = None

    TRAIN_CAPS = {
        1: 200_000,   # Greenery
        2: 200_000,   # Sand
        3: None,      # Water -> keep all
        4: None,      # Cement -> keep all
    }

    VAL_TEST_CAPS = {
        1: 100_000,   # Greenery
        2: 100_000,   # Sand
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

    LABEL_TO_XGB = {1: 0, 2: 1, 3: 2, 4: 3}
    XGB_TO_LABEL = {v: k for k, v in LABEL_TO_XGB.items()}



    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 8
    NUM_EPOCHS = 25
    LR = 3e-4
    NUM_WORKERS = 2
    CEMENT_IMAGE_BOOST = 4.0
    WATER_IMAGE_BOOST = 3.0
    CEMENT_PIXEL_BOOST = 2.0
    WATER_PIXEL_BOOST = 3.0
    CEMENT_CLASS_BOOST = 2.0
    WATER_CLASS_BOOST = 1.5
    DICE_WEIGHT = 0.3

    NUM_CLASSES = 5
    IGNORE_INDEX = 0
    IN_CHANNELS = 17
    

config = Config()
