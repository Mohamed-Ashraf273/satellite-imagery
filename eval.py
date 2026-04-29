import argparse
import re
from pathlib import Path

import numpy as np
import rasterio


def sample_id(path):
    stem = Path(path).stem
    stem = re.sub(r"_prediction$", "", stem)
    stem = re.sub(r"_(Spectral|Mask)(?:_\d+px)?$", "", stem, flags=re.IGNORECASE)
    return stem


def tif_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    paths = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        paths.extend(path.glob(pattern))
    return sorted(paths)


def read_mask(path):
    with rasterio.open(path) as src:
        return src.read(1)


def compute_miou(y_true, y_pred):
    scores = []
    for cls in (1, 2, 3, 4):
        true_cls = y_true == cls
        pred_cls = y_pred == cls
        union = np.logical_or(true_cls, pred_cls).sum()
        if union > 0:
            scores.append(np.logical_and(true_cls, pred_cls).sum() / union)
    return float(np.mean(scores)) if scores else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--masks-dir", default="/home/mohamed-ashraf/Desktop/projects/sat-project/data/test/samples_prepared/masks")
    args = parser.parse_args()

    pred_paths = {sample_id(path): path for path in tif_paths(args.predictions)}
    mask_paths = {sample_id(path): path for path in tif_paths(args.masks_dir)}

    y_true_parts = []
    y_pred_parts = []
    for key in sorted(pred_paths.keys() & mask_paths.keys()):
        pred = read_mask(pred_paths[key])
        mask = read_mask(mask_paths[key])
        if pred.shape != mask.shape:
            raise RuntimeError(f"{key}: prediction shape {pred.shape} does not match mask shape {mask.shape}")
        valid = mask != 0
        if np.any(valid):
            y_true_parts.append(mask[valid].reshape(-1))
            y_pred_parts.append(pred[valid].reshape(-1))

    if not y_true_parts:
        print("nan")
        return

    y_true = np.concatenate(y_true_parts)
    y_pred = np.concatenate(y_pred_parts)
    print(compute_miou(y_true, y_pred))


if __name__ == "__main__":
    main()
