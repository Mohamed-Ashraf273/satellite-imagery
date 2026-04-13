import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from config import config
from scipy.ndimage import binary_dilation, label, median_filter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score

def show_sample(img, mask, pred=None):
    raw_rgb = img[[3, 2, 1]]
    raw_rgb = raw_rgb * config.BAND_STDS[[3, 2, 1], None, None] + config.BAND_MEANS[[3, 2, 1], None, None]
    rgb = np.transpose(raw_rgb, (1, 2, 0))
    rgb = np.clip(rgb * 3.0, 0, 1)

    ncols = 3 if pred is not None else 2
    plt.figure(figsize=(5 * ncols, 5))

    plt.subplot(1, ncols, 1)
    plt.imshow(rgb)
    plt.title("RGB")
    plt.axis("off")

    plt.subplot(1, ncols, 2)
    plt.imshow(mask, vmin=0, vmax=4)
    plt.title("Mask")
    plt.axis("off")

    if pred is not None:
        plt.subplot(1, ncols, 3)
        plt.imshow(pred, vmin=0, vmax=4)
        plt.title("Prediction")
        plt.axis("off")

    plt.show()


def extract_sample_id(path):
    stem = Path(path).stem
    stem = re.sub(r'_(Spectral|Mask)(?:_\d+px)?$', '', stem, flags=re.IGNORECASE)
    return stem


def build_pairs_dataframe(data_dir):
    img_paths = {extract_sample_id(p): p for p in (data_dir / 'imgs').glob('*.tif')}
    mask_paths = {extract_sample_id(p): p for p in (data_dir / 'masks').glob('*.tif')}
    common_ids = sorted(set(img_paths.keys()) & set(mask_paths.keys()), key=str)
    rows = []
    for sample_id in common_ids:
        rows.append(
            {
                'sample_id': sample_id,
                'img_path': str(img_paths[sample_id]),
                'mask_path': str(mask_paths[sample_id]),
            }
        )

    return pd.DataFrame(rows, columns=['sample_id', 'img_path', 'mask_path'])


def mask_summary(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    values, counts = np.unique(mask, return_counts=True)
    summary = {int(v): int(c) for v, c in zip(values, counts)}
    total = int(mask.size)

    return {
        'total_pixels': total,
        'has_water': int(3 in summary),
        'has_cement': int(4 in summary),
        'has_sand': int(2 in summary),
        **{f'count_{cls}': summary.get(cls, 0) for cls in range(5)},
    }


def build_metadata(data_dir):
    pairs = build_pairs_dataframe(data_dir)
    summaries = [mask_summary(path) for path in tqdm(pairs['mask_path'], desc='Scanning masks')]
    meta = pd.concat([pairs, pd.DataFrame(summaries)], axis=1)
    return meta


def _connected_structure(connectivity=2):
    if connectivity == 1:
        return np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.uint8,
        )
    return np.ones((3, 3), dtype=np.uint8)


def build_pixel_quality_mask(img):
    bands = img[:12].astype(np.float32)
    band_mean = np.mean(bands, axis=0)
    band_max = np.max(bands, axis=0)
    saturated_band_count = np.sum(bands >= config.SATURATED_BAND_THRESHOLD, axis=0)

    dark_pixels = (
        (band_mean <= config.DARK_PIXEL_MEAN_THRESHOLD)
        & (band_max <= config.DARK_PIXEL_MAX_THRESHOLD)
    )
    bright_pixels = (
        (band_mean >= config.BRIGHT_PIXEL_MEAN_THRESHOLD)
        | (saturated_band_count >= config.MAX_SATURATED_BANDS)
    )
    outlier_pixels = build_pixel_outlier_mask(bands)
    return ~(dark_pixels | bright_pixels | outlier_pixels)


def build_nan_pixel_mask(img):
    return np.isnan(np.asarray(img, dtype=np.float32)).any(axis=0)


def build_pixel_outlier_mask(
    bands,
    window_size=config.OUTLIER_WINDOW_SIZE,
    mean_diff_threshold=config.OUTLIER_MEAN_DIFF_THRESHOLD,
    max_diff_threshold=config.OUTLIER_MAX_DIFF_THRESHOLD,
    min_band_count=config.OUTLIER_MIN_BAND_COUNT,
):
    bands = np.asarray(bands, dtype=np.float32)
    local_median = median_filter(
        bands,
        size=(1, window_size, window_size),
        mode='nearest',
    )
    abs_diff = np.abs(bands - local_median)
    mean_abs_diff = np.mean(abs_diff, axis=0)
    max_abs_diff = np.max(abs_diff, axis=0)
    inconsistent_band_count = np.sum(abs_diff >= mean_diff_threshold, axis=0)

    return (
        (mean_abs_diff >= mean_diff_threshold)
        & (max_abs_diff >= max_diff_threshold)
        & (inconsistent_band_count >= min_band_count)
    )


def denoise_img_for_features(img):
    img = np.asarray(img, dtype=np.float32)
    outlier_mask = build_pixel_outlier_mask(img)
    if not np.any(outlier_mask):
        return img

    local_median = median_filter(
        img,
        size=(1, config.OUTLIER_WINDOW_SIZE, config.OUTLIER_WINDOW_SIZE),
        mode='nearest',
    )
    clean = img.copy()
    clean[:, outlier_mask] = local_median[:, outlier_mask]
    return clean


def refine_mask_small_components(
    mask,
    max_component_size=128,
    min_neighbor_pixels=8,
    iterations=2,
    connectivity=2,
    target_classes=None,
    return_details=False,
):
    refined = np.asarray(mask)
    if refined.dtype != np.uint8:
        refined = refined.astype(np.uint8)
    
    refined = refined.copy()
    
    changed_mask = np.zeros(refined.shape, dtype=bool)
    structure = _connected_structure(connectivity=connectivity)
    if target_classes is None:
        classes = [int(v) for v in np.unique(refined)]
    else:
        classes = [int(v) for v in target_classes]

    for _ in range(iterations):
        changed = False

        for cls in classes:
            class_mask = refined == cls
            if not np.any(class_mask):
                continue

            labeled, num_components = label(class_mask, structure=structure)

            for component_id in range(1, num_components + 1):
                component = labeled == component_id
                component_size = int(component.sum())
                should_relabel = component_size <= max_component_size
                if not should_relabel:
                    continue

                border = binary_dilation(component, structure=structure) & ~component
                neighbor_labels = refined[border]

                if neighbor_labels.size == 0:
                    continue

                valid = neighbor_labels != cls
                if not np.any(valid):
                    continue

                neighbor_labels = neighbor_labels[valid]
                scores = {}
                for neighbor_cls in np.unique(neighbor_labels):
                    cls_pixels = neighbor_labels == neighbor_cls
                    scores[int(neighbor_cls)] = float(cls_pixels.sum())

                new_cls, best_score = max(scores.items(), key=lambda item: item[1])
                if best_score < min_neighbor_pixels:
                    continue

                refined[component] = new_cls
                changed_mask[component] = True
                changed = True

        if not changed:
            break

    if return_details:
        return refined, changed_mask.astype(np.uint8)
    return refined


def refine_mask_from_path(
    mask_path,
    max_component_size=128,
    min_neighbor_pixels=8,
    iterations=2,
    connectivity=2,
    target_classes=None,
    return_details=False,
):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    return refine_mask_small_components(
        mask=mask,
        max_component_size=max_component_size,
        min_neighbor_pixels=min_neighbor_pixels,
        iterations=iterations,
        connectivity=connectivity,
        target_classes=target_classes,
        return_details=return_details,
    )


def preprocess_img(img_path, mask_path, train=False):
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)

    img_h, img_w = img.shape[1], img.shape[2]
    mask_h, mask_w = mask.shape[0], mask.shape[1]
    if (img_h != mask_h) or (img_w != mask_w):
        h = min(img_h, mask_h)
        w = min(img_w, mask_w)
        img = img[:, :h, :w]
        mask = mask[:h, :w]

    nan_pixel_mask = build_nan_pixel_mask(img)
    refined_mask, _ = refine_mask_small_components(
        mask,
        return_details=True,
        **config.REFINE_KWARGS,
    )

    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, 10000) / 10000.0
    pixel_valid = build_pixel_quality_mask(img) & (~nan_pixel_mask)

    if train:
        refined_mask[~pixel_valid] = 0

    return img, refined_mask, pixel_valid


def build_strata(df):
    coarse_strata = np.where(
        (df['has_water'] == 1) & (df['has_cement'] == 1),
        'water_and_cement',
        np.where(
            df['has_water'] == 1,
            'water_only',
            np.where(df['has_cement'] == 1, 'cement_only', 'base')
        ),
    )

    coarse_strata = pd.Series(coarse_strata, index=df.index)

    bins = [-1, 0, 100, 1_000, 10_000, np.inf]
    labels = ['0', '1_100', '101_1k', '1k_10k', 'gt_10k']

    water_bin = pd.cut(df['count_3'], bins=bins, labels=labels).astype(str)
    cement_bin = pd.cut(df['count_4'], bins=bins, labels=labels).astype(str)
    fine_strata = coarse_strata + '__w_' + water_bin + '__c_' + cement_bin

    counts = fine_strata.value_counts()
    strata = fine_strata.where(fine_strata.map(counts) >= 2, coarse_strata)

    counts = strata.value_counts()
    strata = strata.where(strata.map(counts) >= 2, 'base')
    return strata


def safe_stratify_labels(labels, test_size, split_name):
    counts = labels.value_counts()
    num_classes = int(counts.shape[0])
    min_count = int(counts.min()) if num_classes > 0 else 0
    n_samples = int(labels.shape[0])
    n_test = int(np.ceil(n_samples * test_size)) if isinstance(test_size, float) else int(test_size)
    n_train = n_samples - n_test

    if min_count < 2:
        too_small = counts[counts < 2].index.tolist()
        print(
            f"[{split_name}] Falling back to non-stratified split: "
            f"strata with fewer than 2 samples: {too_small}"
        )
        return None

    if n_train < num_classes or n_test < num_classes:
        print(
            f"[{split_name}] Falling back to non-stratified split: "
            f"n_train={n_train}, n_test={n_test}, num_strata={num_classes}"
        )
        return None

    return labels


def split_metadata(meta, random_state=config.RANDOM_STATE):
    meta = meta.copy()
    valid_rows = []

    for row in meta.itertuples(index=False):
        if is_valid_image(row.img_path):
            valid_rows.append(row)

    meta = pd.DataFrame(valid_rows)
    meta['stratum'] = build_strata(meta)
    stratify_labels = safe_stratify_labels(meta['stratum'], test_size=0.20, split_name='train/temp')

    train_df, temp_df = train_test_split(
        meta,
        test_size=0.20,
        random_state=random_state,
        stratify=stratify_labels,
    )

    temp_df = temp_df.copy()
    temp_df['stratum'] = build_strata(temp_df)
    stratify_labels = safe_stratify_labels(temp_df['stratum'], test_size=0.50, split_name='val/test')
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def print_split_summary(name, df):
    print(f'[{name}] images: {len(df)}')
    for cls in range(5):
        pixels = int(df[f'count_{cls}'].sum())
        images = int((df[f'count_{cls}'] > 0).sum())
        print(f"  {config.CLASS_NAMES[cls]:<8} pixels={pixels:>9,} images={images:>3}")
    print('  strata:', df['stratum'].value_counts().to_dict())

def plot_class_distribution(y, title):
    classes, counts = np.unique(y, return_counts=True)
    labels = [f"{int(cls)} - {config.CLASS_NAMES.get(int(cls), f'Class {int(cls)}')}" for cls in classes]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(
        labels,
        counts,
        color=["#4CAF50", "#D2B48C", "#1E88E5", "#9E9E9E"][: len(labels)],
    )
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Pixel count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{int(count):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def count_by_class(y):
    values, counts = np.unique(y, return_counts=True)
    return {config.CLASS_NAMES[int(v)]: int(c) for v, c in zip(values, counts)}


def sample_pixels(X, y, caps, pixel_weight=None, random_state=config.RANDOM_STATE):
    rng = np.random.RandomState(random_state)
    chosen = []

    for cls in [1, 2, 3, 4]:
        cls_idx = np.where(y == cls)[0]
        cap = caps.get(cls, None)

        if cap is None:
            take = len(cls_idx)
            selected = cls_idx
        else:
            take = min(cap, len(cls_idx))
            selected = rng.choice(cls_idx, size=take, replace=False)

        chosen.append(selected)
        print(f'{config.CLASS_NAMES[cls]:<8} available={len(cls_idx):>9,} sampled={take:>9,}')

    chosen = np.concatenate(chosen)
    rng.shuffle(chosen)
    
    if pixel_weight is not None:
        pixel_weight_chosen = pixel_weight[chosen]
    else:
        pixel_weight_chosen = None
        
    return X[chosen], y[chosen], pixel_weight_chosen


def encode_labels(y):
    return np.array([config.LABEL_TO_XGB[int(v)] for v in y], dtype=np.uint8)


def decode_labels(y):
    return np.array([config.XGB_TO_LABEL[int(v)] for v in y], dtype=np.uint8)


def is_valid_image(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    nan_fraction = float(build_nan_pixel_mask(img).mean())
    return nan_fraction < config.MAX_NAN_PIXEL_FRACTION


def evaluate_split(name, model, X, y):
    y_pred = decode_labels(model.predict(X))
    print(f'===== {name} =====')
    print(classification_report(y, y_pred, digits=4, labels=[1, 2, 3, 4]))
    cm = confusion_matrix(y, y_pred, labels=[1, 2, 3, 4])
    print('Confusion Matrix:\n', cm)

    macro_iou = jaccard_score(y, y_pred, average='macro', labels=[1, 2, 3, 4])
    per_class_iou = jaccard_score(y, y_pred, average=None, labels=[1, 2, 3, 4])
    print('mIoU:', macro_iou)
    print('Per-class IoU:', {config.CLASS_NAMES[cls]: float(score) for cls, score in zip([1, 2, 3, 4], per_class_iou)})

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4), [config.CLASS_NAMES[c] for c in [1, 2, 3, 4]], rotation=45, ha='right')
    ax.set_yticks(range(4), [config.CLASS_NAMES[c] for c in [1, 2, 3, 4]])
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{name} Confusion Matrix')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def get_feature_corr_ranking(X, y, feature_names=None):
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    classes = np.unique(y)
    X_centered = X - X.mean(axis=0)
    corrs_per_class = []

    for c in classes:
        y_binary = (y == c).astype(np.float32)
        y_centered = y_binary - y_binary.mean()

        numerator = (X_centered * y_centered[:, None]).sum(axis=0)
        denominator = np.sqrt(
            (X_centered**2).sum(axis=0) * (y_centered**2).sum()
        )

        corr = numerator / (denominator + 1e-8)
        corrs_per_class.append(corr)

    corrs_per_class = np.array(corrs_per_class)
    max_corr = np.max(np.abs(corrs_per_class), axis=0)
    sorted_idx = np.argsort(-max_corr)

    if feature_names is not None:
        sorted_features = [(feature_names[i], max_corr[i]) for i in sorted_idx]
        return sorted_features
    else:
        return sorted_idx, max_corr[sorted_idx]
