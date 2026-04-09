from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
CLASS_NAMES = {
    0: "Unknown",
    1: "Greenery",
    2: "Sand",
    3: "Water",
    4: "Cement",
}


@dataclass(frozen=True)
class Pair:
    sample_id: str
    image_path: Path
    mask_path: Path


def extract_sample_id(path: Path) -> str:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not extract numeric id from {path}")
    return match.group(1)


def load_pairs(root: Path) -> list[Pair]:
    image_paths = sorted((root / "data" / "imgs").glob("*.tif"))
    mask_paths = sorted((root / "data" / "masks").glob("*.tif"))

    if len(image_paths) != len(mask_paths):
        raise ValueError(f"Mismatched file counts. images={len(image_paths)} masks={len(mask_paths)}")

    pairs: list[Pair] = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image_id = extract_sample_id(image_path)
        mask_id = extract_sample_id(mask_path)
        if image_id != mask_id:
            raise ValueError(f"Mismatched pair: {image_path.name} vs {mask_path.name}")
        pairs.append(Pair(sample_id=image_id, image_path=image_path, mask_path=mask_path))

    return pairs


def read_mask(mask_path: Path) -> np.ndarray:
    return np.array(Image.open(mask_path))


def split_pairs(pairs: list[Pair]) -> dict[str, list[Pair]]:
    sample_ids = [pair.sample_id for pair in pairs]
    train_ids, val_ids = train_test_split(sample_ids, test_size=0.3, random_state=RANDOM_STATE)
    val_ids, test_ids = train_test_split(val_ids, test_size=0.5, random_state=RANDOM_STATE)

    by_id = {pair.sample_id: pair for pair in pairs}
    return {
        "train": [by_id[sample_id] for sample_id in train_ids],
        "val": [by_id[sample_id] for sample_id in val_ids],
        "test": [by_id[sample_id] for sample_id in test_ids],
    }


def summarize_split(name: str, pairs: list[Pair]) -> None:
    pixel_counts = Counter()
    images_with_class = Counter()
    top_examples = defaultdict(list)

    for pair in pairs:
        mask = read_mask(pair.mask_path)
        classes, counts = np.unique(mask, return_counts=True)
        class_counts = {int(cls): int(count) for cls, count in zip(classes, counts)}

        for cls, count in class_counts.items():
            pixel_counts[cls] += count
            images_with_class[cls] += 1
            top_examples[cls].append((count, pair.mask_path.name))

    total_pixels = sum(pixel_counts.values())
    print(f"\n[{name}]")
    print(f"images: {len(pairs)}")
    print(f"pixels: {total_pixels:,}")
    print("pixel counts:")
    for cls in sorted(CLASS_NAMES):
        count = pixel_counts.get(cls, 0)
        pct = (100.0 * count / total_pixels) if total_pixels else 0.0
        print(f"  {cls} {CLASS_NAMES[cls]:<8} {count:>9,}  ({pct:6.2f}%)")

    print("images containing class:")
    for cls in sorted(CLASS_NAMES):
        count = images_with_class.get(cls, 0)
        pct = (100.0 * count / len(pairs)) if pairs else 0.0
        print(f"  {cls} {CLASS_NAMES[cls]:<8} {count:>9,}  ({pct:6.2f}%)")

    for cls in (3, 4):
        ranked = sorted(top_examples[cls], reverse=True)[:5]
        preview = ", ".join(f"{name}={count:,}" for count, name in ranked) or "none"
        print(f"top {CLASS_NAMES[cls].lower()} masks: {preview}")


def main() -> None:
    root = Path(__file__).resolve().parent
    pairs = load_pairs(root)
    splits = split_pairs(pairs)

    print(f"paired samples: {len(pairs)}")
    for split_name in ("train", "val", "test"):
        summarize_split(split_name, splits[split_name])

    print("\nKey checks:")
    print("- The current split is image-random, not stratified by class presence.")
    print("- Validation has very few water- and cement-containing masks, so metrics for rare classes are unstable.")
    print("- The notebook's balanced validation subset does not match the natural test distribution used by the project.")


if __name__ == "__main__":
    main()
