#!/usr/bin/env python3
import argparse
import base64
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
DEFAULT_SPECTRAL_DIR = ROOT / "spectral"
DEFAULT_MASK_DIR = ROOT / "team_6"

CLASS_NAMES = {
    0: "Unknown",
    1: "Greenery",
    2: "Sand",
    3: "Water",
    4: "Cement",
}

MASK_COLOR_NAMES = {
    0: "black",
    1: "red",
    2: "teal",
    3: "blue",
    4: "orange",
    5: "purple",
    6: "yellow",
    7: "mint green",
}

MASK_PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [42, 157, 143],
        [69, 123, 157],
        [244, 162, 97],
        [131, 56, 236],
        [255, 209, 102],
        [6, 214, 160],
    ],
    dtype=np.uint8,
)


TIFF_TYPE_SIZES = {
    1: 1,   # BYTE
    2: 1,   # ASCII
    3: 2,   # SHORT
    4: 4,   # LONG
    5: 8,   # RATIONAL
    11: 4,  # FLOAT
    12: 8,  # DOUBLE
}


def numeric_key(path):
    stem = path.stem.replace("_Mask", "")
    return int(stem) if stem.isdigit() else stem


def read_tiff_with_rasterio(path):
    try:
        import rasterio
    except ImportError:
        return None

    with rasterio.open(path) as src:
        data = src.read()
    return np.moveaxis(data, 0, -1)


def _read_tiff_values(f, endian, type_id, count, value_or_offset):
    size = TIFF_TYPE_SIZES[type_id] * count
    raw_offset = struct.pack(endian + "I", value_or_offset)

    if size <= 4:
        raw = raw_offset[:size]
    else:
        current = f.tell()
        f.seek(value_or_offset)
        raw = f.read(size)
        f.seek(current)

    if type_id == 1:
        return list(raw)
    if type_id == 2:
        return raw.rstrip(b"\x00").decode("ascii", errors="replace")
    if type_id == 3:
        return list(struct.unpack(endian + f"{count}H", raw))
    if type_id == 4:
        return list(struct.unpack(endian + f"{count}I", raw))
    if type_id == 11:
        return list(struct.unpack(endian + f"{count}f", raw))
    if type_id == 12:
        return list(struct.unpack(endian + f"{count}d", raw))
    raise ValueError(f"Unsupported TIFF tag type: {type_id}")


def read_simple_tiff(path):
    """Read the uncompressed TIFF layout used by the project spectral files."""
    with open(path, "rb") as f:
        byte_order = f.read(2)
        if byte_order == b"II":
            endian = "<"
        elif byte_order == b"MM":
            endian = ">"
        else:
            raise ValueError(f"{path} is not a TIFF file")

        magic, ifd_offset = struct.unpack(endian + "HI", f.read(6))
        if magic != 42:
            raise ValueError(f"{path} has an unsupported TIFF magic number")

        f.seek(ifd_offset)
        entry_count = struct.unpack(endian + "H", f.read(2))[0]
        tags = {}

        for _ in range(entry_count):
            tag, type_id, count, value_or_offset = struct.unpack(endian + "HHII", f.read(12))
            if type_id in TIFF_TYPE_SIZES:
                tags[tag] = _read_tiff_values(f, endian, type_id, count, value_or_offset)

        width = tags[256][0]
        height = tags[257][0]
        bits_per_sample = tags[258]
        compression = tags[259][0]
        strip_offsets = tags[273]
        samples_per_pixel = tags.get(277, [1])[0]
        rows_per_strip = tags.get(278, [height])[0]
        strip_byte_counts = tags[279]
        planar_config = tags.get(284, [1])[0]
        sample_format = tags.get(339, [1] * samples_per_pixel)

        if compression != 1:
            raise ValueError(f"{path} is compressed; install rasterio to read it")
        if planar_config != 1:
            raise ValueError(f"{path} uses planar TIFF data; install rasterio to read it")
        if len(set(bits_per_sample)) != 1 or len(set(sample_format)) != 1:
            raise ValueError(f"{path} has mixed sample types; install rasterio to read it")

        bits = bits_per_sample[0]
        fmt = sample_format[0]
        if fmt == 1 and bits == 8:
            dtype = np.uint8
        elif fmt == 1 and bits == 16:
            dtype = endian + "u2"
        elif fmt == 2 and bits == 16:
            dtype = endian + "i2"
        elif fmt == 3 and bits == 32:
            dtype = endian + "f4"
        elif fmt == 3 and bits == 64:
            dtype = endian + "f8"
        else:
            raise ValueError(f"{path} uses unsupported sample format/bits: {fmt}/{bits}")

        rows = []
        values_per_row = width * samples_per_pixel
        for offset, byte_count in zip(strip_offsets, strip_byte_counts):
            f.seek(offset)
            strip = np.frombuffer(f.read(byte_count), dtype=dtype)
            strip_rows = len(strip) // values_per_row
            rows.append(strip[: strip_rows * values_per_row].reshape(strip_rows, width, samples_per_pixel))

        return np.concatenate(rows, axis=0)[:height]


def read_spectral(path):
    data = read_tiff_with_rasterio(path)
    if data is None:
        data = read_simple_tiff(path)
    if data.ndim == 2:
        data = data[:, :, None]
    return data


def normalize_rgb(data, bands):
    band_indexes = [band - 1 for band in bands]
    missing = [band for band, index in zip(bands, band_indexes) if index < 0 or index >= data.shape[-1]]
    if missing:
        raise ValueError(f"Requested band(s) {missing}, but {data.shape[-1]} band(s) are available")

    rgb = data[:, :, band_indexes].astype(np.float32)
    output = np.zeros_like(rgb, dtype=np.float32)

    for channel in range(3):
        values = rgb[:, :, channel]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.percentile(values[finite], (2, 98))
        if high <= low:
            low, high = values[finite].min(), values[finite].max()
        if high > low:
            output[:, :, channel] = (values - low) / (high - low)

    return Image.fromarray((np.clip(output, 0, 1) * 255).astype(np.uint8), mode="RGB")


def colorize_mask(mask_image):
    mask = np.array(mask_image)
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    return Image.fromarray(MASK_PALETTE[mask % len(MASK_PALETTE)], mode="RGB")


def print_mask_color_legend(classes=(1, 2, 3, 4)):
    print("Mask color legend:")
    for cls in classes:
        color_name = MASK_COLOR_NAMES.get(cls, "unknown color")
        class_name = CLASS_NAMES.get(cls, f"Class {cls}")
        print(f"  {color_name}: Class {cls} ({class_name})")


def make_pair_image(spectral_path, mask_path, bands, max_side):
    rgb = normalize_rgb(read_spectral(spectral_path), bands)
    mask = colorize_mask(Image.open(mask_path))

    if max_side:
        rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        mask.thumbnail((max_side, max_side), Image.Resampling.NEAREST)

    gap = 14
    label_height = 28
    width = rgb.width + mask.width + gap
    height = max(rgb.height, mask.height) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    canvas.paste(rgb, (0, label_height))
    canvas.paste(mask, (rgb.width + gap, label_height))
    title = f"{spectral_path.name}    RGB bands {bands[0]},{bands[1]},{bands[2]}"
    draw.text((4, 6), title, fill=(0, 0, 0), font=ImageFont.load_default())
    draw.text((rgb.width + gap + 4, 6), mask_path.name, fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def collect_pairs(spectral_dir, mask_dir):
    spectral_files = sorted(spectral_dir.glob("*.tif"), key=numeric_key)
    pairs = []

    for spectral_path in spectral_files:
        mask_path = mask_dir / f"{spectral_path.stem}_Mask.tif"
        if mask_path.exists():
            pairs.append((spectral_path, mask_path))
        else:
            print(f"Skipping {spectral_path.name}: missing {mask_path.name}")

    return pairs


class PairViewer:
    def __init__(self, pairs, bands, max_side):
        import tkinter as tk

        self.tk = tk
        self.root = tk.Tk()
        self.root.title("Spectral RGB / Mask Viewer")
        self.pairs = pairs
        self.bands = bands
        self.max_side = max_side
        self.index = 0
        self.photo = None

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10, pady=10)
        self.status = tk.Label(self.root, anchor="center")
        self.status.pack(fill="x", padx=10, pady=(0, 10))

        self.root.bind("<Right>", lambda event: self.next())
        self.root.bind("<space>", lambda event: self.next())
        self.root.bind("<Left>", lambda event: self.previous())
        self.root.bind("<Escape>", lambda event: self.root.destroy())
        self.show()

    def show(self):
        spectral_path, mask_path = self.pairs[self.index]
        image = make_pair_image(spectral_path, mask_path, self.bands, self.max_side)
        self.photo = self._photo_image(image)
        self.image_label.configure(image=self.photo)
        self.status.configure(text=f"{self.index + 1}/{len(self.pairs)}  |  Left/Right arrows, Space, Esc")

    def _photo_image(self, image):
        image = image.convert("RGB")
        header = f"P6 {image.width} {image.height} 255\n".encode("ascii")
        ppm = header + image.tobytes()
        data = base64.b64encode(ppm)
        return self.tk.PhotoImage(data=data, format="PPM")

    def next(self):
        self.index = (self.index + 1) % len(self.pairs)
        self.show()

    def previous(self):
        self.index = (self.index - 1) % len(self.pairs)
        self.show()

    def run(self):
        self.root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="Display spectral RGB TIFFs next to their masks.")
    parser.add_argument("--spectral-dir", type=Path, default=DEFAULT_SPECTRAL_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--bands", type=int, nargs=3, default=(4, 3, 2), metavar=("R", "G", "B"))
    parser.add_argument("--max-side", type=int, default=512, help="Maximum displayed side length for each panel.")
    parser.add_argument("--save-dir", type=Path, help="Save paired previews instead of opening the GUI.")
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = collect_pairs(args.spectral_dir, args.mask_dir)
    if not pairs:
        raise SystemExit("No matching spectral/mask TIFF pairs were found.")

    print_mask_color_legend()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for spectral_path, mask_path in pairs:
            image = make_pair_image(spectral_path, mask_path, tuple(args.bands), args.max_side)
            output_path = args.save_dir / f"{spectral_path.stem}_pair.png"
            image.save(output_path)
            print(output_path)
        return

    try:
        viewer = PairViewer(pairs, tuple(args.bands), args.max_side)
        viewer.run()
    except Exception as exc:
        raise SystemExit(f"Could not open GUI viewer: {exc}\nTry: python display.py --save-dir previews")


if __name__ == "__main__":
    main()
