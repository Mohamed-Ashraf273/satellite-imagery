import argparse
import pickle
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch
from torch import nn

from config import config
from utils import build_bright_pixel_mask, compress_bright_pixels, decode_labels, extract_features


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class CementFeatureStem(nn.Module):
    def __init__(self, channels=12, hidden=64, index_channels=8, dropout=0.1):
        super().__init__()
        self.index_proj = nn.Sequential(
            nn.Conv2d(index_channels, hidden // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(inplace=True),
        )
        self.raw_proj = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        fused_channels = hidden + hidden // 2
        self.local = nn.Sequential(
            nn.Conv2d(fused_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.dilated2 = nn.Sequential(
            nn.Conv2d(fused_channels, hidden, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.dilated4 = nn.Sequential(
            nn.Conv2d(fused_channels, hidden, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.ReLU(inplace=True)

    def make_indices(self, x):
        blue = x[:, 1:2]
        green = x[:, 2:3]
        red = x[:, 3:4]
        nir = x[:, 7:8]
        swir1 = x[:, 10:11]
        eps = 1e-6
        ndbi = (swir1 - nir) / (swir1 + nir + eps)
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + eps)
        mndwi = (green - swir1) / (green + swir1 + eps)
        ndvi = (nir - red) / (nir + red + eps)
        visible = torch.cat([blue, green, red], dim=1)
        brightness = visible.mean(dim=1, keepdim=True)
        visible_std = visible.std(dim=1, keepdim=True)
        swir1_red_ratio = swir1 / (red + eps)
        swir1_nir_ratio = swir1 / (nir + eps)
        return torch.cat([ndbi, bsi, mndwi, ndvi, brightness, visible_std, swir1_red_ratio, swir1_nir_ratio], dim=1)

    def forward(self, x):
        indices = self.make_indices(x)
        raw = self.raw_proj(x)
        idx = self.index_proj(indices)
        fused = torch.cat([raw, idx], dim=1)
        multi_scale = torch.cat([self.local(fused), self.dilated2(fused), self.dilated4(fused)], dim=1)
        out = self.fuse(multi_scale)
        out = self.se(out)
        return self.act(x + out)


class UNet(nn.Module):
    def __init__(self, encoder_name="resnet50", num_classes=5):
        super().__init__()
        self.encoder_name = encoder_name
        self.stem = CementFeatureStem(channels=12, hidden=64, dropout=0.1)
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=12,
            classes=num_classes,
        )

    def forward(self, x):
        x = self.stem(x)
        return self.unet(x)


def image_paths(imgs_dir):
    imgs_dir = Path(imgs_dir)
    paths = []
    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        paths.extend(imgs_dir.glob(pattern))
    return sorted(paths)


def preprocess_image(img_path):
    with rasterio.open(img_path) as src:
        img = src.read().astype(np.float32)
    img = np.nan_to_num(img, nan=0.0)
    img = np.clip(img, 0, 10000) / 10000.0
    bright_pixel_mask = build_bright_pixel_mask(img)
    img = compress_bright_pixels(img, bright_mask=bright_pixel_mask)
    return img.astype(np.float32)


def read_profile(img_path):
    with rasterio.open(img_path) as src:
        return src.profile.copy()


def load_model(model_type, model_path):
    if model_type == "ml":
        with open(model_path, "rb") as f:
            return pickle.load(f)
    model = UNet(encoder_name="resnet50", num_classes=5).to(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()
    return model


def predict_ml(model, img):
    features, _ = extract_features(img)
    x = features.reshape(features.shape[0], -1).T.astype(np.float32)
    return decode_labels(model.predict(x)).astype(np.uint8)


@torch.no_grad()
def predict_dl(model, img):
    h, w = img.shape[1], img.shape[2]
    x = torch.from_numpy(img).unsqueeze(0).to(config.DEVICE)
    logits = model(x)
    pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    if pred.shape != (h, w):
        raise RuntimeError(f"prediction shape {pred.shape} does not match image shape {(h, w)}")
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs-dir", default="/home/mohamed-ashraf/Desktop/projects/sat-project/data/samples/imgs")
    parser.add_argument("--model-type", choices=["ml", "dl"], default="dl")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output-dir", default="predictions")
    args = parser.parse_args()

    model_path = args.model_path or ("model.pkl" if args.model_type == "ml" else "best_unet.pth")
    model = load_model(args.model_type, model_path)
    paths = image_paths(args.imgs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in paths:
        img = preprocess_image(img_path)
        pred = predict_ml(model, img) if args.model_type == "ml" else predict_dl(model, img)
        pred = pred.reshape(img.shape[1], img.shape[2]).astype(np.uint8)
        profile = read_profile(img_path)
        profile.update(count=1, dtype=rasterio.uint8, nodata=0)
        output_path = output_dir / f"{img_path.stem}_prediction.tif"
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(pred, 1)


if __name__ == "__main__":
    main()
