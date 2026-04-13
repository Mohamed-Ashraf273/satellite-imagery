# Sat Project

Land-cover classification and segmentation on Sentinel-2 samples using two parallel approaches:

- `train_p1.ipynb`: classical pixel-wise classification with XGBoost
- `train_p2.ipynb`: semantic segmentation with a U-Net-style model in PyTorch

The dataset is built from exported Sentinel-2 spectral tiles and remapped Dynamic World masks.

## Classes

The project uses 5 labels:

- `0`: Unknown / ignored
- `1`: Greenery
- `2`: Sand
- `3`: Water
- `4`: Cement

## Repo Layout

```text
.
├── config.py
├── utils.py
├── generate_samples_gee.ipynb
├── data_diagnosis.ipynb
├── train_p1.ipynb
├── train_p2.ipynb
├── test_samples.ipynb
├── data.ipynb
├── model.pkl
├── best_unet.pth
└── data/
    └── samples/
        ├── imgs/
        └── masks/
```

Main files:

- `config.py`: shared paths, thresholds, label maps, and training settings
- `utils.py`: metadata building, preprocessing, mask cleanup, evaluation helpers
- `generate_samples_gee.ipynb`: Earth Engine export notebook
- `data_diagnosis.ipynb`: random sample inspection and preprocessing diagnostics
- `train_p1.ipynb`: feature extraction + XGBoost training
- `train_p2.ipynb`: dataset building + segmentation training
- `test_samples.ipynb`: quick qualitative prediction checks

## Setup

Use Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Jupyter:

```bash
python -m ipykernel install --user --name sat-project
```

## Data Layout

Expected local dataset structure:

```text
data/
└── samples/
    ├── imgs/
    │   ├── Alexandria_g00_Spectral_300px.tif
    │   └── ...
    └── masks/
        ├── Alexandria_g00_Mask_300px.tif
        └── ...
```

Image and mask pairing is based on the shared sample prefix, for example:

- `Alexandria_g00_Spectral_300px.tif`
- `Alexandria_g00_Mask_300px.tif`

## Workflow

### 1. Generate or collect samples

Use `generate_samples_gee.ipynb` to create Earth Engine export tasks for Sentinel-2 imagery and Dynamic World masks.

Then place the exported GeoTIFFs into:

- `data/samples/imgs`
- `data/samples/masks`

### 2. Inspect data quality

Run `data_diagnosis.ipynb`.

It now shows 5 random samples on each run and reports:

- NaN pixels
- dark pixels
- bright / saturated pixels
- local outlier pixels
- pixels excluded by preprocessing
- mask distributions before and after cleanup

### 3. Train the classical model

Run `train_p1.ipynb`.

Pipeline summary:

- build metadata from masks
- stratified train/val/test split
- preprocess images
- extract hand-crafted spectral and texture features
- sample pixels per class
- train an `XGBClassifier`
- save `model.pkl`

### 4. Train the segmentation model

Run `train_p2.ipynb`.

Pipeline summary:

- build metadata and split
- preprocess image/mask pairs
- crop or pad to a fixed training size
- train a U-Net-based segmentation model
- save `best_unet.pth`

### 5. Visualize predictions

Use `test_samples.ipynb` to load saved checkpoints and inspect predictions visually.

## Preprocessing

Shared preprocessing lives in `utils.py`.

Current behavior:

- scales Sentinel-2 values to `[0, 1]` after clipping to `[0, 10000]`
- drops tiles only when NaN coverage is `>= 95%`
- converts remaining NaNs to zero before feature/model input
- marks invalid pixels using:
  - NaN mask
  - dark-pixel detection
  - bright / saturation detection
  - local outlier detection
- cleans small connected components in the mask for target class `0`

Important detail:

- masks are expected to already be remapped during data generation
- training ignores label `0`

## Outputs

Typical saved artifacts:

- `model.pkl`: trained XGBoost model
- `best_unet.pth`: trained segmentation checkpoint
- `data_config.pkl`: saved normalization / configuration artifact if produced by notebooks

## Notes

- `generate_samples_gee.ipynb` requires an authenticated Google Earth Engine account.
- `data.ipynb` includes some extra data-handling utilities and optional browser/download automation.
- The notebooks are the main entry points; this repo is notebook-driven rather than packaged as a Python module.

## License / Data

No explicit project license is included in this repository right now.  
Check the original dataset and Google Earth Engine source terms before redistributing data or derived exports.
