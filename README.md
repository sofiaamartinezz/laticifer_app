# Laticifer Annotation App

Desktop annotation app built on napari for labeling laticifer structures and generating AI-assisted masks.

## What it does

- Open microscopy images in napari.
- Draw or edit label masks.
- Generate a mask with a U-Net model (SE-ResNeXt50 encoder).
- Enhance contrast using CLAHE for visual aid.
- Load existing masks.
- Save images, masks, and metadata in a consistent dataset layout.

## Project structure

```text
laticifer_app/
├─ main.py                 # Entry point: launches napari + widget
├─ ui.py                   # UI logic and user workflow
├─ model.py                # Model loading + inference pipeline
├─ dataset.py              # Dataset IO + CSV logging
├─ utils/
│  ├─ preprocessing.py     # CLAHE utilities
│  └─ inference.py         # Patch-based inference
│  └─ quantification.py    # Laticifer quantification utilities
```

## Module responsibilities

### main.py
- Creates the napari viewer and attaches the annotation widget.

### ui.py
- Handles button actions, layer management, and user workflow.
- Keeps the original image layer and any enhanced layers organized.

### model.py
- Loads the trained checkpoint from `models/`.
- Applies CLAHE preprocessing and runs sliding-window inference.
- Returns a binary mask for napari labels.

### dataset.py
- Locates the dataset root `dataset/`.
- Loads existing masks when available.
- Saves images, masks, and a row in `annotations.csv`.
- Validates image/mask shape consistency.

### utils/preprocessing.py
- `apply_clahe(image)` for contrast enhancement.

### utils/inference.py
- Patch-based prediction with padding and reconstruction.

### utils/quantification.py
- Quantification helpers for laticifer measurements.

## Installation

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate laticifer-env
```

### Option B: Pip

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Run the app

```bash
python main.py
```

Run from inside `laticifer_app/`.

## Dataset format

```text
dataset/
├─ images/
│  └─ <name>.tif
├─ masks/
│  └─ <name>_mask.tif
└─ annotations.csv
```
