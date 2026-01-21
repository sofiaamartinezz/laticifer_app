# 🧬 Laticifer Annotation App

A desktop application built with **napari** that allows biologists to:

- Open microscopy images
- Manually annotate laticifer structures
- Automatically generate segmentation masks using an AI model (U-Net with SE-ResNeXt50 encoder)
- Enhance contrast using CLAHE for easier visual identification
- Load existing masks
- Save images, masks, and annotation metadata in a clean dataset structure

The app is designed for ease of use, even for users with minimal technical background, while maintaining a modular and professional codebase.

---

# 📁 Project Structure

```text
laticifer_annotation_app/
│
├─ main.py # Entry point: launches napari and the annotation widget
├─ ui.py # UI logic: buttons, layer handling, user workflow
├─ model.py # AI model loading and mask prediction pipeline
├─ dataset.py # Dataset handling: saving annotations, loading masks, CSV logging
│
├─ utils/
│ ├─ preprocessing.py # Generic preprocessing utilities (e.g., CLAHE)
│ └─ inference.py # Patch-based sliding-window inference (predict_image)
│
└─ models/
└─ best_model_soft_clDice.pth # Trained U-Net model (required for AI masks)
```

---

# 🧠 Responsibilities of Each Module

### `main.py`
- Minimal entry point.
- Creates the napari viewer and attaches the annotation widget.
- Keeps GUI boot logic separate from app functionality.

### `ui.py`
Contains the **LaticiferAnnotationWidget**, which manages all user interactions:

- Load images via napari’s native “Open File”
- Create empty masks
- Generate AI-based masks
- Load existing masks
- Enhance the current image with CLAHE (visual aid only)
- Save annotations into the dataset structure

Also handles:
- Activation/deactivation of buttons
- Tracking of the “original image layer” vs. enhanced layers
- Management of napari layers and editing behavior

### `model.py`
Responsible for all AI inference logic:

- Loads the U-Net model checkpoint from `models/`
- Applies CLAHE preprocessing
- Calls the sliding-window inference function from `utils/inference.py`
- Returns a clean **0/1** binary mask ready for napari Labels

### `dataset.py`
Implements robust dataset handling:

- Automatically detects the dataset root folder (`dataset/`)
- Validates folder structure and corrects user selections
- Loads existing masks when available
- Saves:
  - Raw image (original, not enhanced)
  - Mask (converted to binary 0/255)
  - Metadata row into `annotations.csv`

Ensures consistent and safe dataset storage for training or fine-tuning segmentation models.

### `utils/preprocessing.py`
Contains **only preprocessing utilities**, currently:

- `apply_clahe(image)` → returns enhanced grayscale image for better visibility and IA preprocessing.

### `utils/inference.py`
Provides the **patch-based sliding-window inference** used by the AI model:

- Automatically pads images so the model covers the full area
- Runs inference in overlapping patches
- Reconstructs the full-size probability map
- Thresholds to output a **0/255 uint8 mask**

---

## 📦 Installation

It is strongly recommended to create a dedicated environment:

### Option A — Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate laticifer-env
```

This will:
1. Open a napari window
2. Load the annotation widget on the right side
3. Allow you to open images and annotate them directly

### Option B — Using requirements.txt
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

After installing the environment, run the application:
```bash
python main.py
```
Make sure to run the application from inside the laticifer_annotation_app/ folder.

# 📌 Dataset Format

When saving an annotation, the app ensures the following structure:

```text
dataset/
│
├─ images/
│   └─ <name>.tif
│
├─ masks/
│   └─ <name>_mask.tif
│
└─ annotations.csv
```

# 🧪 Features Designed for Reliability

Ensures the dataset root is always the correct folder named dataset
Verifies shape consistency between image and mask
Allows enhanced visualization (CLAHE)
Keeps UI state consistent with napari layer events

