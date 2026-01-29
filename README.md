# Laticifer Annotation App

A comprehensive desktop tool built on [napari](https://napari.org/) for segmenting, refining, and quantifying laticifer structures in plant microscopy images. It supports both interactive "Human-in-the-Loop" annotation and high-throughput batch processing.

## Key Features

### Interactive Editor
- **AI-Assisted Segmentation:** Generate initial masks using a U-Net model (SE-ResNeXt50).
- **Preprocessing:** Apply CLAHE contrast enhancement for better visibility.
- **Mask Refinement:**
  - **Morphology:** Dilate and Erode masks.
  - **Cleaning:** "Remove Small Objects" to remove noise.
  - **Topology:** "Skeletonize" to view the structural network.
- **Advanced Quantification:**
  - **Pixel Density:** Calculate density relative to the whole image or **automatically detected leaf tissue area**.
  - **Transect Method:** Virtual stereology lines to count network intersections.
- **Data Management:** Auto-saves images, masks, and a persistent `annotations.csv` log.

### Batch Processing
- **Bulk Inference:** Process entire folders of images automatically in the background.
- **CSV Reporting:** Generates a summary CSV containing density metrics for every image in the folder.

---

## Installation & Usage

### Option A: For End-Users (Windows Only)
This method requires no prior installation of Python or other tools.

1.  **Download:** Download the distribution ZIP file from the URL: https://github.com/sofiaamartinezz/laticifer_app/releases/download/v1.0/LaticiferSegmentationApp.zip. 
2. **Unzip and open:** Right-click the file and select **"Extract All..."**. Navigate into the extracted folder.
3.  **Run the App:** Double-click the file named **`Start_App.bat`**.

*Note: The first run may take up to ~10 minutes to configure the internal environment. Subsequent runs will be instant.*

### Option B: For Developers (Running from source)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/laticifer_app.git
    cd laticifer_app
    ```

2.  **Create a virtual environment:**
    ```bash
    # Create env
    python -m venv venv
    
    # Activate (Windows)
    venv\Scripts\activate
    
    # Activate (Linux/Mac)
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    # Run as a module from the root directory
    python src/main.py
    ```

---

## Project Structure

```text
laticifer_app/
├── models/
│   └── best_model.pth          # Trained PyTorch model
├── src/
│   ├── main.py                 # Entry point: launches napari
│   ├── ui.py                   # UI Widgets (Tabs, Dialogs, Buttons)
│   ├── model.py                # Model loading & inference logic
│   ├── dataset.py              # File I/O, path inference, CSV logging
│   └── utils/
│       ├── inference.py        # Tiled inference with overlap
│       ├── preprocessing.py    # CLAHE & image normalization
│       ├── postprocessing.py   # Morphology & Skeletonization logic
│       └── quantification.py   # Density & Transect math
├── Start_App.bat               # Windows Launcher
├── start.sh                    # Linux Launcher
├── environment.yml             # Conda environment spec
└── requirements.txt            # Pip requirements
```

## Module Overview

### `src/ui.py`
Contains the main GUI logic using `qtpy`.
- **InteractiveEditorWidget:** Handles the single-image workflow (Load -> Enhance -> Predict -> Refine -> Save).
- **BatchProcessingWidget:** Handles the bulk processing tab with background threading.
- **QuantificationDialog:** Settings for density calculation methods.

### `src/model.py`
- Loads the U-Net checkpoint.
- Handles input validation (grayscale conversion, type casting) to ensure the model receives expected data.

### `src/dataset.py`
- Manages the dataset folder structure.
- **`save_annotation`:** Atomically saves the image, the binary mask, and appends metrics to `annotations.csv`.

### `src/utils/postprocessing.py`
- **`skeletonize_mask`**: Reduces masks to centerlines.
- **`dilate/erode`**: Standard morphological operations.

### `src/utils/quantification.py`
- **`analyze_density_pixel_ratio`**: Calculates density. Includes logic to generate a **Tissue Mask** (morphological envelope) to exclude empty background from calculations.
- **`analyze_density_transect`**: Generates grid lines and counts intersections with the laticifers.

## Dataset Output Format

The application enforces a consistent structure for reproducibility when saving a new or refined mask:

```text
dataset_folder/
├── images/
│   └── sample_01.tif
├── masks/
│   └── sample_01_mask.tif
├── annotations.csv        # Contains density metrics for all saved images
```