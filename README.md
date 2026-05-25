# LatexLens - Laticifer Annotation App

A desktop tool built on [napari](https://napari.org/) for segmenting, refining, and quantifying laticifer structures in plant microscopy images. It supports interactive human-in-the-loop annotation, density analysis, skeleton/network measurements, and high-throughput batch processing.

## Key Features

### Interactive Editor
- **AI-Assisted Segmentation:** Generate initial masks using a U-Net model (SE-ResNeXt50).
- **Preprocessing:** Apply CLAHE contrast enhancement for better visibility.
- **Scale Calibration:** Work in pixels or set a pixel size manually. The app can also detect a baked-in scale bar and convert measurements to real units.
- **Mask Refinement:**
  - **Morphology:** Dilate and erode masks.
  - **Cleaning:** Remove small objects and fill small holes.
  - **Manual Editing:** Load existing masks or draw/edit labels directly in napari.
- **Advanced Quantification:**
  - **Pixel Density:** Calculate density relative to the whole image or automatically estimated tissue area.
  - **Transect Method:** Generate editable horizontal, vertical, or combined transects and count laticifer intersections.
  - **Network Analysis:** Skeleton-based expansion, branching, thickness, and connectivity metrics.
- **Data Management:** Auto-saves images, masks, and a persistent `annotations.csv` log.

### Batch Processing
- **Bulk Inference:** Process entire folders of images automatically in the background.
- **CSV Reporting:** Generates `batch_results.csv` with density, transect, network, and optional real-unit metrics for every image.

---

## Installation & Usage

### Option A: For End-Users (Windows Only)
This method requires no prior installation of Python or other tools.

1. **Download:** Download the distribution ZIP file from the release page:
   https://github.com/sofiaamartinezz/laticifer_app/releases/download/v1.0/LaticiferSegmentationApp.zip
2. **Unzip and open:** Right-click the file and select **"Extract All..."**. Navigate into the extracted folder.
3. **Run the app:** Double-click **`Start_App.bat`**.

*Note: The first run may take several minutes while `micromamba` creates the local environment. Later launches should be much faster.*

### Option B: For Developers (Running from source)

1. **Clone the repository:**
    ```bash
    git clone https://github.com/sofiaamartinezz/laticifer_app.git
    cd laticifer_app
    ```

2. **Create the environment:**

   Recommended, using the provided conda/micromamba environment:
   ```bash
   conda env create -f environment.yml
   conda activate latiseg
   ```

   Or with `venv` and pip:
    ```bash
    python -m venv venv

    # Windows
    venv\Scripts\activate

    # Linux/macOS
    source venv/bin/activate
    ```

3. **Install dependencies when using pip:**
    ```bash
    pip install -r requirements.txt
    pip install segmentation-models-pytorch pandas scipy
    ```

4. **Check the model file:**

   The predictor expects the trained checkpoint at:
   ```text
   src/models/best_model_soft_clDice.pth
   ```

5. **Run the application:**
    ```bash
    python src/main.py
    ```

---

## Project Structure

```text
laticifer_app/
├── resources/
│   └── app_icon.ico            # Application icon
├── src/
│   ├── main.py                 # Entry point: launches napari
│   ├── data/
│   │   ├── annotations.py      # annotations.csv logging and dataset resolution
│   │   ├── batch.py            # Folder-level batch processing and CSV export
│   │   └── io.py               # Image/mask path inference, loading, saving
│   ├── model/
│   │   ├── inference.py        # Sliding-window patch inference
│   │   └── predictor.py        # Model loading and mask prediction
│   ├── models/
│   │   └── best_model_soft_clDice.pth
│   ├── ui/
│   │   ├── dialogs.py          # Quantification settings dialogs
│   │   ├── transect_controller.py
│   │   └── widgets.py          # Main napari dock widgets and tabs
│   └── utils/
│       ├── network_analysis.py # Skeleton/network metrics
│       ├── postprocessing.py   # Mask cleanup and morphology
│       ├── preprocessing.py    # CLAHE and image normalization
│       ├── quantification.py   # Density and transect math
│       └── scalebar.py         # Scale-bar detection
├── Start_App.bat               # Windows Launcher
├── environment.yml             # Conda environment spec
├── run_laticifer_app.sh        # Linux helper script
└── requirements.txt            # Pip requirements
```

## Module Overview

### `src/main.py`
- Launches the napari viewer.
- Sets the window title and icon when available.
- Adds the `LaticiferAnnotationWidget` dock widget.

### `src/ui/widgets.py`
Contains the main GUI logic using `qtpy`.
- **InteractiveEditorWidget:** Handles the single-image workflow through the Prepare, Mask, Density, and Network tabs.
- **BatchProcessingWidget:** Handles folder-level processing with background threading.
- **LaticiferAnnotationWidget:** Combines the interactive editor and batch processor in the main dock widget.

### `src/model/predictor.py`
- Loads the U-Net checkpoint from `src/models/best_model_soft_clDice.pth`.
- Converts RGB or grayscale inputs to the format expected by the model.
- Runs patch-based prediction through `src/model/inference.py`.

### `src/data/annotations.py`
- Resolves or asks for the dataset folder.
- **`save_annotation`:** Saves the image, binary mask, and a row in `annotations.csv`.

### `src/data/batch.py`
- Runs prediction, density quantification, transect analysis, and optional network analysis over a folder.
- Saves generated masks under the selected output folder.
- Writes the final `batch_results.csv`.

### `src/utils/postprocessing.py`
- **`remove_small_objects`**: Removes small isolated mask components.
- **`fill_small_holes`**: Fills small gaps inside laticifer regions.
- **`dilate_mask` / `erode_mask`**: Standard morphological operations.

### `src/utils/quantification.py`
- **`analyze_density_pixel_ratio`**: Calculates density against the whole image or an estimated tissue mask.
- **`analyze_density_transect`**: Generates transect lines and counts intersections with laticifers.
- **`analyze_density_from_lines`**: Recomputes transect density from edited napari line geometry.

### `src/utils/network_analysis.py`
- Computes skeleton length, connected components, branch counts, bifurcation/end-point counts, branch lengths, bifurcation angles, thickness, and branch-node ratio.
- Returns both scalar metrics and geometry arrays for napari visualization.

### `src/utils/scalebar.py`
- Detects bright or dark scale bars in common image regions.
- Converts a detected bar length plus a user-supplied real-world length into pixel size.

## Dataset Output Format

The application enforces a consistent structure for reproducibility when saving a new or refined mask:

```text
dataset_folder/
├── images/
│   └── sample_01.tif
├── masks/
│   └── sample_01_mask.tif
└── annotations.csv        # Contains density and transect metrics for saved images
```

`annotations.csv` contains one row per saved annotation, including:

```text
image_path, mask_path, timestamp, initialized_from_model,
image_shape_y, image_shape_x, laticifer_pixels,
density_tissue, density, transect_direction,
transect_num_lines, transect_mean_intersections_per_line
```

Batch processing writes a separate output folder:

```text
output_folder/
├── masks/
│   └── sample_01_mask.tif
└── batch_results.csv      # Density, transect, network, and scale metrics
```
