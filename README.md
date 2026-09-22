# LatexLens — Laticifer Annotation App

LatexLens is a desktop application built with [napari](https://napari.org/) for segmenting, editing, and quantifying laticifer networks in microscopy images. It supports an interactive human-in-the-loop workflow and folder-level batch processing.

## Main features

- AI-assisted segmentation with a U-Net using a SE-ResNeXt50 encoder.
- Manual creation, loading, painting, and refinement of masks.
- Non-destructive CLAHE enhancement: it creates a new image layer and keeps the original unchanged.
- Scale calibration from a user-drawn reference line or a known pixel size.
- Pixel-density, editable-transect, and skeleton/network measurements.
- Physical measurements in µm and µm² when a scale is active; otherwise results remain explicitly in pixels.
- Versioned analysis sessions that restore the image, editable mask, scale, transects, parameters, and workflow position.
- Persistent user settings for transects, mask cleanup, CLAHE, and scale validation.
- Background batch processing with progress, cancellation, per-image failure reporting, and CSV export.

LatexLens does **not** infer scale automatically from image metadata. A scale only becomes active after the user applies a reference-line calibration or enters the pixel size directly.

## Installation from source

Python 3.10 is recommended.

### Using `venv` and pip

From the project root on Windows Command Prompt:

```bat
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

In PowerShell, activate with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If Python 3.10 is not installed, install it first or use an available compatible Python version.

### Using conda or micromamba

```bash
conda env create -f environment.yml
conda activate latiseg
```

### Model checkpoint

The trained checkpoint is not stored in Git. Place it at:

```text
src/models/best_model_soft_clDice.pth
```

Without this file, manual mask editing and non-AI analyses remain available, but automatic mask generation will report that the model is missing.

## Running the application

```bat
python src\main.py
```

A packaged Windows distribution can use `Start_App.bat`; it creates its local micromamba environment on the first run and then starts the application.

## Interactive workflow

Open an image using napari's **File → Open** command or drag it into the viewer. The **Interactive Editor** contains four workflow tabs.

### 1 · Prepare

The controls are separated by purpose:

- **Session**
  - **Open session…** restores a previously saved analysis.
  - **Save session…** stores the current analysis for later.
- **Image**
  - Displays the active source image and its dimensions.
  - **Create contrast-enhanced copy** adds a CLAHE-enhanced layer without changing the original.
- **Scale calibration**
  - Draw a line over a known distance, enter its real length and unit, review the calculated µm/px value, and apply it.
  - Alternatively, enter a known pixel size directly.
  - Use **Remove scale · use pixels only** to return to pixel measurements.

Reference lines that are too short are rejected. Unusually small or large scales require confirmation. The active scale is also shown beside measurement results.

### 2 · Mask

- Generate an initial mask with the AI model.
- Load an existing mask or create an empty one.
- Paint and erase directly in napari.
- Remove small objects, fill holes, erode, or dilate using the configured defaults.

Deleting the source-image layer resets the complete analysis: mask, derived layers, calibration, transects, cached results, and interface state.

### 3 · Density

- Calculate laticifer pixel coverage against the whole image or estimated tissue area.
- Generate horizontal, vertical, or combined transects.
- Edit or delete transect lines and recalculate from their current geometry.
- Save the image, mask, and measurements to the selected dataset folder.

### 4 · Network

Run skeleton analysis to calculate:

- Total skeleton length and connected components.
- Bifurcations, endpoints, branches, branch lengths, and bifurcation angles.
- Mean, standard deviation, and median diameter.
- Branch-to-node connectivity ratio.

Derived points, maps, and histograms can be displayed from the result controls.

## Saving and opening sessions

Choose **Interactive Editor → 1 · Prepare → Save session…**. For a session named `sample.json`, LatexLens creates:

```text
sample.json
sample_mask.tif    # created when the analysis has a mask
```

The JSON contains no image pixels. It stores a reference to the original image plus the calibration, mask reference, edited transects, parameters, and active workflow tab. Keep the JSON and mask sidecar together. The original image must remain accessible; paths inside the session are relative where possible to make moving the files easier.

To resume work, start LatexLens, choose **Open session…**, and select the JSON file. The file is validated before the current analysis is replaced. Corrupt files, unsupported versions, missing images or masks, invalid scales, and invalid transect geometry are reported without clearing the current work.

## Persistent settings

The top-level **Settings** tab controls:

- Default number of transect lines.
- Minimum object size and maximum hole area.
- Morphological-operation radius.
- CLAHE clip limit and tile size.
- Minimum reference-line length.
- Lower and upper limits used to flag unusual scales.

**Save settings** applies and persists the values immediately. **Restore defaults** removes the saved preferences and restores safe defaults. Settings are stored in the platform-standard Qt user-settings location, outside the repository.

## Batch processing

The **Batch Processing** tab processes supported images from an input folder. Each execution creates a timestamped subfolder in the selected output folder, keeping previous runs separate.

- Choose pixel-only results or enter one shared scale for all images. Density percentages and transect intersections do not require a physical scale; the scale converts network lengths and diameters to physical units.
- Do not combine images with different acquisition scales in one shared-scale batch.
- Network metrics can be disabled when only density and transect results are needed.
- A confirmation summary shows the image count, scale, transects, network setting, and destination before processing starts.
- Grayscale, RGB, and RGBA images are accepted; ambiguous multidimensional images such as Z-stacks are reported and skipped.
- Processing runs in the background.
- Results are checkpointed after every completed image, so an interrupted run keeps its completed CSV rows.
- Mask filenames include the source extension to avoid collisions such as `sample.png` and `sample.tif`.
- **Cancel after current image** lets the active image finish, saves completed rows, and prevents the next image from starting.
- A failed image does not stop the remaining batch.
- `analysis_status` is `success`, `partial`, or `failed`; `error_reason` explains incomplete results.

## Exported data

Saving an interactive annotation creates or updates:

```text
dataset/
├── images/
│   └── sample.tif
├── masks/
│   └── sample_mask.tif
└── annotations.csv
```

`annotations.csv` records source and saved paths, timestamp with timezone, application version, image dimensions, mask origin, density, transect parameters, calibration provenance, and explicit measurement units. Existing CSV files are migrated to the current column schema before new rows are appended.

Batch output uses:

```text
output/
└── batch_YYYYMMDD_HHMMSS/
    ├── masks/
    │   └── sample_tif_mask.tif
    └── batch_results.csv
```

The batch CSV includes provenance, parameters, units, scale, density, transect and network metrics, plus per-image status and errors.

## Error handling

Long-running mask generation, network analysis, and batch processing recover their controls after errors. User-facing dialogs explain the failure, stale worker results are ignored after a session reset, and batch failures are preserved in the CSV.

## Running tests

Install development dependencies and run from the project root:

```bat
python -m pip install -r requirements-dev.txt
python -m pytest
```

With coverage:

```bat
python -m pytest --cov=src --cov-report=term-missing
```

## Project structure

```text
laticifer_app/
├── resources/
│   └── app_icon.ico
├── sessions/                    # Local session workspace (contents ignored by Git)
├── src/
│   ├── main.py
│   ├── data/
│   │   ├── annotations.py       # Interactive CSV export
│   │   ├── batch.py             # Batch processing and CSV export
│   │   ├── errors.py            # User-facing error formatting
│   │   ├── io.py                # Image/mask I/O and scale calibration
│   │   ├── provenance.py        # Version, timestamp, and unit provenance
│   │   ├── session.py           # Versioned analysis sessions
│   │   └── settings.py          # Validated persistent settings
│   ├── model/
│   │   ├── inference.py         # Sliding-window inference
│   │   └── predictor.py         # Model loading and prediction
│   ├── models/
│   │   └── best_model_soft_clDice.pth
│   ├── ui/
│   │   ├── dialogs.py
│   │   ├── transect_controller.py
│   │   └── widgets.py
│   └── utils/
│       ├── network_analysis.py
│       ├── postprocessing.py
│       ├── preprocessing.py
│       └── quantification.py
├── tests/
├── environment.yml
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
└── Start_App.bat
```

## Current limitations

- Session files reference the original image instead of embedding it.
- Restored network metrics must be recalculated from the restored mask.
- A batch-level scale applies equally to every image in that batch.
- The trained model checkpoint must be distributed separately from the Git repository.
