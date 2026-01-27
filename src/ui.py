# src/ui.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import napari
import pandas as pd
import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QSpinBox,
    QTabWidget,
    QLineEdit,
    QProgressBar,
)
from napari.qt.threading import thread_worker
from skimage import io as skio

# Import logic from other modules
from model import predict_laticifer_mask
from dataset import (
    ensure_dataset_root,
    infer_mask_path_for_image,
    load_mask_from_path,
    save_annotation as save_annotation_to_dataset,
)

import sys
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.preprocessing import apply_clahe
from utils.quantification import (
    analyze_density_pixel_ratio,
    analyze_density_transect,
)

# -----------------------------------------------------------------------------
#  DIALOG: Quantification Settings
# -----------------------------------------------------------------------------
class QuantificationDialog(QDialog):
    def __init__(self, parent: QWidget, viewer: napari.Viewer, image_layer: Optional[napari.layers.Image]):
        super().__init__(parent)
        self.viewer = viewer
        self.image_layer = image_layer
        self.setWindowTitle("Quantification settings")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Method
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Pixel ratio (area fraction)", "Transect lines (intersections)"])
        form.addRow("Method", self.method_combo)

        # Direction
        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["horizontal", "vertical", "both"])
        form.addRow("Direction", self.direction_combo)

        # Num lines
        self.num_lines_spin = QSpinBox()
        self.num_lines_spin.setRange(1, 10000)
        self.num_lines_spin.setValue(10)
        form.addRow("Number of lines", self.num_lines_spin)

        # Points overlay
        self.show_points_cb = QCheckBox("Show intersection points (0→1 entries)")
        self.show_points_cb.setChecked(True)
        form.addRow("", self.show_points_cb)

        layout.addLayout(form)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Wire UI toggles
        self.method_combo.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self):
        is_pixel = self.method_combo.currentIndex() == 0
        self.direction_combo.setEnabled(not is_pixel)
        self.num_lines_spin.setEnabled(not is_pixel)
        self.show_points_cb.setEnabled(not is_pixel)

    def get_params(self):
        is_pixel = self.method_combo.currentIndex() == 0
        if is_pixel:
            return {"method": "pixel_ratio"}

        return {
            "method": "transect",
            "direction": self.direction_combo.currentText(),
            "num_lines": int(self.num_lines_spin.value()),
            "show_points": bool(self.show_points_cb.isChecked()),
        }

# -----------------------------------------------------------------------------
#  WIDGET: Interactive Editor (Single Image)
# -----------------------------------------------------------------------------
class InteractiveEditorWidget(QWidget):
    """
    The original interactive annotation UI.
    """
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.labels_layer: Optional[napari.layers.Labels] = None
        self.dataset_root: Optional[Path] = None
        self.initialized_from_model: bool = False
        self.base_image_layer: Optional[napari.layers.Image] = None

        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        self.last_density_ratio = None
        self.last_density_percent = None
        self.last_transect_num_lines = 10
        self.last_transect_direction = "horizontal"

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Use napari's 'Open File' to load an image")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)

        # Image tools
        image_group = QGroupBox("Image tools")
        image_layout = QVBoxLayout()
        self.enhance_btn = QPushButton("Enhance image (CLAHE)")
        self.enhance_btn.clicked.connect(self.enhance_current_image)
        self.enhance_btn.setEnabled(False)
        image_layout.addWidget(self.enhance_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # Mask loading
        load_group = QGroupBox("Mask loading")
        load_layout = QVBoxLayout()
        self.load_mask_btn = QPushButton("Load existing mask")
        self.load_mask_btn.clicked.connect(self.load_existing_mask)
        self.load_mask_btn.setEnabled(False)
        load_layout.addWidget(self.load_mask_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # Mask creation
        mask_group = QGroupBox("Mask creation")
        mask_layout = QVBoxLayout()
        self.empty_mask_btn = QPushButton("Create empty mask")
        self.empty_mask_btn.clicked.connect(self.create_empty_mask)
        self.empty_mask_btn.setEnabled(False)
        mask_layout.addWidget(self.empty_mask_btn)

        self.auto_mask_btn = QPushButton("Auto-generate mask (AI)")
        self.auto_mask_btn.clicked.connect(self.auto_generate_mask)
        self.auto_mask_btn.setEnabled(False)
        mask_layout.addWidget(self.auto_mask_btn)
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        layout.addSpacing(15)

        # Quantification
        quant_group = QGroupBox("Quantification")
        quant_layout = QVBoxLayout()
        self.density_btn = QPushButton("Calculate laticifer density")
        self.density_btn.clicked.connect(self.compute_laticifer_density)
        self.density_btn.setEnabled(False)
        quant_layout.addWidget(self.density_btn)
        quant_group.setLayout(quant_layout)
        layout.addWidget(quant_group)

        layout.addSpacing(15)

        # Save
        self.save_btn = QPushButton("Save annotation")
        self.save_btn.clicked.connect(self.save_annotation)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        layout.addStretch()
        self.setLayout(layout)

    # --- Layer Events ---
    def _on_layer_inserted(self, event) -> None:
        layer = event.value
        if isinstance(layer, napari.layers.Labels):
            if not layer.metadata.get("is_debug_tissue_mask", False):
                self.labels_layer = layer
                self.labels_layer.editable = True
                self.labels_layer.mode = "paint"
                self.labels_layer.selected_label = 1
        if isinstance(layer, napari.layers.Image) and self.base_image_layer is None:
            self.base_image_layer = layer
        self._update_button_states()

    def _on_layer_removed(self, event) -> None:
        layer = event.value
        if layer is self.labels_layer:
            self.labels_layer = None
        if layer is self.base_image_layer:
            self.base_image_layer = None
        self._update_button_states()

    def _update_button_states(self) -> None:
        has_image = self._get_original_image_layer() is not None
        has_mask = self.labels_layer is not None
        self.enhance_btn.setEnabled(has_image)
        self.empty_mask_btn.setEnabled(has_image and not has_mask)
        self.auto_mask_btn.setEnabled(has_image and not has_mask)
        self.load_mask_btn.setEnabled(has_image and not has_mask)
        self.density_btn.setEnabled(has_mask)
        self.save_btn.setEnabled(has_image and has_mask)

    def _get_current_image_layer(self) -> Optional[napari.layers.Image]:
        if isinstance(self.viewer.layers.selection.active, napari.layers.Image):
            return self.viewer.layers.selection.active
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return layer
        return None

    def _get_original_image_layer(self) -> Optional[napari.layers.Image]:
        if (
            self.base_image_layer is not None
            and self.base_image_layer in self.viewer.layers
            and isinstance(self.base_image_layer, napari.layers.Image)
        ):
            return self.base_image_layer
        return self._get_current_image_layer()

    # --- Actions ---
    def create_empty_mask(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before creating a mask.")
            return
        image_shape = image_layer.data.shape[:2]
        mask_data = np.zeros(image_shape, dtype=np.uint8)
        self._add_labels_layer(mask_data, from_model=False)

    def auto_generate_mask(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before generating a mask.")
            return
        image_data = np.asarray(image_layer.data)
        try:
            mask = predict_laticifer_mask(image_data)
            self._add_labels_layer(mask.astype(np.uint8), from_model=True)
        except Exception as exc:
            QMessageBox.critical(self, "Prediction error", f"Failed to generate mask: {exc}")

    def _add_labels_layer(self, data: np.ndarray, from_model: bool, opacity: float = 1.0) -> None:
        if self.labels_layer is not None:
            try:
                self.viewer.layers.remove(self.labels_layer)
            except ValueError:
                pass
            self.labels_layer = None
        self.labels_layer = self.viewer.add_labels(
            data,
            name="Laticifer mask",
            opacity=opacity,
            blending="translucent",
        )
        self.labels_layer.editable = True
        self.labels_layer.mode = "paint"
        self.labels_layer.selected_label = 1
        self.viewer.layers.selection.active = self.labels_layer
        self.initialized_from_model = from_model
        self._update_button_states()

    def load_existing_mask(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before loading a mask.")
            return
        img_shape = np.asarray(image_layer.data).shape[:2]
        mask_path = infer_mask_path_for_image(image_layer, self.dataset_root)
        if mask_path is None:
            mask_path_str, _ = QFileDialog.getOpenFileName(
                self, "Select mask file", "", "Image files (*.tif *.tiff *.png *.jpg *.jpeg)"
            )
            if not mask_path_str: return
            mask_path = Path(mask_path_str)
        mask_labels = load_mask_from_path(mask_path, img_shape)
        if mask_labels is None:
            QMessageBox.warning(self, "Load error", "Failed to load mask.")
            return
        self._add_labels_layer(mask_labels, from_model=False)

    def enhance_current_image(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None: return
        data = np.asarray(image_layer.data)
        enhanced = apply_clahe(data)
        enhanced_layer = self.viewer.add_image(
            enhanced,
            name=f"{image_layer.name} [enhanced]",
            blending="additive",
            colormap="gray",
        )
        enhanced_layer.metadata["is_preprocessed"] = True
        self.viewer.layers.selection.active = enhanced_layer

    def compute_laticifer_density(self) -> None:
        if self.labels_layer is None:
            QMessageBox.warning(self, "No mask", "Create or load a mask first.")
            return

        mask = np.asarray(self.labels_layer.data)
        image_layer = self._get_original_image_layer()

        dlg = QuantificationDialog(self, viewer=self.viewer, image_layer=image_layer)
        if dlg.exec_() != QDialog.Accepted:
            return

        params = dlg.get_params()

        # -------------------------------------------------
        # PIXEL RATIO (Whole image vs Tissue-only)
        # -------------------------------------------------
        if params["method"] == "pixel_ratio":
            # Better choice dialog
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("Density calculation area")
            msg.setText("Choose the area used as denominator for density.")
            msg.setInformativeText(
                "• Whole image: Density = laticifer pixels / (width × height)\n"
                "• Tissue only: Density = laticifer pixels / (detected leaf/tissue area)\n\n"
                "Tip: Use 'Tissue only' if your image contains background."
            )
            msg.setStandardButtons(QMessageBox.Cancel)
            btn_whole = msg.addButton("Whole image", QMessageBox.AcceptRole)
            btn_tissue = msg.addButton("Tissue only (auto)", QMessageBox.AcceptRole)

            # Default selection (tissue is usually safer if there is background)
            msg.setDefaultButton(btn_tissue)

            msg.exec_()
            if msg.clickedButton() == msg.button(QMessageBox.Cancel):
                return

            use_tissue_mask = (msg.clickedButton() == btn_tissue)

            # Compute stats
            stats = analyze_density_pixel_ratio(mask, use_tissue_mask=use_tissue_mask)

            # Show tissue mask overlay only if used
            if use_tissue_mask:
                debug_mask = stats.get("debug_tissue_mask")
                if debug_mask is not None:
                    if "Computed Tissue Area" in self.viewer.layers:
                        self.viewer.layers.remove("Computed Tissue Area")

                    tissue_layer = self.viewer.add_labels(
                        debug_mask,
                        name="Computed Tissue Area",
                        opacity=0.3,
                        metadata={"is_debug_tissue_mask": True},
                    )
                    tissue_layer.editable = False
                    if self.labels_layer is not None:
                        self.viewer.layers.selection.active = self.labels_layer
                    print("[DEBUG] Tissue mask added to viewer.")
            else:
                # If user chose whole image, remove old debug layer if present
                if "Computed Tissue Area" in self.viewer.layers:
                    self.viewer.layers.remove("Computed Tissue Area")

            # Store + report
            self.last_density_ratio = float(stats["pixel_ratio"])
            self.last_density_percent = float(stats["density_percentage"])

            denom_text = "Detected tissue area" if use_tissue_mask else "Total image area"
            QMessageBox.information(
                self,
                "Laticifer density",
                f"Density: {self.last_density_percent:.2f}%\n"
                f"(Laticifers / {denom_text})"
            )

        # -------------------------------------------------
        # TRANSECT METHOD (unchanged)
        # -------------------------------------------------
        else:
            direction = str(params["direction"]).lower().strip()
            num_lines = max(1, int(params.get("num_lines", 10)))
            self.last_transect_num_lines = num_lines
            self.last_transect_direction = direction

            stats, lines, pts = analyze_density_transect(
                mask, num_lines=num_lines, direction=direction
            )

            # Visualize lines/points
            if "Transect lines" in self.viewer.layers:
                self.viewer.layers.remove("Transect lines")
            self.viewer.add_shapes(lines, name="Transect lines", shape_type="line", edge_width=2)

            if "Intersection points" in self.viewer.layers:
                self.viewer.layers.remove("Intersection points")
            if bool(params.get("show_points", True)):
                self.viewer.add_points(pts, name="Intersection points", size=6)

            mean = stats.get("mean_intersections_per_line", float("nan"))
            QMessageBox.information(self, "Transect Results", f"Mean intersections: {mean:.2f}")

    def save_annotation(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None or self.labels_layer is None:
            QMessageBox.warning(self, "Missing data", "Ensure both image and mask are present.")
            return
        root = ensure_dataset_root(self, image_layer, self.dataset_root)
        if root is None: return
        self.dataset_root = root
        save_annotation_to_dataset(
            parent_widget=self,
            image_layer=image_layer,
            labels_layer=self.labels_layer,
            dataset_root=self.dataset_root,
            initialized_from_model=self.initialized_from_model,
            transect_num_lines=getattr(self, "last_transect_num_lines", 10),
            transect_direction=getattr(self, "last_transect_direction", "both"),
        )

# -----------------------------------------------------------------------------
#  WIDGET: Batch Processing (Multiple Images)
# -----------------------------------------------------------------------------
class BatchProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        
        # Input
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select input folder...")
        btn_in = QPushButton("Browse Input")
        btn_in.clicked.connect(self._select_input)
        layout.addWidget(QLabel("Input Images:"))
        layout.addWidget(self.input_dir_edit)
        layout.addWidget(btn_in)
        layout.addSpacing(10)

        # Output
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output folder...")
        btn_out = QPushButton("Browse Output")
        btn_out.clicked.connect(self._select_output)
        layout.addWidget(QLabel("Output Folder:"))
        layout.addWidget(self.output_dir_edit)
        layout.addWidget(btn_out)
        layout.addSpacing(20)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_lbl = QLabel("Ready")
        layout.addWidget(self.status_lbl)
        layout.addWidget(self.progress_bar)

        # Run
        self.run_btn = QPushButton("Start Batch Processing")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self._start_batch)
        layout.addWidget(self.run_btn)
        
        layout.addStretch()
        self.setLayout(layout)

    def _select_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input")
        if d: self.input_dir_edit.setText(d)

    def _select_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output")
        if d: self.output_dir_edit.setText(d)

    def _start_batch(self):
        in_d = self.input_dir_edit.text()
        out_d = self.output_dir_edit.text()
        if not in_d or not out_d:
            QMessageBox.warning(self, "Error", "Select both folders.")
            return

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        
        # Launch worker
        worker = self.run_batch_logic(in_d, out_d)
        worker.yielded.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.start()

    def _on_progress(self, data):
        curr, total, name = data
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.status_lbl.setText(f"Processing: {name}")

    def _on_finished(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_lbl.setText("Complete!")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Batch Processing")
        QMessageBox.information(self, "Done", "Batch processing complete.\nSee 'batch_results.csv' in output folder.")

    # --- Background Worker ---
    @thread_worker
    def run_batch_logic(self, in_dir_str, out_dir_str):
        import pandas as pd
        from pathlib import Path
        # Local imports to avoid circular issues
        from model import predict_laticifer_mask
        from utils.quantification import analyze_density_pixel_ratio, analyze_density_transect
        from utils.preprocessing import apply_clahe

        in_path = Path(in_dir_str)
        out_path = Path(out_dir_str)
        masks_out = out_path / "masks"
        masks_out.mkdir(parents=True, exist_ok=True)

        files = []
        for ext in ['*.tif', '*.tiff', '*.jpg', '*.png']:
            files.extend(list(in_path.glob(ext)))
        
        results = []
        total = len(files)

        for i, f in enumerate(files):
            yield (i + 1, total, f.name)
            try:
                img = skio.imread(f)
                img_enh = apply_clahe(img) # Ensure enhancement is applied
                mask = predict_laticifer_mask(img_enh)
                
                # Metrics
                px = analyze_density_pixel_ratio(mask)
                tr, _, _ = analyze_density_transect(mask, num_lines=10, direction="both")
                
                # Save mask
                mask_name = f"{f.stem}_mask.tif"
                skio.imsave(masks_out / mask_name, (mask * 255).astype(np.uint8))
                
                results.append({
                    "filename": f.name,
                    "density_percent": px["density_percentage"],
                    "transect_mean_intersections": tr.get("mean_intersections_per_line", "")
                })
            except Exception as e:
                print(f"Error on {f.name}: {e}")

        if results:
            pd.DataFrame(results).to_csv(out_path / "batch_results.csv", index=False)

# -----------------------------------------------------------------------------
#  MAIN WIDGET: Combined Tabs
# -----------------------------------------------------------------------------
class LaticiferAnnotationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        self.editor = InteractiveEditorWidget(viewer)
        self.batch = BatchProcessingWidget()
        
        self.tabs.addTab(self.editor, "Interactive Editor")
        self.tabs.addTab(self.batch, "Batch Processing")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
