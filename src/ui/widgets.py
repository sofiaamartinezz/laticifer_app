# ui/widgets.py
"""
Main Qt widgets for the interactive editor and batch processing UI.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QDialog, QFileDialog, QFrame, QGroupBox, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget, QLineEdit, QProgressBar,
)

from model.predictor import predict_laticifer_mask
from data.io import infer_mask_path, load_mask
from data.annotations import ensure_dataset_root, save_annotation
from data.batch import run_batch_processing, write_batch_csv
from utils.preprocessing import apply_clahe
from utils.quantification import analyze_density_pixel_ratio
from utils.postprocessing import (
    remove_small_objects, dilate_mask, erode_mask, skeletonize_mask,
)
from ui.dialogs import QuantificationDialog
from ui.transect_controller import TransectController


# -----------------------------------------------------------------------------
#  WIDGET: Interactive Editor
# -----------------------------------------------------------------------------

class InteractiveEditorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.labels_layer: Optional[napari.layers.Labels] = None
        self.dataset_root: Optional[Path] = None
        self.initialized_from_model: bool = False
        self.base_image_layer: Optional[napari.layers.Image] = None

        self.last_density_ratio: Optional[float] = None
        self.last_density_percent: Optional[float] = None
        self.last_transect_num_lines: int = 10
        self.last_transect_direction: str = "horizontal"
        self.last_transect_mean: Optional[float] = None  # result the user last saw

        self._transect_ctrl = TransectController(
            viewer=self.viewer,
            on_state_change=self._refresh_transect_ui,
        )

        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        info = QLabel("Use napari's 'Open File' to load an image")
        info.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info)

        # --- Image tools ---
        image_group = QGroupBox("Image tools")
        image_layout = QVBoxLayout()
        self.enhance_btn = QPushButton("Enhance image (CLAHE)")
        self.enhance_btn.clicked.connect(self.enhance_current_image)
        self.enhance_btn.setEnabled(False)
        image_layout.addWidget(self.enhance_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)

        # --- Mask loading ---
        load_group = QGroupBox("Mask loading")
        load_layout = QVBoxLayout()
        self.load_mask_btn = QPushButton("Load existing mask")
        self.load_mask_btn.clicked.connect(self.load_existing_mask)
        self.load_mask_btn.setEnabled(False)
        load_layout.addWidget(self.load_mask_btn)
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # --- Mask creation ---
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

        self._ai_status_lbl = QLabel("")
        self._ai_status_lbl.setStyleSheet("color: gray; font-style: italic;")
        self._ai_status_lbl.setVisible(False)
        mask_layout.addWidget(self._ai_status_lbl)

        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)

        # --- Mask refinement ---
        refine_group = QGroupBox("Mask Refinement")
        refine_layout = QVBoxLayout()

        row_clean = QHBoxLayout()
        self.spin_min_size = QSpinBox()
        self.spin_min_size.setRange(1, 10000)
        self.spin_min_size.setValue(100)
        self.spin_min_size.setSuffix(" px")
        self.spin_min_size.setToolTip("Remove objects smaller than this")
        self.btn_remove_small = QPushButton("Remove Small")
        self.btn_remove_small.clicked.connect(self._on_remove_small_objects)
        self.btn_remove_small.setEnabled(False)
        row_clean.addWidget(self.spin_min_size)
        row_clean.addWidget(self.btn_remove_small)
        refine_layout.addLayout(row_clean)

        row_topo = QHBoxLayout()
        self.btn_skeleton = QPushButton("Skeletonize")
        self.btn_skeleton.setToolTip("Reduce mask to 1-pixel wide centerlines")
        self.btn_skeleton.clicked.connect(self._on_skeletonize)
        self.btn_skeleton.setEnabled(False)
        row_topo.addWidget(QLabel("Topology:"))
        row_topo.addWidget(self.btn_skeleton)
        refine_layout.addLayout(row_topo)

        row_morph = QHBoxLayout()
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 20)
        self.spin_radius.setValue(1)
        self.spin_radius.setSuffix(" px")
        self.btn_erode = QPushButton("Erode")
        self.btn_erode.clicked.connect(self._on_erode)
        self.btn_erode.setEnabled(False)
        self.btn_dilate = QPushButton("Dilate")
        self.btn_dilate.clicked.connect(self._on_dilate)
        self.btn_dilate.setEnabled(False)
        row_morph.addWidget(QLabel("Radius:"))
        row_morph.addWidget(self.spin_radius)
        row_morph.addWidget(self.btn_erode)
        row_morph.addWidget(self.btn_dilate)
        refine_layout.addLayout(row_morph)

        refine_group.setLayout(refine_layout)
        layout.addWidget(refine_group)

        layout.addSpacing(15)

        # --- Quantification ---
        quant_group = QGroupBox("Quantification")
        quant_layout = QVBoxLayout()

        self.density_btn = QPushButton("Calculate laticifer density")
        self.density_btn.clicked.connect(self.compute_laticifer_density)
        self.density_btn.setEnabled(False)
        quant_layout.addWidget(self.density_btn)

        quant_layout.addSpacing(6)
        quant_layout.addWidget(_h_rule())

        quant_layout.addWidget(QLabel("<b>Editable transects</b>"))

        hint = QLabel(
            "To delete a line: select the \"Transect lines\" layer, "
            "click a line and press the Delete key."
        )
        hint.setStyleSheet("color: gray; font-style: italic;")
        hint.setWordWrap(True)
        quant_layout.addWidget(hint)

        self._pending_lbl = QLabel("⚠ Transects edited — recalculate to update results")
        self._pending_lbl.setStyleSheet("color: orange; font-style: italic;")
        self._pending_lbl.setVisible(False)
        quant_layout.addWidget(self._pending_lbl)

        self._recalc_btn = QPushButton("Recalculate density with transects")
        self._recalc_btn.setToolTip(
            "Recalculate using the current transect geometry. Lines are NOT regenerated."
        )
        self._recalc_btn.clicked.connect(self._on_recalculate_density)
        self._recalc_btn.setEnabled(False)
        quant_layout.addWidget(self._recalc_btn)

        quant_group.setLayout(quant_layout)
        layout.addWidget(quant_group)

        layout.addSpacing(15)

        self.save_btn = QPushButton("Save annotation")
        self.save_btn.clicked.connect(self.save_annotation)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        layout.addStretch()
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Layer events
    # ------------------------------------------------------------------

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
        has_image = self._get_image_layer() is not None
        has_mask  = self.labels_layer is not None
        self.enhance_btn.setEnabled(has_image)
        self.empty_mask_btn.setEnabled(has_image and not has_mask)
        self.auto_mask_btn.setEnabled(has_image)
        self.load_mask_btn.setEnabled(has_image and not has_mask)
        self.density_btn.setEnabled(has_mask)
        self.save_btn.setEnabled(has_image and has_mask)
        self.btn_skeleton.setEnabled(has_mask)
        self.btn_remove_small.setEnabled(has_mask)
        self.btn_dilate.setEnabled(has_mask)
        self.btn_erode.setEnabled(has_mask)
        self._refresh_transect_ui()

    def _refresh_transect_ui(self) -> None:
        """
        Sync transect UI state with the controller:
        - Recalculate button enabled only when there are pending edits.
        - Pending label visible when edits haven't been recalculated yet.
        """
        pending = self._transect_ctrl.pending
        self._recalc_btn.setEnabled(pending)
        self._pending_lbl.setVisible(pending)

    # ------------------------------------------------------------------
    # Image layer helper
    # ------------------------------------------------------------------

    def _get_image_layer(self) -> Optional[napari.layers.Image]:
        """Return the base image layer, falling back to any Image in the viewer."""
        if (
            self.base_image_layer is not None
            and self.base_image_layer in self.viewer.layers
            and isinstance(self.base_image_layer, napari.layers.Image)
        ):
            return self.base_image_layer
        if isinstance(self.viewer.layers.selection.active, napari.layers.Image):
            return self.viewer.layers.selection.active
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return layer
        return None

    # ------------------------------------------------------------------
    # Mask actions
    # ------------------------------------------------------------------

    def create_empty_mask(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before creating a mask.")
            return
        if self.labels_layer is not None:
            reply = QMessageBox.question(
                self, "Overwrite mask?",
                "A mask already exists. Creating a new one will replace it. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return
        mask_data = np.zeros(image_layer.data.shape[:2], dtype=np.uint8)
        self._add_labels_layer(mask_data, from_model=False)

    def auto_generate_mask(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before generating a mask.")
            return
        if self.labels_layer is not None:
            reply = QMessageBox.question(
                self, "Overwrite mask?",
                "A mask already exists. Generating a new one will replace it. Continue?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
            )
            if reply != QMessageBox.Yes:
                return

        image_data = np.asarray(image_layer.data)
        self.auto_mask_btn.setEnabled(False)
        self.auto_mask_btn.setText("Generating…")
        self._ai_status_lbl.setText("⏳ Running AI model, please wait…")
        self._ai_status_lbl.setVisible(True)

        @thread_worker
        def _run():
            return predict_laticifer_mask(image_data)

        worker = _run()

        def _on_done(mask):
            self._add_labels_layer(mask.astype(np.uint8), from_model=True)
            self._ai_status_lbl.setText("✓ Mask generated successfully.")
            from qtpy.QtCore import QTimer
            QTimer.singleShot(4000, lambda: self._ai_status_lbl.setVisible(False))

        def _on_error(exc_info):
            self._ai_status_lbl.setVisible(False)
            QMessageBox.critical(self, "Prediction error", f"Failed to generate mask: {exc_info[1]}")

        def _on_finished():
            self.auto_mask_btn.setText("Auto-generate mask (AI)")
            self._update_button_states()

        worker.returned.connect(_on_done)
        worker.errored.connect(_on_error)
        worker.finished.connect(_on_finished)
        worker.start()

    def _add_labels_layer(self, data: np.ndarray, from_model: bool) -> None:
        if self.labels_layer is not None:
            try:
                self.viewer.layers.remove(self.labels_layer)
            except ValueError:
                pass
            self.labels_layer = None
        self.labels_layer = self.viewer.add_labels(
            data, name="Laticifer mask", opacity=1.0, blending="translucent",
        )
        self.labels_layer.editable = True
        self.labels_layer.mode = "paint"
        self.labels_layer.selected_label = 1
        self.viewer.layers.selection.active = self.labels_layer
        self.initialized_from_model = from_model
        self._update_button_states()

    def load_existing_mask(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before loading a mask.")
            return
        img_shape = np.asarray(image_layer.data).shape[:2]
        mask_path = infer_mask_path(image_layer, self.dataset_root)
        if mask_path is None:
            path_str, _ = QFileDialog.getOpenFileName(
                self, "Select mask file", "",
                "Image files (*.tif *.tiff *.png *.jpg *.jpeg)",
            )
            if not path_str:
                return
            from pathlib import Path
            mask_path = Path(path_str)
        mask_labels = load_mask(mask_path, img_shape)
        if mask_labels is None:
            QMessageBox.warning(self, "Load error", "Failed to load mask.")
            return
        self._add_labels_layer(mask_labels, from_model=False)

    def enhance_current_image(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            return
        enhanced = apply_clahe(np.asarray(image_layer.data))
        layer = self.viewer.add_image(
            enhanced,
            name=f"{image_layer.name} [enhanced]",
            blending="additive",
            colormap="gray",
        )
        layer.metadata["is_preprocessed"] = True
        self.viewer.layers.selection.active = layer

    # ------------------------------------------------------------------
    # Mask refinement
    # ------------------------------------------------------------------

    def _get_mask_data(self) -> Optional[np.ndarray]:
        return np.asarray(self.labels_layer.data) if self.labels_layer is not None else None

    def _update_mask_data(self, new_data: np.ndarray) -> None:
        if self.labels_layer is not None:
            self.labels_layer.data = new_data
            self.labels_layer.refresh()

    def _on_remove_small_objects(self) -> None:
        mask = self._get_mask_data()
        if mask is not None:
            self._update_mask_data(remove_small_objects(mask, self.spin_min_size.value()))

    def _on_skeletonize(self) -> None:
        mask = self._get_mask_data()
        if mask is not None:
            self._update_mask_data(skeletonize_mask(mask))

    def _on_dilate(self) -> None:
        mask = self._get_mask_data()
        if mask is not None:
            self._update_mask_data(dilate_mask(mask, self.spin_radius.value()))

    def _on_erode(self) -> None:
        mask = self._get_mask_data()
        if mask is not None:
            self._update_mask_data(erode_mask(mask, self.spin_radius.value()))

    # ------------------------------------------------------------------
    # Quantification
    # ------------------------------------------------------------------

    def compute_laticifer_density(self) -> None:
        if self.labels_layer is None:
            QMessageBox.warning(self, "No mask", "Create or load a mask first.")
            return
        mask = np.asarray(self.labels_layer.data)
        dlg = QuantificationDialog(self, viewer=self.viewer, image_layer=self._get_image_layer())
        if dlg.exec_() != QDialog.Accepted:
            return
        params = dlg.get_params()
        if params["method"] == "pixel_ratio":
            self._run_pixel_ratio(mask)
        else:
            self._run_transect(mask, params)

    def _run_pixel_ratio(self, mask: np.ndarray) -> None:
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
        btn_whole  = msg.addButton("Whole image",        QMessageBox.AcceptRole)
        btn_tissue = msg.addButton("Tissue only (auto)", QMessageBox.AcceptRole)
        msg.setDefaultButton(btn_tissue)
        msg.exec_()

        if msg.clickedButton() == msg.button(QMessageBox.Cancel):
            return

        use_tissue = msg.clickedButton() == btn_tissue
        stats = analyze_density_pixel_ratio(mask, use_tissue_mask=use_tissue)

        if use_tissue:
            debug_mask = stats.get("debug_tissue_mask")
            if debug_mask is not None:
                if "Computed Tissue Area" in self.viewer.layers:
                    self.viewer.layers.remove("Computed Tissue Area")
                tissue_layer = self.viewer.add_labels(
                    debug_mask, name="Computed Tissue Area", opacity=0.3,
                    metadata={"is_debug_tissue_mask": True},
                )
                tissue_layer.editable = False
                if self.labels_layer is not None:
                    self.viewer.layers.selection.active = self.labels_layer
        else:
            if "Computed Tissue Area" in self.viewer.layers:
                self.viewer.layers.remove("Computed Tissue Area")

        self.last_density_ratio   = float(stats["pixel_ratio"])
        self.last_density_percent = float(stats["density_percentage"])
        denom = "Detected tissue area" if use_tissue else "Total image area"
        QMessageBox.information(
            self, "Laticifer density",
            f"Density: {self.last_density_percent:.2f}%\n(Laticifers / {denom})",
        )

    def _run_transect(self, mask: np.ndarray, params: dict) -> None:
        direction = str(params["direction"]).lower().strip()
        num_lines = max(1, int(params["num_lines"]))
        self.last_transect_num_lines = num_lines
        self.last_transect_direction = direction

        self._transect_ctrl.generate(
            mask_shape=tuple(mask.shape[:2]),
            num_lines=num_lines,
            direction=direction,
        )
        stats = self._transect_ctrl.recalculate(mask, show_points=bool(params["show_points"]))
        mean  = stats.get("mean_intersections_per_line", float("nan")) if stats else float("nan")
        self.last_transect_mean = mean if np.isfinite(mean) else None
        QMessageBox.information(self, "Transect Results", f"Mean intersections: {mean:.2f}")

    def _on_recalculate_density(self) -> None:
        """Recalculate using current transect geometry — lines are not regenerated."""
        if self.labels_layer is None:
            QMessageBox.warning(self, "No mask", "Create or load a mask first.")
            return
        mask  = np.asarray(self.labels_layer.data)
        stats = self._transect_ctrl.recalculate(mask, show_points=True)
        if stats is None:
            QMessageBox.warning(self, "Error", "Recalculation failed — no transects found.")
            return
        self.last_transect_num_lines = int(stats.get("num_lines", self.last_transect_num_lines))
        self.last_transect_direction = str(stats.get("direction", self.last_transect_direction))
        mean = stats.get("mean_intersections_per_line", float("nan"))
        self.last_transect_mean = mean if np.isfinite(mean) else None
        QMessageBox.information(self, "Transect Results", f"Mean intersections: {mean:.2f}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_annotation(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None or self.labels_layer is None:
            QMessageBox.warning(self, "Missing data", "Ensure both image and mask are present.")
            return
        root = ensure_dataset_root(self, image_layer, self.dataset_root)
        if root is None:
            return
        self.dataset_root = root
        save_annotation(
            parent_widget=self,
            image_layer=image_layer,
            labels_layer=self.labels_layer,
            dataset_root=self.dataset_root,
            initialized_from_model=self.initialized_from_model,
            transect_num_lines=self.last_transect_num_lines,
            transect_direction=self.last_transect_direction,
            transect_mean=self.last_transect_mean,
        )


# -----------------------------------------------------------------------------
#  WIDGET: Batch Processing
# -----------------------------------------------------------------------------

class BatchProcessingWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()

        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select input folder...")
        btn_in = QPushButton("Browse Input")
        btn_in.clicked.connect(self._select_input)
        layout.addWidget(QLabel("Input Images:"))
        layout.addWidget(self.input_dir_edit)
        layout.addWidget(btn_in)
        layout.addSpacing(10)

        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output folder...")
        btn_out = QPushButton("Browse Output")
        btn_out.clicked.connect(self._select_output)
        layout.addWidget(QLabel("Output Folder:"))
        layout.addWidget(self.output_dir_edit)
        layout.addWidget(btn_out)
        layout.addSpacing(20)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.status_lbl = QLabel("Ready")
        layout.addWidget(self.status_lbl)
        layout.addWidget(self.progress_bar)

        self.run_btn = QPushButton("Start Batch Processing")
        self.run_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        self.run_btn.clicked.connect(self._start_batch)
        layout.addWidget(self.run_btn)

        layout.addStretch()
        self.setLayout(layout)

    def _select_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input")
        if d:
            self.input_dir_edit.setText(d)

    def _select_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output")
        if d:
            self.output_dir_edit.setText(d)

    def _start_batch(self):
        in_d  = self.input_dir_edit.text()
        out_d = self.output_dir_edit.text()
        if not in_d or not out_d:
            QMessageBox.warning(self, "Error", "Select both folders.")
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")
        worker = self._run_batch_worker(in_d, out_d)
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
        QMessageBox.information(
            self, "Done",
            "Batch processing complete.\nSee 'batch_results.csv' in output folder.",
        )

    @thread_worker
    def _run_batch_worker(self, in_dir_str, out_dir_str):
        results = []
        for curr, total, name, row in run_batch_processing(in_dir_str, out_dir_str, num_lines=10):
            yield (curr, total, name)
            results.append(row)
        write_batch_csv(out_dir_str, results)


# -----------------------------------------------------------------------------
#  MAIN WIDGET
# -----------------------------------------------------------------------------

class LaticiferAnnotationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        layout = QVBoxLayout()
        tabs = QTabWidget()
        tabs.addTab(InteractiveEditorWidget(viewer), "Interactive Editor")
        tabs.addTab(BatchProcessingWidget(),          "Batch Processing")
        layout.addWidget(tabs)
        self.setLayout(layout)


# -----------------------------------------------------------------------------
#  Utility
# -----------------------------------------------------------------------------

def _h_rule() -> QFrame:
    """Thin horizontal separator line."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line