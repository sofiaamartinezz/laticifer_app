# ui.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import napari
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
)
from skimage import io as skio

from model import predict_laticifer_mask
from dataset import (
    ensure_dataset_root,
    infer_mask_path_for_image,
    load_mask_from_path,
    save_annotation as save_annotation_to_dataset,
)

# Make sure utils is importable if running as scripts
import sys
sys.path.append(str(Path(__file__).parent / "utils"))

from utils.preprocessing import apply_clahe
from utils.quantification import (
    analyze_density_pixel_ratio,
    analyze_density_transect,
)

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


class LaticiferAnnotationWidget(QWidget):
    """Napari dock widget with controls for laticifer annotation."""

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

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Use napari's 'Open File' to load images")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)

        # Image tools (CLAHE, etc.)
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

        # Quantification group
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


    # --------- Layer events ------------
    def _on_layer_inserted(self, event) -> None:
        layer = event.value

        if isinstance(layer, napari.layers.Labels):
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

    # -------- Helpers for image layers --------
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

    # ------------- Actions ----------------
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
        except FileNotFoundError as exc:
            QMessageBox.critical(
                self,
                "Model not found",
                f"Could not find the AI model:\n{exc}\n\nPlease ensure your model is saved in the models/ directory.",
            )
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

        if from_model:
            print("AI-generated mask ready for editing. Use label 1 for laticifers, 0 for background.")
        else:
            print("Mask ready for editing. Use label 1 for laticifers, 0 for background.")

    def load_existing_mask(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before loading a mask.")
            return

        img_shape = np.asarray(image_layer.data).shape[:2]

        mask_path = infer_mask_path_for_image(image_layer, self.dataset_root)

        if mask_path is None:
            mask_path_str, _ = QFileDialog.getOpenFileName(
                self,
                "Select mask file",
                "",
                "Image files (*.tif *.tiff *.png *.jpg *.jpeg)",
            )
            if not mask_path_str:
                return
            mask_path = Path(mask_path_str)

        mask_labels = load_mask_from_path(mask_path, img_shape)
        if mask_labels is None:
            QMessageBox.warning(
                self,
                "Load error",
                "Failed to load mask or shape mismatch with the current image.",
            )
            return

        self._add_labels_layer(mask_labels, from_model=False)
        print(
            f"[DEBUG] Loaded existing mask from {mask_path}, "
            f"unique={np.unique(mask_labels)}"
        )

    def enhance_current_image(self) -> None:
        """
        Apply preprocessing (CLAHE) to the original image and show it
        as a new grayscale image layer, without replacing the original.
        """
        image_layer = self._get_original_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before enhancing.")
            return

        data = np.asarray(image_layer.data)

        # apply_clahe now handles RGB/gray and float/uint8 internally
        enhanced = apply_clahe(data)

        enhanced_layer = self.viewer.add_image(
            enhanced,
            name=f"{image_layer.name} [enhanced]",
            blending="additive",
            colormap="gray",
        )

        enhanced_layer.metadata["is_preprocessed"] = True
        self.viewer.layers.selection.active = enhanced_layer

        print(
            f"[DEBUG] Enhanced image created from '{image_layer.name}' "
            f"-> '{enhanced_layer.name}', shape={enhanced.shape}, "
            f"dtype={enhanced.dtype}"
        )

    def compute_laticifer_density(self) -> None:
        """
        Interactive quantification:
        - Pixel ratio (area fraction)
        - Transect intersections (entry counts 0->1), with overlay layers
        """
        if self.labels_layer is None:
            QMessageBox.warning(self, "No mask", "Create or load a mask first.")
            return

        mask = np.asarray(self.labels_layer.data)
        if mask.size == 0:
            QMessageBox.warning(self, "Empty mask", "Mask is empty.")
            return

        # Try to infer pixel size from image layer metadata/scale (used only to
        # convert spacing -> num_lines if user selects spacing mode)
        default_px_um = 1.0
        image_layer = self._get_original_image_layer()

        if image_layer is not None:
            try:
                md = getattr(image_layer, "metadata", {}) or {}
                if "pixel_size_um" in md and md["pixel_size_um"] is not None:
                    default_px_um = float(md["pixel_size_um"])
                else:
                    if hasattr(image_layer, "scale") and image_layer.scale is not None:
                        sy = float(image_layer.scale[0]) if len(image_layer.scale) > 0 else 1.0
                        sx = float(image_layer.scale[1]) if len(image_layer.scale) > 1 else sy
                        default_px_um = float((abs(sy) + abs(sx)) / 2.0) if (sy != 0 and sx != 0) else 1.0
            except Exception:
                default_px_um = 1.0

        dlg = QuantificationDialog(self, viewer=self.viewer, image_layer=image_layer)

        if dlg.exec_() != QDialog.Accepted:
            return

        params = dlg.get_params()

        # ----- Pixel ratio -----
        if params["method"] == "pixel_ratio":
            stats = analyze_density_pixel_ratio(mask)

            ratio = float(stats["pixel_ratio"])
            percent = float(stats["density_percentage"])
            lt = int(stats["laticifer_pixels"])
            tot = int(stats["total_pixels"])

            self.last_density_ratio = ratio
            self.last_density_percent = percent

            QMessageBox.information(
                self,
                "Laticifer density (pixel ratio)",
                f"Densidad (pixel ratio): {percent:.2f}%\n"
                f"Laticifer pixels: {lt:,}\n"
                f"Total pixels: {tot:,}",
            )
            return

        # ----- Transect method -----
        direction = str(params["direction"]).lower().strip()
        num_lines = max(1, int(params.get("num_lines", 10)))

        self.last_transect_num_lines = num_lines
        self.last_transect_direction = direction

        stats, lines, pts = analyze_density_transect(
            mask,
            num_lines=num_lines,
            direction=direction,
            roi=None,
            start_frac=0.1,
            end_frac=0.9,
        )

        # Add/update shapes layer for lines
        lines_name = "Transect lines"
        if lines_name in self.viewer.layers:
            layer = self.viewer.layers[lines_name]
            try:
                layer.data = lines
            except Exception:
                self.viewer.layers.remove(layer)
                self.viewer.add_shapes(lines, name=lines_name, shape_type="line", edge_width=2)
        else:
            self.viewer.add_shapes(lines, name=lines_name, shape_type="line", edge_width=2)

        # Add/update points layer (optional)
        points_name = "Intersection points"
        show_points = bool(params.get("show_points", True))
        if show_points:
            if points_name in self.viewer.layers:
                p_layer = self.viewer.layers[points_name]
                try:
                    p_layer.data = pts
                except Exception:
                    self.viewer.layers.remove(p_layer)
                    self.viewer.add_points(pts, name=points_name, size=6)
            else:
                self.viewer.add_points(pts, name=points_name, size=6)
        else:
            if points_name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[points_name])

        # Build results message aligned with *your current* transect definition:
        # intersections are entries 0->1
        mean_all = stats.get("mean_intersections_per_line", float("nan"))
        std_all = stats.get("std_intersections_per_line", float("nan"))
        mean_h = stats.get("mean_horizontal_intersections", float("nan"))
        mean_v = stats.get("mean_vertical_intersections", float("nan"))

        msg = (
            "Laticifer density (transect method)\n\n"
            f"Orientation: {direction}\n"
            f"Number of transects: {num_lines}\n\n"
            "Mean number of laticifer interceptions per transect:\n"
            f"{mean_all:.2f} ± {std_all:.2f}\n\n"
            "Each interception corresponds to one laticifer crossing\n"
            "a sampling transect.\n\n"
            "Transects were evenly distributed across the image."
        )

        QMessageBox.information(self, "Laticifer density (transect)", msg)

    def save_annotation(self) -> None:
        image_layer = self._get_original_image_layer()
        if image_layer is None or self.labels_layer is None:
            QMessageBox.warning(self, "Missing data", "Ensure both image and mask are present.")
            return

        root = ensure_dataset_root(self, image_layer, self.dataset_root)
        if root is None:
            return
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

