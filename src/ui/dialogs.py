# ui/dialogs.py
"""
Qt dialogs used by the interactive editor.
Kept separate from widgets.py to make each file focused and testable.
"""
from __future__ import annotations

from typing import Optional

import napari
from qtpy.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFormLayout, QSpinBox, QVBoxLayout, QWidget,
)


class QuantificationDialog(QDialog):
    """
    Modal dialog that lets the user choose between pixel-ratio and
    transect-based density methods, and configure the relevant parameters.
    """

    def __init__(
        self,
        parent: QWidget,
        viewer: napari.Viewer,
        image_layer: Optional[napari.layers.Image],
    ):
        super().__init__(parent)
        self.viewer = viewer
        self.image_layer = image_layer
        self.setWindowTitle("Quantification settings")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Pixel ratio (area fraction)",
            "Transect lines (intersections)",
        ])
        form.addRow("Method", self.method_combo)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["horizontal", "vertical", "both"])
        form.addRow("Direction", self.direction_combo)

        self.num_lines_spin = QSpinBox()
        self.num_lines_spin.setRange(1, 10000)
        self.num_lines_spin.setValue(10)
        form.addRow("Number of lines", self.num_lines_spin)

        self.show_points_cb = QCheckBox("Show intersection points (0\u21921 entries)")
        self.show_points_cb.setChecked(True)
        form.addRow("", self.show_points_cb)

        layout.addLayout(form)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        self.method_combo.currentIndexChanged.connect(self._update_enabled)
        self._update_enabled()

    def _update_enabled(self) -> None:
        is_pixel = self.method_combo.currentIndex() == 0
        self.direction_combo.setEnabled(not is_pixel)
        self.num_lines_spin.setEnabled(not is_pixel)
        self.show_points_cb.setEnabled(not is_pixel)

    def get_params(self) -> dict:
        if self.method_combo.currentIndex() == 0:
            return {"method": "pixel_ratio"}
        return {
            "method":      "transect",
            "direction":   self.direction_combo.currentText(),
            "num_lines":   int(self.num_lines_spin.value()),
            "show_points": bool(self.show_points_cb.isChecked()),
        }