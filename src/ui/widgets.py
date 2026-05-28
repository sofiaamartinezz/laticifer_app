# ui/widgets.py
"""
LatexLens – redesigned dock widget.

Workflow tabs:
  1 · Prepare   – load image, CLAHE, scale calibration
  2 · Mask      – detect / load / draw / refine mask
  3 · Density   – pixel-ratio + editable transects, save annotation
  4 · Network   – skeleton analysis: Expansion / Branching / Thickness / Connectivity
  Batch         – unchanged batch-processing tab
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSizePolicy,
    QSpinBox, QTabWidget, QVBoxLayout, QWidget,
)

from utils.scalebar import detect_scalebar, ScalebarHints
from model.predictor import predict_laticifer_mask
from data.io import infer_mask_path, load_mask
from data.annotations import ensure_dataset_root, save_annotation
from data.batch import run_batch_processing, write_batch_csv
from utils.preprocessing import apply_clahe
from utils.quantification import analyze_density_pixel_ratio
from utils.postprocessing import (
    remove_small_objects, dilate_mask, erode_mask,
    fill_small_holes,
)
from utils.network_analysis import run_network_analysis, NetworkStats
from ui.dialogs import QuantificationDialog
from ui.transect_controller import TransectController


# ---------------------------------------------------------------------------
#  Style helpers
# ---------------------------------------------------------------------------

_PANEL_STYLE = """
QGroupBox {
    font-weight: bold;
    font-size: 11px;
    color: #aaa;
    border: none;
    border-top: 1px solid #333;
    margin-top: 8px;
    padding-top: 10px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 0px;
    top: 0px;
    padding: 0 2px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
QPushButton {
    padding: 6px 10px;
    border: 1px solid #444;
    border-radius: 4px;
    background: #2a2a2a;
    color: #ddd;
    font-size: 12px;
    text-align: left;
}
QPushButton:hover  { background: #333; border-color: #555; }
QPushButton:pressed { background: #222; }
QPushButton:disabled { color: #555; border-color: #333; }
QPushButton[accent="true"] {
    border-color: #2d6a2d;
    color: #5ca;
    background: rgba(93,202,165,0.07);
}
QPushButton[accent="true"]:hover { background: rgba(93,202,165,0.14); }
QPushButton[primary="true"] { background: #333; font-weight: bold; }
QLabel { font-size: 12px; color: #ccc; }
QLabel[muted="true"] { color: #777; font-style: italic; font-size: 11px; }
QLabel[warning="true"] { color: #d4a017; font-size: 11px; }
QLabel[metric_val="true"] { font-size: 16px; font-weight: bold; color: #eee; }
QLabel[metric_lbl="true"] { font-size: 10px; color: #888; }
QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {
    padding: 4px 6px;
    border: 1px solid #444;
    border-radius: 4px;
    background: #1e1e1e;
    color: #ddd;
    font-size: 12px;
}
QScrollArea { border: none; background: transparent; }
"""

_ACCENT_BTN_STYLE = "border-color:#2d6a2d;color:#5ca;background:rgba(93,202,165,0.07);"
_WARN_STYLE = "color:#d4a017;font-style:italic;font-size:11px;"
_MUTED_STYLE = "color:#777;font-style:italic;font-size:11px;"


def _h_rule() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setStyleSheet("color:#333;")
    return line


def _group(title: str, tooltip: str = "") -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    if tooltip:
        box.setToolTip(tooltip)
    lay = QVBoxLayout()
    lay.setSpacing(5)
    lay.setContentsMargins(0, 6, 0, 6)
    box.setLayout(lay)
    return box, lay


def _metric_pair(val_text: str, lbl_text: str) -> QWidget:
    """Small metric card: big value + small label."""
    w = QFrame()
    w.setStyleSheet("background:#252525;border-radius:4px;padding:4px;")
    lay = QVBoxLayout()
    lay.setSpacing(1)
    lay.setContentsMargins(8, 6, 8, 6)
    val = QLabel(val_text)
    val.setStyleSheet("font-size:16px;font-weight:bold;color:#eee;")
    lbl = QLabel(lbl_text)
    lbl.setStyleSheet("font-size:10px;color:#888;")
    lay.addWidget(val)
    lay.addWidget(lbl)
    w.setLayout(lay)
    return w, val, lbl


def _metric_grid(*pairs) -> QWidget:
    """2-column grid of metric cards."""
    w = QWidget()
    row = QHBoxLayout()
    row.setSpacing(5)
    row.setContentsMargins(0, 0, 0, 0)
    for card, _, _ in pairs:
        row.addWidget(card)
    w.setLayout(row)
    return w


def _small_label(style: str, text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(style)
    lbl.setWordWrap(True)
    return lbl


def _scroll_wrap(inner: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sc.setWidget(inner)
    return sc


# ---------------------------------------------------------------------------
#  Tab 1 – Prepare
# ---------------------------------------------------------------------------

class PrepareTab(QWidget):
    def __init__(self, editor: "InteractiveEditorWidget") -> None:
        super().__init__()
        self._ed = editor
        self._detected_bar_px: Optional[float] = None

        inner = QWidget()
        lay = QVBoxLayout()
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(2)

        # ── Image ────────────────────────────────────────────────────────────
        img_box, img_lay = _group("Image")
        self._img_lbl = QLabel("Open an image via File → Open or drag & drop")
        self._img_lbl.setStyleSheet(_MUTED_STYLE)
        self._img_lbl.setWordWrap(True)
        img_lay.addWidget(self._img_lbl)
        self.enhance_btn = QPushButton("⚡  Enhance contrast (CLAHE)")
        self.enhance_btn.setEnabled(False)
        self.enhance_btn.clicked.connect(editor.enhance_current_image)
        img_lay.addWidget(self.enhance_btn)
        lay.addWidget(img_box)

        # ── Scale calibration ────────────────────────────────────────────────
        scale_box, scale_lay = _group(
            "Scale calibration",
            "Set the pixel size so all measurements are shown in real units.\n"
            "Leave empty to work in pixels only."
        )

        self._scale_warn = QLabel("⚠  No scale set — results will be in pixels")
        self._scale_warn.setStyleSheet(_WARN_STYLE)
        self._scale_warn.setWordWrap(True)
        scale_lay.addWidget(self._scale_warn)

        # ── Auto-detect ───────────────────────────────────────────────────────
        auto_box, auto_lay = _group(
            "Auto-detect scale bar",
            "Detects a scale bar baked into the image pixels.\n"
            "Only works when the bar is part of the image data, not a\n"
            "napari viewer overlay."
        )

        # Color hint
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Bar color:"))
        self._color_combo = QComboBox()
        self._color_combo.addItems(["auto", "white / bright", "black / dark"])
        self._color_combo.setCurrentIndex(2)  # default: black / dark
        self._color_combo.setToolTip(
            "auto: tries white then black.\n"
            "white: bright bar on dark background (fluorescence).\n"
            "black: dark bar on bright background (brightfield)."
        )
        color_row.addWidget(self._color_combo)
        auto_lay.addLayout(color_row)

        # Region hint
        region_row = QHBoxLayout()
        region_row.addWidget(QLabel("Bar location:"))
        self._region_combo = QComboBox()
        self._region_combo.addItems(["bottom", "top", "anywhere"])
        self._region_combo.setToolTip(
            "Which part of the image to search.\n"
            "Limiting the region avoids false positives in the tissue."
        )
        region_row.addWidget(self._region_combo)
        auto_lay.addLayout(region_row)

        # Detect button
        self._detect_btn = QPushButton("🔍  Detect scale bar")
        self._detect_btn.setEnabled(False)
        self._detect_btn.clicked.connect(self._on_detect)
        auto_lay.addWidget(self._detect_btn)

        # Status label — shows "Detecting…", success, or failure message
        self._detect_status = QLabel("")
        self._detect_status.setStyleSheet("font-size:11px;")
        self._detect_status.setWordWrap(True)
        self._detect_status.setVisible(False)
        auto_lay.addWidget(self._detect_status)

        # Real-world length row — revealed only after a successful detection
        self._rw_widget = QWidget()
        rw_lay = QVBoxLayout()
        rw_lay.setContentsMargins(0, 4, 0, 0)
        rw_lay.setSpacing(4)
        rw_lay.addWidget(_small_label(
            "color:#aaa;font-size:11px;",
            "Enter the real-world length printed next to the scale bar:"
        ))
        rw_row = QHBoxLayout()
        self._bar_len_spin = QDoubleSpinBox()
        self._bar_len_spin.setRange(0.001, 999999.0)
        self._bar_len_spin.setDecimals(2)
        self._bar_len_spin.setValue(1.0)
        rw_row.addWidget(self._bar_len_spin)

        self._bar_unit_combo = QComboBox()
        self._bar_unit_combo.addItems(["µm", "nm", "mm"])
        self._bar_unit_combo.setCurrentText("mm")
        rw_row.addWidget(self._bar_unit_combo)
        apply_auto_btn = QPushButton("Apply scale")
        apply_auto_btn.setStyleSheet(_ACCENT_BTN_STYLE)
        apply_auto_btn.clicked.connect(self._apply_from_detection)
        rw_row.addWidget(apply_auto_btn)
        rw_lay.addLayout(rw_row)
        self._rw_widget.setLayout(rw_lay)
        self._rw_widget.setVisible(False)
        auto_lay.addWidget(self._rw_widget)

        # ── Manual entry ──────────────────────────────────────────────────────
        manual_box, manual_lay = _group(
            "Enter pixel size directly",
            "Use this when you know the pixel size from microscope metadata."
        )
        manual_lay.addWidget(_small_label(
            "color:#666;font-size:10px;",
            "Check microscope software, image metadata, or calibration slide."
        ))
        man_row = QHBoxLayout()
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0, 9999.0)
        self._scale_spin.setDecimals(4)
        self._scale_spin.setValue(0.9302)
        self._scale_spin.setSpecialValueText("—")
        self._scale_spin.setToolTip("e.g. 0.65 means 1 px = 0.65 µm")
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["µm/px", "nm/px"])
        apply_man_btn = QPushButton("Apply")
        apply_man_btn.setMaximumWidth(55)
        apply_man_btn.clicked.connect(self._apply_manual)
        man_row.addWidget(self._scale_spin)
        man_row.addWidget(self._unit_combo)
        man_row.addWidget(apply_man_btn)
        manual_lay.addLayout(man_row)

        scale_lay.addWidget(manual_box)
        scale_lay.addWidget(auto_box)

        # Keep reset visually separated to avoid accidental clicks
        scale_lay.addSpacing(14)

        reset_scale_btn = QPushButton("↺  Reset scale / use pixels only")
        reset_scale_btn.setToolTip("Remove the current scale and return to pixel-only measurements.")
        reset_scale_btn.setStyleSheet(
            "color:#d4a017;"
            "border-color:#5a4610;"
            "background:rgba(212,160,23,0.06);"
        )
        reset_scale_btn.clicked.connect(self._reset_scale)
        scale_lay.addWidget(reset_scale_btn)

        scale_lay.addSpacing(6)

        # Active scale confirmation label
        self._scale_active_lbl = QLabel("")
        self._scale_active_lbl.setStyleSheet(
            "color:#5ca;font-size:11px;font-weight:bold;"
        )
        self._scale_active_lbl.setWordWrap(True)
        scale_lay.addWidget(self._scale_active_lbl)
        lay.addWidget(scale_box)

        lay.addStretch()
        inner.setLayout(lay)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_wrap(inner))
        self.setLayout(outer)

    # ── Public ───────────────────────────────────────────────────────────────

    def set_image_info(self, name: str, shape: tuple) -> None:
        self._img_lbl.setText(f"{name}  ·  {shape[1]} × {shape[0]} px")
        self._img_lbl.setStyleSheet("color:#5ca;font-size:11px;")
        self.enhance_btn.setEnabled(True)
        self._detect_btn.setEnabled(True)
        # Clear any previous detection result when a new image is loaded
        self._detected_bar_px = None
        self._detect_status.setVisible(False)
        self._rw_widget.setVisible(False)

    # ── Auto-detect ───────────────────────────────────────────────────────────

    def _on_detect(self) -> None:
        image_layer = self._ed._get_image_layer()
        if image_layer is None:
            return

        # Map UI selections to ScalebarHints fields
        color_map  = {0: "auto", 1: "white", 2: "black"}
        region_map = {
            "bottom": "bottom", "top": "top",
            "anywhere": "any",
        }
        hints = ScalebarHints(
            color=color_map.get(self._color_combo.currentIndex(), "auto"),
            region=region_map.get(self._region_combo.currentText(), "bottom"),
        )

        # Show "Detecting…" immediately before the (potentially slow) detection
        self._detect_status.setText("⏳  Detecting…")
        self._detect_status.setStyleSheet("color:#aaa;font-size:11px;")
        self._detect_status.setVisible(True)
        self._detect_btn.setEnabled(False)
        from qtpy.QtWidgets import QApplication
        QApplication.processEvents()

        try:
            img = np.asarray(image_layer.data)
            # Fast path: only search the selected region instead of the full image
            H = img.shape[0]
            region = self._region_combo.currentText()

            y_offset = 0
            img_search = img

            if region == "bottom":
                y_offset = int(H * 0.75)      # search only bottom 25%
                img_search = img[y_offset:, ...]
            elif region == "top":
                y_offset = 0
                img_search = img[:int(H * 0.25), ...]
            else:
                y_offset = 0
                img_search = img

            result, message = detect_scalebar(img_search, hints)

            # If detection was done on a crop, shift coordinates back to full image
            if result is not None and y_offset > 0:
                result.y0 += y_offset
                result.y1 += y_offset
        except Exception as exc:
            self._detect_status.setText(f"❌  Error during detection: {exc}")
            self._detect_status.setStyleSheet(_WARN_STYLE)
            self._rw_widget.setVisible(False)
            return
        finally:
            self._detect_btn.setEnabled(True)

        if result is None:
            # Detection failed — show the detailed reason so the user knows
            # exactly what to change (color, region, or switch to manual)
            self._detect_status.setText(
                f"❌  Not detected.\n\n{message}\n\n"
                "Try changing the bar color or location hint, "
                "or enter the pixel size manually below."
            )
            self._detect_status.setStyleSheet(_WARN_STYLE)
            self._rw_widget.setVisible(False)
            return

        # ── Success ───────────────────────────────────────────────────────────
        self._detected_bar_px = float(result.width_px)
        self._detect_status.setText(
            f"✓  {result.color.capitalize()} bar found: "
            f"{result.width_px} px wide.\n"
            "Enter its real-world length below and click Apply scale."
        )
        self._detect_status.setStyleSheet("color:#5ca;font-size:11px;")
        self._rw_widget.setVisible(True)

        # Draw a highlighting rectangle in the viewer so the user can confirm
        # that the correct bar was detected
        self._show_detection_layer(result, image_layer)

    def _show_detection_layer(self, result, image_layer) -> None:
        """Add a Shapes layer that outlines the detected scale bar."""
        LAYER_NAME = "Detected scale bar"
        viewer = self._ed.viewer

        # Remove any previous detection layer
        if LAYER_NAME in viewer.layers:
            viewer.layers.remove(LAYER_NAME)

        # Build a rectangle: [[y0,x0],[y0,x1],[y1,x1],[y1,x0]]
        rect = np.array([
            [result.y0, result.x0],
            [result.y0, result.x1],
            [result.y1, result.x1],
            [result.y1, result.x0],
        ], dtype=float)

        # Also draw the centre-line of the bar as a line shape
        y_mid = (result.y0 + result.y1) / 2.0
        line  = np.array([
            [y_mid, float(result.x0)],
            [y_mid, float(result.x1)],
        ], dtype=float)

        shapes_data  = [rect, line]
        shape_types  = ["rectangle", "line"]
        edge_colors  = ["#ffdd00", "#ffdd00"]   # bright yellow — visible on any background
        face_colors  = ["transparent", "transparent"]

        layer = viewer.add_shapes(
            shapes_data,
            shape_type=shape_types,
            name=LAYER_NAME,
            edge_color=edge_colors,
            face_color=face_colors,
            edge_width=2,
            opacity=0.9,
        )
        layer.editable = False
        # Bring the image back to the front after adding the overlay
        viewer.layers.selection.active = image_layer

    # ── Apply scale from detection ────────────────────────────────────────────

    def _apply_from_detection(self) -> None:
        if not self._detected_bar_px or self._detected_bar_px <= 0:
            QMessageBox.warning(
                self, "No detection",
                "Run the auto-detector first, or enter the pixel size manually."
            )
            return
        real_len = self._bar_len_spin.value()
        unit     = self._bar_unit_combo.currentText()
        unit_to_um = {"µm": 1.0, "nm": 0.001, "mm": 1000.0}
        real_len_um = real_len * unit_to_um.get(unit, 1.0)
        um_per_px   = real_len_um / self._detected_bar_px
        self._commit_scale(
            um_per_px,
            f"{real_len} {unit} bar ({self._detected_bar_px:.1f} px) "
            f"→ {um_per_px:.4f} µm/px",
        )

    # ── Manual entry ─────────────────────────────────────────────────────────

    def _apply_manual(self) -> None:
        val  = self._scale_spin.value()
        unit = self._unit_combo.currentText()
        if val <= 0:
            self._ed.um_per_px = None
            self._scale_active_lbl.setText("")
            self._scale_warn.setVisible(True)
            return
        factor = val if "µm" in unit else val / 1000.0
        self._commit_scale(factor, f"1 px = {val} {unit}")

    def _reset_scale(self) -> None:
        self._ed.um_per_px = None
        self._scale_spin.setValue(0.9302)
        self._unit_combo.setCurrentText("µm/px")
        self._scale_active_lbl.setText("")
        self._scale_warn.setVisible(True)

        self._detected_bar_px = None
        self._detect_status.setVisible(False)
        self._rw_widget.setVisible(False)

        if "Detected scale bar" in self._ed.viewer.layers:
            self._ed.viewer.layers.remove("Detected scale bar")

    # ── Shared ───────────────────────────────────────────────────────────────

    def _commit_scale(self, um_per_px: float, description: str) -> None:
        self._ed.um_per_px = um_per_px
        self._scale_active_lbl.setText(f"✓ Scale active — {description}")
        self._scale_warn.setVisible(False)


# ---------------------------------------------------------------------------
#  Tab 2 – Mask
# ---------------------------------------------------------------------------

class MaskTab(QWidget):
    def __init__(self, editor: "InteractiveEditorWidget") -> None:
        super().__init__()
        self._ed = editor
        inner = QWidget()
        lay = QVBoxLayout()
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(2)

        # Source
        src_box, src_lay = _group("Mask source")
        self.auto_btn = QPushButton("🤖  Auto-detect with AI model")
        self.auto_btn.setStyleSheet(_ACCENT_BTN_STYLE)
        self.auto_btn.setEnabled(False)
        self.auto_btn.clicked.connect(editor.auto_generate_mask)
        src_lay.addWidget(self.auto_btn)

        self._ai_lbl = QLabel("")
        self._ai_lbl.setStyleSheet(_MUTED_STYLE)
        self._ai_lbl.setVisible(False)
        src_lay.addWidget(self._ai_lbl)

        src_lay.addWidget(_h_rule())

        self.load_btn = QPushButton("📂  Load existing mask")
        self.load_btn.setEnabled(False)
        self.load_btn.clicked.connect(editor.load_existing_mask)
        src_lay.addWidget(self.load_btn)

        self.empty_btn = QPushButton("✏️  Create empty mask (draw manually)")
        self.empty_btn.setEnabled(False)
        self.empty_btn.clicked.connect(editor.create_empty_mask)
        src_lay.addWidget(self.empty_btn)
        lay.addWidget(src_box)

        # Refinement
        ref_box, ref_lay = _group(
            "Refine mask",
            "Remove artefacts and close gaps in the detected network.\n"
            "Changes are applied directly to the mask layer."
        )
        # Remove small
        rrow = QHBoxLayout()
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 10000)
        self.spin_min.setValue(100)
        self.spin_min.setSuffix(" px")
        self.rm_small_btn = QPushButton("Remove small objects")
        self.rm_small_btn.setEnabled(False)
        self.rm_small_btn.clicked.connect(self._on_rm_small)
        rrow.addWidget(self.spin_min)
        rrow.addWidget(self.rm_small_btn)
        ref_lay.addLayout(rrow)

        # Fill holes
        frow = QHBoxLayout()
        self.spin_hole = QSpinBox()
        self.spin_hole.setRange(1, 50000)
        self.spin_hole.setValue(500)
        self.spin_hole.setSuffix(" px²")
        self.fill_holes_btn = QPushButton("Fill small holes")
        self.fill_holes_btn.setEnabled(False)
        self.fill_holes_btn.clicked.connect(self._on_fill_holes)
        frow.addWidget(self.spin_hole)
        frow.addWidget(self.fill_holes_btn)
        ref_lay.addLayout(frow)

        ref_lay.addWidget(_h_rule())
        ref_lay.addWidget(QLabel("Morphological operations:"))

        mrow = QHBoxLayout()
        self.spin_radius = QSpinBox()
        self.spin_radius.setRange(1, 20)
        self.spin_radius.setValue(1)
        self.spin_radius.setSuffix(" px")
        self.erode_btn = QPushButton("Erode")
        self.erode_btn.setEnabled(False)
        self.erode_btn.clicked.connect(self._on_erode)
        self.dilate_btn = QPushButton("Dilate")
        self.dilate_btn.setEnabled(False)
        self.dilate_btn.clicked.connect(self._on_dilate)
        mrow.addWidget(QLabel("Radius:"))
        mrow.addWidget(self.spin_radius)
        mrow.addWidget(self.erode_btn)
        mrow.addWidget(self.dilate_btn)
        ref_lay.addLayout(mrow)
        lay.addWidget(ref_box)

        lay.addStretch()
        inner.setLayout(lay)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_wrap(inner))
        self.setLayout(outer)

    def update_states(self, has_image: bool, has_mask: bool) -> None:
        self.auto_btn.setEnabled(has_image)
        self.load_btn.setEnabled(has_image and not has_mask)
        self.empty_btn.setEnabled(has_image and not has_mask)
        for b in (self.rm_small_btn, self.fill_holes_btn,
                  self.erode_btn, self.dilate_btn):
            b.setEnabled(has_mask)

    def set_ai_status(self, text: str, visible: bool) -> None:
        self._ai_lbl.setText(text)
        self._ai_lbl.setVisible(visible)

    def _on_rm_small(self) -> None:
        mask = self._ed._get_mask_data()
        if mask is not None:
            self._ed._update_mask_data(remove_small_objects(mask, self.spin_min.value()))

    def _on_fill_holes(self) -> None:
        mask = self._ed._get_mask_data()
        if mask is not None:
            self._ed._update_mask_data(fill_small_holes(mask, self.spin_hole.value()))

    def _on_erode(self) -> None:
        mask = self._ed._get_mask_data()
        if mask is not None:
            self._ed._update_mask_data(erode_mask(mask, self.spin_radius.value()))

    def _on_dilate(self) -> None:
        mask = self._ed._get_mask_data()
        if mask is not None:
            self._ed._update_mask_data(dilate_mask(mask, self.spin_radius.value()))


# ---------------------------------------------------------------------------
#  Tab 3 – Density
# ---------------------------------------------------------------------------

class DensityTab(QWidget):
    def __init__(self, editor: "InteractiveEditorWidget") -> None:
        super().__init__()
        self._ed = editor
        inner = QWidget()
        lay = QVBoxLayout()
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(2)

        # Pixel-ratio density
        px_box, px_lay = _group(
            "Pixel density",
            "Fraction of pixels classified as laticifer.\n"
            "Choose 'Tissue area' to exclude background."
        )
        area_row = QHBoxLayout()
        area_row.addWidget(QLabel("Reference area:"))
        self._area_combo = QComboBox()
        self._area_combo.addItems(["Tissue area (auto)", "Whole image"])
        area_row.addWidget(self._area_combo)
        px_lay.addLayout(area_row)

        self.px_btn = QPushButton("Calculate pixel ratio")
        self.px_btn.setEnabled(False)
        self.px_btn.clicked.connect(self._run_pixel_ratio)
        px_lay.addWidget(self.px_btn)

        # result cards
        self._px_cov_card, self._px_cov_val, _ = _metric_pair("—", "Coverage (%)")
        self._px_area_card, self._px_area_val, _ = _metric_pair("—", "Laticifer area")
        px_lay.addWidget(_metric_grid(
            (self._px_cov_card, None, None),
            (self._px_area_card, None, None),
        ))
        lay.addWidget(px_box)

        # Transect
        tr_box, tr_lay = _group(
            "Transect analysis",
            "Count laticifer boundary crossings along scan lines.\n"
            "Reflects structural density independent of pixel brightness."
        )
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Direction:"))
        self._dir_combo = QComboBox()
        self._dir_combo.addItems(["both", "horizontal", "vertical"])
        self._dir_combo.setCurrentText("horizontal")
        dir_row.addWidget(self._dir_combo)
        tr_lay.addLayout(dir_row)

        n_row = QHBoxLayout()
        n_row.addWidget(QLabel("Lines:"))
        self._nlines_spin = QSpinBox()
        self._nlines_spin.setRange(1, 10000)
        self._nlines_spin.setValue(10)
        n_row.addWidget(self._nlines_spin)
        tr_lay.addLayout(n_row)

        self.gen_btn = QPushButton("▶  Generate transects")
        self.gen_btn.setStyleSheet(_ACCENT_BTN_STYLE)
        self.gen_btn.setEnabled(False)
        self.gen_btn.clicked.connect(self._run_transect)
        tr_lay.addWidget(self.gen_btn)

        self._pending_lbl = QLabel("⚠  Transects edited — recalculate to update")
        self._pending_lbl.setStyleSheet(_WARN_STYLE)
        self._pending_lbl.setWordWrap(True)
        self._pending_lbl.setVisible(False)
        tr_lay.addWidget(self._pending_lbl)

        self._recalc_btn = QPushButton("↻  Recalculate with edited transects")
        self._recalc_btn.setEnabled(False)
        self._recalc_btn.setToolTip("Recalculate using current line geometry — lines are NOT regenerated.")
        self._recalc_btn.clicked.connect(self._on_recalculate)
        tr_lay.addWidget(self._recalc_btn)

        hint = QLabel("To delete a line: select 'Transect lines' layer → click line → Delete.")
        hint.setStyleSheet("color:#666;font-size:10px;")
        hint.setWordWrap(True)
        tr_lay.addWidget(hint)

        self._tr_mean_card, self._tr_mean_val, _ = _metric_pair("—", "Mean intersections / line")
        self._tr_std_card,  self._tr_std_val,  _ = _metric_pair("—", "Std dev")
        tr_lay.addWidget(_metric_grid(
            (self._tr_mean_card, None, None),
            (self._tr_std_card,  None, None),
        ))
        lay.addWidget(tr_box)

        # Save
        sav_box, sav_lay = _group("Save annotation")
        self.save_btn = QPushButton("💾  Save annotation + metrics")
        self.save_btn.setStyleSheet(_ACCENT_BTN_STYLE)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(editor.save_annotation)
        sav_lay.addWidget(self.save_btn)
        lay.addWidget(sav_box)

        lay.addStretch()
        inner.setLayout(lay)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_wrap(inner))
        self.setLayout(outer)

    def update_states(self, has_mask: bool, has_image: bool) -> None:
        self.px_btn.setEnabled(has_mask)
        self.gen_btn.setEnabled(has_mask)
        self.save_btn.setEnabled(has_mask and has_image)

    def refresh_transect_ui(self, pending: bool) -> None:
        self._pending_lbl.setVisible(pending)
        self._recalc_btn.setEnabled(pending)

    def show_pixel_results(self, coverage_pct: float, area_px: int, um_per_px: Optional[float]) -> None:
        self._px_cov_val.setText(f"{coverage_pct:.2f}%")
        if um_per_px and um_per_px > 0:
            area_um = area_px * (um_per_px ** 2)
            self._px_area_val.setText(f"{area_um:.1f} µm²")
        else:
            self._px_area_val.setText(f"{area_px:,} px")

    def show_transect_results(self, mean: float, std: float) -> None:
        if math.isfinite(mean):
            self._tr_mean_val.setText(f"{mean:.2f}")
            self._tr_std_val.setText(f"±{std:.2f}")
        else:
            self._tr_mean_val.setText("—")
            self._tr_std_val.setText("—")

    def _run_pixel_ratio(self) -> None:
        self._ed._run_pixel_ratio_from_tab()

    def _run_transect(self) -> None:
        self._ed._run_transect_from_tab(
            direction=self._dir_combo.currentText(),
            num_lines=self._nlines_spin.value(),
        )

    def _on_recalculate(self) -> None:
        self._ed._on_recalculate_density()


# ---------------------------------------------------------------------------
#  Tab 4 – Network
# ---------------------------------------------------------------------------

class NetworkTab(QWidget):
    def __init__(self, editor: "InteractiveEditorWidget") -> None:
        super().__init__()
        self._ed = editor
        inner = QWidget()
        lay = QVBoxLayout()
        lay.setAlignment(Qt.AlignTop)
        lay.setSpacing(2)

        # Run button
        run_box, run_lay = _group(
            "Skeleton analysis",
            "Reduces the mask to a 1-px centreline and extracts the full\n"
            "topology of the laticifer network."
        )
        run_lay.addWidget(QLabel(
            "Detects bifurcation nodes, traces individual branches,\n"
            "measures diameters and angles from the current mask.",
        ))
        self.run_btn = QPushButton("▶  Run skeleton analysis")
        self.run_btn.setStyleSheet(_ACCENT_BTN_STYLE)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run_analysis)
        run_lay.addWidget(self.run_btn)
        self._running_lbl = QLabel("")
        self._running_lbl.setStyleSheet(_MUTED_STYLE)
        self._running_lbl.setVisible(False)
        run_lay.addWidget(self._running_lbl)
        lay.addWidget(run_box)

        # --- Expansion ---
        exp_box, exp_lay = _group(
            "Expansion",
            "How far the laticifer network extends through the tissue.\n"
            "Total skeleton length sums all centreline pixels."
        )
        self._skel_len_card, self._skel_len_val, self._skel_len_lbl = _metric_pair("—", "Total length (px)")
        self._skel_um_card,  self._skel_um_val,  self._skel_um_lbl  = _metric_pair("—", "Total length (µm)")
        grid = _metric_grid(
            (self._skel_len_card, None, None),
            (self._skel_um_card,  None, None),
        )
        exp_lay.addWidget(grid)
        self._cc_lbl = QLabel("Connected components: —")
        self._cc_lbl.setStyleSheet("color:#999;font-size:11px;")
        exp_lay.addWidget(self._cc_lbl)
        lay.addWidget(exp_box)

        # --- Branching ---
        br_box, br_lay = _group(
            "Branching",
            "How the network divides. More bifurcations = more complex network.\n"
            "Bifurcation angle: opening angle at each branching point."
        )
        self._bif_card,    self._bif_val,    _ = _metric_pair("—", "Bifurcations")
        self._ep_card,     self._ep_val,     _ = _metric_pair("—", "Endpoints")
        self._br_card,     self._br_val,     _ = _metric_pair("—", "Branches")
        self._brlen_card,  self._brlen_val,  _ = _metric_pair("—", "Mean branch length")
        self._ang_card,    self._ang_val,    _ = _metric_pair("—", "Mean angle (°)")
        self._angstd_card, self._angstd_val, _ = _metric_pair("—", "Angle std (°)")

        br_lay.addWidget(_metric_grid(
            (self._bif_card, None, None),
            (self._ep_card,  None, None),
        ))
        br_lay.addWidget(_metric_grid(
            (self._br_card,    None, None),
            (self._brlen_card, None, None),
        ))
        br_lay.addWidget(_metric_grid(
            (self._ang_card,    None, None),
            (self._angstd_card, None, None),
        ))

        self.show_bif_btn    = QPushButton("Show bifurcations + endpoints in viewer")
        self.show_bif_btn.setEnabled(False)
        self.show_bif_btn.clicked.connect(self._show_bifurcations)
        br_lay.addWidget(self.show_bif_btn)

        self.hist_angle_btn = QPushButton("📊  Angle histogram")
        self.hist_angle_btn.setEnabled(False)
        self.hist_angle_btn.clicked.connect(self._show_angle_histogram)
        br_lay.addWidget(self.hist_angle_btn)

        self.hist_brlen_btn = QPushButton("📊  Branch length histogram")
        self.hist_brlen_btn.setEnabled(False)
        self.hist_brlen_btn.clicked.connect(self._show_brlen_histogram)
        br_lay.addWidget(self.hist_brlen_btn)
        lay.addWidget(br_box)

        # --- Thickness ---
        th_box, th_lay = _group(
            "Thickness",
            "Estimated laticifer diameter at each skeleton point via\n"
            "the distance transform of the mask (diameter = 2 × local radius)."
        )
        self._diam_card, self._diam_val, _ = _metric_pair("—", "Mean diameter")
        self._dstd_card, self._dstd_val, _ = _metric_pair("—", "Std dev")
        self._dmed_card, self._dmed_val, _ = _metric_pair("—", "Median diameter")
        th_lay.addWidget(_metric_grid(
            (self._diam_card, None, None),
            (self._dstd_card, None, None),
        ))
        th_lay.addWidget(_metric_grid(
            (self._dmed_card, None, None),
        ))

        self.show_diam_btn = QPushButton("Show diameter map in viewer")
        self.show_diam_btn.setEnabled(False)
        self.show_diam_btn.clicked.connect(self._show_diameter_map)
        th_lay.addWidget(self.show_diam_btn)

        self.hist_diam_btn = QPushButton("📊  Diameter histogram")
        self.hist_diam_btn.setEnabled(False)
        self.hist_diam_btn.clicked.connect(self._show_diameter_histogram)
        th_lay.addWidget(self.hist_diam_btn)
        lay.addWidget(th_box)

        # --- Connectivity ---
        co_box, co_lay = _group(
            "Connectivity",
            "Branch-to-node ratio: higher values indicate a more interconnected network.\n"
            "Ratio > 1 means more branches than nodes (branchy network)."
        )
        self._bnr_card, self._bnr_val, _ = _metric_pair("—", "Branch / node ratio")
        co_lay.addWidget(_metric_grid(
            (self._bnr_card, None, None),
        ))
        lay.addWidget(co_box)

        lay.addStretch()
        inner.setLayout(lay)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(_scroll_wrap(inner))
        self.setLayout(outer)

        self._geom = None  
        self._stats: Optional[NetworkStats] = None

    def update_states(self, has_mask: bool) -> None:
        self.run_btn.setEnabled(has_mask)

    def _run_analysis(self) -> None:
        mask = self._ed._get_mask_data()
        if mask is None:
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…")
        self._running_lbl.setText("⏳  Skeletonizing and computing network metrics…")
        self._running_lbl.setVisible(True)
        um = self._ed.um_per_px

        @thread_worker
        def _worker():
            return run_network_analysis(mask, um_per_px=um)

        w = _worker()
        w.returned.connect(self._on_analysis_done)
        w.errored.connect(self._on_analysis_error)
        w.start()

    def _on_analysis_done(self, result) -> None:
        stats, geom = result
        self._stats = stats
        self._geom  = geom
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Run skeleton analysis")
        self._running_lbl.setVisible(False)
        self._populate(stats)
        # Auto-display nodes so the user immediately sees something useful
        self._show_bifurcations()

    def _on_analysis_error(self, exc_info) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Run skeleton analysis")
        self._running_lbl.setVisible(False)
        QMessageBox.critical(self._ed, "Analysis error",
                             f"Failed to run network analysis:\n{exc_info[1]}")

    def _fmt_px_um(self, px_val: float, um_per_px: Optional[float], decimals: int = 1) -> str:
        if not math.isfinite(px_val):
            return "—"
        if um_per_px and um_per_px > 0:
            return f"{px_val * um_per_px:.{decimals}f} µm"
        return f"{px_val:.{decimals}f} px"

    def _populate(self, s: NetworkStats) -> None:
        um = s.um_per_px

        # Expansion
        self._skel_len_val.setText(f"{s.total_skeleton_length_px:,.0f} px")
        if s.total_skeleton_length_um is not None:
            self._skel_um_val.setText(f"{s.total_skeleton_length_um:,.1f} µm")
        else:
            self._skel_um_val.setText("— (no scale)")
        self._cc_lbl.setText(f"Connected components: {s.connected_components}")

        # Branching
        self._bif_val.setText(str(s.bifurcation_count))
        self._ep_val.setText(str(s.endpoint_count))
        self._br_val.setText(str(s.branch_count))
        self._brlen_val.setText(self._fmt_px_um(s.mean_branch_length_px, um))
        ang = s.mean_bifurcation_angle_deg
        self._ang_val.setText(f"{ang:.1f}°" if math.isfinite(ang) else "—")
        astd = s.std_bifurcation_angle_deg
        self._angstd_val.setText(f"±{astd:.1f}°" if math.isfinite(astd) else "—")

        # Thickness
        self._diam_val.setText(self._fmt_px_um(s.mean_diameter_px, um))
        self._dstd_val.setText(self._fmt_px_um(s.std_diameter_px, um))
        self._dmed_val.setText(self._fmt_px_um(s.median_diameter_px, um))

        # Connectivity
        bnr = s.branch_node_ratio
        self._bnr_val.setText(f"{bnr:.2f}" if math.isfinite(bnr) else "—")

        # Enable secondary buttons
        for b in (self.show_bif_btn, self.show_diam_btn,
                  self.hist_angle_btn, self.hist_brlen_btn, self.hist_diam_btn):
            b.setEnabled(True)

    # Viewer overlays ----------------------------------------------------------

    def _show_bifurcations(self) -> None:
        if self._geom is None:
            return
        viewer = self._ed.viewer

        # Bifurcation nodes
        BIF_NAME = "Bifurcation nodes"
        if BIF_NAME in viewer.layers:
            viewer.layers.remove(BIF_NAME)

        bif_pts = self._geom.bifurcation_points
        if bif_pts.shape[0] > 0:
            viewer.add_points(
                bif_pts,
                name=BIF_NAME,
                size=18,
                face_color="#ff2d78",
                border_color="#ffffff",
                border_width=2,
                border_width_is_relative=False,
                opacity=0.95,
                symbol="disc",
            )

        # Endpoints
        EP_NAME = "Laticifer endpoints"
        if EP_NAME in viewer.layers:
            viewer.layers.remove(EP_NAME)

        ep_pts = self._geom.endpoint_points
        if ep_pts.shape[0] > 0:
            viewer.add_points(
                ep_pts,
                name=EP_NAME,
                size=14,
                face_color="#00cfff",
                border_color="#ffffff",
                border_width=1,
                border_width_is_relative=False,
                opacity=0.85,
                symbol="triangle_up",
            )

    def _show_diameter_map(self) -> None:
        if self._geom is None:
            return
        viewer = self._ed.viewer
        NAME = "Diameter map"
        if NAME in viewer.layers:
            viewer.layers.remove(NAME)
        dmap = self._geom.diameter_map
        if dmap.size > 0:
            viewer.add_image(dmap, name=NAME, colormap="inferno", opacity=0.85,
                             blending="additive")

    # Histograms ---------------------------------------------------------------

    def _show_histogram(self, data: np.ndarray, title: str, xlabel: str) -> None:
        """Launch a minimal histogram dialog using matplotlib."""
        if data.size == 0:
            QMessageBox.information(self._ed, title, "No data to display.")
            return
        try:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
            ax.hist(data, bins=40, color="#5ca882", edgecolor="#2d6a2d")
            ax.set_title(title, fontsize=13)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.spines[["top", "right"]].set_visible(False)
            plt.show()
        except ImportError:
            QMessageBox.information(self._ed, title,
                                    "Install matplotlib to view histograms:\n  pip install matplotlib")

    def _show_angle_histogram(self) -> None:
        if self._geom is None:
            return
        self._show_histogram(
            self._geom.bifurcation_angles_deg,
            "Bifurcation angle distribution",
            "Angle (°)",
        )

    def _show_brlen_histogram(self) -> None:
        if self._geom is None:
            return
        um = self._stats.um_per_px if self._stats else None
        data = self._geom.branch_lengths_px
        xlabel = "Length (px)"
        if um and um > 0:
            data = data * um
            xlabel = "Length (µm)"
        self._show_histogram(data, "Branch length distribution", xlabel)

    def _show_diameter_histogram(self) -> None:
        if self._geom is None:
            return
        dmap = self._geom.diameter_map
        skel_vals = dmap[dmap > 0]
        um = self._stats.um_per_px if self._stats else None
        xlabel = "Diameter (px)"
        if um and um > 0:
            skel_vals = skel_vals * um
            xlabel = "Diameter (µm)"
        self._show_histogram(skel_vals, "Laticifer diameter distribution", xlabel)


# ---------------------------------------------------------------------------
#  Interactive Editor (coordinates all tabs)
# ---------------------------------------------------------------------------

class InteractiveEditorWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        self.viewer = viewer
        self.labels_layer: Optional[napari.layers.Labels] = None
        self.dataset_root: Optional[Path] = None
        self.initialized_from_model: bool = False
        self.base_image_layer: Optional[napari.layers.Image] = None
        self.um_per_px: Optional[float] = None

        self.last_transect_num_lines: int = 10
        self.last_transect_direction: str = "horizontal"
        self.last_transect_mean: Optional[float] = None

        self._transect_ctrl = TransectController(
            viewer=self.viewer,
            on_state_change=self._refresh_transect_ui,
        )

        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        self._build_ui()

    def _build_ui(self) -> None:
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setDocumentMode(True)
        self._tabs.setStyleSheet("""
            QTabBar::tab { padding: 7px 12px; font-size: 11px; min-width: 60px; }
            QTabBar::tab:selected { font-weight: bold; }
        """)

        self.tab_prepare = PrepareTab(self)
        self.tab_mask    = MaskTab(self)
        self.tab_density = DensityTab(self)
        self.tab_network = NetworkTab(self)

        self._tabs.addTab(self.tab_prepare, "1 · Prepare")
        self._tabs.addTab(self.tab_mask,    "2 · Mask")
        self._tabs.addTab(self.tab_density, "3 · Density")
        self._tabs.addTab(self.tab_network, "4 · Network")

        lay.addWidget(self._tabs)
        self.setLayout(lay)
        self.setStyleSheet(_PANEL_STYLE)

    # ------------------------------------------------------------------
    # Layer events
    # ------------------------------------------------------------------

    def _on_layer_inserted(self, event) -> None:
        layer = event.value
        if isinstance(layer, napari.layers.Labels):
            if not layer.metadata.get("is_debug_tissue_mask", False):
                self.labels_layer = layer
                layer.editable = True
                layer.mode = "paint"
                layer.selected_label = 1
        if isinstance(layer, napari.layers.Image) and self.base_image_layer is None:
            self.base_image_layer = layer
        self._update_all_states()
        if isinstance(layer, napari.layers.Image):
            img = np.asarray(layer.data)
            self.tab_prepare.set_image_info(layer.name, img.shape[:2])

    def _on_layer_removed(self, event) -> None:
        layer = event.value
        if layer is self.labels_layer:
            self.labels_layer = None
        if layer is self.base_image_layer:
            self.base_image_layer = None
        self._update_all_states()

    def _update_all_states(self) -> None:
        has_image = self._get_image_layer() is not None
        has_mask  = self.labels_layer is not None
        self.tab_mask.update_states(has_image, has_mask)
        self.tab_density.update_states(has_mask, has_image)
        self.tab_network.update_states(has_mask)

    def _refresh_transect_ui(self) -> None:
        self.tab_density.refresh_transect_ui(self._transect_ctrl.pending)

    # ------------------------------------------------------------------
    # Image layer helper
    # ------------------------------------------------------------------

    def _get_image_layer(self) -> Optional[napari.layers.Image]:
        if (self.base_image_layer is not None
                and self.base_image_layer in self.viewer.layers
                and isinstance(self.base_image_layer, napari.layers.Image)):
            return self.base_image_layer
        if isinstance(self.viewer.layers.selection.active, napari.layers.Image):
            return self.viewer.layers.selection.active
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Image):
                return layer
        return None

    # ------------------------------------------------------------------
    # Mask helpers (called from MaskTab)
    # ------------------------------------------------------------------

    def create_empty_mask(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before creating a mask.")
            return
        if self.labels_layer is not None:
            if QMessageBox.question(
                self, "Overwrite?",
                "A mask already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            ) != QMessageBox.Yes:
                return
        data = np.zeros(np.asarray(image_layer.data).shape[:2], dtype=np.uint8)
        self._add_labels_layer(data, from_model=False)

    def auto_generate_mask(self) -> None:
        image_layer = self._get_image_layer()
        if image_layer is None:
            QMessageBox.warning(self, "No image", "Load an image before generating a mask.")
            return
        if self.labels_layer is not None:
            if QMessageBox.question(
                self, "Overwrite?",
                "A mask already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            ) != QMessageBox.Yes:
                return
        image_data = np.asarray(image_layer.data)
        self.tab_mask.auto_btn.setEnabled(False)
        self.tab_mask.auto_btn.setText("Generating…")
        self.tab_mask.set_ai_status("⏳ Running AI model, please wait…", True)

        @thread_worker
        def _run():
            return predict_laticifer_mask(image_data)

        w = _run()

        def _done(mask):
            self._add_labels_layer(mask.astype(np.uint8), from_model=True)
            self.tab_mask.set_ai_status("✓ Mask generated successfully.", True)
            from qtpy.QtCore import QTimer
            QTimer.singleShot(4000, lambda: self.tab_mask.set_ai_status("", False))

        def _err(exc):
            self.tab_mask.set_ai_status("", False)
            QMessageBox.critical(self, "Prediction error", f"Failed: {exc[1]}")

        def _fin():
            self.tab_mask.auto_btn.setText("🤖  Auto-detect with AI model")
            self._update_all_states()

        w.returned.connect(_done)
        w.errored.connect(_err)
        w.finished.connect(_fin)
        w.start()

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
        self._update_all_states()

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
    # Mask data helpers
    # ------------------------------------------------------------------

    def _get_mask_data(self) -> Optional[np.ndarray]:
        return np.asarray(self.labels_layer.data) if self.labels_layer is not None else None

    def _update_mask_data(self, new_data: np.ndarray) -> None:
        if self.labels_layer is not None:
            self.labels_layer.data = new_data
            self.labels_layer.refresh()

    # ------------------------------------------------------------------
    # Density (called from DensityTab)
    # ------------------------------------------------------------------

    def _run_pixel_ratio_from_tab(self) -> None:
        if self.labels_layer is None:
            return
        mask = self._get_mask_data()
        use_tissue = self.tab_density._area_combo.currentIndex() == 1
        stats = analyze_density_pixel_ratio(mask, use_tissue_mask=use_tissue)

        if use_tissue:
            dbg = stats.get("debug_tissue_mask")
            if dbg is not None:
                if "Computed Tissue Area" in self.viewer.layers:
                    self.viewer.layers.remove("Computed Tissue Area")
                tl = self.viewer.add_labels(
                    dbg, name="Computed Tissue Area", opacity=0.3,
                    metadata={"is_debug_tissue_mask": True},
                )
                tl.editable = False
                if self.labels_layer is not None:
                    self.viewer.layers.selection.active = self.labels_layer
        else:
            if "Computed Tissue Area" in self.viewer.layers:
                self.viewer.layers.remove("Computed Tissue Area")

        self.tab_density.show_pixel_results(
            float(stats["density_percentage"]),
            int(stats["laticifer_pixels"]),
            self.um_per_px,
        )

    def _run_transect_from_tab(self, direction: str, num_lines: int) -> None:
        if self.labels_layer is None:
            return
        mask = self._get_mask_data()
        self.last_transect_num_lines = num_lines
        self.last_transect_direction = direction
        self._transect_ctrl.generate(
            mask_shape=tuple(mask.shape[:2]),
            num_lines=num_lines,
            direction=direction,
        )
        stats = self._transect_ctrl.recalculate(mask, show_points=True)
        if stats:
            mean = stats.get("mean_intersections_per_line", float("nan"))
            std  = stats.get("std_intersections_per_line",  float("nan"))
            self.last_transect_mean = mean if math.isfinite(mean) else None
            self.tab_density.show_transect_results(mean, std)

    def _on_recalculate_density(self) -> None:
        if self.labels_layer is None:
            return
        mask  = self._get_mask_data()
        stats = self._transect_ctrl.recalculate(mask, show_points=True)
        if stats is None:
            QMessageBox.warning(self, "Error", "No transects found.")
            return
        mean = stats.get("mean_intersections_per_line", float("nan"))
        std  = stats.get("std_intersections_per_line",  float("nan"))
        self.last_transect_num_lines = int(stats.get("num_lines", self.last_transect_num_lines))
        self.last_transect_direction = str(stats.get("direction", self.last_transect_direction))
        self.last_transect_mean = mean if math.isfinite(mean) else None
        self.tab_density.show_transect_results(mean, std)

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


# ---------------------------------------------------------------------------
#  Batch Processing (unchanged logic, minor style)
# ---------------------------------------------------------------------------

class BatchProcessingWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._build_ui()
 
    def _build_ui(self) -> None:
        lay = QVBoxLayout()
        lay.setSpacing(8)
 
        # Input/output folders
        lay.addWidget(QLabel("Input images folder:"))
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select input folder…")
        btn_in = QPushButton("Browse…")
        btn_in.clicked.connect(self._select_input)
        row_in = QHBoxLayout()
        row_in.addWidget(self.input_dir_edit)
        row_in.addWidget(btn_in)
        lay.addLayout(row_in)
 
        lay.addWidget(QLabel("Output folder:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output folder…")
        btn_out = QPushButton("Browse…")
        btn_out.clicked.connect(self._select_output)
        row_out = QHBoxLayout()
        row_out.addWidget(self.output_dir_edit)
        row_out.addWidget(btn_out)
        lay.addLayout(row_out)
 
        # Optional Scale
        scale_box, scale_lay = _group(
            "Scale (optional)",
            "When set, µm columns are added to the CSV alongside the pixel columns.\n"
            "Leave at 0 to export pixel values only."
        )
        scale_lay.addWidget(_small_label(
            "color:#666;font-size:10px;",
            "Check your microscope metadata or use the auto-detector in the\n"
            "Interactive Editor tab to find the pixel size for your images."
        ))
 
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Pixel size:"))
        self._scale_spin = QDoubleSpinBox()
        self._scale_spin.setRange(0.0, 9999.0)
        self._scale_spin.setDecimals(4)
        self._scale_spin.setValue(0.0)
        self._scale_spin.setSpecialValueText("— (pixels only)")
        self._scale_spin.setToolTip("e.g. 0.65 → 1 px = 0.65 µm")
        scale_row.addWidget(self._scale_spin)
        self._unit_combo = QComboBox()
        self._unit_combo.addItems(["µm/px", "nm/px"])
        scale_row.addWidget(self._unit_combo)
        scale_lay.addLayout(scale_row)
 
        self._scale_preview = QLabel("")
        self._scale_preview.setStyleSheet("color:#5ca;font-size:11px;")
        scale_lay.addWidget(self._scale_preview)
 
        self._scale_spin.valueChanged.connect(self._update_scale_preview)
        self._unit_combo.currentIndexChanged.connect(self._update_scale_preview)
 
        lay.addWidget(scale_box)
 
        # Progress
        lay.addSpacing(4)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setStyleSheet("color:#aaa;font-size:11px;")
        lay.addWidget(self.status_lbl)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        lay.addWidget(self.progress_bar)
 
        # Run
        self.run_btn = QPushButton("▶  Start batch processing")
        self.run_btn.setStyleSheet(
            "background:#2d6a2d;color:#c8f0d8;font-weight:bold;padding:10px;"
            "border:none;border-radius:4px;"
        )
        self.run_btn.clicked.connect(self._start_batch)
        lay.addWidget(self.run_btn)
 
        lay.addStretch()
        self.setLayout(lay)
        self.setStyleSheet(_PANEL_STYLE)
 
    # Helpers
 
    def _um_per_px(self) -> Optional[float]:
        """Return the scale in µm/px, or None if not set."""
        val = self._scale_spin.value()
        if val <= 0:
            return None
        if self._unit_combo.currentText() == "nm/px":
            return val / 1000.0
        return float(val)
 
    def _update_scale_preview(self) -> None:
        um = self._um_per_px()
        if um and um > 0:
            self._scale_preview.setText(f"→ 1 px = {um:.4f} µm  ·  µm columns will be exported")
        else:
            self._scale_preview.setText("")
 
    def _select_input(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Input")
        if d:
            self.input_dir_edit.setText(d)
 
    def _select_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output")
        if d:
            self.output_dir_edit.setText(d)
 
    def _start_batch(self) -> None:
        in_d  = self.input_dir_edit.text()
        out_d = self.output_dir_edit.text()
        if not in_d or not out_d:
            QMessageBox.warning(self, "Error", "Select both folders.")
            return
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing…")
        um = self._um_per_px()
        w = self._run_batch_worker(in_d, out_d, um)
        w.yielded.connect(self._on_progress)
        w.finished.connect(self._on_finished)
        w.start()
 
    def _on_progress(self, data) -> None:
        curr, total, name = data
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(curr)
        self.status_lbl.setText(f"Processing: {name}")
 
    def _on_finished(self) -> None:
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_lbl.setText("Complete!")
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶  Start batch processing")
        QMessageBox.information(
            self, "Done",
            "Batch processing complete.\nSee 'batch_results.csv' in the output folder.",
        )
 
    @thread_worker
    def _run_batch_worker(self, in_dir_str: str, out_dir_str: str, um_per_px: Optional[float]):
        results = []
        for curr, total, name, row in run_batch_processing(
            in_dir_str, out_dir_str, num_lines=10, um_per_px=um_per_px
        ):
            yield (curr, total, name)
            results.append(row)
        write_batch_csv(out_dir_str, results)


# ---------------------------------------------------------------------------
#  Root widget
# ---------------------------------------------------------------------------

class LaticiferAnnotationWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()
        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(InteractiveEditorWidget(viewer), "Interactive Editor")
        tabs.addTab(BatchProcessingWidget(),          "Batch Processing")
        lay.addWidget(tabs)
        self.setLayout(lay)