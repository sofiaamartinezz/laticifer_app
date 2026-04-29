# main.py
"""
LatexLens — Laticifer Annotation App
Entry point: launches the napari viewer and attaches the annotation widget.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import napari
from qtpy.QtGui import QIcon

from ui.widgets import LaticiferAnnotationWidget


def main() -> None:
    # Windows taskbar icon fix
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "upv.latiseg.assist.v1"
            )
        except Exception:
            pass

    viewer = napari.Viewer(title="LatexLens")

    icon_path = Path(__file__).parent.parent / "resources" / "app_icon.ico"
    if icon_path.exists():
        viewer.window._qt_window.setWindowIcon(QIcon(str(icon_path)))
    else:
        print(f"[WARN] Icon not found at {icon_path}")

    widget = LaticiferAnnotationWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")

    napari.run()


if __name__ == "__main__":
    main()