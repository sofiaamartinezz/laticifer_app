"""
Laticifer Annotation App
-----------------------
A simple desktop tool built with napari to help biologists annotate laticifer 
structures in microscopy images.

Usage:
    python laticifer_annotation_app.py
"""

from __future__ import annotations
import napari
from ui import LaticiferAnnotationWidget


def main() -> None:
    """Launch the napari viewer and dock widget."""
    viewer = napari.Viewer()
    widget = LaticiferAnnotationWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()


if __name__ == "__main__":
    main()
