"""
Laticifer Annotation App
-----------------------
A simple desktop tool built with napari to help biologists annotate laticifer 
structures in microscopy images.

Usage:
    python laticifer_annotation_app.py
"""

from __future__ import annotations
import sys
import os
import ctypes
from pathlib import Path
import napari
from qtpy.QtGui import QIcon

# Ensure we can import your modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ui import LaticiferAnnotationWidget

def main():
    # --- 1. WINDOWS TASKBAR FIX ---
    # This tells Windows: "I am a specific app, not just generic Python"
    if os.name == 'nt':
        myappid = 'upv.latiseg.assist.v1' # Arbitrary string
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    # --- 2. LAUNCH VIEWER ---
    viewer = napari.Viewer(title="LaticiferSegmentationApp")
    
    # --- 3. SET THE ICON ---
    # Calculate path relative to this script
    # Assuming structure: LatiSeg_App/src/main.py and LatiSeg_App/resources/app_icon.ico
    current_dir = Path(__file__).parent
    icon_path = current_dir.parent / "resources" / "app_icon.ico"
    
    if icon_path.exists():
        # Access the underlying Qt window of Napari
        viewer.window._qt_window.setWindowIcon(QIcon(str(icon_path)))
    else:
        print(f"Warning: Icon not found at {icon_path}")

    # --- 4. LOAD WIDGET ---
    widget = LaticiferAnnotationWidget(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    
    napari.run()

if __name__ == "__main__":
    main()