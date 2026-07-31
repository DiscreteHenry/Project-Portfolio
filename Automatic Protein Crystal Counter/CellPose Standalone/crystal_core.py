"""
Core logic for crystal detection, counting, concentration, and overlay drawing.
GUI-independent so it can be imported or tested separately.
"""

import os
import sys
import numpy as np
import cv2
import pandas as pd
from cellpose import models

# --- Version-safe GPU detection ---
try:
    from cellpose.core import use_gpu
except ImportError:
    def use_gpu():
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


# ==========================================
# LAB CONFIGURATION LOOKUP TABLES
# ==========================================
MAGNIFICATION_DIAMETERS = {"10x": 15, "20x": 30, "40x": 60}

HEMOCYTOMETER_SPECS = {
    "Neubauer": {"depth_mm": 0.1, "frame_area_mm2": 1.0},
    "Fuchs":    {"depth_mm": 0.2, "frame_area_mm2": 0.5},
}

_MODEL = None  # cached model instance


def _bundled_model_path():
    """If running as a frozen exe, point to bundled model weights."""
    if getattr(sys, "frozen", False):
        return os.path.join(sys._MEIPASS, "cellpose_models")
    return None


def get_model(model_type="cyto3", force_cpu=True):
    """Load (and cache) the Cellpose model. CPU-only by default for distribution."""
    global _MODEL
    if _MODEL is None:
        bundled = _bundled_model_path()
        if bundled and os.path.isdir(bundled):
            os.environ["CELLPOSE_LOCAL_MODELS_PATH"] = bundled
        gpu = False if force_cpu else use_gpu()
        _MODEL = models.CellposeModel(gpu=gpu, model_type=model_type)
    return _MODEL


def _run_eval(model, image, diameter):
    """Handle Cellpose API differences across versions (3- or 4-tuple, channels arg)."""
    try:
        result = model.eval(image, diameter=diameter, channels=[0, 0])
    except TypeError:
        result = model.eval(image, diameter=diameter)
    return result[0]


def draw_detections(image, masks, draw_boxes=True, draw_outlines=False,
                    draw_labels=True):
    """Return a BGR copy of `image` with detections drawn on it."""
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    n = int(masks.max())
    box_color = (0, 255, 0)        # green (BGR)
    outline_color = (0, 165, 255)  # orange
    text_color = (0, 255, 0)

    for crystal_id in range(1, n + 1):
        single = (masks == crystal_id).astype(np.uint8)
        if single.sum() == 0:
            continue

        ys, xs = np.where(single)
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        if draw_boxes:
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max),
                          box_color, thickness=2)

        if draw_outlines:
            contours, _ = cv2.findContours(single, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, outline_color, thickness=1)

        if draw_labels:
            cv2.putText(overlay, str(crystal_id),
                        (x_min, max(y_min - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1,
                        cv2.LINE_AA)

    return overlay


# ============ CHANGED: explicit output folders ============
def process_image(img_path, slide_type, magnification, model,
                  overlay_folder, labels_folder=None,
                  save_overlay=True, save_labels=False,
                  draw_boxes=True, draw_outlines=False):
    """
    Process one image; return a result dict (or None on failure/skip).

    overlay_folder : where verification overlays are written (required)
    labels_folder  : where binary mask labels are written (only if save_labels)
    """
    filename = os.path.basename(img_path)

    if slide_type not in HEMOCYTOMETER_SPECS or magnification not in MAGNIFICATION_DIAMETERS:
        return None

    image = cv2.imread(img_path)
    if image is None:
        return None

    diameter = MAGNIFICATION_DIAMETERS[magnification]
    depth = HEMOCYTOMETER_SPECS[slide_type]["depth_mm"]
    area = HEMOCYTOMETER_SPECS[slide_type]["frame_area_mm2"]

    masks = _run_eval(model, image, diameter)
    count = int(masks.max())

    volume_ml = (area * depth) * 0.001
    conc = (count / volume_ml) if volume_ml > 0 else 0

    overlay_path = None
    if save_overlay:
        os.makedirs(overlay_folder, exist_ok=True)
        overlay = draw_detections(image, masks,
                                  draw_boxes=draw_boxes,
                                  draw_outlines=draw_outlines)
        overlay_path = os.path.join(
            overlay_folder, f"verify_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(overlay_path, overlay)

    if save_labels and labels_folder:
        os.makedirs(labels_folder, exist_ok=True)
        binary = np.where(masks > 0, 255, 0).astype(np.uint8)
        cv2.imwrite(
            os.path.join(labels_folder, f"mask_{os.path.splitext(filename)[0]}.png"),
            binary)
    # ==========================================================

    return {
        "Filename": filename,
        "Slide_Type": slide_type,
        "Magnification": magnification,
        "Crystal_Count": count,
        "Total_Volume_Imaged_mL": volume_ml,
        "Concentration_Crystals_per_mL": conc,
        "Overlay_Path": overlay_path,
    }


def export_results(results, csv_path):
    """Write results list to CSV (drops the internal Overlay_Path column)."""
    df = pd.DataFrame(results)
    if "Overlay_Path" in df.columns:
        df = df.drop(columns=["Overlay_Path"])
    df.to_csv(csv_path, index=False)


def default_output_dir():
    """Sensible default output location for a distributed app."""
    home = os.path.expanduser("~")
    docs = os.path.join(home, "Documents")
    base = docs if os.path.isdir(docs) else home
    return os.path.join(base, "CrystalCounter")