"""Verification script and acceptance test suite for the 4-layer colored marker stroke pipeline."""

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stroke_detector.config import Config
from stroke_detector.pipeline import run_pipeline
from stroke_detector.viz import save_stage_visualizations


def run_test_image_verification(save_dir: Path | None = None) -> None:
    """Run the full pipeline on test_images/2_colour_test.jpg, print summary counts, and verify acceptance criteria."""
    img_path = ROOT / "test_images" / "2_colour_test.jpg"
    assert img_path.exists(), f"Test image not found: {img_path}"
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None, f"Failed to load image: {img_path}"

    cfg = Config()
    res, artifacts = run_pipeline(img_bgr, cfg)

    if save_dir is not None:
        save_stage_visualizations(artifacts, save_dir)

    all_entries = res["curves"]
    bed_entries = [c for c in all_entries if c.get("layer") == "bed"]
    strokes = [c for c in all_entries if c.get("layer") != "bed"]

    color_counts = Counter(str(c.get("color")) for c in strokes)
    model_counts = Counter(str(c.get("model")) for c in strokes)
    residuals = [float(c.get("residual_mm", 0.0)) for c in strokes]
    mean_res_mm = float(np.mean(residuals)) if residuals else 0.0
    median_res_mm = float(np.median(residuals)) if residuals else 0.0

    print("=== Test Image Verification: test_images/2_colour_test.jpg ===")
    print(f"Status                : {res['status']}")
    print(f"Total stroke count    : {len(strokes)} (excluding {len(bed_entries)} bed reference entries)")
    print(f"Per-color counts      : red={color_counts.get('red', 0)}, green={color_counts.get('green', 0)}, blue={color_counts.get('blue', 0)}")
    print(f"Per-model counts      : {dict(model_counts)}")
    print(f"Residual (mm)         : mean={mean_res_mm:.2f} mm, median={median_res_mm:.2f} mm")
    print("\nDetailed Stroke List:")
    for idx, c in enumerate(strokes, start=1):
        print(
            f"  #{idx:02d} | layer={c['layer']:5s} | color={c['color']:5s} | "
            f"model={c['model']:8s} | residual={c['residual_mm']:.2f} mm | len={c['length_px']:.1f} px"
        )

    # Verify SVG structure and 4 layers in exact order: bed, red, green, blue
    root = ET.fromstring(artifacts.svg_string)
    assert root.attrib.get("viewBox") == "0 0 775.0 1065.0" or root.attrib.get("viewBox") == "0 0 775 1065"
    g_layers = [el for el in root if el.tag.endswith("g")]
    layer_ids = [el.attrib.get("id") for el in g_layers]
    assert layer_ids == ["bed", "red", "green", "blue"], f"Unexpected SVG layer order: {layer_ids}"
    assert g_layers[0].attrib.get("data-role") == "reference"

    # Acceptance assertions
    assert 20 <= len(strokes) <= 25, f"Expected 20-25 strokes, got {len(strokes)}"
    assert len(bed_entries) == 2, "Expected 2 reference entries in bed layer (bed outline + cardboard polygon)"
    assert model_counts.get("rect", 0) == 3, f"Expected 3 rectangles (squares), got {model_counts.get('rect', 0)}"
    assert model_counts.get("polygon", 0) == 3, f"Expected 3 closed polygons (2 trapezoids + 1 triangle), got {model_counts.get('polygon', 0)}"
    assert model_counts.get("circle", 0) == 2, f"Expected 2 circles, got {model_counts.get('circle', 0)}"
    assert model_counts.get("bezier", 0) == 1, f"Expected 1 bezier path (wave), got {model_counts.get('bezier', 0)}"
    assert model_counts.get("polyline", 0) >= 1, "Expected zigzag polyline"
    assert mean_res_mm < 1.0, f"Expected mean residual < 1.0 mm, got {mean_res_mm:.2f} mm"
    print("\n[PASS] Real test image verification passed all acceptance criteria.\n")


def run_synthetic_blue_verification() -> None:
    """Recolor green strokes in a copy of the flattened ROI to blue and verify <g id='blue'> classification."""
    img_path = ROOT / "test_images" / "2_colour_test.jpg"
    img_bgr = cv2.imread(str(img_path))
    assert img_bgr is not None

    cfg = Config()
    _, base_art = run_pipeline(img_bgr, cfg)

    # Take the cropped ROI image and recolor the green strokes to blue marker ink
    roi_bgr = base_art.capture.image_bgr.copy()
    green_mask_work = base_art.thresh.color_masks["green"]
    h_roi, w_roi = roi_bgr.shape[:2]
    green_mask_roi = cv2.resize(
        green_mask_work.astype(np.uint8) * 255,
        (w_roi, h_roi),
        interpolation=cv2.INTER_NEAREST,
    )
    # Dilate mask to cover full stroke width and anti-aliasing
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    recolor_zone = cv2.dilate(green_mask_roi, k_dil) > 0

    # Recolor in LAB chroma space: neutral a (0 green), negative b (blue ink)
    lab_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    lab_roi[recolor_zone, 1] = 135
    lab_roi[recolor_zone, 2] = 85
    roi_bgr = cv2.cvtColor(lab_roi, cv2.COLOR_LAB2BGR)

    # Run pipeline directly on the synthetic blue+red ROI with matching physical dimensions
    phys_w = float(w_roi) / max(1e-6, base_art.capture.px_per_mm)
    phys_h = float(h_roi) / max(1e-6, base_art.capture.px_per_mm)
    cfg_syn = Config(
        enable_aruco_warp=False,
        auto_detect_roi=False,
        default_phys_width_mm=phys_w,
        default_phys_height_mm=phys_h,
    )
    res_syn, art_syn = run_pipeline(roi_bgr, cfg_syn)
    strokes_syn = [c for c in res_syn["curves"] if c.get("layer") != "bed"]
    color_counts_syn = Counter(str(c.get("color")) for c in strokes_syn)

    print("=== Synthetic Blue Stroke Verification ===")
    print(f"Total stroke count    : {len(strokes_syn)}")
    print(
        f"Per-color counts      : red={color_counts_syn.get('red', 0)}, "
        f"green={color_counts_syn.get('green', 0)}, blue={color_counts_syn.get('blue', 0)}"
    )

    root_syn = ET.fromstring(art_syn.svg_string)
    blue_g = next(el for el in root_syn if el.tag.endswith("g") and el.attrib.get("id") == "blue")
    blue_svg_elements = list(blue_g)
    print(f"Elements in <g id='blue'> : {len(blue_svg_elements)}")

    assert color_counts_syn.get("blue", 0) >= 8, f"Expected >= 8 blue strokes, got {color_counts_syn.get('blue', 0)}"
    assert color_counts_syn.get("green", 0) == 0, f"Expected 0 green strokes after recoloring to blue, got {color_counts_syn.get('green', 0)}"
    assert len(blue_svg_elements) == color_counts_syn.get("blue", 0)
    print("[PASS] Synthetic blue stroke verification passed with zero misclassification.\n")


if __name__ == "__main__":
    run_test_image_verification()
    run_synthetic_blue_verification()
