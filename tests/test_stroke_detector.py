"""Synthetic tests for thin low-contrast strokes on textured cardboard across all 9 stages."""

import tempfile
import time
import unittest
from pathlib import Path
from typing import Callable, Tuple
import cv2
import numpy as np

from stroke_detector.config import Config
from stroke_detector.pipeline import detect_strokes, run_pipeline
from stroke_detector.stages import (
    stage1_capture_gate,
    stage2_preprocess,
    stage3_enhance_strokes,
    stage4_hysteresis_threshold,
    stage5_skeleton_and_grouping,
    stage6_vectorize,
)
from stroke_detector.viz import save_stage_visualizations


def make_cardboard_background(
    width: int = 1000, height: int = 800, seed: int = 123
) -> np.ndarray:
    """Create a realistic textured cardboard BGR image with uneven lighting, flecks, and JPEG artifacts."""
    rng = np.random.default_rng(seed)

    # Base warm cardboard color in BGR: (B=115, G=155, R=190)
    base = np.zeros((height, width, 3), dtype=np.float32)
    base[:, :, 0] = 115.0
    base[:, :, 1] = 155.0
    base[:, :, 2] = 190.0

    # Smooth uneven lighting gradient (+/- 35 gray levels across the frame)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    illum = (
        28.0 * np.sin(xx / width * np.pi * 1.1)
        + 22.0 * np.cos(yy / height * np.pi * 0.9)
        - 15.0 * ((xx - width * 0.3) ** 2 + (yy - height * 0.4) ** 2) / (width ** 2)
    )
    base += illum[:, :, None]

    # Fine cardboard fiber grain
    grain = rng.normal(0.0, 3.2, size=(height, width)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (3, 3), 0.7)
    base += grain[:, :, None]

    # Random dark cardboard flecks (small 1-2 px pepper spots)
    fleck_mask = rng.random((height, width)) < 0.0035
    fleck_drop = rng.uniform(10.0, 22.0, size=(height, width)).astype(np.float32)
    base -= (fleck_mask * fleck_drop)[:, :, None]

    return np.clip(base, 0, 255).astype(np.uint8)


def apply_stroke_and_jpeg(
    cardboard_bgr: np.ndarray,
    draw_fn: Callable[[np.ndarray], None],
    contrast_drop: float = 16.0,
    jpeg_quality: int = 50,
) -> np.ndarray:
    """Draw a thin anti-aliased stroke only a few gray levels darker than the board, then JPEG-compress."""
    h, w = cardboard_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    draw_fn(mask)

    # Smooth stroke mask slightly to simulate pen ink profile (width ~ 3-4 px)
    stroke_alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (5, 5), 1.0)
    stroke_alpha = np.clip(stroke_alpha * 1.35, 0.0, 1.0)

    out_f = cardboard_bgr.astype(np.float32) - (stroke_alpha[:, :, None] * contrast_drop)
    out_u8 = np.clip(out_f, 0, 255).astype(np.uint8)

    # Compress with JPEG at quality 40-60 to introduce 8x8 DCT block artifacts
    ok, enc = cv2.imencode(".jpg", out_u8, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    assert ok
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    assert dec is not None
    return dec


def test_pure_texture_not_detected():
    """Pure textured cardboard with flecks, uneven lighting, and JPEG Q=45 must return not_detected."""
    bg = make_cardboard_background(1000, 800, seed=10)
    img = apply_stroke_and_jpeg(bg, lambda m: None, contrast_drop=0.0, jpeg_quality=45)
    res = detect_strokes(img)
    assert res["status"] == "not_detected"
    strokes = [c for c in res["curves"] if c.get("layer") != "bed"]
    assert len(strokes) == 0


def test_line_detection():
    """Detect a low-contrast straight line on textured JPEG cardboard and verify endpoint accuracy."""
    bg = make_cardboard_background(1000, 800, seed=20)
    gt_p1 = (180, 210)
    gt_p2 = (820, 590)

    def draw(m: np.ndarray) -> None:
        cv2.line(m, gt_p1, gt_p2, 255, 3, cv2.LINE_AA)

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=16.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    assert res["confidence"] >= 0.5
    curves = [c for c in res["curves"] if c.get("layer") != "bed"]
    assert len(curves) >= 1
    c = curves[0]
    assert c["type"] == "line"
    assert c["fit_error_px"] < 2.5

    # Verify line angle and distance from ground truth endpoints to fitted line
    p = c["params"]
    fx1, fy1, fx2, fy2 = p["x1"], p["y1"], p["x2"], p["y2"]
    gt_angle = np.degrees(np.arctan2(gt_p2[1] - gt_p1[1], gt_p2[0] - gt_p1[0])) % 180.0
    fit_angle = np.degrees(np.arctan2(fy2 - fy1, fx2 - fx1)) % 180.0
    angle_err = min(abs(gt_angle - fit_angle), 180.0 - abs(gt_angle - fit_angle))
    assert angle_err < 2.0


def test_circle_detection():
    """Detect a low-contrast circle on textured JPEG cardboard and assert center/radius error < 6 px."""
    bg = make_cardboard_background(1000, 800, seed=30)
    gt_cx, gt_cy, gt_r = 500, 400, 210

    def draw(m: np.ndarray) -> None:
        cv2.circle(m, (gt_cx, gt_cy), gt_r, 255, 3, cv2.LINE_AA)

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=16.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    assert res["confidence"] >= 0.5
    c = [c for c in res["curves"] if c.get("layer") != "bed"][0]
    assert c["type"] == "circle"
    p = c["params"]
    assert abs(p["cx"] - gt_cx) < 6.0
    assert abs(p["cy"] - gt_cy) < 6.0
    assert abs(p["r"] - gt_r) < 6.0


def test_cropped_arc_detection():
    """Detect a circle partially cut off by the frame edge (becomes a circular arc)."""
    bg = make_cardboard_background(1000, 800, seed=35)
    gt_cx, gt_cy, gt_r = 500, 50, 260  # Top half cropped off by y=0 edge

    def draw(m: np.ndarray) -> None:
        cv2.circle(m, (gt_cx, gt_cy), gt_r, 255, 3, cv2.LINE_AA)

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=17.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    c = [c for c in res["curves"] if c.get("layer") != "bed"][0]
    assert c["type"] in ("arc", "circle", "bezier")
    if c["type"] in ("arc", "circle"):
        p = c["params"]
        assert abs(p["cx"] - gt_cx) < 12.0
        assert abs(p["r"] - gt_r) < 12.0


def test_ellipse_detection():
    """Detect a low-contrast ellipse on textured JPEG cardboard and assert center/radii accuracy."""
    bg = make_cardboard_background(1000, 800, seed=40)
    gt_cx, gt_cy, gt_rx, gt_ry, gt_ang = 500, 400, 270, 145, 25.0

    def draw(m: np.ndarray) -> None:
        cv2.ellipse(
            m, (gt_cx, gt_cy), (gt_rx, gt_ry), gt_ang, 0, 360, 255, 3, cv2.LINE_AA
        )

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=16.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    assert res["confidence"] >= 0.5
    c = [c for c in res["curves"] if c.get("layer") != "bed"][0]
    assert c["type"] == "ellipse"
    p = c["params"]
    assert abs(p["cx"] - gt_cx) < 8.0
    assert abs(p["cy"] - gt_cy) < 8.0
    assert abs(p["rx"] - gt_rx) < 10.0
    assert abs(p["ry"] - gt_ry) < 10.0


def test_bezier_detection():
    """Detect a cubic Bezier S-curve on textured JPEG cardboard."""
    bg = make_cardboard_background(1000, 800, seed=50)
    p0 = np.array([150.0, 580.0])
    p1 = np.array([350.0, 100.0])
    p2 = np.array([650.0, 700.0])
    p3 = np.array([850.0, 220.0])

    t = np.linspace(0.0, 1.0, 300)
    omt = 1.0 - t
    curve_pts = (
        np.outer(omt ** 3, p0)
        + np.outer(3 * omt ** 2 * t, p1)
        + np.outer(3 * omt * t ** 2, p2)
        + np.outer(t ** 3, p3)
    )

    def draw(m: np.ndarray) -> None:
        pts_i = np.round(curve_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts_i], False, 255, 3, cv2.LINE_AA)

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=17.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    assert res["confidence"] >= 0.5
    c = [c for c in res["curves"] if c.get("layer") != "bed"][0]
    assert c["type"] == "bezier"
    assert c["fit_error_px"] < 2.5


def test_freehand_detection():
    """Detect a multi-oscillation freehand curve that cannot be represented by a single cubic Bezier."""
    bg = make_cardboard_background(1000, 800, seed=60)
    xs = np.linspace(120.0, 880.0, 400)
    ys = 400.0 + 130.0 * np.sin((xs - 120.0) / 760.0 * 3.5 * np.pi)
    freehand_pts = np.column_stack([xs, ys])

    def draw(m: np.ndarray) -> None:
        pts_i = np.round(freehand_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(m, [pts_i], False, 255, 3, cv2.LINE_AA)

    img = apply_stroke_and_jpeg(bg, draw, contrast_drop=17.0, jpeg_quality=50)
    res = detect_strokes(img)

    assert res["status"] == "detected"
    assert res["confidence"] >= 0.5
    c = [c for c in res["curves"] if c.get("layer") != "bed"][0]
    assert c["type"] in ("freehand", "bezier")
    assert len(c["points"]) >= 6


def test_stages_1_to_6_runtime_and_viz(tmp_path=None):
    """Verify stages 1-6 runtime on a 1 MP image and verify viz.py outputs all 9 stage images."""
    bg = make_cardboard_background(1150, 870, seed=77)  # ~1.0 MP
    cv2.circle(bg, (575, 435), 220, (95, 135, 170), 3, cv2.LINE_AA)

    cfg = Config()
    # Warm up once
    cap = stage1_capture_gate(bg, cfg)
    prep = stage2_preprocess(cap, cfg)
    enh = stage3_enhance_strokes(prep, cfg)
    thr = stage4_hysteresis_threshold(enh, cfg)
    skl = stage5_skeleton_and_grouping(thr, cfg)
    _ = stage6_vectorize(skl, prep, cfg)

    t0 = time.perf_counter()
    cap = stage1_capture_gate(bg, cfg)
    prep = stage2_preprocess(cap, cfg)
    enh = stage3_enhance_strokes(prep, cfg)
    thr = stage4_hysteresis_threshold(enh, cfg)
    skl = stage5_skeleton_and_grouping(thr, cfg)
    vec = stage6_vectorize(skl, prep, cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(vec.curves) >= 1
    assert elapsed_ms < 2500.0  # Includes 3-channel LAB baseline + per-color segmentation

    _, artifacts = run_pipeline(bg, cfg)
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            saved = save_stage_visualizations(artifacts, td)
            assert len(saved) >= 10
    else:
        saved = save_stage_visualizations(artifacts, tmp_path)
        assert len(saved) >= 10  # 01..10 PNGs + output.svg


def test_roi_zoom_and_offset_scaling():
    """Verify that zooming into a small ROI on a large bed detects a small stroke and maps coordinates back to the full bed."""
    # Full 2000x1600 bed with a 360x360 ROI at offset (850, 620)
    bed = np.full((1600, 2000, 3), (35, 35, 35), dtype=np.uint8)
    roi_x, roi_y, roi_w, roi_h = 850, 620, 360, 360
    cardboard_roi = make_cardboard_background(roi_w, roi_h, seed=91)
    # Draw a circle (radius 95 px) inside the ROI
    local_cx, local_cy, gt_r = 180, 180, 95
    gt_bed_cx = roi_x + local_cx
    gt_bed_cy = roi_y + local_cy

    def draw(m: np.ndarray) -> None:
        cv2.circle(m, (local_cx, local_cy), gt_r, 255, 2, cv2.LINE_AA)

    roi_with_stroke = apply_stroke_and_jpeg(cardboard_roi, draw, contrast_drop=18.0, jpeg_quality=60)
    bed[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi_with_stroke

    cfg = Config(roi_bbox=(roi_x, roi_y, roi_w, roi_h))
    res = detect_strokes(bed, cfg)

    assert res["status"] == "detected"
    strokes = [c for c in res["curves"] if c.get("layer") != "bed"]
    assert len(strokes) >= 1
    c = strokes[0]
    assert c["type"] == "circle"
    p = c["params"]
    assert abs(p["cx"] - gt_bed_cx) < 6.0
    assert abs(p["cy"] - gt_bed_cy) < 6.0
    assert abs(p["r"] - gt_r) < 6.0


class TestStrokeDetectorSuite(unittest.TestCase):
    def test_01_pure_texture(self):
        test_pure_texture_not_detected()

    def test_02_line(self):
        test_line_detection()

    def test_03_circle(self):
        test_circle_detection()

    def test_04_cropped_arc(self):
        test_cropped_arc_detection()

    def test_05_ellipse(self):
        test_ellipse_detection()

    def test_06_bezier(self):
        test_bezier_detection()

    def test_07_freehand(self):
        test_freehand_detection()

    def test_08_runtime_and_viz(self):
        test_stages_1_to_6_runtime_and_viz()

    def test_09_roi_zoom_and_offset_scaling(self):
        test_roi_zoom_and_offset_scaling()



if __name__ == "__main__":
    unittest.main(verbosity=2)

