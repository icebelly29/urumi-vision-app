"""Visualization utility saving side-by-side input/output debug images for stages 01 through 09."""

import os
from pathlib import Path
from typing import List
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .pipeline import PipelineArtifacts


def _save_side_by_side(
    out_path: Path,
    left_img: np.ndarray,
    right_img: np.ndarray,
    title_left: str,
    title_right: str,
    suptitle: str,
    cmap_left: str = "gray",
    cmap_right: str = "gray",
) -> str:
    """Save a two-panel matplotlib figure showing stage input (left) and stage output (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=110)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    if left_img.ndim == 3:
        axes[0].imshow(left_img)
    else:
        axes[0].imshow(left_img, cmap=cmap_left)
    axes[0].set_title(title_left, fontsize=10)
    axes[0].axis("off")

    if right_img.ndim == 3:
        axes[1].imshow(right_img)
    else:
        axes[1].imshow(right_img, cmap=cmap_right)
    axes[1].set_title(title_right, fontsize=10)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _build_chroma_angle_vis(artifacts: PipelineArtifacts) -> np.ndarray:
    """Build an RGB visualization of the LAB chroma-angle classification map at working resolution."""
    prep = artifacts.prep
    h, w = prep.flat_field.shape
    if prep.chroma_angle_deg is None or prep.ink_strength is None:
        return cv2.cvtColor(prep.resized_bgr, cv2.COLOR_BGR2RGB)

    # HSV wheel where Hue = chroma angle (0..360 -> 0..180 in OpenCV), Value = normalized ink strength
    ang = prep.chroma_angle_deg
    strength = prep.ink_strength
    p99 = max(float(np.percentile(strength, 99.5)), 5.0)
    val = np.clip(strength / p99, 0.0, 1.0)

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.round((ang % 360.0) * 0.5).astype(np.uint8)
    hsv[:, :, 1] = np.where(val > 0.12, 230, 40).astype(np.uint8)
    hsv[:, :, 2] = np.round(np.clip(val * 255.0 + 25.0, 25.0, 255.0)).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _build_color_masks_composite(artifacts: PipelineArtifacts) -> np.ndarray:
    """Build an RGB composite of the three per-color masks (red, green, blue)."""
    h, w = artifacts.thresh.mask.shape
    comp = np.full((h, w, 3), 18, dtype=np.uint8)
    cm = artifacts.thresh.color_masks
    if "red" in cm and np.any(cm["red"]):
        comp[cm["red"]] = [229, 57, 53]
    if "green" in cm and np.any(cm["green"]):
        comp[cm["green"]] = [67, 160, 71]
    if "blue" in cm and np.any(cm["blue"]):
        comp[cm["blue"]] = [30, 136, 229]
    return comp


def render_qc_overlay(artifacts: PipelineArtifacts, alpha: float = 0.50) -> np.ndarray:
    """Render all 4 SVG layers (bed, red, green, blue) onto the flattened ROI photo at ~50% opacity at full ROI resolution (returns BGR uint8)."""
    base_bgr = artifacts.capture.image_bgr.copy()
    h, w = base_bgr.shape[:2]
    draw_bgr = base_bgr.copy()

    offset_vec = np.array(
        [float(artifacts.prep.roi_offset_x), float(artifacts.prep.roi_offset_y)],
        dtype=np.float64,
    )
    bed_h, bed_w = artifacts.prep.full_bed_shape
    if bed_w <= 0:
        bed_w = w
    dots_per_mm = float(bed_w) / max(1.0, float(artifacts.capture.phys_width_mm))
    stroke_px = max(2, int(round(artifacts.cfg.svg_stroke_width_mm * dots_per_mm)))

    # 1. Layer "bed" (reference outline in #888 dashed)
    cv2.rectangle(draw_bgr, (2, 2), (w - 3, h - 3), (136, 136, 136), 2, cv2.LINE_AA)
    if (
        artifacts.capture.cardboard_polygon_bed is not None
        and len(artifacts.capture.cardboard_polygon_bed) >= 3
    ):
        cb_roi = np.round(
            artifacts.capture.cardboard_polygon_bed - offset_vec
        ).astype(np.int32)
        cv2.polylines(
            draw_bgr,
            [cb_roi.reshape(-1, 1, 2)],
            True,
            (136, 136, 136),
            2,
            cv2.LINE_AA,
        )

    # 2. Layers "red", "green", "blue"
    color_bgr = {
        "red": (53, 57, 229),    # #e53935 in BGR
        "green": (71, 160, 67),  # #43a047 in BGR
        "blue": (229, 136, 30),  # #1e88e5 in BGR
    }
    for layer_name in ("red", "green", "blue"):
        col = color_bgr[layer_name]
        for fc in artifacts.fitted_curves:
            fc_col = fc.color or (
                "red"
                if fc.layer == "score"
                else ("green" if fc.layer == "crease" else "blue")
            )
            if fc_col != layer_name:
                continue
            ctype = fc.model or fc.curve_type
            p = fc.params
            if ctype == "circle" and "cx" in p and "cy" in p and "r" in p:
                center = (
                    int(round(float(p["cx"]) - offset_vec[0])),
                    int(round(float(p["cy"]) - offset_vec[1])),
                )
                radius = max(1, int(round(float(p["r"]))))
                cv2.circle(draw_bgr, center, radius, col, stroke_px, cv2.LINE_AA)
            elif ctype == "ellipse" and "cx" in p and "cy" in p and "rx" in p and "ry" in p:
                center = (
                    int(round(float(p["cx"]) - offset_vec[0])),
                    int(round(float(p["cy"]) - offset_vec[1])),
                )
                axes = (
                    max(1, int(round(float(p["rx"])))),
                    max(1, int(round(float(p["ry"])))),
                )
                ang = float(p.get("angle_deg", 0.0))
                cv2.ellipse(
                    draw_bgr, center, axes, ang, 0, 360, col, stroke_px, cv2.LINE_AA
                )
            elif ctype == "line" and "x1" in p and "y1" in p and "x2" in p and "y2" in p:
                pt1 = (
                    int(round(float(p["x1"]) - offset_vec[0])),
                    int(round(float(p["y1"]) - offset_vec[1])),
                )
                pt2 = (
                    int(round(float(p["x2"]) - offset_vec[0])),
                    int(round(float(p["y2"]) - offset_vec[1])),
                )
                cv2.line(draw_bgr, pt1, pt2, col, stroke_px, cv2.LINE_AA)
            else:
                pts_i = np.round(fc.points - offset_vec).astype(np.int32)
                is_closed_shape = ctype in ("rect", "polygon")
                if len(pts_i) >= 2:
                    cv2.polylines(
                        draw_bgr,
                        [pts_i.reshape(-1, 1, 2)],
                        is_closed_shape,
                        col,
                        stroke_px,
                        cv2.LINE_AA,
                    )

    return cv2.addWeighted(draw_bgr, alpha, base_bgr, 1.0 - alpha, 0.0)


def save_stage_visualizations(
    artifacts: PipelineArtifacts, out_dir: str | Path
) -> List[str]:
    """Save debug visualizations for stages 01 through 10 (including per-color masks, chroma map, graph, and full-ROI QC overlay)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []

    rgb_orig = cv2.cvtColor(artifacts.capture.image_bgr, cv2.COLOR_BGR2RGB)
    rgb_work = cv2.cvtColor(artifacts.prep.resized_bgr, cv2.COLOR_BGR2RGB)
    l_chan = artifacts.prep.lab_channels[0]
    flat_f = artifacts.prep.flat_field
    s_map = artifacts.enh.strokeness
    mask = artifacts.thresh.mask
    raw_skel = artifacts.skel.raw_skeleton
    final_skel = artifacts.skel.skeleton

    # 01_capture.png
    overlay_vis = rgb_orig.copy()
    if np.any(artifacts.capture.ignore_mask):
        overlay_vis[artifacts.capture.ignore_mask] = [255, 60, 60]
    q = artifacts.capture.quality
    saved_files.append(
        _save_side_by_side(
            out_path / "01_capture.png",
            rgb_orig,
            overlay_vis,
            f"Flattened ROI ({rgb_orig.shape[1]}x{rgb_orig.shape[0]})",
            f"Quality Gate (blur={q.blur_score:.1f}, bright={q.brightness:.1f})",
            "Stage 01: Capture & Quality Gate",
        )
    )

    # 02_preprocess.png (Ink Strength + Chroma-Angle Classification Map)
    chroma_vis = _build_chroma_angle_vis(artifacts)
    ink_s = (
        artifacts.prep.ink_strength
        if artifacts.prep.ink_strength is not None
        else flat_f
    )
    saved_files.append(
        _save_side_by_side(
            out_path / "02_preprocess.png",
            ink_s,
            chroma_vis,
            f"LAB Baseline-Subtracted Ink Strength ({l_chan.shape[1]}x{l_chan.shape[0]})",
            "Chroma-Angle (da*, db*) Classification Map",
            "Stage 02: Per-Color LAB Baseline Subtraction & Chroma Classification",
            cmap_left="magma",
        )
    )

    # 03_enhancement.png
    saved_files.append(
        _save_side_by_side(
            out_path / "03_enhancement.png",
            rgb_work,
            s_map,
            "Working ROI Input",
            f"Per-Color Multi-Scale Sato Ridge Response (max={artifacts.enh.raw_max:.2f})",
            "Stage 03: Per-Color Multi-Scale Sato Ridge Enhancement",
            cmap_right="magma",
        )
    )

    # 04_threshold.png (Combined + Per-Color Masks RGB Composite)
    color_comp_rgb = _build_color_masks_composite(artifacts)
    saved_files.append(
        _save_side_by_side(
            out_path / "04_threshold.png",
            s_map,
            color_comp_rgb,
            f"Strokeness (low={artifacts.cfg.color_hyst_low:.2f}, high={artifacts.cfg.color_hyst_high:.2f})",
            "Per-Color Hysteresis Masks (Red / Green / Blue)",
            "Stage 04: Per-Color Hysteresis & Overlap Assignment",
            cmap_left="magma",
        )
    )

    # 04b_color_masks.png (3-panel showing individual Red, Green, and Blue binary masks)
    fig_m, axes_m = plt.subplots(1, 3, figsize=(13, 4.5), dpi=110)
    fig_m.suptitle("Stage 04 Debug: Individual Per-Color Masks (Red, Green, Blue)", fontsize=12, fontweight="bold")
    for idx_c, c_name in enumerate(("red", "green", "blue")):
        cm_arr = artifacts.thresh.color_masks.get(
            c_name, np.zeros_like(mask, dtype=bool)
        )
        axes_m[idx_c].imshow(cm_arr.astype(np.uint8) * 255, cmap="gray")
        axes_m[idx_c].set_title(f"{c_name.upper()} Mask ({int(np.sum(cm_arr))} px)", fontsize=10)
        axes_m[idx_c].axis("off")
    plt.tight_layout()
    masks_path = out_path / "04b_color_masks.png"
    fig_m.savefig(str(masks_path), bbox_inches="tight")
    plt.close(fig_m)
    saved_files.append(str(masks_path))

    # 05_skeleton.png (Skeleton + Junction/Endpoint Graph)
    graph_rgb = (
        cv2.cvtColor(artifacts.skel.graph_nodes_vis, cv2.COLOR_BGR2RGB)
        if artifacts.skel.graph_nodes_vis is not None
        else final_skel.astype(np.uint8) * 255
    )
    saved_files.append(
        _save_side_by_side(
            out_path / "05_skeleton.png",
            color_comp_rgb,
            graph_rgb,
            "Per-Color Masks",
            "Topological Graph (Junctions=Yellow, Endpoints=Cyan)",
            "Stage 05: Per-Color Skeletonization & Topological Graph",
        )
    )

    # 06_vectorize.png
    vec_vis = rgb_work.copy()
    col_rgb_map = {"red": (229, 57, 53), "green": (67, 160, 71), "blue": (30, 136, 229)}
    for tc in artifacts.vec.curves:
        pts_i = np.round(tc.simplified_working).astype(np.int32)
        c_rgb = col_rgb_map.get(tc.color, (0, 230, 100))
        if len(pts_i) >= 2:
            cv2.polylines(
                vec_vis, [pts_i.reshape(-1, 1, 2)], tc.is_closed, c_rgb, 2, cv2.LINE_AA
            )
        for pt in pts_i:
            cv2.circle(vec_vis, (int(pt[0]), int(pt[1])), 3, (255, 255, 0), -1)

    saved_files.append(
        _save_side_by_side(
            out_path / "06_vectorize.png",
            graph_rgb,
            vec_vis,
            "Skeleton Graph",
            f"Smoothed & RDP Chains ({len(artifacts.vec.curves)} chains)",
            "Stage 06: Chain Smoothing & Vectorization",
        )
    )

    # 07_shape_selection.png
    offset_vec = np.array(
        [float(artifacts.prep.roi_offset_x), float(artifacts.prep.roi_offset_y)],
        dtype=np.float64,
    )
    traced_orig_vis = rgb_orig.copy()
    for tc in artifacts.vec.curves:
        pts_i = np.round(tc.points_orig - offset_vec).astype(np.int32)
        c_rgb = col_rgb_map.get(tc.color, (255, 180, 0))
        if len(pts_i) >= 2:
            cv2.polylines(
                traced_orig_vis, [pts_i.reshape(-1, 1, 2)], tc.is_closed, c_rgb, 2, cv2.LINE_AA
            )

    fit_orig_vis = rgb_orig.copy()
    for fc in artifacts.fitted_curves:
        pts_i = np.round(fc.points - offset_vec).astype(np.int32)
        c_rgb = col_rgb_map.get(fc.color, (0, 255, 80))
        if len(pts_i) >= 2:
            cv2.polylines(
                fit_orig_vis,
                [pts_i.reshape(-1, 1, 2)],
                fc.model in ("rect", "polygon", "circle", "ellipse"),
                c_rgb,
                3,
                cv2.LINE_AA,
            )
        if len(pts_i) > 0:
            label = f"{fc.color}:{fc.model} ({fc.residual_mm:.2f}mm)"
            p0 = (max(10, int(pts_i[0, 0])), max(25, int(pts_i[0, 1]) - 8))
            cv2.putText(
                fit_orig_vis, label, p0, cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_rgb, 2
            )

    saved_files.append(
        _save_side_by_side(
            out_path / "07_shape_selection.png",
            traced_orig_vis,
            fit_orig_vis,
            "Per-Color Traced Chains (ROI Res)",
            f"Fitted Geometric Shapes ({len(artifacts.fitted_curves)} strokes)",
            "Stage 07: Geometric Shape Model Selection",
        )
    )

    # 08_validation.png
    val_vis = fit_orig_vis.copy()
    for idx, fc in enumerate(artifacts.fitted_curves[:28]):
        txt = (
            f"#{idx+1:02d} [{fc.color}] {fc.model}: res={fc.residual_mm:.2f}mm, conf={fc.confidence:.2f}"
        )
        cv2.putText(
            val_vis,
            txt,
            (15, 28 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 60),
            2,
        )

    saved_files.append(
        _save_side_by_side(
            out_path / "08_validation.png",
            chroma_vis,
            val_vis,
            "Chroma Classification Map",
            f"Validation (status={artifacts.status}, strokes={len(artifacts.fitted_curves)})",
            "Stage 08: Confidence & Residual Validation",
        )
    )

    # 09_output.png
    svg_canvas = np.full_like(rgb_orig, 250)
    cv2.rectangle(
        svg_canvas,
        (2, 2),
        (svg_canvas.shape[1] - 3, svg_canvas.shape[0] - 3),
        (136, 136, 136),
        2,
    )
    for fc in artifacts.fitted_curves:
        pts_i = np.round(fc.points - offset_vec).astype(np.int32)
        c_rgb = col_rgb_map.get(fc.color, (30, 136, 229))
        if len(pts_i) >= 2:
            cv2.polylines(
                svg_canvas,
                [pts_i.reshape(-1, 1, 2)],
                fc.model in ("rect", "polygon", "circle", "ellipse"),
                c_rgb,
                3,
                cv2.LINE_AA,
            )

    saved_files.append(
        _save_side_by_side(
            out_path / "09_output.png",
            rgb_orig,
            svg_canvas,
            "Flattened ROI Photo",
            f"4-Layer SVG Output ({len(artifacts.fitted_curves)} strokes)",
            "Stage 09: Final 4-Layer SVG & JSON Output",
        )
    )

    # 10_qc_overlay.png: Full ROI resolution (1:1 pixels, 50% opacity overlay of all 4 SVG layers)
    qc_bgr = render_qc_overlay(artifacts, alpha=0.50)
    qc_file = out_path / "10_qc_overlay.png"
    cv2.imwrite(str(qc_file), qc_bgr)
    saved_files.append(str(qc_file))

    svg_file = out_path / "output.svg"
    svg_file.write_text(artifacts.svg_string, encoding="utf-8")
    saved_files.append(str(svg_file))

    return saved_files

