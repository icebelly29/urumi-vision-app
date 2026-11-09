"""Stage 9: Pipeline orchestrator, fallback retry logic, and millimeter-accurate SVG export."""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import (
    CaptureOutput,
    Config,
    EnhancementOutput,
    FittedCurve,
    PreprocessOutput,
    SkeletonOutput,
    ThresholdOutput,
    VectorizeOutput,
)
from .fitting import stage7_and_8_fit_and_validate
from .stages import (
    stage1_capture_gate,
    stage2_preprocess,
    stage3_enhance_strokes,
    stage4_hysteresis_threshold,
    stage5_skeleton_and_grouping,
    stage6_vectorize,
)


@dataclass
class PipelineArtifacts:
    """Intermediate outputs across all stages for visualization and debugging."""
    cfg: Config
    capture: CaptureOutput
    prep: PreprocessOutput
    enh: EnhancementOutput
    thresh: ThresholdOutput
    skel: SkeletonOutput
    vec: VectorizeOutput
    fitted_curves: List[FittedCurve]
    status: str
    confidence: float
    attempt_used: int
    svg_string: str


def generate_layered_svg(
    fitted_curves: List[FittedCurve],
    img_width: int,
    img_height: int,
    phys_width_mm: float = 775.0,
    phys_height_mm: float = 1065.0,
    cardboard_polygon_bed: Optional[np.ndarray] = None,
    stroke_width_mm: float = 1.0,
) -> Tuple[str, Dict[str, float]]:
    """Generate 4-layer millimeter-accurate SVG (viewBox in bed mm: 775x1065).

    Layers in order:
      1. <g id="bed" data-role="reference">: bed outline rect + detected cardboard sheet polygon
      2. <g id="red">
      3. <g id="green">
      4. <g id="blue">
    """
    dots_per_mm = float(img_width) / float(phys_width_mm)

    def to_mm(px: float) -> float:
        return float(px) / dots_per_mm

    sw_str = f"{stroke_width_mm:g}"
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {phys_width_mm:g} {phys_height_mm:g}" width="100%" style="display:block;">',
        f'  <g id="bed" data-role="reference">',
        f'    <rect x="0" y="0" width="{phys_width_mm:g}" height="{phys_height_mm:g}" fill="none" stroke="#888" stroke-width="0.5" stroke-dasharray="4,4" data-role="reference"/>',
    ]
    if cardboard_polygon_bed is not None and len(cardboard_polygon_bed) >= 3:
        pts_bed = " ".join(
            f"{to_mm(float(pt[0])):.2f},{to_mm(float(pt[1])):.2f}"
            for pt in cardboard_polygon_bed
        )
        lines.append(
            f'    <polygon points="{pts_bed}" fill="none" stroke="#888" stroke-width="0.5" stroke-dasharray="4,4" data-role="reference"/>'
        )
    lines.append("  </g>")

    layers: Dict[str, List[FittedCurve]] = {
        "red": [],
        "green": [],
        "blue": [],
    }
    default_hex = {"red": "#e53935", "green": "#43a047", "blue": "#1e88e5"}
    for c in fitted_curves:
        c_key = c.color if c.color in layers else (
            "red" if c.layer == "score" else ("green" if c.layer == "crease" else "blue")
        )
        layers[c_key].append(c)

    for layer_id in ("red", "green", "blue"):
        curve_list = layers[layer_id]
        lines.append(f'  <g id="{layer_id}">')
        for c in curve_list:
            col = c.color_hex or default_hex[layer_id]
            common_attrs = f'fill="none" stroke="{col}" stroke-width="{sw_str}" stroke-linecap="round" stroke-linejoin="round"'
            p = c.params
            ctype = c.model or c.curve_type
            if ctype == "line":
                x1, y1 = to_mm(float(p["x1"])), to_mm(float(p["y1"]))
                x2, y2 = to_mm(float(p["x2"])), to_mm(float(p["y2"]))
                lines.append(
                    f'    <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" {common_attrs}/>'
                )
            elif ctype == "circle":
                cx, cy = to_mm(float(p["cx"])), to_mm(float(p["cy"]))
                r = to_mm(float(p["r"]))
                lines.append(
                    f'    <circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r:.3f}" {common_attrs}/>'
                )
            elif ctype == "ellipse":
                cx, cy = to_mm(float(p["cx"])), to_mm(float(p["cy"]))
                rx, ry = to_mm(float(p["rx"])), to_mm(float(p["ry"]))
                ang = float(p.get("angle_deg", 0.0))
                lines.append(
                    f'    <ellipse cx="{cx:.3f}" cy="{cy:.3f}" rx="{rx:.3f}" ry="{ry:.3f}" transform="rotate({ang:.2f} {cx:.3f} {cy:.3f})" {common_attrs}/>'
                )
            elif ctype == "rect":
                ang = float(p.get("angle_deg", 0.0))
                if abs(ang) <= 1e-2 and "x" in p and "y" in p:
                    rx = to_mm(float(p["x"]))
                    ry = to_mm(float(p["y"]))
                    rw = to_mm(float(p["width"]))
                    rh = to_mm(float(p["height"]))
                    lines.append(
                        f'    <rect x="{rx:.3f}" y="{ry:.3f}" width="{rw:.3f}" height="{rh:.3f}" {common_attrs}/>'
                    )
                else:
                    pts_u = c.points[:-1] if len(c.points) > 4 else c.points
                    pts_str = " ".join(
                        f"{to_mm(float(pt[0])):.3f},{to_mm(float(pt[1])):.3f}"
                        for pt in pts_u
                    )
                    lines.append(f'    <polygon points="{pts_str}" {common_attrs}/>')
            elif ctype == "polygon":
                pts_u = (
                    c.points[:-1]
                    if (
                        len(c.points) > 3
                        and np.hypot(
                            c.points[-1, 0] - c.points[0, 0],
                            c.points[-1, 1] - c.points[0, 1],
                        )
                        < 3.0
                    )
                    else c.points
                )
                pts_str = " ".join(
                    f"{to_mm(float(pt[0])):.3f},{to_mm(float(pt[1])):.3f}"
                    for pt in pts_u
                )
                lines.append(f'    <polygon points="{pts_str}" {common_attrs}/>')
            elif ctype == "bezier":
                if "beziers" in p and p["beziers"]:
                    bez_list = p["beziers"]  # type: ignore[index]
                    first_seg = bez_list[0]  # type: ignore[index]
                    d_parts = [
                        f"M {to_mm(float(first_seg[0][0])):.3f},{to_mm(float(first_seg[0][1])):.3f}"  # type: ignore[index]
                    ]
                    for seg in bez_list:  # type: ignore[union-attr]
                        c1 = (to_mm(float(seg[1][0])), to_mm(float(seg[1][1])))  # type: ignore[index]
                        c2 = (to_mm(float(seg[2][0])), to_mm(float(seg[2][1])))  # type: ignore[index]
                        c3 = (to_mm(float(seg[3][0])), to_mm(float(seg[3][1])))  # type: ignore[index]
                        d_parts.append(
                            f"C {c1[0]:.3f},{c1[1]:.3f} {c2[0]:.3f},{c2[1]:.3f} {c3[0]:.3f},{c3[1]:.3f}"
                        )
                    d_str = " ".join(d_parts)
                    lines.append(f'    <path d="{d_str}" {common_attrs}/>')
                elif "control_points" in p:
                    cp = p["control_points"]  # type: ignore[index]
                    p0 = (to_mm(cp[0][0]), to_mm(cp[0][1]))  # type: ignore[index]
                    p1 = (to_mm(cp[1][0]), to_mm(cp[1][1]))  # type: ignore[index]
                    p2 = (to_mm(cp[2][0]), to_mm(cp[2][1]))  # type: ignore[index]
                    p3 = (to_mm(cp[3][0]), to_mm(cp[3][1]))  # type: ignore[index]
                    d_str = (
                        f"M {p0[0]:.3f},{p0[1]:.3f} "
                        f"C {p1[0]:.3f},{p1[1]:.3f} {p2[0]:.3f},{p2[1]:.3f} {p3[0]:.3f},{p3[1]:.3f}"
                    )
                    lines.append(f'    <path d="{d_str}" {common_attrs}/>')
            else:
                pts = c.points
                if len(pts) >= 2:
                    pts_str = " ".join(
                        f"{to_mm(float(pt[0])):.3f},{to_mm(float(pt[1])):.3f}"
                        for pt in pts
                    )
                    lines.append(f'    <polyline points="{pts_str}" {common_attrs}/>')
        lines.append("  </g>")

    lines.append("</svg>")
    meta = {
        "dots_per_mm": round(dots_per_mm, 4),
        "physical_width": phys_width_mm,
        "physical_height": phys_height_mm,
    }
    return "\n".join(lines), meta


def _run_single_attempt(
    capture: CaptureOutput, cfg: Config
) -> Tuple[
    PreprocessOutput,
    EnhancementOutput,
    ThresholdOutput,
    SkeletonOutput,
    VectorizeOutput,
    List[FittedCurve],
    float,
]:
    """Execute Stages 2 through 8 once with the given Config."""
    prep = stage2_preprocess(capture, cfg)
    enh = stage3_enhance_strokes(prep, cfg)
    thresh = stage4_hysteresis_threshold(enh, cfg, prep)
    skel = stage5_skeleton_and_grouping(thresh, cfg, prep)
    vec = stage6_vectorize(skel, prep, cfg)
    fitted = stage7_and_8_fit_and_validate(vec.curves, prep, enh, cfg)

    if not fitted:
        overall_conf = 0.0
    else:
        overall_conf = float(max(c.confidence for c in fitted))

    return prep, enh, thresh, skel, vec, fitted, overall_conf


def run_pipeline(
    image_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Tuple[Dict[str, object], PipelineArtifacts]:
    """Run all 9 stages of the stroke detection pipeline with up to 3 fallback retries.

    Args:
        image_bgr: Input BGR uint8 image.
        cfg: Optional Config instance (uses defaults if None).

    Returns:
        Tuple of (JSON-serializable result dictionary, PipelineArtifacts for visualization).
    """
    if cfg is None:
        cfg = Config()

    capture = stage1_capture_gate(image_bgr, cfg)

    # Build list of retry configurations (Attempt 0 = primary, Attempts 1..3 = retries)
    retry_configs: List[Config] = [cfg]
    if cfg.max_retries >= 1:
        # Retry 1: Larger ridge sigmas + slightly looser hysteresis percentiles
        retry_configs.append(
            replace(
                cfg,
                ridge_sigmas=(2.0, 3.5),
                hysteresis_high_pct=max(94.0, cfg.hysteresis_high_pct - 1.5),
                hysteresis_low_pct=max(85.0, cfg.hysteresis_low_pct - 3.0),
                min_skeleton_length=max(70, int(cfg.min_skeleton_length * 0.75)),
            )
        )
    if cfg.max_retries >= 2:
        # Retry 2: Frangi filter instead of Sato
        retry_configs.append(
            replace(
                cfg,
                ridge_filter="frangi",
                ridge_sigmas=(1.5, 2.5, 3.5),
                min_raw_strokeness=0.0005,
                min_absolute_high=0.10,
                hysteresis_high_pct=96.0,
                hysteresis_low_pct=88.0,
            )
        )
    if cfg.max_retries >= 3:
        # Retry 3: Meijering ridge filter with broader scales
        retry_configs.append(
            replace(
                cfg,
                ridge_filter="meijering",
                ridge_sigmas=(1.5, 2.5, 3.5),
                min_raw_strokeness=0.05,
                hysteresis_high_pct=96.5,
                hysteresis_low_pct=89.0,
            )
        )

    best_tuple = None
    best_conf = -1.0
    best_attempt = 0

    for attempt_idx, try_cfg in enumerate(retry_configs):
        prep, enh, thresh, skel, vec, fitted, conf = _run_single_attempt(
            capture, try_cfg
        )
        if conf > best_conf or best_tuple is None:
            best_conf = conf
            best_tuple = (try_cfg, prep, enh, thresh, skel, vec, fitted)
            best_attempt = attempt_idx
        if conf >= cfg.confidence_threshold and len(fitted) > 0:
            break

    assert best_tuple is not None
    used_cfg, prep, enh, thresh, skel, vec, fitted = best_tuple

    if (
        capture.quality.rejected
        or best_conf < cfg.confidence_threshold
        or len(fitted) == 0
    ):
        status = "not_detected"
        final_curves: List[FittedCurve] = []
        final_conf = round(max(0.0, best_conf), 3) if len(fitted) > 0 else 0.0
    else:
        status = "detected"
        final_curves = [
            c for c in fitted if c.confidence >= (cfg.confidence_threshold * 0.85)
        ]
        if not final_curves:
            final_curves = [fitted[0]]
        final_conf = round(best_conf, 3)

    bed_h, bed_w = prep.full_bed_shape
    if bed_w <= 0 or bed_h <= 0:
        bed_h, bed_w = prep.original_shape
    svg_str, svg_meta = generate_layered_svg(
        final_curves,
        bed_w,
        bed_h,
        capture.phys_width_mm,
        capture.phys_height_mm,
        cardboard_polygon_bed=capture.cardboard_polygon_bed,
        stroke_width_mm=used_cfg.svg_stroke_width_mm,
    )

    curves_json: List[Dict[str, object]] = []
    # Include reference bed layer entries in JSON as layer: "bed"
    curves_json.append(
        {
            "type": "rect",
            "model": "rect",
            "color": "#888",
            "layer": "bed",
            "role": "reference",
            "residual_mm": 0.0,
            "params": {
                "x": 0.0,
                "y": 0.0,
                "width": float(bed_w),
                "height": float(bed_h),
            },
            "points": [
                [0.0, 0.0],
                [float(bed_w), 0.0],
                [float(bed_w), float(bed_h)],
                [0.0, float(bed_h)],
                [0.0, 0.0],
            ],
            "length_px": round(2.0 * float(bed_w + bed_h), 2),
            "mean_contrast": 1.0,
            "fit_error_px": 0.0,
            "inlier_ratio": 1.0,
            "angular_coverage_deg": 360.0,
            "confidence": 1.0,
        }
    )
    if (
        capture.cardboard_polygon_bed is not None
        and len(capture.cardboard_polygon_bed) >= 3
    ):
        cb_pts = [
            [round(float(p[0]), 2), round(float(p[1]), 2)]
            for p in capture.cardboard_polygon_bed
        ]
        cb_closed = cb_pts + [cb_pts[0]]
        cb_diffs = np.diff(np.array(cb_closed, dtype=np.float64), axis=0)
        cb_len = float(np.sum(np.hypot(cb_diffs[:, 0], cb_diffs[:, 1])))
        curves_json.append(
            {
                "type": "polygon",
                "model": "polygon",
                "color": "#888",
                "layer": "bed",
                "role": "reference",
                "residual_mm": 0.0,
                "params": {"num_vertices": len(cb_pts), "closed": True, "vertices": cb_pts},
                "points": cb_closed,
                "length_px": round(cb_len, 2),
                "mean_contrast": 1.0,
                "fit_error_px": 0.0,
                "inlier_ratio": 1.0,
                "angular_coverage_deg": 360.0,
                "confidence": 1.0,
            }
        )

    for c in final_curves:
        pts_list = [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in c.points]
        c_color = c.color or (
            "red" if c.layer == "score" else ("green" if c.layer == "crease" else "blue")
        )
        c_model = c.model or c.curve_type
        c_layer = c.color if c.color in ("red", "green", "blue") else c.layer
        curves_json.append(
            {
                "type": c.curve_type,
                "model": c_model,
                "color": c_color,
                "residual_mm": c.residual_mm,
                "layer": c_layer,
                "params": c.params,
                "points": pts_list,
                "length_px": c.length_px,
                "mean_contrast": c.mean_contrast,
                "fit_error_px": c.fit_error_px,
                "inlier_ratio": c.inlier_ratio,
                "angular_coverage_deg": c.angular_coverage_deg,
                "confidence": c.confidence,
            }
        )

    result_dict: Dict[str, object] = {
        "status": status,
        "confidence": final_conf,
        "curves": curves_json,
        "quality": {
            "blur_score": round(capture.quality.blur_score, 2),
            "brightness": round(capture.quality.brightness, 2),
            "overlay_masked": capture.quality.overlay_masked,
            "aruco_warped": capture.warped_from_aruco,
            "aruco_markers_found": capture.aruco_markers_found,
        },
        "svg": svg_str,
        "svg_meta": svg_meta,
        "debug": {
            "strokeness": f"float32 array {enh.strokeness.shape}, raw_max={enh.raw_max:.4f}",
            "mask": f"bool array {thresh.mask.shape}, active_px={int(np.sum(thresh.mask))}",
            "skeleton": f"bool array {skel.skeleton.shape}, skeleton_px={int(np.sum(skel.skeleton))}",
            "attempt_used": best_attempt,
        },
    }

    artifacts = PipelineArtifacts(
        cfg=used_cfg,
        capture=capture,
        prep=prep,
        enh=enh,
        thresh=thresh,
        skel=skel,
        vec=vec,
        fitted_curves=final_curves,
        status=status,
        confidence=final_conf,
        attempt_used=best_attempt,
        svg_string=svg_str,
    )
    return result_dict, artifacts


def detect_strokes(
    image_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Dict[str, object]:
    """Public convenience API returning the JSON-serializable stroke detection output."""
    result_dict, _ = run_pipeline(image_bgr, cfg)
    return result_dict

