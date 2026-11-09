"""Multi-Color Marker Extraction (Red, Green, Blue) & Machine Bed Layer Pipeline.

This module implements a 5-stage computer vision pipeline for extracting
multi-color hand-drawn marker sketches (Red, Green, Blue) and the cardboard
workpiece boundary from a top-down calibrated image of a CNC flatbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:
    _sk_skeletonize = None


@dataclass
class BedSubstrateResult:
    """Output of Stage 1: Machine Bed & Workpiece Sheet Detection."""

    bed_polygon: np.ndarray  # (4, 2) outer machine bed frame in pixel coords
    substrate_polygon: np.ndarray  # (N, 2) convex/approximated cardboard contour in pixel coords
    substrate_bbox: Tuple[int, int, int, int]  # (x, y, w, h) of cardboard ROI
    substrate_mask: np.ndarray  # uint8 (H, W) mask (255 inside valid cardboard interior)
    roi_bgr: np.ndarray  # BGR image masked to substrate interior


def detect_bed_and_substrate(
    img_warped: np.ndarray,
    border_margin_px: int = 14,
    frame_margin_px: int = 80,
) -> BedSubstrateResult:
    """Stage 1: Identify machine bed frame and cardboard workpiece boundary.

    Args:
        img_warped: Top-down calibrated BGR image (H, W, 3).
        border_margin_px: Inward erosion margin in pixels to suppress edge shadows.
        frame_margin_px: Outer peripheral margin to mask ArUco markers and bed clamps.

    Returns:
        BedSubstrateResult containing bed bounds, cardboard polygon, mask, and masked BGR.
    """
    if img_warped is None or img_warped.ndim != 3:
        raise ValueError("Input img_warped must be a 3-channel BGR image.")

    h, w = img_warped.shape[:2]
    bed_polygon = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]],
        dtype=np.float32,
    )

    # Isolate brown/tan kraft cardboard in HSV (Hue 7..42, Saturation >= 25, Value >= 35)
    hsv = cv2.cvtColor(img_warped, cv2.COLOR_BGR2HSV)
    candidate_mask = cv2.inRange(
        hsv,
        np.array([7, 25, 35], dtype=np.uint8),
        np.array([42, 255, 255], dtype=np.uint8),
    )

    # Mask peripheral machine hardware and ArUco markers along the bed frame border
    if h > 2 * frame_margin_px and w > 2 * frame_margin_px:
        # Only zero out outer frame margin if the image is not already a tight crop
        cb_ratio = float(np.count_nonzero(candidate_mask)) / float(max(1, h * w))
        if cb_ratio < 0.80:
            candidate_mask[:frame_margin_px, :] = 0
            candidate_mask[h - frame_margin_px :, :] = 0
            candidate_mask[:, :frame_margin_px] = 0
            candidate_mask[:, w - frame_margin_px :] = 0

    # Morphological closing to bridge internal marker strokes inside the sheet
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_k)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, open_k)

    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [c for c in contours if cv2.contourArea(c) > 8000.0]

    if valid_contours:
        largest_cnt = max(valid_contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest_cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.018 * peri, True)
        substrate_polygon = approx.reshape(-1, 2).astype(np.float32)
        x, y, cw, ch = cv2.boundingRect(hull)
        substrate_bbox = (int(x), int(y), int(cw), int(ch))

        substrate_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(substrate_mask, [hull], -1, 255, thickness=cv2.FILLED)
    else:
        substrate_polygon = bed_polygon.copy()
        substrate_bbox = (0, 0, w, h)
        substrate_mask = np.full((h, w), 255, dtype=np.uint8)

    if border_margin_px > 0:
        erode_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * border_margin_px + 1, 2 * border_margin_px + 1)
        )
        substrate_mask = cv2.erode(substrate_mask, erode_k)

    roi_bgr = cv2.bitwise_and(img_warped, img_warped, mask=substrate_mask)
    return BedSubstrateResult(
        bed_polygon=bed_polygon,
        substrate_polygon=substrate_polygon,
        substrate_bbox=substrate_bbox,
        substrate_mask=substrate_mask,
        roi_bgr=roi_bgr,
    )


def _hysteresis_threshold(
    response: np.ndarray, high_thresh: float, low_thresh: float
) -> np.ndarray:
    """Connect weak pixels above low_thresh that touch strong seed pixels above high_thresh."""
    high_mask = (response >= high_thresh).astype(np.uint8)
    low_mask = (response >= low_thresh).astype(np.uint8)
    if not np.any(high_mask):
        return np.zeros_like(high_mask)

    num_labels, labels = cv2.connectedComponents(low_mask, connectivity=8)
    if num_labels <= 1:
        return high_mask * 255

    seed_labels = np.unique(labels[high_mask > 0])
    seed_labels = seed_labels[seed_labels > 0]
    keep = np.isin(labels, seed_labels)
    return (keep.astype(np.uint8)) * 255


def segment_ink_colors(
    roi_bgr: np.ndarray,
    substrate_mask: Optional[np.ndarray] = None,
    median_ksize: int = 51,
    t_red: float = 9.5,
    t_green: float = 7.5,
    t_blue: float = 11.0,
) -> Dict[str, np.ndarray]:
    """Stage 2: Chromatic substrate normalization and 3-color (Red, Green, Blue) segmentation.

    Args:
        roi_bgr: Input BGR image (H, W, 3).
        substrate_mask: Optional uint8 mask (H, W) of valid cardboard region.
        median_ksize: Odd kernel size (>= 41) for background substrate estimation.
        t_red: Positive Delta a* threshold for Red ink.
        t_green: Negative Delta a* magnitude threshold for Green ink.
        t_blue: Negative Delta b* magnitude threshold for Blue ink.

    Returns:
        Dictionary mapping {'red': mask_r, 'green': mask_g, 'blue': mask_b} as uint8 (0 or 255).
    """
    if median_ksize < 41:
        median_ksize = 41
    if median_ksize % 2 == 0:
        median_ksize += 1

    # 1. Bilateral filter to suppress cardboard fiber grain while preserving ink stroke edges
    filtered_bgr = cv2.bilateralFilter(roi_bgr, d=9, sigmaColor=45, sigmaSpace=45)

    # 2. Convert to CIELAB color space
    lab = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    # Fill exterior non-substrate pixels with median cardboard color before large median blur
    # so dark background outside the sheet does not bias boundary chrominance estimates.
    if substrate_mask is not None and np.any(substrate_mask > 0):
        valid = substrate_mask > 0
        med_l = int(np.median(l_ch[valid]))
        med_a = int(np.median(a_ch[valid]))
        med_b = int(np.median(b_ch[valid]))
        l_pad = np.where(valid, l_ch, med_l).astype(np.uint8)
        a_pad = np.where(valid, a_ch, med_a).astype(np.uint8)
        b_pad = np.where(valid, b_ch, med_b).astype(np.uint8)
    else:
        valid = np.ones(l_ch.shape, dtype=bool)
        l_pad, a_pad, b_pad = l_ch, a_ch, b_ch

    # 3. Estimate local background substrate baseline (L*_bg, a*_bg, b*_bg)
    l_bg = cv2.medianBlur(l_pad, median_ksize).astype(np.float32)
    a_bg = cv2.medianBlur(a_pad, median_ksize).astype(np.float32)
    b_bg = cv2.medianBlur(b_pad, median_ksize).astype(np.float32)

    # 4. Signed chromatic deviations and relative ink darkness
    delta_l = l_bg - l_ch.astype(np.float32)  # positive when darker than cardboard
    delta_a = a_ch.astype(np.float32) - a_bg  # positive = red/magenta, negative = green
    delta_b = b_ch.astype(np.float32) - b_bg  # negative = blue

    # Supplemental RGB channel contrast to reinforce faint strokes on warm kraft paper
    b_rgb = filtered_bgr[:, :, 0].astype(np.float32)
    g_rgb = filtered_bgr[:, :, 1].astype(np.float32)
    r_rgb = filtered_bgr[:, :, 2].astype(np.float32)

    # Red score: strong +Delta a* and red excess over green
    red_response = delta_a + 0.25 * np.maximum(0.0, (r_rgb - g_rgb) - 25.0)
    red_response = np.where(valid & (delta_l > -5.0), red_response, 0.0)

    # Green score: strong -Delta a* and green dominance over red on dark ink pixels
    green_response = (-delta_a) + 0.30 * np.maximum(0.0, (g_rgb - r_rgb) + 18.0) * (
        delta_l > 4.0
    )
    green_response = np.where(
        valid & (delta_a < -2.0) & (delta_l > 2.0), green_response, 0.0
    )

    # Blue score: strong -Delta b* combined with non-green a* (delta_a > -4.0) and B >= G - 10
    blue_response = (-delta_b) + 0.25 * np.maximum(0.0, b_rgb - g_rgb)
    blue_response = np.where(
        valid
        & (delta_b < -6.0)
        & (delta_a > -4.0)
        & (delta_a < t_red * 0.75)
        & (b_rgb >= g_rgb - 8.0)
        & (delta_l > 4.0),
        blue_response,
        0.0,
    )

    # 5. Hysteresis thresholding per color channel to capture faint stroke ends (e.g. arrow wings)
    mask_red = _hysteresis_threshold(red_response, t_red, t_red * 0.58)
    mask_green = _hysteresis_threshold(green_response, t_green, t_green * 0.58)
    mask_blue = _hysteresis_threshold(blue_response, t_blue, t_blue * 0.62)

    # Mutual exclusivity at non-junction pixels: assign ambiguous border pixels to strongest response
    # while allowing true cross-color intersections to stay connected via morphological closing.
    el_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    el_5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    out_masks: Dict[str, np.ndarray] = {}
    for name, raw_m in [
        ("red", mask_red),
        ("green", mask_green),
        ("blue", mask_blue),
    ]:
        # Close tiny intra-stroke gaps, then open to drop isolated fiber speckles
        cleaned = cv2.morphologyEx(raw_m, cv2.MORPH_CLOSE, el_5x5)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, el_3x3)

        # Remove tiny disconnected noise blobs (< 25 px area)
        n_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        filtered = np.zeros_like(cleaned)
        for lbl_idx in range(1, n_cc):
            area = int(cc_stats[lbl_idx, cv2.CC_STAT_AREA])
            w_box = int(cc_stats[lbl_idx, cv2.CC_STAT_WIDTH])
            h_box = int(cc_stats[lbl_idx, cv2.CC_STAT_HEIGHT])
            if area >= 25 or max(w_box, h_box) >= 12:
                filtered[cc_labels == lbl_idx] = 255

        out_masks[name] = filtered

    return out_masks


def _zhang_suen_thinning(binary_bool: np.ndarray) -> np.ndarray:
    """Fallback morphological skeletonization if scikit-image is unavailable."""
    if _sk_skeletonize is not None:
        return _sk_skeletonize(binary_bool)

    img = (binary_bool.astype(np.uint8)) * 255
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel > 0


def _compute_neighbor_degree(skel_u8: np.ndarray) -> np.ndarray:
    """Compute 8-connected neighbor count for each foreground skeleton pixel."""
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    nbrs = cv2.filter2D(skel_u8, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    return nbrs * skel_u8


def skeletonize_and_prune(
    binary_mask: np.ndarray,
    spur_factor: float = 3.0,
    min_component_length: int = 12,
) -> np.ndarray:
    """Stage 3: Distance-regularized centerline extraction and terminal hair spur pruning.

    Args:
        binary_mask: Single-color binary mask (uint8, 0 or 255).
        spur_factor: Multiplier on median stroke width for terminal branch pruning.
        min_component_length: Minimum pixel count for an isolated skeleton component.

    Returns:
        Pruned 1-pixel skeleton as uint8 mask (0 or 255).
    """
    if binary_mask is None or not np.any(binary_mask > 0):
        return np.zeros_like(binary_mask, dtype=np.uint8)

    mask_u8 = (binary_mask > 0).astype(np.uint8)

    # 1. Euclidean Distance Transform to estimate stroke width and regularize rough fiber edges
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    fg_dists = dist[dist > 0]
    median_radius = float(np.median(fg_dists)) if fg_dists.size > 0 else 2.0
    stroke_width = max(3.0, 2.0 * median_radius)
    max_spur_len = int(np.clip(round(spur_factor * stroke_width), 10, 16))

    # Light Gaussian smoothing on distance field to suppress 1-px boundary indentations before thinning
    smooth_mask = cv2.GaussianBlur((mask_u8 * 255).astype(np.uint8), (3, 3), 0) >= 110
    skel_bool = _zhang_suen_thinning(smooth_mask)
    skel = skel_bool.astype(np.uint8)

    # 2. Iteratively strip terminal hair spurs (shorter than max_spur_len) attached to junctions
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    h, w = skel.shape

    for _ in range(3):
        deg = _compute_neighbor_degree(skel)
        endpoints = np.argwhere((skel > 0) & (deg == 1))
        if endpoints.size == 0:
            break

        pruned_any = False
        for ey, ex in endpoints:
            if skel[ey, ex] == 0:
                continue
            path: List[Tuple[int, int]] = [(int(ey), int(ex))]
            cy, cx = int(ey), int(ex)
            py, px = -1, -1
            hit_junction = False

            for _step in range(max_spur_len):
                nxt = None
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] > 0:
                        if ny != py or nx != px:
                            nxt = (ny, nx)
                            break
                if nxt is None:
                    break
                ny, nx = nxt
                if deg[ny, nx] >= 3:
                    hit_junction = True
                    break
                path.append((ny, nx))
                py, px = cy, cx
                cy, cx = ny, nx

            # Only prune if the branch terminated at a junction (deg >= 3) within max_spur_len
            if hit_junction and len(path) <= max_spur_len:
                for ry, rx in path:
                    skel[ry, rx] = 0
                pruned_any = True

        if not pruned_any:
            break

    # 3. Filter tiny isolated skeleton fragments
    n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(skel, connectivity=8)
    clean_skel = np.zeros_like(skel)
    for idx in range(1, n_cc):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= min_component_length:
            clean_skel[labels == idx] = 255

    return clean_skel


def _trace_skeleton_polylines(
    skel_mask: np.ndarray,
    rdp_epsilon: float = 2.0,
) -> List[np.ndarray]:
    """Stage 4: Trace ordered pixel chains from a 1-px skeleton and simplify via RDP.

    Splits at junction clusters so T-junctions and X-junctions preserve crisp topology,
    then simplifies each continuous chain with Ramer-Douglas-Peucker (cv2.approxPolyDP).
    """
    skel = (skel_mask > 0).astype(np.uint8)
    if not np.any(skel):
        return []

    deg = _compute_neighbor_degree(skel)
    junction_mask = ((skel > 0) & (deg >= 3)).astype(np.uint8)

    # Dilate junctions by 1 px so branches separate cleanly into degree<=2 chains
    j_dilated = cv2.dilate(junction_mask, np.ones((3, 3), dtype=np.uint8))
    j_Zone = j_dilated & skel
    n_j, j_labels, _, j_centroids = cv2.connectedComponentsWithStats(
        j_Zone, connectivity=8
    )

    edges_only = (skel > 0) & (j_Zone == 0)
    n_e, e_labels = cv2.connectedComponents(edges_only.astype(np.uint8), connectivity=8)

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = skel.shape
    polylines: List[np.ndarray] = []

    def _touching_junction(pt_xy: Tuple[int, int]) -> int:
        px, py = pt_xy
        y0, y1 = max(0, py - 2), min(h, py + 3)
        x0, x1 = max(0, px - 2), min(w, px + 3)
        patch = j_labels[y0:y1, x0:x1]
        nz = patch[patch > 0]
        return int(nz[0]) if nz.size > 0 else -1

    for e_idx in range(1, n_e):
        ys, xs = np.where(e_labels == e_idx)
        if ys.size < 2:
            continue

        comp_set = set(zip(xs.tolist(), ys.tolist()))
        # Find an endpoint (<= 1 neighbor inside comp_set) to start ordering
        start_pt = (int(xs[0]), int(ys[0]))
        for x_i, y_i in comp_set:
            nbr_cnt = sum(
                1 for dy, dx in dirs if (x_i + dx, y_i + dy) in comp_set
            )
            if nbr_cnt <= 1:
                start_pt = (x_i, y_i)
                break

        ordered: List[Tuple[float, float]] = [
            (float(start_pt[0]), float(start_pt[1]))
        ]
        visited = {start_pt}
        curr = start_pt
        while True:
            cx, cy = curr
            nxt_pt = None
            for dy, dx in dirs:
                cand = (cx + dx, cy + dy)
                if cand in comp_set and cand not in visited:
                    nxt_pt = cand
                    break
            if nxt_pt is None:
                break
            visited.add(nxt_pt)
            ordered.append((float(nxt_pt[0]), float(nxt_pt[1])))
            curr = nxt_pt

        # Attach junction centroids at start/end so T and X intersections meet seamlessly
        j_start = _touching_junction((int(ordered[0][0]), int(ordered[0][1])))
        j_end = _touching_junction((int(ordered[-1][0]), int(ordered[-1][1])))

        if j_start > 0:
            jc = j_centroids[j_start]
            ordered.insert(0, (float(jc[0]), float(jc[1])))
        if j_end > 0 and (len(ordered) > 2 or j_end != j_start):
            jc = j_centroids[j_end]
            ordered.append((float(jc[0]), float(jc[1])))

        pts_arr = np.array(ordered, dtype=np.float32)
        is_closed = (
            len(pts_arr) >= 12
            and float(np.hypot(*(pts_arr[0] - pts_arr[-1]))) <= 4.0
        )
        simp = cv2.approxPolyDP(pts_arr, epsilon=rdp_epsilon, closed=is_closed).reshape(
            -1, 2
        )
        if is_closed and len(simp) >= 3:
            simp = np.vstack([simp, simp[0:1]])
        if len(simp) >= 2:
            polylines.append(simp)

    return polylines


def _polyline_to_svg_path(
    pts: np.ndarray,
    mm_per_px: float,
    fit_bezier: bool = False,
) -> str:
    """Convert a 2D point array (N, 2) into an SVG path 'd' attribute string."""
    scaled = pts.astype(np.float64) * float(mm_per_px)
    if len(scaled) < 2:
        return ""

    if not fit_bezier or len(scaled) < 4:
        segs = [f"M {scaled[0, 0]:.2f},{scaled[0, 1]:.2f}"]
        segs.extend(f"L {p[0]:.2f},{p[1]:.2f}" for p in scaled[1:])
        return " ".join(segs)

    # Smooth Catmull-Rom to Cubic Bezier conversion
    d_parts = [f"M {scaled[0, 0]:.2f},{scaled[0, 1]:.2f}"]
    n = len(scaled)
    for i in range(n - 1):
        p0 = scaled[max(0, i - 1)]
        p1 = scaled[i]
        p2 = scaled[i + 1]
        p3 = scaled[min(n - 1, i + 2)]

        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        d_parts.append(
            f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}"
        )
    return " ".join(d_parts)


def trace_and_generate_svg(
    layers_dict: Dict[str, Union[np.ndarray, List[np.ndarray]]],
    image_dims: Tuple[int, int],
    mm_per_px: float = 1.0,
    rdp_epsilon: float = 2.0,
    fit_bezier: bool = False,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Stage 4 & 5: Trace skeleton masks and generate a multi-layer production SVG.

    Args:
        layers_dict: Dictionary containing:
            - 'bed': (N, 2) polygon array (or list of polygons) of the cardboard/bed outline
            - 'red': binary skeleton mask (H, W) or list of polylines
            - 'green': binary skeleton mask (H, W) or list of polylines
            - 'blue': binary skeleton mask (H, W) or list of polylines
        image_dims: Tuple of (height, width) in pixels.
        mm_per_px: Scale factor converting pixels to physical millimeters.
        rdp_epsilon: Ramer-Douglas-Peucker simplification tolerance in pixels (1.5 - 2.5).
        fit_bezier: If True, emit smooth cubic Bezier splines; otherwise emit polylines.
        output_path: Optional file path to write the SVG document.

    Returns:
        Formatted SVG XML string.
    """
    h_px, w_px = int(image_dims[0]), int(image_dims[1])
    width_mm = w_px * float(mm_per_px)
    height_mm = h_px * float(mm_per_px)

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{width_mm:.2f}mm",
            "height": f"{height_mm:.2f}mm",
            "viewBox": f"0 0 {width_mm:.2f} {height_mm:.2f}",
            "data-units": "mm",
            "data-mm-per-px": f"{mm_per_px:.6f}",
        },
    )

    # 1. Machine Bed / Workpiece Boundary Layer
    g_bed = ET.SubElement(
        svg,
        "g",
        {
            "id": "layer_bed",
            "data-cam-layer": "bed_boundary",
            "stroke": "#424242",
            "stroke-width": "1.5",
            "fill": "none",
        },
    )
    bed_data = layers_dict.get("bed")
    if bed_data is not None:
        polys = [bed_data] if isinstance(bed_data, np.ndarray) else list(bed_data)
        for poly in polys:
            poly_arr = np.asarray(poly, dtype=np.float64).reshape(-1, 2) * float(
                mm_per_px
            )
            if len(poly_arr) >= 3:
                pts_str = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in poly_arr)
                ET.SubElement(g_bed, "polygon", {"points": pts_str})

    # 2. Marker Color Layers (Red, Green, Blue)
    layer_specs = [
        ("red", "layer_red", "#E53935", "cut"),
        ("green", "layer_green", "#43A047", "crease"),
        ("blue", "layer_blue", "#1E88E5", "score"),
    ]

    for key, group_id, stroke_hex, cam_op in layer_specs:
        g_el = ET.SubElement(
            svg,
            "g",
            {
                "id": group_id,
                "data-cam-operation": cam_op,
                "stroke": stroke_hex,
                "stroke-width": "2",
                "fill": "none",
                "stroke-linecap": "round",
                "stroke-linejoin": "round",
            },
        )
        layer_val = layers_dict.get(key)
        if layer_val is None:
            continue

        if isinstance(layer_val, np.ndarray) and layer_val.ndim == 2:
            polylines = _trace_skeleton_polylines(layer_val, rdp_epsilon=rdp_epsilon)
        else:
            polylines = [np.asarray(p, dtype=np.float32) for p in layer_val]

        for idx, poly in enumerate(polylines):
            d_str = _polyline_to_svg_path(
                poly, mm_per_px=mm_per_px, fit_bezier=fit_bezier
            )
            if d_str:
                ET.SubElement(
                    g_el,
                    "path",
                    {"id": f"{key}_path_{idx}", "d": d_str},
                )

    ET.indent(svg, space="  ")
    svg_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
        svg, encoding="unicode"
    )

    if output_path is not None:
        Path(output_path).write_text(svg_str, encoding="utf-8")

    return svg_str


def run_multi_color_pipeline(
    image_path: Union[str, Path],
    output_svg_path: Optional[Union[str, Path]] = None,
    mm_per_px: float = 0.5,
    rdp_epsilon: float = 2.0,
    fit_bezier: bool = False,
) -> Tuple[str, Dict[str, np.ndarray]]:
    """Run the full 5-stage pipeline end-to-end on an input image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # Stage 1: Bed & Substrate Detection
    bed_res = detect_bed_and_substrate(img)

    # Stage 2: Chromatic Substrate Normalization & 3-Color Masking
    color_masks = segment_ink_colors(
        bed_res.roi_bgr, substrate_mask=bed_res.substrate_mask
    )

    # Stage 3: Independent Centerline Extraction & Graph Pruning
    skeletons: Dict[str, np.ndarray] = {}
    for color_name, mask in color_masks.items():
        skeletons[color_name] = skeletonize_and_prune(mask)

    # Stage 4 & 5: Topological Polyline Tracing & Multi-Layer SVG Generation
    layers_payload: Dict[str, Union[np.ndarray, List[np.ndarray]]] = {
        "bed": [bed_res.bed_polygon, bed_res.substrate_polygon],
        "red": skeletons["red"],
        "green": skeletons["green"],
        "blue": skeletons["blue"],
    }
    svg_xml = trace_and_generate_svg(
        layers_dict=layers_payload,
        image_dims=(img.shape[0], img.shape[1]),
        mm_per_px=mm_per_px,
        rdp_epsilon=rdp_epsilon,
        fit_bezier=fit_bezier,
        output_path=output_svg_path,
    )
    return svg_xml, skeletons


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Color Marker & Machine Bed Extraction Pipeline"
    )
    parser.add_argument("input", help="Path to input top-down image")
    parser.add_argument(
        "-o", "--output", default="output_multi_layer.svg", help="Output SVG path"
    )
    parser.add_argument(
        "--mm-per-px", type=float, default=0.5, help="Millimeters per pixel scale"
    )
    parser.add_argument(
        "--epsilon", type=float, default=2.0, help="RDP simplification epsilon (px)"
    )
    parser.add_argument(
        "--bezier", action="store_true", help="Fit smooth cubic Bezier curves"
    )
    args = parser.parse_args()

    svg_out, skels = run_multi_color_pipeline(
        args.input,
        output_svg_path=args.output,
        mm_per_px=args.mm_per_px,
        rdp_epsilon=args.epsilon,
        fit_bezier=args.bezier,
    )
    print(f"Saved SVG to {args.output} ({len(svg_out)} bytes)")
