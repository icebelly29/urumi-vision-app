"""Stages 1 through 6 of the hand-drawn stroke detection pipeline."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import apply_hysteresis_threshold, frangi, meijering, sato
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

from .config import (
    CaptureOutput,
    Config,
    EnhancementOutput,
    PreprocessOutput,
    QualityReport,
    SkeletonOutput,
    ThresholdOutput,
    TracedCurve,
    VectorizeOutput,
)


def detect_and_warp_aruco(
    image_bgr: np.ndarray, cfg: Config
) -> Tuple[np.ndarray, bool, float, float, int]:
    """Detect 4 ArUco markers (DICT_4X4_50) and flatten perspective to physical metrics.

    Mirrors WarpEngine.js:
    - IDs 12,13,14,15 -> cropPhysW=775mm, cropPhysH=1065mm, frame 830x1130mm
    - IDs 8..11 -> 270x350mm, margin 16mm
    - IDs 4..7 -> 210x290mm, margin 17mm
    - IDs 0..3 -> 150x230mm, margin 11mm
    """
    src_img = image_bgr
    if abs(cfg.radial_undistort_k1) > 1e-6:
        ih, iw = src_img.shape[:2]
        f_len = float(max(iw, ih))
        cam_mat = np.array(
            [[f_len, 0.0, iw * 0.5], [0.0, f_len, ih * 0.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        dist_coeffs = np.array([cfg.radial_undistort_k1, 0.0, 0.0, 0.0], dtype=np.float64)
        src_img = cv2.undistort(src_img, cam_mat, dist_coeffs)

    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)

    marker_centers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            c = corners[i][0]
            marker_centers[int(marker_id)] = (float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))

    # Fallback: if 1..3 markers found (e.g. small/angled corner marker near image border),
    # upscale 2x with cubic interpolation to recover the missing corner marker.
    if 1 <= len(marker_centers) < 4:
        up_gray = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        c2, ids2, _ = detector.detectMarkers(up_gray)
        if ids2 is not None:
            for i, marker_id in enumerate(ids2.flatten()):
                mid = int(marker_id)
                if mid not in marker_centers:
                    c = c2[i][0] * 0.5
                    marker_centers[mid] = (float(np.mean(c[:, 0])), float(np.mean(c[:, 1])))

    detected_ids = sorted(marker_centers.keys())
    if len(detected_ids) < 4:
        return (
            image_bgr,
            False,
            cfg.default_phys_width_mm,
            cfg.default_phys_height_mm,
            len(detected_ids),
        )

    # Select 4 corner IDs
    if all(k in marker_centers for k in (12, 13, 14, 15)):
        use_ids = [12, 13, 14, 15]
    else:
        use_ids = detected_ids[:4]

    tl = marker_centers[use_ids[0]]
    tr = marker_centers[use_ids[1]]
    br = marker_centers[use_ids[2]]
    bl = marker_centers[use_ids[3]]

    if any(k in use_ids for k in (12, 13, 14, 15)):
        phys_w = cfg.aruco_frame_width_mm
        phys_h = cfg.aruco_frame_height_mm
        crop_w = cfg.default_phys_width_mm
        crop_h = cfg.default_phys_height_mm
        margin_x = (phys_w - crop_w) / 2.0
        margin_y = (phys_h - crop_h) / 2.0
    else:
        phys_w, phys_h, legacy_margin = 210.0, 290.0, 17.0
        if any(k in use_ids for k in (8, 9, 10, 11)):
            phys_w, phys_h, legacy_margin = 270.0, 350.0, 16.0
        elif any(k in use_ids for k in (0, 1, 2, 3)):
            phys_w, phys_h, legacy_margin = 150.0, 230.0, 11.0
        crop_w = phys_w - 2.0 * legacy_margin
        crop_h = phys_h - 2.0 * legacy_margin
        margin_x = legacy_margin
        margin_y = legacy_margin

    out_w = cfg.aruco_output_width
    dots_per_mm = out_w / crop_w
    out_h = int(round(dots_per_mm * crop_h))
    margin_x_px = margin_x * dots_per_mm
    margin_y_px = margin_y * dots_per_mm

    src_tri = np.array([tl, tr, br, bl], dtype=np.float32)
    dst_tri = np.array(
        [
            [-margin_x_px, -margin_y_px],
            [out_w + margin_x_px, -margin_y_px],
            [out_w + margin_x_px, out_h + margin_y_px],
            [-margin_x_px, out_h + margin_y_px],
        ],
        dtype=np.float32,
    )

    mat = cv2.getPerspectiveTransform(src_tri, dst_tri)
    warped = cv2.warpPerspective(
        src_img,
        mat,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, True, crop_w, crop_h, len(detected_ids)


def detect_cardboard_roi(
    warped_bgr: np.ndarray, shave_px: int = 18, frame_margin: int = 80
) -> Optional[Tuple[int, int, int, int]]:
    """Detect cardboard workpiece bounding box on the warped bed (mirrors ContourRoiDetector.js)."""
    rows, cols = warped_bgr.shape[:2]
    hsv = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
    # Brown/tan kraft cardboard has Hue 7..42, Saturation >= 25, Value >= 35
    mask = cv2.inRange(hsv, np.array([7, 25, 35], dtype=np.uint8), np.array([42, 255, 255], dtype=np.uint8))

    m_top = min(frame_margin, rows)
    m_bot = max(0, rows - frame_margin)
    m_left = min(frame_margin, cols)
    m_right = max(0, cols - frame_margin)
    mask[:m_top, :] = 0
    mask[m_bot:, :] = 0
    mask[:, :m_left] = 0
    mask[:, m_right:] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    max_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 8000 and area > max_area:
            max_area = area
            best_rect = cv2.boundingRect(cnt)

    if best_rect is None:
        return None

    bx, by, bw, bh = best_rect
    # If the cardboard mask already covers >85% of the image, the input is already a cardboard close-up
    if bw * bh > 0.85 * rows * cols:
        return None

    inset_x = min(shave_px, int(bw * 0.35))
    inset_y = min(shave_px, int(bh * 0.35))
    bx += inset_x
    by += inset_y
    bw = max(20, bw - 2 * inset_x)
    bh = max(20, bh - 2 * inset_y)
    return (bx, by, bw, bh)


def stage1_capture_gate(image_bgr: np.ndarray, cfg: Config) -> CaptureOutput:
    """Stage 1: Capture/quality gate, optional ArUco flatten + ROI zoom crop, and UI overlay mask.

    Args:
        image_bgr: Input BGR uint8 image.
        cfg: Pipeline configuration.

    Returns:
        CaptureOutput containing the cropped ROI (or full) image, ignore mask, and quality metrics.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty or None.")

    work_img = image_bgr
    warped_flag = False
    phys_w = cfg.default_phys_width_mm
    phys_h = cfg.default_phys_height_mm
    markers_found = 0

    if cfg.enable_aruco_warp:
        # Only run full ArUco marker scan on full-bed frames (or when dark bed border is present)
        ih, iw = image_bgr.shape[:2]
        if max(ih, iw) >= 1400 or float(np.percentile(image_bgr[::8, ::8], 10)) < 75.0:
            work_img, warped_flag, phys_w, phys_h, markers_found = detect_and_warp_aruco(
                image_bgr, cfg
            )

    full_h, full_w = work_img.shape[:2]
    warped_bed_bgr = work_img.copy()
    cardboard_poly_bed = np.array(
        [[0.0, 0.0], [float(full_w), 0.0], [float(full_w), float(full_h)], [0.0, float(full_h)]],
        dtype=np.float64,
    )
    roi_x, roi_y = 0, 0

    bbox = cfg.roi_bbox
    if bbox is None and cfg.auto_detect_roi:
        # Only auto-crop when warped from ArUco or when the frame has a dark cutting bed border
        if warped_flag or float(np.percentile(work_img[::8, ::8], 10)) < 75.0:
            hsv_full = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
            cb_full = cv2.inRange(
                hsv_full,
                np.array([7, 25, 35], dtype=np.uint8),
                np.array([42, 255, 255], dtype=np.uint8),
            )
            cb_ratio = float(np.count_nonzero(cb_full)) / float(max(1, full_h * full_w))
            if warped_flag or cb_ratio < 0.75:
                # Extract cardboard polygon on full bed before cropping
                cb_mask_poly = cb_full.copy()
                fm = 80 if warped_flag else 10
                if full_h > 2 * fm and full_w > 2 * fm:
                    cb_mask_poly[:fm, :] = 0
                    cb_mask_poly[full_h - fm :, :] = 0
                    cb_mask_poly[:, :fm] = 0
                    cb_mask_poly[:, full_w - fm :] = 0
                k_poly = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
                cb_closed_poly = cv2.morphologyEx(cb_mask_poly, cv2.MORPH_CLOSE, k_poly)
                cnts_p, _ = cv2.findContours(cb_closed_poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_p:
                    best_c = max(cnts_p, key=cv2.contourArea)
                    if cv2.contourArea(best_c) > 8000:
                        hull = cv2.convexHull(best_c)
                        peri = cv2.arcLength(hull, True)
                        approx = cv2.approxPolyDP(hull, 0.015 * peri, True).reshape(-1, 2)
                        cardboard_poly_bed = approx.astype(np.float64)
                bbox = detect_cardboard_roi(
                    work_img,
                    shave_px=cfg.roi_shave_px,
                    frame_margin=80 if warped_flag else 10,
                )

    if bbox is not None:
        bx, by, bw, bh = bbox
        bx = max(0, min(int(bx), full_w - 10))
        by = max(0, min(int(by), full_h - 10))
        bw = max(10, min(int(bw), full_w - bx))
        bh = max(10, min(int(bh), full_h - by))
        work_img = work_img[by : by + bh, bx : bx + bw].copy()
        roi_x, roi_y = bx, by
        cardboard_poly_bed = np.array(
            [
                [float(bx), float(by)],
                [float(bx + bw), float(by)],
                [float(bx + bw), float(by + bh)],
                [float(bx), float(by + bh)],
            ],
            dtype=np.float64,
        )

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    brightness = float(cv2.mean(gray)[0])

    # Fast near-white UI overlay check before full LAB conversion
    b_ch, g_ch, r_ch = cv2.split(work_img)
    min_rgb = cv2.min(cv2.min(b_ch, g_ch), r_ch)
    if int(np.max(min_rgb)) > 215:
        lab = cv2.cvtColor(work_img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0]
        overlay_raw = l_chan > cfg.overlay_l_threshold
    else:
        overlay_raw = np.zeros(gray.shape, dtype=bool)

    # Mask non-cardboard dark cutting bed pixels (and cardboard shadow edges) when auto-detected ROI cropping was used
    if bbox is not None and cfg.roi_bbox is None:
        hsv_roi = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
        cb_mask = cv2.inRange(
            hsv_roi,
            np.array([5, 25, 65], dtype=np.uint8),
            np.array([42, 255, 255], dtype=np.uint8),
        )
        # First open to sever any dusty/brownish cutting-bed patches attached to the cardboard border
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        cb_opened = cv2.morphologyEx(cb_mask, cv2.MORPH_OPEN, k_open)
        # Close interior holes (dark marker strokes inside the cardboard) so strokes are not masked!
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        cb_closed = cv2.morphologyEx(cb_opened, cv2.MORPH_CLOSE, k_close)
        # Keep only the largest external contour (the cardboard sheet silhouette)
        cnts_cb, _ = cv2.findContours(cb_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cb_filled = np.zeros_like(cb_closed)
        if cnts_cb:
            largest_cb = max(cnts_cb, key=cv2.contourArea)
            if cv2.contourArea(largest_cb) > 2000:
                cv2.drawContours(cb_filled, [largest_cb], -1, 255, thickness=cv2.FILLED)
        # Erode inward so the high-contrast cardboard cut edge and flat-field blur halo never trigger false strokes.
        erode_px = max(1, int(cfg.cardboard_safety_erosion_px))
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        cb_interior = cv2.erode(cb_filled, k_erode)
        non_cardboard = cb_interior == 0
        if np.count_nonzero(cb_interior) > 0.30 * gray.size:
            overlay_raw = overlay_raw | non_cardboard

    overlay_masked = bool(np.any(overlay_raw))

    if overlay_masked and cfg.overlay_dilate_px > 0 and bbox is None:
        ksize = 2 * cfg.overlay_dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        ignore_u8 = cv2.dilate(overlay_raw.astype(np.uint8) * 255, kernel)
        ignore_mask = ignore_u8 > 0
    else:
        ignore_mask = overlay_raw.copy()

    # An explicit ROI can end exactly at a cardboard/cutting-mat step edge, in
    # which case contour detection above has no exterior context.  Protect the
    # crop perimeter in all cases; Stage 2 rescales this same physical margin.
    frame_px = max(0, int(cfg.border_ignore_px))
    if frame_px and gray.shape[0] > 2 * frame_px and gray.shape[1] > 2 * frame_px:
        ignore_mask[:frame_px, :] = True
        ignore_mask[-frame_px:, :] = True
        ignore_mask[:, :frame_px] = True
        ignore_mask[:, -frame_px:] = True

    rejected = False
    warnings = []
    if blur_score < cfg.min_blur_score:
        rejected = True
        warnings.append(f"Blur score {blur_score:.1f} below minimum {cfg.min_blur_score}")
    elif blur_score < cfg.warn_blur_score:
        warnings.append(f"Low blur score {blur_score:.1f}")

    if brightness < cfg.min_brightness or brightness > cfg.max_brightness:
        warnings.append(f"Extreme brightness {brightness:.1f}")

    quality = QualityReport(
        blur_score=blur_score,
        brightness=brightness,
        overlay_masked=overlay_masked,
        rejected=rejected,
        warning="; ".join(warnings) if warnings else None,
    )
    base_px_per_mm = (
        float(full_w) / max(1.0, phys_w)
        if (warped_flag and phys_w > 0)
        else (float(image_bgr.shape[1]) / max(1.0, phys_w) if phys_w > 0 else 2.5806)
    )
    zoom_factor = (
        float(full_w) / max(1.0, float(work_img.shape[1]))
        if (warped_flag and work_img.shape[1] > 0)
        else 1.0
    )

    return CaptureOutput(
        image_bgr=work_img,
        ignore_mask=ignore_mask,
        quality=quality,
        warped_from_aruco=warped_flag,
        phys_width_mm=phys_w,
        phys_height_mm=phys_h,
        aruco_markers_found=markers_found,
        roi_offset_x=roi_x,
        roi_offset_y=roi_y,
        full_bed_width=full_w,
        full_bed_height=full_h,
        warped_bed_bgr=warped_bed_bgr,
        cardboard_polygon_bed=cardboard_poly_bed,
        px_per_mm=base_px_per_mm,
        zoom_factor=zoom_factor,
    )


def stage2_preprocess(capture: CaptureOutput, cfg: Config) -> PreprocessOutput:
    """Stage 2: Resize, LAB extraction, ignore-mask median fill, flat-fielding, and bilateral denoise.

    Args:
        capture: Output from Stage 1.
        cfg: Pipeline configuration.

    Returns:
        PreprocessOutput with float32 flat-fielded image and scale factors.
    """
    img = capture.image_bgr
    orig_h, orig_w = img.shape[:2]
    long_side = max(orig_h, orig_w)

    if long_side < cfg.min_side_for_upscale:
        new_w = int(round(orig_w * cfg.upscale_factor))
        new_h = int(round(orig_h * cfg.upscale_factor))
        interp = cv2.INTER_CUBIC
    elif long_side > cfg.target_long_side:
        scale = cfg.target_long_side / float(long_side)
        new_w = int(round(orig_w * scale))
        new_h = int(round(orig_h * scale))
        interp = cv2.INTER_AREA
    else:
        new_w, new_h = orig_w, orig_h
        interp = cv2.INTER_LINEAR

    if (new_w, new_h) != (orig_w, orig_h):
        resized_bgr = cv2.resize(img, (new_w, new_h), interpolation=interp)
        ignore_u8 = cv2.resize(
            capture.ignore_mask.astype(np.uint8) * 255,
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST,
        )
        ignore_working = ignore_u8 > 0
    else:
        resized_bgr = img.copy()
        ignore_working = capture.ignore_mask.copy()

    # Keep the Stage-1 crop-edge safety band at the working scale as well.
    # This matters after cubic upscaling, where the interpolation halo would
    # otherwise reintroduce a border ridge between adjacent mask pixels.
    frame_px = max(0, int(round(cfg.border_ignore_px * (new_w / float(max(1, orig_w))))))
    if frame_px and new_h > 2 * frame_px and new_w > 2 * frame_px:
        ignore_working[:frame_px, :] = True
        ignore_working[-frame_px:, :] = True
        ignore_working[:, :frame_px] = True
        ignore_working[:, -frame_px:] = True

    scale_x = float(orig_w) / float(new_w)
    scale_y = float(orig_h) / float(new_h)

    lab = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    if cfg.channel_mode == "a":
        active_chan = a_chan.copy()
    elif cfg.channel_mode == "b":
        active_chan = b_chan.copy()
    else:
        active_chan = l_chan.copy()

    if np.any(ignore_working):
        valid_pixels = active_chan[~ignore_working]
        med_val = float(np.median(valid_pixels)) if valid_pixels.size > 0 else 128.0
        active_chan[ignore_working] = int(round(med_val))

    chan_f32 = active_chan.astype(np.float32)
    if cfg.flat_field_sigma >= 12.0 and min(new_h, new_w) >= 128:
        small_w = max(16, new_w // 4)
        small_h = max(16, new_h // 4)
        chan_small = cv2.resize(chan_f32, (small_w, small_h), interpolation=cv2.INTER_AREA)
        blur_small = cv2.GaussianBlur(
            chan_small,
            (0, 0),
            sigmaX=cfg.flat_field_sigma * (small_w / float(new_w)),
            sigmaY=cfg.flat_field_sigma * (small_h / float(new_h)),
        )
        bg_blur = cv2.resize(blur_small, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        bg_blur = cv2.GaussianBlur(
            chan_f32, (0, 0), sigmaX=cfg.flat_field_sigma, sigmaY=cfg.flat_field_sigma
        )
    l_diff = cv2.subtract(chan_f32, bg_blur)

    # Suppress horizontal/vertical corrugated cardboard flute ripples without weakening dark or colored ink
    if cfg.channel_mode == "L" and min(new_h, new_w) >= 128:
        flute_h = cv2.GaussianBlur(l_diff, (0, 0), sigmaX=18.0, sigmaY=0.8)
        flute_v = cv2.GaussianBlur(l_diff, (0, 0), sigmaX=0.8, sigmaY=18.0)
        flute_ripple = np.where(np.abs(flute_h) >= np.abs(flute_v), flute_h, flute_v)
        # Only subtract low-amplitude periodic fluting (|l_diff| < 12 gray levels)
        ripple_weight = np.clip(1.0 - np.abs(l_diff) / 14.0, 0.0, 1.0).astype(np.float32)
        l_diff = l_diff - flute_ripple * ripple_weight * np.float32(0.75)

    # Enhance colored marker strokes (red, green, blue) relative to neutral tan cardboard
    # while keeping pure neutral cardboard grain/flute shadows unamplified.
    if cfg.channel_mode == "L":
        sub_ig = ignore_working[::8, ::8]
        sub_a = a_chan[::8, ::8]
        sub_b = b_chan[::8, ::8]
        valid_sub = ~sub_ig
        if np.any(valid_sub):
            med_a = float(np.median(sub_a[valid_sub]))
            med_b = float(np.median(sub_b[valid_sub]))
        else:
            med_a, med_b = 128.0, 128.0
        idx_256 = np.arange(256, dtype=np.float32)
        da_lut = idx_256 - np.float32(med_a)
        da_lut[da_lut < 0.0] *= np.float32(1.85)
        db_lut = idx_256 - np.float32(med_b)
        chroma_sq = cv2.add(
            cv2.LUT(a_chan, da_lut * da_lut),
            cv2.LUT(b_chan, db_lut * db_lut),
        )
        if float(np.max(chroma_sq)) > 27.04:
            chroma_dev = cv2.sqrt(chroma_sq)
            dark_w = cv2.min(cv2.max(l_diff * np.float32(-1.0 / 3.5), 0.0), 1.0)
            chroma_boost = cv2.min(
                dark_w * cv2.max(chroma_dev - np.float32(5.2), 0.0) * np.float32(0.95),
                12.0,
            )
            if np.any(ignore_working):
                chroma_boost[ignore_working] = np.float32(0.0)
            flat_raw = cv2.subtract(l_diff, chroma_boost)
        else:
            flat_raw = l_diff
    else:
        flat_raw = l_diff

    flat_denoised = cv2.bilateralFilter(
        flat_raw,
        d=cfg.bilateral_d,
        sigmaColor=float(cfg.bilateral_sigma_color),
        sigmaSpace=float(cfg.bilateral_sigma_space),
    )

    # Part A: Per-color LAB residual baseline estimation (median blur ~24 mm + Gaussian sigma ~1 px)
    work_px_per_mm = capture.px_per_mm * (float(new_w) / float(max(1, orig_w)))
    l_pad = l_chan.copy()
    a_pad = a_chan.copy()
    b_pad = b_chan.copy()
    valid_px = ~ignore_working
    if np.any(ignore_working) and np.any(valid_px):
        l_pad[ignore_working] = int(round(float(np.median(l_pad[valid_px]))))
        a_pad[ignore_working] = int(round(float(np.median(a_pad[valid_px]))))
        b_pad[ignore_working] = int(round(float(np.median(b_pad[valid_px]))))

    med_k = int(round(cfg.lab_median_kernel_mm * work_px_per_mm)) | 1
    if med_k < 3:
        med_k = 61
    if med_k >= 21 and min(new_h, new_w) >= 128:
        sub_w = max(16, new_w // 4)
        sub_h = max(16, new_h // 4)
        sub_k = (med_k // 4) | 1
        l_sub = cv2.resize(l_pad, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        a_sub = cv2.resize(a_pad, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        b_sub = cv2.resize(b_pad, (sub_w, sub_h), interpolation=cv2.INTER_AREA)
        l_bg = cv2.resize(
            cv2.medianBlur(l_sub, sub_k), (new_w, new_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        a_bg = cv2.resize(
            cv2.medianBlur(a_sub, sub_k), (new_w, new_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        b_bg = cv2.resize(
            cv2.medianBlur(b_sub, sub_k), (new_w, new_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
    else:
        l_bg = cv2.medianBlur(l_pad, med_k).astype(np.float32)
        a_bg = cv2.medianBlur(a_pad, med_k).astype(np.float32)
        b_bg = cv2.medianBlur(b_pad, med_k).astype(np.float32)

    res_sig = float(cfg.lab_residual_sigma)
    dL = cv2.GaussianBlur(l_pad.astype(np.float32) - l_bg, (0, 0), sigmaX=res_sig, sigmaY=res_sig)
    da = cv2.GaussianBlur(a_pad.astype(np.float32) - a_bg, (0, 0), sigmaX=res_sig, sigmaY=res_sig)
    db = cv2.GaussianBlur(b_pad.astype(np.float32) - b_bg, (0, 0), sigmaX=res_sig, sigmaY=res_sig)

    chroma = cv2.magnitude(da, db)
    ink_s = chroma + np.float32(cfg.ink_l_weight_k) * np.maximum(-dL, np.float32(0.0))

    # Background noise floor estimation on ink strength s
    s_sub = ink_s[::2, ::2]
    m_s = float(np.median(s_sub))
    sigma_s = float(1.4826 * np.median(np.abs(s_sub - m_s)))
    ink_mask_s = (ink_s > m_s + float(cfg.k_chroma_sigma) * sigma_s).astype(np.float32)
    if np.any(ignore_working):
        ink_mask_s[ignore_working] = 0.0

    # Normalized convolution smoothing for chroma vector (da, db) restricted to ink mask
    sigma_conv_px = max(1.0, float(cfg.chroma_smooth_sigma_mm) * work_px_per_mm)
    k_conv = int(round(sigma_conv_px * 4.0)) | 1
    da_filt = cv2.GaussianBlur(da * ink_mask_s, (k_conv, k_conv), sigma_conv_px)
    db_filt = cv2.GaussianBlur(db * ink_mask_s, (k_conv, k_conv), sigma_conv_px)
    norm_filt = cv2.GaussianBlur(ink_mask_s, (k_conv, k_conv), sigma_conv_px) + 1e-6
    sm_da = da_filt / norm_filt
    sm_db = db_filt / norm_filt
    sm_chroma_angle = (np.degrees(np.arctan2(sm_db, sm_da)) + 360.0) % 360.0
    sm_mag = np.hypot(sm_da, sm_db)

    # Estimate dominant corrugation orientation from structure tensor of dL in non-ink background
    bg_mask_s = ink_s <= (m_s + 2.0 * sigma_s)
    if np.any(ignore_working):
        bg_mask_s &= ~ignore_working
    Ix_L = cv2.Sobel(dL, cv2.CV_32F, 1, 0, ksize=3) * bg_mask_s.astype(np.float32)
    Iy_L = cv2.Sobel(dL, cv2.CV_32F, 0, 1, ksize=3) * bg_mask_s.astype(np.float32)
    if np.any(bg_mask_s):
        jxx = float(np.mean((Ix_L * Ix_L)[bg_mask_s]))
        jyy = float(np.mean((Iy_L * Iy_L)[bg_mask_s]))
        jxy = float(np.mean((Ix_L * Iy_L)[bg_mask_s]))
        flute_grad = (np.degrees(0.5 * np.arctan2(2.0 * jxy, jxx - jyy)) + 180.0) % 180.0
        flute_angle_deg = (flute_grad + 90.0) % 180.0
    else:
        flute_angle_deg = 90.0

    color_signals = {}
    for c_spec in cfg.colors:
        ang_diff = np.abs(((sm_chroma_angle - float(c_spec.center_deg) + 180.0) % 360.0) - 180.0)
        hw = max(1.0, float(c_spec.half_width_deg))
        w_ang = np.clip((hw - ang_diff) / (hw * 0.45), 0.0, 1.0).astype(np.float32)
        min_c = max(0.5, float(c_spec.min_chroma))
        w_chr = np.clip((sm_mag - min_c * 0.70) / (min_c * 0.45), 0.0, 1.0).astype(np.float32)
        sig_c = (ink_s * w_ang * w_chr * ink_mask_s).astype(np.float32)
        if np.any(ignore_working):
            sig_c[ignore_working] = 0.0
        color_signals[c_spec.name] = sig_c

    return PreprocessOutput(
        flat_field=flat_denoised.astype(np.float32),
        lab_channels=(l_chan, a_chan, b_chan),
        resized_bgr=resized_bgr,
        ignore_mask_working=ignore_working,
        scale_x=scale_x,
        scale_y=scale_y,
        original_shape=(orig_h, orig_w),
        roi_offset_x=capture.roi_offset_x,
        roi_offset_y=capture.roi_offset_y,
        full_bed_shape=(
            capture.full_bed_height if capture.full_bed_height > 0 else orig_h,
            capture.full_bed_width if capture.full_bed_width > 0 else orig_w,
        ),
        dL=dL,
        da=da,
        db=db,
        ink_strength=ink_s,
        chroma_angle_deg=sm_chroma_angle.astype(np.float32),
        color_signals=color_signals,
        phys_width_mm=capture.phys_width_mm,
        phys_height_mm=capture.phys_height_mm,
        px_per_mm=work_px_per_mm,
        flute_angle_deg=flute_angle_deg,
        smoothed_da=sm_da,
        smoothed_db=sm_db,
        ink_sigma=sigma_s,
        ink_median=m_s,
    )


def _fast_sato_2d(
    img_f: np.ndarray, sigmas: List[float], black_ridges: bool
) -> np.ndarray:
    """Fast OpenCV C++ implementation of multi-scale 2D Sato tubeness filter.

    Uses OpenCV SIMD GaussianBlur and Sobel second derivatives to compute the
    scale-normalized Hessian eigenvalues, matching `skimage.filters.sato` in 2D
    while running ~4x faster on 1 MP images.
    """
    resp = np.zeros_like(img_f, dtype=np.float32)
    # 1/(2*2) normalization factor for OpenCV ksize=3 Sobel 2nd derivative
    sobel_norm = 0.25
    for s in sigmas:
        s_val = float(s)
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigmaX=s_val, sigmaY=s_val)
        scale_factor = (s_val * s_val) * sobel_norm
        hxx = cv2.Sobel(blurred, cv2.CV_32F, 2, 0, ksize=3, scale=scale_factor)
        hyy = cv2.Sobel(blurred, cv2.CV_32F, 0, 2, ksize=3, scale=scale_factor)
        hxy = cv2.Sobel(blurred, cv2.CV_32F, 1, 1, ksize=3, scale=scale_factor)

        tr = cv2.add(hxx, hyy)
        diff = cv2.subtract(hxx, hyy)
        disc = cv2.magnitude(diff, hxy * np.float32(2.0))
        if black_ridges:
            # Dark ridge on light background -> largest positive curvature eigenvalue
            eig = cv2.addWeighted(tr, 0.5, disc, 0.5, 0.0)
        else:
            # Bright ridge on dark background -> negated most-negative eigenvalue
            eig = cv2.addWeighted(disc, 0.5, tr, -0.5, 0.0)
        resp = cv2.max(resp, eig)
    return resp


def stage3_enhance_strokes(prep: PreprocessOutput, cfg: Config) -> EnhancementOutput:
    """Stage 3: Multi-scale ridge enhancement (Sato / Frangi / Meijering / black-hat).

    Args:
        prep: Output from Stage 2.
        cfg: Pipeline configuration.

    Returns:
        EnhancementOutput containing normalized float32 strokeness map in [0..1].
    """
    img_f = prep.flat_field
    mode = cfg.ridge_filter.lower()
    sigmas = list(cfg.ridge_sigmas)

    if mode == "sato":
        resp = _fast_sato_2d(img_f, sigmas, cfg.black_ridges)
    elif mode == "frangi":
        resp = frangi(
            img_f, sigmas=sigmas, black_ridges=cfg.black_ridges, mode="reflect"
        ).astype(np.float32)
    elif mode == "meijering":
        resp = meijering(
            img_f, sigmas=sigmas, black_ridges=cfg.black_ridges, mode="reflect"
        ).astype(np.float32)
    elif mode == "blackhat":
        ksize = int(max(3, round(max(sigmas) * 4))) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if cfg.black_ridges:
            resp = cv2.morphologyEx(img_f, cv2.MORPH_BLACKHAT, kernel)
        else:
            resp = cv2.morphologyEx(img_f, cv2.MORPH_TOPHAT, kernel)
        resp = resp.astype(np.float32)
    else:
        raise ValueError(f"Unsupported ridge_filter: {cfg.ridge_filter}")

    # Zero response inside ignore-mask and image border (avoids filter boundary artifacts)
    mask_zero = prep.ignore_mask_working.copy()
    if np.any(mask_zero):
        k_ig = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        mask_zero = cv2.dilate(mask_zero.astype(np.uint8) * 255, k_ig) > 0
    b_px = cfg.border_ignore_px
    if b_px > 0 and resp.shape[0] > 2 * b_px and resp.shape[1] > 2 * b_px:
        mask_zero[:b_px, :] = True
        mask_zero[-b_px:, :] = True
        mask_zero[:, :b_px] = True
        mask_zero[:, -b_px:] = True

    resp[mask_zero] = 0.0
    resp = np.maximum(resp, 0.0)

    # Estimate background floor via percentiles to reject pure texture images
    valid_resp = resp[~mask_zero]
    if valid_resp.size > 0:
        sub = valid_resp[::4]  # fast 4x subsample for percentile estimation
        p50, p70, p80, p95, p995, p999 = np.percentile(
            sub, [50.0, 70.0, 80.0, 95.0, 99.5, 99.9]
        )
        snr_95 = float(p999 - p50) / max(1e-6, float(p95 - p50))
        snr_80 = float(p999 - p50) / max(1e-6, float(p80 - p50))
        snr_70 = float(p999 - p50) / max(1e-6, float(p70 - p50))
    else:
        p995 = 0.0
        snr_95 = 0.0
        snr_80 = 0.0
        snr_70 = 0.0

    raw_max = float(np.max(resp))
    # Require sufficient raw response and peak-to-background SNR (supports both sparse and dense multi-shape drawings)
    if raw_max < cfg.min_raw_strokeness or (
        snr_95 < 3.6 and snr_80 < 22.0 and snr_70 < 32.0
    ):
        strokeness = np.zeros_like(resp, dtype=np.float32)
    else:
        norm_denom = max(float(p995), raw_max * 0.72)
        strokeness = np.clip(resp / max(1e-6, norm_denom), 0.0, 1.0).astype(np.float32)
        if cfg.black_ridges:
            ink_dark = np.maximum(0.0, -img_f)
            ink_score = np.clip((ink_dark - 12.0) / 25.0, 0.0, 1.0) * np.float32(0.75)
            strokeness = np.maximum(strokeness, ink_score).astype(np.float32)
            strokeness[mask_zero] = 0.0

    # Per-color bright ridge Sato filter with physical sigmas and noise-calibrated thresholds
    work_px_per_mm = prep.px_per_mm
    sigmas_px = [float(s * work_px_per_mm) for s in cfg.ridge_sigmas_mm]

    # Measure real stroke width w_mm from distance transform on strong ink
    ink_binary = (prep.ink_strength > (prep.ink_median + 3.0 * prep.ink_sigma)) & (~mask_zero)
    sk_ref = skeletonize(ink_binary)
    dist_map = cv2.distanceTransform(ink_binary.astype(np.uint8), cv2.DIST_L2, 5)
    sk_dists = dist_map[sk_ref]
    if len(sk_dists) > 20:
        w_px = float(2.0 * np.median(sk_dists[sk_dists > 1.0]))
        measured_w_mm = float(w_px / max(1e-6, work_px_per_mm))
    else:
        measured_w_mm = 1.6

    # If measured stroke width differs from config nominal (0.5+0.9=1.4mm) by >30%, re-scale sigmas
    nominal_w_mm = sum(cfg.ridge_sigmas_mm)
    if abs(measured_w_mm - nominal_w_mm) / max(1e-6, nominal_w_mm) > 0.30:
        sigmas_px = [
            float(measured_w_mm * 0.50 * work_px_per_mm),
            float(measured_w_mm * 0.75 * work_px_per_mm),
        ]

    color_strokeness = {}
    color_raw_max = {}
    color_noise_stats = {}
    bg_mask = (prep.ink_strength <= (prep.ink_median + 2.0 * prep.ink_sigma)) & (~mask_zero)

    for c_spec in cfg.colors:
        sig_c = prep.color_signals.get(c_spec.name)
        if sig_c is None:
            continue
        c_sig_max = float(np.max(sig_c[::4, ::4]))
        if c_sig_max <= 1e-4 and float(np.max(sig_c)) <= 1e-4:
            color_raw_max[c_spec.name] = 0.0
            color_noise_stats[c_spec.name] = (0.0, 0.02)
            color_strokeness[c_spec.name] = np.zeros_like(sig_c, dtype=np.float32)
            continue
        resp_c = _fast_sato_2d(sig_c, sigmas_px, black_ridges=False)
        resp_c[mask_zero] = 0.0
        resp_c = np.maximum(resp_c, 0.0)
        c_max = float(np.max(resp_c))
        color_raw_max[c_spec.name] = c_max
        bg_vals = resp_c[bg_mask] if np.any(bg_mask) else resp_c
        bg_sample = bg_vals[::4] if len(bg_vals) > 4000 else bg_vals
        m_bg = float(np.median(bg_sample))
        sigma_bg = float(1.4826 * np.median(np.abs(bg_sample - m_bg)))
        if sigma_bg < 1e-4:
            sigma_bg = float(np.std(bg_sample))
        if sigma_bg < 1e-4:
            sigma_bg = 0.02
        color_noise_stats[c_spec.name] = (m_bg, sigma_bg)
        color_strokeness[c_spec.name] = resp_c

    return EnhancementOutput(
        strokeness=strokeness,
        raw_max=raw_max,
        color_strokeness=color_strokeness,
        color_raw_max=color_raw_max,
        color_noise_stats=color_noise_stats,
        measured_stroke_width_mm=measured_w_mm,
    )


def stage4_hysteresis_threshold(
    enh: EnhancementOutput, cfg: Config, prep: Optional[PreprocessOutput] = None
) -> ThresholdOutput:
    """Stage 4: Noise-calibrated hysteresis thresholding and speck rejection.

    Args:
        enh: Output from Stage 3.
        cfg: Pipeline configuration.
        prep: Optional PreprocessOutput from Stage 2 for physical scale and color signals.

    Returns:
        ThresholdOutput with boolean stroke mask and per-color masks.
    """
    s_map = enh.strokeness
    color_masks: Dict[str, np.ndarray] = {}
    color_names = [c.name for c in cfg.colors]
    work_px_per_mm = prep.px_per_mm if prep is not None else 1.6
    w_mm = enh.measured_stroke_width_mm
    flute_ang = prep.flute_angle_deg if prep is not None else 90.0

    kept_specks = np.zeros(s_map.shape, dtype=bool)
    rejected_specks = np.zeros(s_map.shape, dtype=bool)
    speck_stats: Dict[str, object] = {}

    if enh.color_strokeness:
        for c_name in color_names:
            cs_map = enh.color_strokeness.get(c_name)
            noise_stat = enh.color_noise_stats.get(c_name)
            if cs_map is None or noise_stat is None:
                color_masks[c_name] = np.zeros(s_map.shape, dtype=bool)
                continue
            m_bg, sigma_bg = noise_stat
            low_t = m_bg + cfg.k_low_sigma * sigma_bg
            high_t = m_bg + cfg.k_high_sigma * sigma_bg

            low_m = cs_map > low_t
            high_m = cs_map >= high_t
            if not np.any(high_m):
                color_masks[c_name] = np.zeros(s_map.shape, dtype=bool)
                continue

            n_lbl_c, lbl_c, stats_c, _ = cv2.connectedComponentsWithStats(
                low_m.astype(np.uint8), connectivity=8
            )
            valid_c = np.zeros(n_lbl_c, dtype=bool)
            valid_c[lbl_c[high_m]] = True
            valid_c[0] = False

            raw_count = int(np.sum(valid_c))
            kept_count = 0
            rej_count = 0

            for c_i in range(1, n_lbl_c):
                if not valid_c[c_i]:
                    continue
                area_px = int(stats_c[c_i, cv2.CC_STAT_AREA])
                comp_m = lbl_c == c_i
                sk = skeletonize(comp_m)
                sk_len_px = float(np.sum(sk))
                if sk_len_px < 2.0:
                    valid_c[c_i] = False
                    rejected_specks[comp_m] = True
                    rej_count += 1
                    continue

                L_mm = sk_len_px / work_px_per_mm
                mean_resp_sigma = (float(np.mean(cs_map[sk])) - m_bg) / max(1e-6, sigma_bg)
                mean_w_mm = (area_px / max(1.0, sk_len_px)) / work_px_per_mm

                # Principal orientation via covariance matrix
                ys_c, xs_c = np.nonzero(comp_m)
                if len(xs_c) > 3:
                    pts_c = np.column_stack([xs_c - np.mean(xs_c), ys_c - np.mean(ys_c)]).astype(np.float64)
                    cov_c = np.cov(pts_c.T)
                    evals, evecs = np.linalg.eigh(cov_c)
                    comp_ang = (np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])) + 180.0) % 180.0
                else:
                    comp_ang = 0.0

                # Flute streak rejection: L < 12 mm, orientation within 15 deg of flute, width < 0.7 w
                diff_f = abs(comp_ang - flute_ang)
                if diff_f > 90.0:
                    diff_f = 180.0 - diff_f
                is_flute = (
                    (L_mm < cfg.flute_reject_len_mm)
                    and (diff_f <= cfg.flute_reject_angle_deg)
                    and (mean_w_mm < cfg.flute_reject_max_width_ratio * w_mm)
                )

                mean_ink = float(np.mean(prep.ink_strength[sk])) if prep is not None and prep.ink_strength is not None else 10.0
                keep_a = (
                    (L_mm >= cfg.min_component_len_mm)
                    and (mean_resp_sigma >= cfg.min_response_sigma_long)
                    and (mean_ink >= 5.5)
                    and (mean_w_mm <= 3.0 * w_mm)
                )
                keep_b = (
                    (cfg.min_short_stroke_len_mm <= L_mm < cfg.min_component_len_mm)
                    and (mean_resp_sigma >= cfg.min_response_sigma_short)
                    and (0.45 * w_mm <= mean_w_mm <= 2.20 * w_mm)
                    and (mean_ink >= 6.0)
                )

                if (keep_a or keep_b) and not is_flute:
                    kept_specks[comp_m] = True
                    kept_count += 1
                else:
                    valid_c[c_i] = False
                    rejected_specks[comp_m] = True
                    rej_count += 1

            speck_stats[c_name] = {"raw": raw_count, "kept": kept_count, "rejected": rej_count}
            c_u8 = (valid_c[lbl_c]).astype(np.uint8) * 255
            cnts_c, hier_c = cv2.findContours(
                c_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hier_c is not None:
                for ci, h_info in enumerate(hier_c[0]):
                    if h_info[3] >= 0 and cv2.contourArea(cnts_c[ci]) < 260.0:
                        cv2.drawContours(
                            c_u8, cnts_c, ci, 255, thickness=cv2.FILLED
                        )
            # Retain pixel in color mask
            color_masks[c_name] = c_u8 > 0

        # Resolve overlaps by assigning overlapping pixels to the color with the larger strokeness response
        for i in range(len(color_names)):
            c1 = color_names[i]
            m1 = color_masks.get(c1)
            r1 = enh.color_strokeness.get(c1)
            if m1 is None or r1 is None:
                continue
            for j in range(i + 1, len(color_names)):
                c2 = color_names[j]
                m2 = color_masks.get(c2)
                r2 = enh.color_strokeness.get(c2)
                if m2 is None or r2 is None:
                    continue
                ol = m1 & m2
                if np.any(ol):
                    m1[ol & (r2 > r1)] = False
                    m2[ol & (r1 >= r2)] = False

        # Drop colors that have no long stroke (L >= 22.0 mm) or insufficient peak response
        for c_name in color_names:
            c_max = enh.color_raw_max.get(c_name, 0.0)
            if c_max < 9.0:
                color_masks[c_name] = np.zeros_like(color_masks.get(c_name, s_map), dtype=bool)
                continue
            cm = color_masks.get(c_name)
            if cm is None or not np.any(cm):
                continue
            sk = skeletonize(cm)
            n_cc, _, stats, _ = cv2.connectedComponentsWithStats(sk.astype(np.uint8))
            has_long = False
            for k in range(1, n_cc):
                if (stats[k, cv2.CC_STAT_AREA] / max(1e-6, work_px_per_mm)) >= 22.0:
                    has_long = True
                    break
            if not has_long:
                color_masks[c_name] = np.zeros_like(cm, dtype=bool)

    if float(np.max(s_map)) <= 1e-6:
        union_c = np.zeros(s_map.shape, dtype=bool)
        for cm in color_masks.values():
            union_c |= cm
        return ThresholdOutput(
            mask=union_c,
            high_val=0.0,
            low_val=0.0,
            color_masks=color_masks,
            kept_speck_mask=kept_specks,
            rejected_speck_mask=rejected_specks,
            speck_stats=speck_stats,
        )

    p80, p95, low_val, high_val = [
        float(v)
        for v in np.percentile(
            s_map[::2, ::2], [80.0, 95.0, cfg.hysteresis_low_pct, cfg.hysteresis_high_pct]
        )
    ]

    # When an image has dense strokes covering >5% of the frame (so p95 is already >2.8x above p80),
    # p97 rises deep inside the strokes. Cap high_val relative to p80 so lighter strokes are not lost.
    if p80 > 0.01 and p95 > 2.8 * p80:
        high_val = min(high_val, max(0.28, 4.5 * p80))
        low_val = max(0.14, min(high_val * 0.70, 2.2 * p80))

    high_val = max(high_val, cfg.min_absolute_high)
    low_val = min(low_val, high_val * 0.85)
    low_val = max(low_val, cfg.min_absolute_high * 0.5)

    low_u8 = (s_map > low_val).astype(np.uint8)
    high_mask = s_map >= high_val
    if not np.any(high_mask):
        return ThresholdOutput(
            mask=np.zeros(s_map.shape, dtype=bool), high_val=high_val, low_val=low_val
        )
    num_lbl, labels = cv2.connectedComponents(low_u8, connectivity=8)
    valid_lbl = np.zeros(num_lbl, dtype=bool)
    valid_lbl[labels[high_mask]] = True
    valid_lbl[0] = False
    hyst_mask = valid_lbl[labels]

    k = cfg.morph_close_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed_u8 = cv2.morphologyEx(
        hyst_mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel
    )

    # Fill small interior pinholes (< 200 px^2) inside thick strokes/junctions (such as the apex of a thick arrow)
    # so topological skeletonization does not create figure-8 loops around tiny hollow centers.
    contours, hierarchy = cv2.findContours(
        closed_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is not None:
        hier = hierarchy[0]
        for c_idx, h_info in enumerate(hier):
            if h_info[3] >= 0:  # interior hole inside a stroke component
                if cv2.contourArea(contours[c_idx]) < 200.0:
                    cv2.drawContours(closed_u8, contours, c_idx, 255, thickness=cv2.FILLED)

    return ThresholdOutput(
        mask=closed_u8 > 0,
        high_val=high_val,
        low_val=low_val,
        color_masks=color_masks,
        kept_speck_mask=kept_specks,
        rejected_speck_mask=rejected_specks,
        speck_stats=speck_stats,
    )


def _find_skeleton_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return boolean mask of skeleton pixels that have exactly 1 8-connected neighbor."""
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skel_u8, cv2.CV_8U, kernel, borderType=cv2.BORDER_CONSTANT)
    return (skel_u8 == 1) & (neighbor_count == 1)


def _prune_spurs(skel: np.ndarray, iterations: int) -> np.ndarray:
    """Iteratively remove degree-1 endpoints to prune short skeleton spurs."""
    out = skel.copy()
    ys_all, xs_all = np.nonzero(out)
    if len(xs_all) == 0:
        return out
    y0, y1 = max(0, int(ys_all.min()) - 2), min(out.shape[0], int(ys_all.max()) + 3)
    x0, x1 = max(0, int(xs_all.min()) - 2), min(out.shape[1], int(xs_all.max()) + 3)
    sub = out[y0:y1, x0:x1]
    for _ in range(iterations):
        endpoints = _find_skeleton_endpoints(sub)
        if not np.any(endpoints):
            break
        sub[endpoints] = False
    return out


def _link_endpoints(skel: np.ndarray, max_dist: float, min_cos: float) -> np.ndarray:
    """Connect nearby endpoints whose local tangent directions point toward each other."""
    out = skel.copy()
    ep_mask = _find_skeleton_endpoints(out)
    ys, xs = np.nonzero(ep_mask)
    n_ep = len(xs)
    if n_ep < 2 or n_ep > 250:
        return out

    # Estimate local outward tangent vector at each endpoint by walking 5 pixels inward
    tangents = np.zeros((n_ep, 2), dtype=np.float64)
    valid_ep = np.ones(n_ep, dtype=bool)
    h, w = out.shape

    for i in range(n_ep):
        cx, cy = int(xs[i]), int(ys[i])
        prev_x, prev_y = cx, cy
        cur_x, cur_y = cx, cy
        for _ in range(6):
            found_next = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = cur_x + dx, cur_y + dy
                    if 0 <= nx < w and 0 <= ny < h and out[ny, nx]:
                        if nx != prev_x or ny != prev_y:
                            prev_x, prev_y = cur_x, cur_y
                            cur_x, cur_y = nx, ny
                            found_next = True
                            break
                if found_next:
                    break
            if not found_next:
                break
        vx = float(cx - cur_x)
        vy = float(cy - cur_y)
        norm = np.hypot(vx, vy)
        if norm < 1e-3:
            valid_ep[i] = False
        else:
            tangents[i, 0] = vx / norm
            tangents[i, 1] = vy / norm

    out_u8 = out.astype(np.uint8) * 255
    _, comp_labels = cv2.connectedComponents((out_u8 > 0).astype(np.uint8), connectivity=8)
    coords = np.column_stack([xs, ys]).astype(np.float64)
    used = set()

    for i in range(n_ep):
        if not valid_ep[i] or i in used:
            continue
        best_j = -1
        best_d = max_dist
        lbl_i = int(comp_labels[int(ys[i]), int(xs[i])])
        for j in range(i + 1, n_ep):
            if not valid_ep[j] or j in used:
                continue
            dvec = coords[j] - coords[i]
            dist = float(np.hypot(dvec[0], dvec[1]))
            if dist < 2.0 or dist > best_d:
                continue
            u = dvec / dist
            cos_i = float(np.dot(tangents[i], u))
            cos_j = float(np.dot(tangents[j], -u))
            lbl_j = int(comp_labels[int(ys[j]), int(xs[j])])
            # 1) Collinear gap bridging up to max_dist
            is_collinear = cos_i >= min_cos and cos_j >= min_cos
            # 2) Corner / V-valley gap (<= 7.0 px) across distinct components where both point toward each other
            is_short_corner = (
                dist <= 7.0
                and lbl_i != lbl_j
                and cos_i > 0.25
                and cos_j > 0.25
            )
            if is_collinear or is_short_corner:
                best_d = dist
                best_j = j
        if best_j >= 0:
            used.add(i)
            used.add(best_j)
            pt1 = (int(xs[i]), int(ys[i]))
            pt2 = (int(xs[best_j]), int(ys[best_j]))
            cv2.line(out_u8, pt1, pt2, 255, 1)

    if np.any(out_u8 > 0):
        if used:
            return skeletonize(out_u8 > 0)
        return out
    return out


def stage5_skeleton_and_grouping(
    thresh: ThresholdOutput, cfg: Config, prep: Optional[PreprocessOutput] = None
) -> SkeletonOutput:
    """Stage 5: Skeletonize each color mask, build junction/endpoint graph, prune spurs,
    merge through degree-2 nodes, close endpoint gaps, and snap T-junctions per color.

    Args:
        thresh: Output from Stage 4.
        cfg: Pipeline configuration.
        prep: Optional Output from Stage 2 for physical scale (px/mm).

    Returns:
        SkeletonOutput with per-color skeletons, traced chains, and graph visualization.
    """
    work_px_per_mm = 1.6
    if prep is not None:
        bed_h, bed_w = prep.full_bed_shape
        if bed_w > 0 and prep.phys_width_mm > 0:
            dots_per_mm = float(bed_w) / float(prep.phys_width_mm)
            avg_scale = 0.5 * (prep.scale_x + prep.scale_y)
            work_px_per_mm = dots_per_mm / max(1e-6, avg_scale)

    spur_px = max(3, int(round(cfg.spur_prune_mm * work_px_per_mm)))
    gap_px = max(2.5, float(cfg.endpoint_gap_mm) * work_px_per_mm)
    t_snap_px = max(4.0, float(cfg.t_junction_snap_mm) * work_px_per_mm)

    color_skeletons: Dict[str, np.ndarray] = {}
    color_chains: Dict[str, List[Tuple[np.ndarray, bool]]] = {}
    h_img, w_img = thresh.mask.shape
    graph_vis = np.full((h_img, w_img, 3), 20, dtype=np.uint8)
    k_nb = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    color_bgr_map: Dict[str, Tuple[int, int, int]] = {}
    for c_spec in cfg.colors:
        hx = c_spec.hex_color.lstrip("#")
        if len(hx) == 6:
            r_v, g_v, b_v = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
            color_bgr_map[c_spec.name] = (b_v, g_v, r_v)
        else:
            color_bgr_map[c_spec.name] = (229, 136, 30)

    if thresh.color_masks:
        for c_name, c_mask in thresh.color_masks.items():
            if not np.any(c_mask):
                color_skeletons[c_name] = np.zeros_like(c_mask, dtype=bool)
                color_chains[c_name] = []
                continue
            ys_c, xs_c = np.nonzero(c_mask)
            cy0 = max(0, int(ys_c.min()) - 2)
            cy1 = min(h_img, int(ys_c.max()) + 3)
            cx0 = max(0, int(xs_c.min()) - 2)
            cx1 = min(w_img, int(xs_c.max()) + 3)
            sk_c = np.zeros_like(c_mask, dtype=bool)
            sk_c[cy0:cy1, cx0:cx1] = skeletonize(c_mask[cy0:cy1, cx0:cx1])

            n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(
                sk_c.astype(np.uint8), connectivity=8
            )
            raw_c_chains: List[Tuple[np.ndarray, bool]] = []
            keep_sk_c = np.zeros_like(sk_c, dtype=bool)
            for c_idx in range(1, n_cc):
                bx = int(cc_stats[c_idx, cv2.CC_STAT_LEFT])
                by = int(cc_stats[c_idx, cv2.CC_STAT_TOP])
                bw = int(cc_stats[c_idx, cv2.CC_STAT_WIDTH])
                bh = int(cc_stats[c_idx, cv2.CC_STAT_HEIGHT])
                b_area = int(cc_stats[c_idx, cv2.CC_STAT_AREA])
                if b_area < 3:
                    continue
                sub_m = cc_lbl[by : by + bh, bx : bx + bw] == c_idx
                keep_sk_c[by : by + bh, bx : bx + bw] |= sub_m
                sub_chains = _trace_component_chains(sub_m, max_spur_px=spur_px)
                for pts_loc, is_cl in sub_chains:
                    raw_c_chains.append(
                        (
                            pts_loc + np.array([float(bx), float(by)], dtype=np.float64),
                            is_cl,
                        )
                    )

            merged_c_chains = _close_gaps_and_snap_t_junctions(
                raw_c_chains,
                min_gap_px=min(2.0, gap_px),
                collinear_gap_px=max(20.0, gap_px * 8.0),
                t_snap_px=t_snap_px,
            )
            # Filter out tiny isolated noise chains (< 13.5 px arc length)
            final_c_chains: List[Tuple[np.ndarray, bool]] = []
            for pts_c, is_cl in merged_c_chains:
                if len(pts_c) < 6:
                    continue
                diffs_c = np.diff(pts_c, axis=0)
                arc_c = float(np.sum(np.hypot(diffs_c[:, 0], diffs_c[:, 1])))
                if arc_c < 13.5:
                    continue
                final_c_chains.append((pts_c, is_cl))

            color_skeletons[c_name] = keep_sk_c
            color_chains[c_name] = final_c_chains

            # Draw skeleton, junctions, and endpoints onto graph_vis
            bgr_col = color_bgr_map.get(c_name, (200, 200, 200))
            graph_vis[keep_sk_c] = bgr_col
            u8_c = keep_sk_c.astype(np.uint8)
            deg_c = (
                cv2.filter2D(u8_c, cv2.CV_8U, k_nb, borderType=cv2.BORDER_CONSTANT)
                * u8_c
            )
            j_ys, j_xs = np.nonzero(deg_c >= 3)
            for jy, jx in zip(j_ys, j_xs):
                cv2.circle(graph_vis, (int(jx), int(jy)), 3, (0, 255, 255), -1)
            for pts_c, is_cl in final_c_chains:
                if not is_cl and len(pts_c) >= 2:
                    p0 = (int(round(float(pts_c[0, 0]))), int(round(float(pts_c[0, 1]))))
                    p1 = (int(round(float(pts_c[-1, 0]))), int(round(float(pts_c[-1, 1]))))
                    cv2.circle(graph_vis, p0, 3, (255, 255, 0), -1)
                    cv2.circle(graph_vis, p1, 3, (255, 255, 0), -1)

        # Prune spurious green edge chains that lie along red strokes or corrugation texture (chroma < 5.0)
        if "green" in color_chains:
            pruned_green = []
            for pts_c, is_cl in color_chains["green"]:
                diffs_c = np.diff(pts_c, axis=0)
                arc_c = float(np.sum(np.hypot(diffs_c[:, 0], diffs_c[:, 1])))
                if arc_c < 22.0 and not is_cl:
                    xs_i = np.clip(np.round(pts_c[:, 0]).astype(int), 0, thresh.mask.shape[1] - 1)
                    ys_i = np.clip(np.round(pts_c[:, 1]).astype(int), 0, thresh.mask.shape[0] - 1)
                    if prep is not None and prep.da is not None and prep.db is not None:
                        s_c = float(np.mean(np.hypot(prep.da[ys_i, xs_i], prep.db[ys_i, xs_i])))
                    else:
                        s_c = 10.0
                    if s_c < 5.0:
                        continue
                pruned_green.append((pts_c, is_cl))
            color_chains["green"] = pruned_green

    if not np.any(thresh.mask):
        empty = np.zeros_like(thresh.mask, dtype=bool)
        return SkeletonOutput(
            raw_skeleton=empty,
            pruned_skeleton=empty,
            skeleton=empty,
            color_skeletons=color_skeletons,
            color_chains=color_chains,
            graph_nodes_vis=graph_vis,
        )

    # Pre-filter tiny isolated noise blobs (< 25 px area and < 15 px span) before skeletonization
    n_pre, pre_labels, pre_stats, _ = cv2.connectedComponentsWithStats(
        thresh.mask.astype(np.uint8), connectivity=8
    )
    keep_pre = np.zeros(n_pre, dtype=bool)
    for c_idx in range(1, n_pre):
        c_area = int(pre_stats[c_idx, cv2.CC_STAT_AREA])
        c_w = int(pre_stats[c_idx, cv2.CC_STAT_WIDTH])
        c_h = int(pre_stats[c_idx, cv2.CC_STAT_HEIGHT])
        if c_area >= 25 or max(c_w, c_h) >= 15:
            keep_pre[c_idx] = True
    clean_pre = keep_pre[pre_labels]

    ys_m, xs_m = np.nonzero(clean_pre)
    if len(xs_m) == 0:
        empty = np.zeros_like(thresh.mask, dtype=bool)
        return SkeletonOutput(
            raw_skeleton=empty,
            pruned_skeleton=empty,
            skeleton=empty,
            color_skeletons=color_skeletons,
            color_chains=color_chains,
            graph_nodes_vis=graph_vis,
        )

    my0, my1 = max(0, int(ys_m.min()) - 2), min(clean_pre.shape[0], int(ys_m.max()) + 3)
    mx0, mx1 = max(0, int(xs_m.min()) - 2), min(clean_pre.shape[1], int(xs_m.max()) + 3)
    raw_skel = np.zeros_like(clean_pre, dtype=bool)
    raw_skel[my0:my1, mx0:mx1] = skeletonize(clean_pre[my0:my1, mx0:mx1])

    pruned = _prune_spurs(raw_skel, min(1, cfg.spur_prune_iters))
    linked = _link_endpoints(pruned, cfg.endpoint_link_dist, cfg.endpoint_cos_sim)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        linked.astype(np.uint8), connectivity=8
    )

    keep = np.zeros(num_labels, dtype=bool)
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        span = float(np.hypot(w, h))
        aspect = float(max(w, h)) / float(max(1, min(w, h)))
        effective_min_len = max(30, cfg.min_skeleton_length - 2 * cfg.spur_prune_iters)
        is_clean_short_tick = (
            cfg.ridge_filter == "sato"
            and area >= 21
            and span >= 20.0
            and aspect >= 4.0
            and h > w
        )
        if area < effective_min_len and not is_clean_short_tick:
            continue
        if cfg.filter_compact_blobs and not is_clean_short_tick:
            if span < 50.0:
                continue
        keep[lbl] = True

    final_skel = keep[labels]
    if color_skeletons:
        union_sk = np.zeros_like(final_skel, dtype=bool)
        for c_sk in color_skeletons.values():
            union_sk |= c_sk
        if np.any(union_sk):
            final_skel = union_sk

    return SkeletonOutput(
        raw_skeleton=raw_skel,
        pruned_skeleton=linked,
        skeleton=final_skel,
        color_skeletons=color_skeletons,
        color_chains=color_chains,
        graph_nodes_vis=graph_vis,
    )


def rdp_simplify(points: np.ndarray, epsilon: float, closed: bool = False) -> np.ndarray:
    """Simplify a 2D polyline or closed loop using Ramer-Douglas-Peucker algorithm."""
    if len(points) <= 2:
        return points.copy()
    if closed and len(points) > 4:
        pts_f32 = points[:-1].astype(np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(pts_f32, epsilon=epsilon, closed=True)
        res = approx.reshape(-1, 2).astype(np.float64)
        return np.vstack([res, res[0:1]])
    pts_f32 = points.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts_f32, epsilon=epsilon, closed=False)
    return approx.reshape(-1, 2).astype(np.float64)


def _remove_skeleton_pimples(skel_bool: np.ndarray) -> np.ndarray:
    """Fast vectorized removal of 1-px degree-2 staircase/pimple pixels whose two neighbors are 8-adjacent."""
    out = skel_bool.copy()
    h, w = out.shape
    k_nb = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    k_3x3 = np.ones((3, 3), dtype=np.uint8)
    for _ in range(3):
        u8 = out.astype(np.uint8)
        deg = cv2.filter2D(u8, cv2.CV_8U, k_nb, borderType=cv2.BORDER_CONSTANT) * u8
        junc_near = cv2.dilate((deg >= 3).astype(np.uint8), k_3x3)
        cand_ys, cand_xs = np.nonzero((deg == 2) & (junc_near > 0))
        if len(cand_xs) == 0:
            break
        changed = False
        for y, x in zip(cand_ys, cand_xs):
            if not out[y, x]:
                continue
            nbs = [
                (ny, nx)
                for ny in range(max(0, y - 1), min(h, y + 2))
                for nx in range(max(0, x - 1), min(w, x + 2))
                if (ny != y or nx != x) and out[ny, nx]
            ]
            if len(nbs) == 2:
                (y1, x1), (y2, x2) = nbs
                if max(abs(y1 - y2), abs(x1 - x2)) <= 1:
                    out[y, x] = False
                    changed = True
        if not changed:
            break
    return out


def _order_simple_chain(chain_mask: np.ndarray) -> np.ndarray:
    """Order a degree<=2 connected skeleton chain from one end to the other via two-pass BFS."""
    ys, xs = np.nonzero(chain_mask)
    n = len(xs)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if n <= 2:
        return np.column_stack([xs, ys]).astype(np.float64)
    h, w = chain_mask.shape
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n, dtype=np.int32)
    adj: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        x, y = int(xs[i]), int(ys[i])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h:
                    nb = int(idx_map[ny, nx])
                    if nb >= 0:
                        adj[i].append(nb)

    start = 0
    for i in range(n):
        if len(adj[i]) <= 1:
            start = i
            break

    def bfs(s_node: int) -> Tuple[int, np.ndarray]:
        dist = np.full(n, -1, dtype=np.int32)
        par = np.full(n, -1, dtype=np.int32)
        q = [s_node]
        dist[s_node] = 0
        far = s_node
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            if dist[u] > dist[far]:
                far = u
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    par[v] = u
                    q.append(v)
        return far, par

    far1, _ = bfs(start)
    far2, par2 = bfs(far1)
    path = []
    cur = far2
    while cur != -1:
        path.append(cur)
        cur = int(par2[cur])
    path.reverse()
    return np.column_stack([xs[path], ys[path]]).astype(np.float64)


def _prune_junction_hairs(skel_bool: np.ndarray, max_hair_len: int = 12) -> np.ndarray:
    """Remove short dead-end side hairs (<= max_hair_len px) attached to degree>=3 junctions
    without eroding genuine open stroke endpoints or crossing ticks."""
    out = _remove_skeleton_pimples(skel_bool)
    h, w = out.shape
    k_nb = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    for _ in range(2):
        u8 = out.astype(np.uint8)
        deg = cv2.filter2D(u8, cv2.CV_8U, k_nb, borderType=cv2.BORDER_CONSTANT) * u8
        if int(np.max(deg)) < 3:
            break
        ep_ys, ep_xs = np.nonzero(deg == 1)
        if len(ep_xs) == 0:
            break
        removed_any = False
        for sy, sx in zip(ep_ys, ep_xs):
            if not out[sy, sx]:
                continue
            path = [(int(sy), int(sx))]
            cy, cx = int(sy), int(sx)
            py, px = -1, -1
            hit_junc = False
            for _step in range(max_hair_len):
                nxt = None
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and out[ny, nx] and (ny != py or nx != px):
                            nxt = (ny, nx)
                            break
                    if nxt is not None:
                        break
                if nxt is None:
                    break
                ny, nx = nxt
                if deg[ny, nx] >= 3:
                    hit_junc = True
                    break
                path.append((ny, nx))
                py, px = cy, cx
                cy, cx = ny, nx
            if hit_junc and len(path) <= max_hair_len:
                for ry, rx in path:
                    out[ry, rx] = False
                removed_any = True
        out = _remove_skeleton_pimples(out)
        if not removed_any:
            break
    return out


def _peel_spurs_and_pimples(skel_bool: np.ndarray, iterations: int) -> np.ndarray:
    """Alternately remove 3-clique pimples (including 2-px end-caps) and degree-1 endpoints."""
    out = skel_bool.copy()
    ys_all, xs_all = np.nonzero(out)
    if len(xs_all) == 0:
        return out
    y0, y1 = max(0, int(ys_all.min()) - 2), min(out.shape[0], int(ys_all.max()) + 3)
    x0, x1 = max(0, int(xs_all.min()) - 2), min(out.shape[1], int(xs_all.max()) + 3)
    sub = out[y0:y1, x0:x1]
    for _ in range(iterations):
        sub_clean = _remove_skeleton_pimples(sub)
        endpoints = _find_skeleton_endpoints(sub_clean)
        if not np.any(endpoints) and np.array_equal(sub_clean, sub):
            sub[:] = sub_clean
            break
        sub_clean[endpoints] = False
        sub[:] = sub_clean
    sub[:] = _remove_skeleton_pimples(sub)
    return out


def _order_closed_cycle(loop_mask: np.ndarray) -> np.ndarray:
    """Walk a 1-px closed skeleton loop neighbor-by-neighbor starting at the farthest corner from centroid."""
    lys, lxs = np.nonzero(loop_mask)
    n = len(lxs)
    if n < 4:
        return np.column_stack([lxs, lys]).astype(np.float64)
    cx, cy = float(np.mean(lxs)), float(np.mean(lys))
    h, w = loop_mask.shape
    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[lys, lxs] = np.arange(n, dtype=np.int32)

    # Start at the pixel farthest from centroid (a true polygon corner on triangles/squares)
    d2 = (lxs - cx) ** 2 + (lys - cy) ** 2
    start_idx = int(np.argmax(d2))

    visited = np.zeros(n, dtype=bool)
    order: List[int] = [start_idx]
    visited[start_idx] = True
    cur = start_idx

    # Prefer 4-connected neighbors before diagonal neighbors for clean cycle traversal
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)]
    while True:
        x, y = int(lxs[cur]), int(lys[cur])
        nxt = -1
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                nb = int(idx_map[ny, nx])
                if nb >= 0 and not visited[nb]:
                    nxt = nb
                    break
        if nxt == -1:
            break
        visited[nxt] = True
        order.append(nxt)
        cur = nxt

    if len(order) >= int(0.85 * n):
        end_idx = order[-1]
        if np.hypot(lxs[end_idx] - lxs[start_idx], lys[end_idx] - lys[start_idx]) <= 5.0:
            order.append(order[0])
            return np.column_stack([lxs[order], lys[order]]).astype(np.float64)

    # Fallback: fill the interior of the loop and extract its external contour
    pad_loop = np.pad(loop_mask.astype(np.uint8) * 255, 2, mode="constant", constant_values=0)
    cnts, _ = cv2.findContours(pad_loop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        best_cnt = max(cnts, key=len).reshape(-1, 2).astype(np.float64) - 2.0
        if len(best_cnt) >= 4:
            d2_c = (best_cnt[:, 0] - cx) ** 2 + (best_cnt[:, 1] - cy) ** 2
            s_c = int(np.argmax(d2_c))
            best_cnt = np.roll(best_cnt, -s_c, axis=0)
            return np.vstack([best_cnt, best_cnt[0:1]])

    return np.column_stack([lxs[order], lys[order]]).astype(np.float64)


def _trace_component_chains(
    comp_mask: np.ndarray, max_spur_px: int = 4
) -> List[Tuple[np.ndarray, bool]]:
    """Decompose a skeleton component into ordered (N, 2) [x, y] chains.

    Handles:
    1. Single simple closed loops (num_holes == 1): 2-core peeling strips corner overhangs/spurs
       and orders the loop cyclically.
    2. Multi-loop planar graphs (num_holes >= 2, e.g. connected box nets / trapezoids): splits at
       true junction clusters so each face edge/flap is vectorized and color-classified cleanly.
    3. Branched / crossing open structures (num_holes == 0, e.g. arrows and '++++' stitch lines):
       splits at junctions, pairs collinear segments straight across X/T-junctions, and prunes
       unpaired dead-end side whiskers.
    """
    ys, xs = np.nonzero(comp_mask)
    n_pts = len(xs)
    if n_pts == 0:
        return []
    if n_pts == 1:
        return [(np.array([[float(xs[0]), float(ys[0])]], dtype=np.float64), False)]

    # Count interior holes >= 260 px^2 (fill tiny corner/tooth pinholes < 260 px^2)
    pad = np.pad(comp_mask.astype(np.uint8) * 255, 2, mode="constant", constant_values=0)
    inv = (pad == 0).astype(np.uint8)
    n_inv, inv_lbl, inv_stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    large_holes = 0
    small_hole_ids: List[int] = []
    for h_idx in range(2, n_inv):
        if int(inv_stats[h_idx, cv2.CC_STAT_AREA]) >= 260:
            large_holes += 1
        else:
            small_hole_ids.append(h_idx)

    base_mask = comp_mask.copy()
    if small_hole_ids:
        for sh_id in small_hole_ids:
            pad[inv_lbl == sh_id] = 255
        base_mask = skeletonize(pad[2:-2, 2:-2] > 0)

    # Case 1: Single simple closed loop (triangle, square, circle, ellipse)
    if large_holes == 1:
        loop_clean = _peel_spurs_and_pimples(base_mask, iterations=40)
        if np.any(loop_clean) and int(np.sum(loop_clean)) >= int(0.75 * n_pts):
            ordered = _order_closed_cycle(loop_clean)
            return [(ordered, True)]

    # Case 2: Multi-loop graph (large_holes >= 2) or open/branched structure (large_holes == 0)
    # Remove short side hairs attached to junctions without shortening genuine open endpoints/ticks.
    hair_len = 6 if n_pts >= 180 else max(3, min(4, max_spur_px))
    work_mask = _prune_junction_hairs(_remove_skeleton_pimples(base_mask), max_hair_len=hair_len)

    if not np.any(work_mask):
        work_mask = base_mask

    k_nb = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    work_u8 = work_mask.astype(np.uint8)
    deg = cv2.filter2D(work_u8, cv2.CV_8U, k_nb, borderType=cv2.BORDER_CONSTANT) * work_u8

    def _split_asymmetric_l_chain(pts_in: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
        """Split an open chain if it consists of a long straight line and a short perpendicular tick meeting at a ~90 deg corner."""
        if len(pts_in) < 25:
            return [(pts_in, False)]
        p_a = pts_in[0]
        p_c = pts_in[-1]
        ac = p_c - p_a
        ac_len = float(np.hypot(ac[0], ac[1]))
        if ac_len < 15.0:
            return [(pts_in, False)]
        n_ac = np.array([-ac[1] / ac_len, ac[0] / ac_len], dtype=np.float64)
        perp_d = np.abs((pts_in - p_a) @ n_ac)
        sp = int(np.argmax(perp_d))
        if sp < 8 or sp > len(pts_in) - 8 or float(perp_d[sp]) < 8.0:
            return [(pts_in, False)]
        p_b = pts_in[sp]
        v1 = p_a - p_b
        v2 = p_c - p_b
        l1 = float(np.hypot(v1[0], v1[1]))
        l2 = float(np.hypot(v2[0], v2[1]))
        if min(l1, l2) < 12.0 or max(l1, l2) / max(1e-6, min(l1, l2)) < 2.2:
            return [(pts_in, False)]
        cos_c = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        ang_c = float(np.degrees(np.arccos(cos_c)))
        if not (65.0 <= ang_c <= 115.0):
            return [(pts_in, False)]
        # Verify both arms A->B and B->C are straight lines (mean point-to-chord residual <= 2.0 px)
        n1 = np.array([-v1[1] / l1, v1[0] / l1], dtype=np.float64)
        n2 = np.array([-v2[1] / l2, v2[0] / l2], dtype=np.float64)
        err1 = float(np.mean(np.abs((pts_in[: sp + 1] - p_b) @ n1)))
        err2 = float(np.mean(np.abs((pts_in[sp:] - p_b) @ n2)))
        if err1 <= 2.0 and err2 <= 2.0:
            return [(pts_in[: sp + 1], False), (pts_in[sp:], False)]
        return [(pts_in, False)]

    # Subcase 2A: Pure unbranched open curve (line, arc, Bezier, sine wave, zigzag)
    if int(np.max(deg)) < 3:
        pts_chain = _order_simple_chain(work_mask)
        if len(pts_chain) < 2:
            return []
        end_gap = float(np.hypot(pts_chain[-1, 0] - pts_chain[0, 0], pts_chain[-1, 1] - pts_chain[0, 1]))
        is_cl = bool(end_gap <= 8.0 and len(pts_chain) > 24)
        if not is_cl:
            return _split_asymmetric_l_chain(pts_chain)
        return [(pts_chain, is_cl)]

    # Subcase 2B: Has junctions -> cluster junctions, extract degree<=2 edge segments
    junc_raw = (deg >= 3).astype(np.uint8)
    junc_dil = cv2.dilate(junc_raw, np.ones((3, 3), dtype=np.uint8)) & work_u8
    n_j, j_lbl, _, j_cent = cv2.connectedComponentsWithStats(junc_dil, connectivity=8)

    edges_mask = work_mask & (junc_dil == 0)
    n_e, e_lbl, _, _ = cv2.connectedComponentsWithStats(edges_mask.astype(np.uint8), connectivity=8)

    h, w = work_mask.shape
    raw_segs: List[Dict[str, object]] = []

    def find_touching_junctions(em_mask: np.ndarray, pt0: np.ndarray, pt1: np.ndarray) -> Tuple[int, int]:
        px0, py0 = int(round(float(pt0[0]))), int(round(float(pt0[1])))
        px1, py1 = int(round(float(pt1[0]))), int(round(float(pt1[1])))
        js0, js1 = -1, -1
        for r in range(1, 6):
            if js0 > 0:
                break
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = px0 + dx, py0 + dy
                    if 0 <= nx < w and 0 <= ny < h and j_lbl[ny, nx] > 0:
                        js0 = int(j_lbl[ny, nx])
                        break
                if js0 > 0:
                    break
        for r in range(1, 6):
            if js1 > 0:
                break
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = px1 + dx, py1 + dy
                    if 0 <= nx < w and 0 <= ny < h and j_lbl[ny, nx] > 0:
                        js1 = int(j_lbl[ny, nx])
                        break
                if js1 > 0:
                    break
        # If both ends resolved to the same junction on a short segment, check if the edge mask touches two distinct junctions
        if js0 > 0 and js0 == js1 and int(np.sum(em_mask)) <= 12:
            em_dil = cv2.dilate(em_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8))
            touched = sorted(list(set(int(v) for v in j_lbl[em_dil > 0] if v > 0)))
            if len(touched) >= 2:
                d_a = np.hypot(pt0[0] - j_cent[touched[0]][0], pt0[1] - j_cent[touched[0]][1])
                d_b = np.hypot(pt0[0] - j_cent[touched[1]][0], pt0[1] - j_cent[touched[1]][1])
                if d_a <= d_b:
                    js0, js1 = touched[0], touched[1]
                else:
                    js0, js1 = touched[1], touched[0]
        return js0, js1

    for e_idx in range(1, n_e):
        em = e_lbl == e_idx
        pts_e = _order_simple_chain(em)
        if len(pts_e) == 0:
            continue
        j_start, j_end = find_touching_junctions(em, pts_e[0], pts_e[-1])
        pts_list = list(pts_e)
        if j_start > 0:
            pts_list.insert(0, np.array([float(j_cent[j_start][0]), float(j_cent[j_start][1])]))
        if j_end > 0 and (len(pts_e) > 1 or j_end != j_start):
            pts_list.append(np.array([float(j_cent[j_end][0]), float(j_cent[j_end][1])]))
        arr_e = np.array(pts_list, dtype=np.float64)
        # Drop tiny self-loops around a single junction cluster
        if j_start > 0 and j_start == j_end and len(arr_e) < 35:
            continue
        raw_segs.append(
            {
                "pts": arr_e,
                "j_start": j_start,
                "j_end": j_end,
                "pixel_count": len(pts_e),
                "active": True,
            }
        )

    # Junction pairing across X/T-junctions and collinear continuation
    if n_j > 1:
        def outward_tangent(pts_arr: np.ndarray, from_start: bool) -> np.ndarray:
            k_step = min(len(pts_arr) - 1, 10)
            if k_step <= 0:
                return np.array([1.0, 0.0])
            if from_start:
                v = pts_arr[k_step] - pts_arr[0]
            else:
                v = pts_arr[-1 - k_step] - pts_arr[-1]
            norm = float(np.hypot(v[0], v[1]))
            return v / norm if norm > 1e-6 else np.array([1.0, 0.0])

        # Only deactivate 1-2 px burrs before junction pairing so crossing tick arms (>= 4 px) can pair first
        for seg in raw_segs:
            js, je = int(seg["j_start"]), int(seg["j_end"])
            px_cnt = int(seg["pixel_count"])
            if ((js > 0 and je == -1) or (je > 0 and js == -1)) and px_cnt <= 2:
                seg["active"] = False

        for j_id in range(1, n_j):
            while True:
                incident = []
                for s_idx, seg in enumerate(raw_segs):
                    if not seg["active"]:
                        continue
                    js, je = int(seg["j_start"]), int(seg["j_end"])  # type: ignore[arg-type]
                    if js == j_id and je != j_id:
                        incident.append((s_idx, True))
                    elif je == j_id and js != j_id:
                        incident.append((s_idx, False))

                if len(incident) < 2:
                    break

                best_pair = None
                if len(incident) == 2:
                    s1, start1 = incident[0]
                    s2, start2 = incident[1]
                    u1 = outward_tangent(raw_segs[s1]["pts"], start1)  # type: ignore[arg-type]
                    u2 = outward_tangent(raw_segs[s2]["pts"], start2)  # type: ignore[arg-type]
                    dot = float(np.dot(u1, u2))
                    if dot < -0.65 or (large_holes == 0 and dot < 0.90):
                        best_pair = (s1, start1, s2, start2)
                elif len(incident) >= 3:
                    best_dot = -0.65  # collinear crossing continuation across X/T-junctions
                    for a_i in range(len(incident)):
                        s1, start1 = incident[a_i]
                        u1 = outward_tangent(raw_segs[s1]["pts"], start1)  # type: ignore[arg-type]
                        for b_i in range(a_i + 1, len(incident)):
                            s2, start2 = incident[b_i]
                            u2 = outward_tangent(raw_segs[s2]["pts"], start2)  # type: ignore[arg-type]
                            dot = float(np.dot(u1, u2))
                            if dot < best_dot:
                                best_dot = dot
                                best_pair = (s1, start1, s2, start2)

                if best_pair is None:
                    break
                s1, start1, s2, start2 = best_pair
                pts1 = raw_segs[s1]["pts"]  # type: ignore[assignment]
                pts2 = raw_segs[s2]["pts"]  # type: ignore[assignment]
                # Orient pts1 to end at j_id, and pts2 to start at j_id
                if start1:
                    pts1 = pts1[::-1]
                    new_j_start = int(raw_segs[s1]["j_end"])  # type: ignore[arg-type]
                else:
                    new_j_start = int(raw_segs[s1]["j_start"])  # type: ignore[arg-type]
                if not start2:
                    pts2 = pts2[::-1]
                    new_j_end = int(raw_segs[s2]["j_start"])  # type: ignore[arg-type]
                else:
                    new_j_end = int(raw_segs[s2]["j_end"])  # type: ignore[arg-type]
                merged_pts = np.vstack([pts1[:-1], pts2])
                raw_segs[s1]["pts"] = merged_pts
                raw_segs[s1]["j_start"] = new_j_start
                raw_segs[s1]["j_end"] = new_j_end
                raw_segs[s1]["pixel_count"] = int(raw_segs[s1]["pixel_count"]) + int(raw_segs[s2]["pixel_count"])  # type: ignore[arg-type]
                raw_segs[s2]["active"] = False

    chains: List[Tuple[np.ndarray, bool]] = []
    for seg in raw_segs:
        if not seg["active"]:
            continue
        pts_arr = seg["pts"]  # type: ignore[assignment]
        px_cnt = int(seg["pixel_count"])  # type: ignore[arg-type]
        js, je = int(seg["j_start"]), int(seg["j_end"])  # type: ignore[arg-type]
        # Prune unpaired dead-end side whiskers attached to a junction (short spurs <= max_spur_px)
        if (js > 0 and je == -1) or (je > 0 and js == -1):
            if px_cnt <= max(4, max_spur_px):
                continue
        if len(pts_arr) >= 3:
            for sub_pts, sub_cl in _split_asymmetric_l_chain(pts_arr):
                chains.append((sub_pts, sub_cl))

    return chains


def _smooth_chain_gaussian(
    pts: np.ndarray, is_closed: bool, sigma: float = 2.0
) -> np.ndarray:
    """Smooth a 2D pixel chain with a 1D Gaussian filter (sigma ~2 px), preserving endpoints on open chains."""
    if len(pts) < 5 or sigma <= 1e-3:
        return pts.copy()
    if is_closed:
        core = pts[:-1] if np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]) < 2.0 else pts
        if len(core) < 5:
            return pts.copy()
        sx = ndimage.gaussian_filter1d(core[:, 0], sigma=sigma, mode="wrap")
        sy = ndimage.gaussian_filter1d(core[:, 1], sigma=sigma, mode="wrap")
        res = np.column_stack([sx, sy]).astype(np.float64)
        return np.vstack([res, res[0:1]])
    sx = ndimage.gaussian_filter1d(pts[:, 0], sigma=sigma, mode="nearest")
    sy = ndimage.gaussian_filter1d(pts[:, 1], sigma=sigma, mode="nearest")
    res = np.column_stack([sx, sy]).astype(np.float64)
    res[0] = pts[0]
    res[-1] = pts[-1]
    return res


def _is_three_sided_trapezoid(pts: np.ndarray) -> bool:
    """Return True if an open chain forms a 3-sided convex trapezoid (2 slant legs + 1 base, with open chord parallel to base)."""
    if len(pts) < 40:
        return False
    for eps_mult in (0.025, 0.035, 0.045, 0.060):
        eps_t = max(4.0, eps_mult * float(len(pts)))
        poly = cv2.approxPolyDP(pts.astype(np.float32), epsilon=eps_t, closed=False).reshape(-1, 2).astype(np.float64)
        if len(poly) != 4:
            continue
        e0 = poly[1] - poly[0]
        e1 = poly[2] - poly[1]
        e2 = poly[3] - poly[2]
        e3 = poly[0] - poly[3]  # open closing chord
        l0 = float(np.hypot(e0[0], e0[1]))
        l1 = float(np.hypot(e1[0], e1[1]))
        l2 = float(np.hypot(e2[0], e2[1]))
        l3 = float(np.hypot(e3[0], e3[1]))
        if min(l0, l2) < 16.0 or min(l1, l3) < 20.0:
            continue
        # Check that open chord e3 is parallel (opposite direction) to base e1 within ~28 deg
        cos_base = float(np.dot(e1 / l1, e3 / l3))
        if cos_base > -0.88:
            continue
        # Check that both legs have comparable height perpendicular to base e1
        n_base = np.array([-e1[1] / l1, e1[0] / l1], dtype=np.float64)
        h0 = abs(float(np.dot(e0, n_base)))
        h2 = abs(float(np.dot(e2, n_base)))
        if min(h0, h2) < 10.0 or max(h0, h2) / max(1e-6, min(h0, h2)) > 1.70:
            continue
        return True
    return False


def _close_gaps_and_snap_t_junctions(
    chains: List[Tuple[np.ndarray, bool]],
    min_gap_px: float = 2.5,
    collinear_gap_px: float = 35.0,
    t_snap_px: float = 5.0,
) -> List[Tuple[np.ndarray, bool]]:
    """Close endpoint-to-endpoint gaps (under ~1 mm, corners, curves, or collinear across crossings) and snap T-junction endpoints onto same-color strokes."""
    work: List[List[object]] = [[pts.copy(), bool(is_cl)] for pts, is_cl in chains if len(pts) >= 2]

    def _out_tangent(pts_arr: np.ndarray, at_end: bool) -> Tuple[np.ndarray, np.ndarray]:
        k = min(len(pts_arr) - 1, 8)
        if k <= 0:
            z = np.array([1.0, 0.0], dtype=np.float64)
            return z, z
        v_loc = (pts_arr[-1] - pts_arr[-1 - k]) if at_end else (pts_arr[0] - pts_arr[k])
        n_loc = float(np.hypot(v_loc[0], v_loc[1]))
        u_loc = v_loc / n_loc if n_loc > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

        v_ch = (pts_arr[-1] - pts_arr[0]) if at_end else (pts_arr[0] - pts_arr[-1])
        n_ch = float(np.hypot(v_ch[0], v_ch[1]))
        u_ch = v_ch / n_ch if n_ch > 1e-6 else u_loc
        return u_loc, u_ch

    # 1. Iterative endpoint-to-endpoint gap closing across open chains of the same color
    max_bridge_gap = max(collinear_gap_px, 35.0)
    changed = True
    while changed:
        changed = False
        n_c = len(work)
        for i in range(n_c):
            pts_a, cl_a = work[i][0], bool(work[i][1])  # type: ignore[index]
            if cl_a or len(pts_a) < 2:  # type: ignore[arg-type]
                continue
            self_gap = float(np.hypot(pts_a[-1, 0] - pts_a[0, 0], pts_a[-1, 1] - pts_a[0, 1]))  # type: ignore[index]
            if len(pts_a) >= 20 and self_gap <= max(8.0, min_gap_px * 2.0):  # type: ignore[arg-type]
                work[i][0] = np.vstack([pts_a, pts_a[0:1]])  # type: ignore[index]
                work[i][1] = True
                changed = True
                break

            best_j = -1
            best_mode = ""
            best_score = 999.0
            ua0_loc, ua0_ch = _out_tangent(pts_a, at_end=False)  # type: ignore[arg-type]
            ua1_loc, ua1_ch = _out_tangent(pts_a, at_end=True)  # type: ignore[arg-type]

            for j in range(i + 1, n_c):
                pts_b, cl_b = work[j][0], bool(work[j][1])  # type: ignore[index]
                if cl_b or len(pts_b) < 2:  # type: ignore[arg-type]
                    continue
                ub0_loc, ub0_ch = _out_tangent(pts_b, at_end=False)  # type: ignore[arg-type]
                ub1_loc, ub1_ch = _out_tangent(pts_b, at_end=True)  # type: ignore[arg-type]

                pairs = [
                    ("A1_B0", pts_a[-1], pts_b[0], ua1_loc, ua1_ch, ub0_loc, ub0_ch),  # type: ignore[index]
                    ("A1_B1", pts_a[-1], pts_b[-1], ua1_loc, ua1_ch, ub1_loc, ub1_ch),  # type: ignore[index]
                    ("A0_B0", pts_a[0], pts_b[0], ua0_loc, ua0_ch, ub0_loc, ub0_ch),  # type: ignore[index]
                    ("A0_B1", pts_a[0], pts_b[-1], ua0_loc, ua0_ch, ub1_loc, ub1_ch),  # type: ignore[index]
                ]
                for mode, p_a, p_b, ua_l, ua_c, ub_l, ub_c in pairs:
                    dvec = p_b - p_a
                    d = float(np.hypot(dvec[0], dvec[1]))
                    if d > max_bridge_gap:
                        continue
                    dot_ab = float(np.dot(ua_l, ub_l))
                    u_ab = dvec / d if d > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
                    cos_a = float(np.dot(ua_l, u_ab)) if d > 1e-6 else 1.0
                    cos_b = float(np.dot(ub_l, -u_ab)) if d > 1e-6 else 1.0
                    ok = False
                    score = d
                    if d <= 1.5:
                        ok = True
                        score = d - 50.0
                    elif d <= max(8.5, min_gap_px * 3.5) and dot_ab <= 0.65:
                        ok = True
                        score = d - 40.0
                    elif d <= max(8.5, min_gap_px * 3.0) and cos_a >= 0.10 and cos_b >= 0.10:
                        ok = True
                        score = d - 10.0 * (cos_a + cos_b)
                    elif d <= max(12.0, min_gap_px * 4.5) and dot_ab <= 0.0 and cos_a >= 0.10 and cos_b >= 0.10:
                        ok = True
                        score = d - 10.0 * (cos_a + cos_b)
                    elif d <= max_bridge_gap and cos_a >= 0.40 and cos_b >= 0.40 and dot_ab <= -0.40:
                        ok = True
                        score = d - 20.0 * (cos_a + cos_b)

                    if ok and score < best_score:
                        best_score = score
                        best_j = j
                        best_mode = mode

            if best_j >= 0:
                pts_b = work[best_j][0]  # type: ignore[assignment]
                if best_mode == "A1_B0":
                    merged = np.vstack([pts_a, pts_b])
                elif best_mode == "A1_B1":
                    merged = np.vstack([pts_a, pts_b[::-1]])
                elif best_mode == "A0_B0":
                    merged = np.vstack([pts_b[::-1], pts_a])
                else:
                    merged = np.vstack([pts_b, pts_a])
                end_gap = float(np.hypot(merged[-1, 0] - merged[0, 0], merged[-1, 1] - merged[0, 1]))
                is_cl = bool(len(merged) >= 20 and end_gap <= max(8.0, min_gap_px * 2.0))
                if is_cl:
                    merged = np.vstack([merged, merged[0:1]])
                work[i][0] = merged
                work[i][1] = is_cl
                work.pop(best_j)
                changed = True
                break

    # 2. Close 3-sided trapezoid flaps so Stage 7 fits them as closed 4-point polygons
    for i in range(len(work)):
        pts_a, cl_a = work[i][0], bool(work[i][1])  # type: ignore[index]
        if not cl_a and _is_three_sided_trapezoid(pts_a):  # type: ignore[arg-type]
            poly_t = None
            for eps_mult in (0.025, 0.035, 0.045, 0.060):
                eps_t = max(4.0, eps_mult * float(len(pts_a)))  # type: ignore[arg-type]
                cand = (
                    cv2.approxPolyDP(pts_a.astype(np.float32), epsilon=eps_t, closed=False)  # type: ignore[union-attr]
                    .reshape(-1, 2)
                    .astype(np.float64)
                )
                if len(cand) == 4:
                    poly_t = cand
                    break
            if poly_t is not None:
                closed_t_pts: List[np.ndarray] = []
                for k_t in range(4):
                    p_start_t = poly_t[k_t]
                    p_end_t = poly_t[(k_t + 1) % 4]
                    seg_len_t = int(max(2, round(float(np.hypot(*(p_end_t - p_start_t))))))
                    ts = np.linspace(0.0, 1.0, seg_len_t, endpoint=False)
                    seg_pts_t = p_start_t[None, :] + ts[:, None] * (p_end_t - p_start_t)[None, :]
                    closed_t_pts.append(seg_pts_t)
                closed_t_pts.append(poly_t[0:1])
                work[i][0] = np.vstack(closed_t_pts)
                work[i][1] = True

    # 3. Snap open chain endpoints (T-junctions) onto nearby interior points of same-color chains
    for i in range(len(work)):
        pts_a, cl_a = work[i][0], bool(work[i][1])  # type: ignore[index]
        if cl_a or len(pts_a) < 2:  # type: ignore[arg-type]
            continue
        for ep_idx in (0, -1):
            ep = pts_a[ep_idx]  # type: ignore[index]
            best_pt = None
            best_d = t_snap_px
            for j in range(len(work)):
                if i == j:
                    continue
                pts_b = work[j][0]  # type: ignore[assignment]
                dists = np.hypot(pts_b[:, 0] - ep[0], pts_b[:, 1] - ep[1])
                k_min = int(np.argmin(dists))
                if float(dists[k_min]) <= best_d:
                    best_d = float(dists[k_min])
                    best_pt = pts_b[k_min].copy()
            if best_pt is not None and best_d > 0.5:
                if ep_idx == 0:
                    pts_a[0] = best_pt  # type: ignore[index]
                else:
                    pts_a[-1] = best_pt  # type: ignore[index]

    return [(pts, bool(is_cl)) for pts, is_cl in work]  # type: ignore[misc]


def stage6_vectorize(
    skel_out: SkeletonOutput, prep: PreprocessOutput, cfg: Config
) -> VectorizeOutput:
    """Stage 6: Trace per-color skeleton chains, smooth, simplify with RDP, and scale to bed.

    Args:
        skel_out: Output from Stage 5.
        prep: Output from Stage 2 (contains scale factors back to original resolution).
        cfg: Pipeline configuration.

    Returns:
        VectorizeOutput containing traced curves in both working and original resolutions.
    """
    scale_vec = np.array([prep.scale_x, prep.scale_y], dtype=np.float64)
    offset_vec = np.array(
        [float(prep.roi_offset_x), float(prep.roi_offset_y)], dtype=np.float64
    )
    color_hex_map = {c.name: c.hex_color for c in cfg.colors}
    curves: List[TracedCurve] = []

    has_color_chains = (
        any(len(chains) > 0 for chains in skel_out.color_chains.values())
        if skel_out.color_chains
        else False
    )
    if has_color_chains:
        for color_name, chain_list in skel_out.color_chains.items():
            c_hex = color_hex_map.get(color_name, "#1e88e5")
            for pts_raw, is_closed in chain_list:
                if len(pts_raw) < 6:
                    continue
                pts_work = _smooth_chain_gaussian(
                    pts_raw, is_closed=is_closed, sigma=cfg.chain_smooth_sigma_px
                )
                diffs_work = np.diff(pts_work, axis=0)
                len_work = float(np.sum(np.hypot(diffs_work[:, 0], diffs_work[:, 1])))
                if len_work < 8.0:
                    continue

                simp_work = rdp_simplify(
                    pts_work, cfg.rdp_epsilon if is_closed else 1.2, closed=is_closed
                )
                pts_orig = pts_work * scale_vec + offset_vec
                simp_orig = simp_work * scale_vec + offset_vec
                diffs_orig = np.diff(pts_orig, axis=0)
                len_orig = float(np.sum(np.hypot(diffs_orig[:, 0], diffs_orig[:, 1])))

                curves.append(
                    TracedCurve(
                        points_working=pts_work,
                        simplified_working=simp_work,
                        points_orig=pts_orig,
                        simplified_orig=simp_orig,
                        is_closed=is_closed,
                        length_working=len_work,
                        length_orig=len_orig,
                        color=color_name,
                        color_hex=c_hex,
                    )
                )
        curves.sort(key=lambda c: c.length_orig, reverse=True)
        return VectorizeOutput(curves=curves)

    skel = skel_out.skeleton
    if not np.any(skel):
        return VectorizeOutput(curves=[])

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8), connectivity=8
    )

    for lbl in range(1, num_labels):
        x0 = int(stats[lbl, cv2.CC_STAT_LEFT])
        y0 = int(stats[lbl, cv2.CC_STAT_TOP])
        cw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        ch = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        comp_mask = labels[y0 : y0 + ch, x0 : x0 + cw] == lbl
        chains = _trace_component_chains(comp_mask)
        for pts_local, is_closed in chains:
            pts_work = pts_local + np.array([float(x0), float(y0)], dtype=np.float64)
            if len(pts_work) < 10:
                continue

            pts_work = _smooth_chain_gaussian(
                pts_work, is_closed=is_closed, sigma=cfg.chain_smooth_sigma_px
            )
            diffs_work = np.diff(pts_work, axis=0)
            len_work = float(np.sum(np.hypot(diffs_work[:, 0], diffs_work[:, 1])))
            if len_work < 8.0:
                continue

            simp_work = rdp_simplify(
                pts_work, cfg.rdp_epsilon if is_closed else 0.8, closed=is_closed
            )

            pts_orig = pts_work * scale_vec + offset_vec
            simp_orig = simp_work * scale_vec + offset_vec

            diffs_orig = np.diff(pts_orig, axis=0)
            len_orig = float(np.sum(np.hypot(diffs_orig[:, 0], diffs_orig[:, 1])))

            curves.append(
                TracedCurve(
                    points_working=pts_work,
                    simplified_working=simp_work,
                    points_orig=pts_orig,
                    simplified_orig=simp_orig,
                    is_closed=is_closed,
                    length_working=len_work,
                    length_orig=len_orig,
                )
            )

    # Sort longest curves first
    curves.sort(key=lambda c: c.length_orig, reverse=True)
    return VectorizeOutput(curves=curves)
