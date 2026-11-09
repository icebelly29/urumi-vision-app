"""Stages 7 and 8: Geometric shape model selection (RANSAC + LS) and confidence validation."""

from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from scipy.optimize import least_squares

from .config import (
    Config,
    EnhancementOutput,
    FittedCurve,
    PreprocessOutput,
    TracedCurve,
)


def _fit_line_ransac_ls(
    pts: np.ndarray, iterations: int, tol: float, rng: np.random.Generator
) -> Tuple[Dict[str, float], float, float, np.ndarray]:
    """Fit a line segment to 2D points using vectorized RANSAC + total least squares (SVD)."""
    n = len(pts)
    if n < 2:
        return {}, 1e9, 0.0, pts

    # Subsample candidate pairs in batches
    idx1 = rng.integers(0, n, size=iterations)
    idx2 = rng.integers(0, n, size=iterations)
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid], idx2[valid]
    if len(idx1) == 0:
        idx1, idx2 = np.array([0]), np.array([n - 1])

    p1 = pts[idx1]  # (K, 2)
    p2 = pts[idx2]  # (K, 2)
    d = p2 - p1
    lengths = np.hypot(d[:, 0], d[:, 1])
    good = lengths > 1e-3
    if not np.any(good):
        return {}, 1e9, 0.0, pts

    p1 = p1[good]
    d = d[good]
    lengths = lengths[good]
    normals = np.column_stack([-d[:, 1] / lengths, d[:, 0] / lengths])  # (K, 2)

    # Evaluate inliers on up to 250 subsampled points for speed
    if n > 250:
        sub_idx = np.linspace(0, n - 1, 250).astype(int)
        eval_pts = pts[sub_idx]
    else:
        eval_pts = pts

    # dist[k, i] = |(eval_pts[i] - p1[k]) . normals[k]|
    c_offsets = np.sum(p1 * normals, axis=1, keepdims=True)  # (K, 1)
    dists = np.abs(normals @ eval_pts.T - c_offsets)  # (K, M)
    inlier_counts = np.sum(dists <= tol, axis=1)
    best_k = int(np.argmax(inlier_counts))

    best_n = normals[best_k]
    best_c = float(c_offsets[best_k, 0])
    full_dists = np.abs(pts @ best_n - best_c)
    inliers = full_dists <= tol

    if np.sum(inliers) >= 2:
        inlier_pts = pts[inliers]
    else:
        inlier_pts = pts

    # Total least squares refinement via SVD on inliers
    centroid = np.mean(inlier_pts, axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]  # unit direction vector
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)

    # Project ALL points onto the line to get endpoints and residual
    all_centered = pts - centroid
    residuals = np.abs(all_centered @ normal)
    mean_err = float(np.mean(residuals))
    inlier_ratio = float(np.mean(residuals <= tol))

    t_vals = all_centered @ direction
    t_min, t_max = float(t_vals[0]), float(t_vals[-1])
    # Use extreme projections along the line
    t_start = float(np.min(t_vals))
    t_end = float(np.max(t_vals))
    if abs(t_vals[-1] - t_vals[0]) > 0 and (t_vals[-1] < t_vals[0]):
        t_start, t_end = t_end, t_start

    p_start = centroid + t_start * direction
    p_end = centroid + t_end * direction

    params = {
        "x1": round(float(p_start[0]), 2),
        "y1": round(float(p_start[1]), 2),
        "x2": round(float(p_end[0]), 2),
        "y2": round(float(p_end[1]), 2),
    }
    fitted_pts = np.array([p_start, p_end], dtype=np.float64)
    return params, mean_err, inlier_ratio, fitted_pts


def _kasa_circle_fit(pts: np.ndarray) -> Tuple[float, float, float]:
    """Algebraic Kasa circle fit refined by Pratt/Taubin scaling."""
    x = pts[:, 0]
    y = pts[:, 1]
    z = x * x + y * y
    A = np.column_stack([x, y, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    cx = 0.5 * float(sol[0])
    cy = 0.5 * float(sol[1])
    r_sq = float(sol[2]) + cx * cx + cy * cy
    r = float(np.sqrt(max(r_sq, 1e-6)))
    return cx, cy, r


def _fit_circle_ransac_ls(
    pts: np.ndarray,
    iterations: int,
    tol: float,
    min_r: float,
    max_r: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], float, float, float, np.ndarray]:
    """Fit a circle or circular arc via vectorized 3-point RANSAC + geometric least squares."""
    n = len(pts)
    if n < 6:
        return {}, 1e9, 0.0, 0.0, pts

    if n > 250:
        sub_idx = np.linspace(0, n - 1, 250).astype(int)
        eval_pts = pts[sub_idx]
    else:
        eval_pts = pts

    m = len(eval_pts)
    i1 = rng.integers(0, m, size=iterations)
    i2 = rng.integers(0, m, size=iterations)
    i3 = rng.integers(0, m, size=iterations)

    p1 = eval_pts[i1]
    p2 = eval_pts[i2]
    p3 = eval_pts[i3]

    # Circumcenter of 3 points (vectorized across iterations)
    ax, ay = p1[:, 0], p1[:, 1]
    bx, by = p2[:, 0], p2[:, 1]
    cx, cy = p3[:, 0], p3[:, 1]

    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    valid = np.abs(d) > 1e-3
    if not np.any(valid):
        return {}, 1e9, 0.0, 0.0, pts

    ax, ay, bx, by, cx, cy, d = (
        ax[valid],
        ay[valid],
        bx[valid],
        by[valid],
        cx[valid],
        cy[valid],
        d[valid],
    )
    a2 = ax * ax + ay * ay
    b2 = bx * bx + by * by
    c2 = cx * cx + cy * cy

    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    radii = np.hypot(ax - ux, ay - uy)

    r_valid = (radii >= min_r) & (radii <= max_r)
    if not np.any(r_valid):
        return {}, 1e9, 0.0, 0.0, pts

    ux = ux[r_valid]
    uy = uy[r_valid]
    radii = radii[r_valid]

    # Evaluate inliers in chunks to keep memory small
    dx = ux[:, None] - eval_pts[None, :, 0]
    dy = uy[:, None] - eval_pts[None, :, 1]
    dists = np.abs(np.hypot(dx, dy) - radii[:, None])
    inlier_counts = np.sum(dists <= tol, axis=1)
    best_k = int(np.argmax(inlier_counts))

    best_cx, best_cy, best_r = float(ux[best_k]), float(uy[best_k]), float(radii[best_k])
    full_dists = np.abs(np.hypot(pts[:, 0] - best_cx, pts[:, 1] - best_cy) - best_r)
    inliers = full_dists <= (tol * 1.5)

    inlier_pts = pts[inliers] if np.sum(inliers) >= 6 else pts
    init_cx, init_cy, init_r = _kasa_circle_fit(inlier_pts)

    # Geometric least squares refinement
    def residuals_fn(params: np.ndarray) -> np.ndarray:
        return np.hypot(inlier_pts[:, 0] - params[0], inlier_pts[:, 1] - params[1]) - params[2]

    res = least_squares(
        residuals_fn,
        x0=np.array([init_cx, init_cy, init_r], dtype=np.float64),
        loss="soft_l1",
        f_scale=tol,
        max_nfev=50,
    )
    cx_f, cy_f, r_f = float(res.x[0]), float(res.x[1]), abs(float(res.x[2]))

    if not (min_r <= r_f <= max_r):
        return {}, 1e9, 0.0, 0.0, pts

    all_res = np.abs(np.hypot(pts[:, 0] - cx_f, pts[:, 1] - cy_f) - r_f)
    mean_err = float(np.mean(all_res))
    inlier_ratio = float(np.mean(all_res <= tol))

    # Angular coverage in 10-degree bins (36 bins total)
    angles_rad = np.arctan2(pts[:, 1] - cy_f, pts[:, 0] - cx_f)
    angles_deg = np.degrees(angles_rad) % 360.0
    bins = np.floor(angles_deg / 10.0).astype(int) % 36
    coverage_deg = float(len(np.unique(bins)) * 10.0)

    start_angle = float(np.degrees(angles_rad[0]))
    end_angle = float(np.degrees(angles_rad[-1]))

    # Sample smooth points along fitted circle/arc
    if coverage_deg >= 290.0:
        t_angles = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=True)
    else:
        # Unwrap angles along the ordered curve so arc follows the drawn path
        unwrapped = np.unwrap(angles_rad)
        t_angles = np.linspace(float(unwrapped[0]), float(unwrapped[-1]), 48)
        start_angle = float(np.degrees(unwrapped[0]))
        end_angle = float(np.degrees(unwrapped[-1]))

    fitted_pts = np.column_stack(
        [cx_f + r_f * np.cos(t_angles), cy_f + r_f * np.sin(t_angles)]
    )

    params = {
        "cx": round(cx_f, 2),
        "cy": round(cy_f, 2),
        "r": round(r_f, 2),
        "start_angle_deg": round(start_angle, 2),
        "end_angle_deg": round(end_angle, 2),
    }
    return params, mean_err, inlier_ratio, coverage_deg, fitted_pts


def _ellipse_Signed_Distance_Approx(
    pts: np.ndarray, cx: float, cy: float, a: float, b: float, angle_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute orthogonal-like distance from 2D points to an ellipse (Sampson/polar approximation)."""
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    xp = dx * cos_t + dy * sin_t
    yp = -dx * sin_t + dy * cos_t

    phi = np.arctan2(yp / max(b, 1e-6), xp / max(a, 1e-6))
    ex = a * np.cos(phi)
    ey = b * np.sin(phi)
    dists = np.hypot(xp - ex, yp - ey)
    return dists, np.degrees(phi) % 360.0


def _fit_ellipse_ransac_ls(
    pts: np.ndarray,
    iterations: int,
    tol: float,
    min_axis: float,
    max_axis: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], float, float, float, np.ndarray]:
    """Fit an ellipse using RANSAC subsampling + OpenCV AMS / Fitzgibbon + least-squares refinement."""
    n = len(pts)
    if n < 12:
        return {}, 1e9, 0.0, 0.0, pts

    best_inliers = -1
    best_box = None

    # Use 6-point subsets for RANSAC (80 iterations of cv2.fitEllipse + full fit candidate)
    n_trials = min(80, max(20, iterations // 20))
    pts_f32 = pts.astype(np.float32)

    candidates = []
    try:
        full_box = cv2.fitEllipseAMS(pts_f32)
        candidates.append(full_box)
    except cv2.error:
        pass

    for _ in range(n_trials):
        idx = rng.choice(n, size=min(n, 8), replace=False)
        sub = pts_f32[idx]
        try:
            box = cv2.fitEllipse(sub)
            candidates.append(box)
        except cv2.error:
            continue

    for box in candidates:
        (cx, cy), (w, h), ang = box
        a, b = w * 0.5, h * 0.5
        if a < min_axis or b < min_axis or a > max_axis or b > max_axis:
            continue
        if max(a, b) / max(1e-6, min(a, b)) > 6.0:
            continue
        dists, _ = _ellipse_Signed_Distance_Approx(pts, cx, cy, a, b, ang)
        cnt = int(np.sum(dists <= tol))
        if cnt > best_inliers:
            best_inliers = cnt
            best_box = (cx, cy, a, b, ang)

    if best_box is None:
        return {}, 1e9, 0.0, 0.0, pts

    cx0, cy0, a0, b0, ang0 = best_box
    dists0, _ = _ellipse_Signed_Distance_Approx(pts, cx0, cy0, a0, b0, ang0)
    inliers = dists0 <= (tol * 1.5)
    inlier_pts = pts[inliers] if np.sum(inliers) >= 8 else pts

    def res_fn(p: np.ndarray) -> np.ndarray:
        d, _ = _ellipse_Signed_Distance_Approx(
            inlier_pts, float(p[0]), float(p[1]), abs(float(p[2])), abs(float(p[3])), float(p[4])
        )
        return d

    opt = least_squares(
        res_fn,
        x0=np.array([cx0, cy0, a0, b0, ang0], dtype=np.float64),
        loss="soft_l1",
        f_scale=tol,
        max_nfev=45,
    )
    cx_f, cy_f = float(opt.x[0]), float(opt.x[1])
    a_f, b_f = abs(float(opt.x[2])), abs(float(opt.x[3]))
    ang_f = float(opt.x[4]) % 180.0

    # Ensure rx >= ry convention
    if b_f > a_f:
        a_f, b_f = b_f, a_f
        ang_f = (ang_f + 90.0) % 180.0

    if a_f < min_axis or b_f < min_axis or a_f > max_axis or b_f > max_axis:
        return {}, 1e9, 0.0, 0.0, pts

    all_dists, phi_deg = _ellipse_Signed_Distance_Approx(pts, cx_f, cy_f, a_f, b_f, ang_f)
    mean_err = float(np.mean(all_dists))
    inlier_ratio = float(np.mean(all_dists <= tol))

    bins = np.floor(phi_deg / 10.0).astype(int) % 36
    coverage_deg = float(len(np.unique(bins)) * 10.0)

    t = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=True)
    theta = np.radians(ang_f)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    ex = a_f * np.cos(t)
    ey = b_f * np.sin(t)
    fx = cx_f + ex * cos_t - ey * sin_t
    fy = cy_f + ex * sin_t + ey * cos_t
    fitted_pts = np.column_stack([fx, fy])

    params = {
        "cx": round(cx_f, 2),
        "cy": round(cy_f, 2),
        "rx": round(a_f, 2),
        "ry": round(b_f, 2),
        "angle_deg": round(ang_f, 2),
    }
    return params, mean_err, inlier_ratio, coverage_deg, fitted_pts


def _fit_single_cubic_schneider(
    pts: np.ndarray, t_hat1: np.ndarray, t_hat2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """Fit a single cubic Bezier curve to pts with unit endpoint tangents t_hat1, t_hat2 (Schneider algorithm)."""
    n = len(pts)
    p0 = pts[0]
    p3 = pts[-1]
    chord = float(np.hypot(p3[0] - p0[0], p3[1] - p0[1]))
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum[-1])
    if total_len < 1e-6:
        c1 = p0 + t_hat1 * (chord / 3.0)
        c2 = p3 + t_hat2 * (chord / 3.0)
        return c1, c2, 0.0, 0
    u = cum / total_len

    def _solve_alpha(u_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        omt = 1.0 - u_arr
        b0 = omt ** 3
        b1 = 3.0 * (omt ** 2) * u_arr
        b2 = 3.0 * omt * (u_arr ** 2)
        b3 = u_arr ** 3
        a1 = np.outer(b1, t_hat1)
        a2 = np.outer(b2, t_hat2)
        c11 = float(np.sum(a1 * a1))
        c12 = float(np.sum(a1 * a2))
        c22 = float(np.sum(a2 * a2))
        rhs = pts - (
            np.outer(b0 + b1, p0) + np.outer(b2 + b3, p3)
        )
        x1 = float(np.sum(rhs * a1))
        x2 = float(np.sum(rhs * a2))
        det = c11 * c22 - c12 * c12
        if abs(det) > 1e-6:
            alpha1 = (x1 * c22 - x2 * c12) / det
            alpha2 = (c11 * x2 - c12 * x1) / det
        else:
            alpha1 = total_len / 3.0
            alpha2 = total_len / 3.0
        max_alpha = max(chord, total_len * 0.75)
        if alpha1 < 1e-3 or alpha2 < 1e-3 or alpha1 > max_alpha or alpha2 > max_alpha:
            alpha1 = float(np.clip(alpha1, max(1.0, 0.1 * chord), max_alpha)) if alpha1 > 1e-3 else total_len / 3.0
            alpha2 = float(np.clip(alpha2, max(1.0, 0.1 * chord), max_alpha)) if alpha2 > 1e-3 else total_len / 3.0
        return p0 + t_hat1 * alpha1, p3 + t_hat2 * alpha2

    def _eval_cubic(u_arr: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
        omt = 1.0 - u_arr
        b0 = omt ** 3
        b1 = 3.0 * (omt ** 2) * u_arr
        b2 = 3.0 * omt * (u_arr ** 2)
        b3 = u_arr ** 3
        return (
            np.outer(b0, p0)
            + np.outer(b1, c1)
            + np.outer(b2, c2)
            + np.outer(b3, p3)
        )

    c1, c2 = _solve_alpha(u)
    q_init = _eval_cubic(u, c1, c2)
    errs = np.hypot(q_init[:, 0] - pts[:, 0], q_init[:, 1] - pts[:, 1])
    best_max_err = float(np.max(errs))
    best_c1, best_c2 = c1, c2
    best_errs = errs

    # Only run Newton-Raphson reparameterization when the single cubic is already within 15 px of the curve
    if best_max_err <= 15.0:
        for _ in range(4):
            omt = 1.0 - u
            q_u = _eval_cubic(u, c1, c2)
            q1_u = (
                3.0 * np.outer(omt ** 2, c1 - p0)
                + 6.0 * np.outer(omt * u, c2 - c1)
                + 3.0 * np.outer(u ** 2, p3 - c2)
            )
            q2_u = 6.0 * np.outer(omt, c2 - 2.0 * c1 + p0) + 6.0 * np.outer(
                u, p3 - 2.0 * c2 + c1
            )
            num = np.sum((q_u - pts) * q1_u, axis=1)
            den = np.sum(q1_u * q1_u, axis=1) + np.sum((q_u - pts) * q2_u, axis=1)
            valid = np.abs(den) > 1e-6
            u_new = u.copy()
            u_new[valid] = u[valid] - num[valid] / den[valid]
            u_new = np.clip(u_new, 0.0, 1.0)
            u_new[0] = 0.0
            u_new[-1] = 1.0
            u_new = np.maximum.accumulate(u_new)
            c1_try, c2_try = _solve_alpha(u_new)
            q_try = _eval_cubic(u_new, c1_try, c2_try)
            errs_try = np.hypot(q_try[:, 0] - pts[:, 0], q_try[:, 1] - pts[:, 1])
            max_try = float(np.max(errs_try))
            if max_try < best_max_err:
                best_max_err = max_try
                best_c1, best_c2 = c1_try, c2_try
                best_errs = errs_try
                u = u_new
                c1, c2 = c1_try, c2_try
            else:
                break

    raw_split = int(np.argmax(best_errs))
    if raw_split < 2 or raw_split > n - 3:
        split_idx = n // 2
    else:
        split_idx = raw_split
    return best_c1, best_c2, best_max_err, split_idx


def _fit_schneider_bezier_path(
    pts: np.ndarray, max_err: float
) -> Tuple[Dict[str, object], float, float, np.ndarray]:
    """Fit a piecewise smooth cubic Bezier path to an ordered open curve using Philip Schneider's algorithm."""
    n = len(pts)
    if n < 4:
        return {}, 1e9, 0.0, pts

    def _unit_vec(v: np.ndarray) -> np.ndarray:
        norm = float(np.hypot(v[0], v[1]))
        return v / norm if norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)

    def _fit_recursive(
        sub: np.ndarray, t1: np.ndarray, t2: np.ndarray, depth: int
    ) -> List[List[List[float]]]:
        if len(sub) <= 5 or depth >= 6:
            p0, p3 = sub[0], sub[-1]
            ch = float(np.hypot(p3[0] - p0[0], p3[1] - p0[1])) / 3.0
            c1 = p0 + t1 * ch
            c2 = p3 + t2 * ch
            return [
                [
                    [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                    [round(float(c1[0]), 2), round(float(c1[1]), 2)],
                    [round(float(c2[0]), 2), round(float(c2[1]), 2)],
                    [round(float(p3[0]), 2), round(float(p3[1]), 2)],
                ]
            ]
        c1, c2, max_d, sp = _fit_single_cubic_schneider(sub, t1, t2)
        if max_d <= max_err or len(sub) < 10:
            p0, p3 = sub[0], sub[-1]
            return [
                [
                    [round(float(p0[0]), 2), round(float(p0[1]), 2)],
                    [round(float(c1[0]), 2), round(float(c1[1]), 2)],
                    [round(float(c2[0]), 2), round(float(c2[1]), 2)],
                    [round(float(p3[0]), 2), round(float(p3[1]), 2)],
                ]
            ]
        k_tan = min(4, sp, len(sub) - 1 - sp)
        v_mid = sub[sp - k_tan] - sub[sp + k_tan]
        t_center1 = _unit_vec(v_mid)
        t_center2 = -t_center1
        left_segs = _fit_recursive(sub[: sp + 1], t1, t_center1, depth + 1)
        right_segs = _fit_recursive(sub[sp:], t_center2, t2, depth + 1)
        return left_segs + right_segs

    k_end = min(6, n - 1)
    t_start = _unit_vec(pts[k_end] - pts[0])
    t_end = _unit_vec(pts[-1 - k_end] - pts[-1])
    bez_segs = _fit_recursive(pts, t_start, t_end, 0)

    # Sample dense points along the piecewise cubic Bezier path
    sampled_list: List[np.ndarray] = []
    t_loc = np.linspace(0.0, 1.0, 32)
    omt = 1.0 - t_loc
    for idx_s, seg in enumerate(bez_segs):
        b_p0 = np.array(seg[0], dtype=np.float64)
        b_p1 = np.array(seg[1], dtype=np.float64)
        b_p2 = np.array(seg[2], dtype=np.float64)
        b_p3 = np.array(seg[3], dtype=np.float64)
        pts_seg = (
            np.outer(omt ** 3, b_p0)
            + np.outer(3.0 * (omt ** 2) * t_loc, b_p1)
            + np.outer(3.0 * omt * (t_loc ** 2), b_p2)
            + np.outer(t_loc ** 3, b_p3)
        )
        if idx_s > 0:
            pts_seg = pts_seg[1:]
        sampled_list.append(pts_seg)
    fitted_pts = np.vstack(sampled_list) if sampled_list else pts.copy()

    dx = pts[:, None, 0] - fitted_pts[None, :, 0]
    dy = pts[:, None, 1] - fitted_pts[None, :, 1]
    min_dists = np.min(np.hypot(dx, dy), axis=1)
    mean_err = float(np.mean(min_dists))
    inlier_ratio = float(np.mean(min_dists <= max_err * 1.25))

    params: Dict[str, object] = {
        "beziers": bez_segs,
        "control_points": bez_segs[0],
        "num_segments": len(bez_segs),
    }
    return params, mean_err, inlier_ratio, fitted_pts


def _extract_closed_polygon(
    pts: np.ndarray, avg_scale: float
) -> Tuple[np.ndarray, float, float]:
    """Extract clean corner vertices of a closed chain, refine sides by line intersection, and return (verts, mean_err, min_angle_deg)."""
    core = (
        pts[:-1]
        if (len(pts) > 3 and np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]) < 3.0 * avg_scale)
        else pts
    )
    if len(core) < 6:
        res = _polyline_point_residuals(pts, np.vstack([core, core[0:1]]))
        return core, float(np.mean(res)), 90.0
    diffs = np.diff(np.vstack([core, core[0:1]]), axis=0)
    perim = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    eps = max(4.5 * avg_scale, 0.025 * perim)
    approx = (
        cv2.approxPolyDP(core.astype(np.float32).reshape(-1, 1, 2), epsilon=eps, closed=True)
        .reshape(-1, 2)
        .astype(np.float64)
    )

    # Collapse short bevel edges (< 6% of perimeter) and nearly-collinear vertices (interior angle > 152 deg)
    verts = approx.copy()
    for _ in range(4):
        m = len(verts)
        if m <= 3:
            break
        merged_any = False
        for i in range(m):
            p_prev = verts[(i - 1) % m]
            p_cur = verts[i]
            p_next = verts[(i + 1) % m]
            seg_len = float(np.hypot(*(p_next - p_cur)))
            if seg_len < 0.065 * perim and m > 3:
                verts[i] = 0.5 * (p_cur + p_next)
                verts = np.delete(verts, (i + 1) % m, axis=0)
                merged_any = True
                break
            v1 = p_prev - p_cur
            v2 = p_next - p_cur
            n1 = float(np.hypot(v1[0], v1[1]))
            n2 = float(np.hypot(v2[0], v2[1]))
            if n1 > 1e-3 and n2 > 1e-3:
                cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cos_a)))
                if ang >= 152.0:
                    verts = np.delete(verts, i, axis=0)
                    merged_any = True
                    break
        if not merged_any:
            break

    m = len(verts)
    if m < 3:
        return verts, 1e9, 180.0

    # Refine 3- or 4-corner polygon vertices by fitting SVD lines to each side's interior points and intersecting adjacent lines
    if m in (3, 4):
        idx_corners = [
            int(np.argmin(np.hypot(core[:, 0] - v[0], core[:, 1] - v[1])))
            for v in verts
        ]
        n_c = len(core)
        side_lines: List[Tuple[np.ndarray, np.ndarray]] = []
        valid_sides = True
        for k in range(m):
            i_start = idx_corners[k]
            i_end = idx_corners[(k + 1) % m]
            if i_end > i_start:
                seg_pts = core[i_start : i_end + 1]
            else:
                seg_pts = np.vstack([core[i_start:], core[: i_end + 1]])
            if len(seg_pts) >= 6:
                trim = max(1, len(seg_pts) // 6)
                mid_pts = seg_pts[trim:-trim] if len(seg_pts) - 2 * trim >= 3 else seg_pts
                c_pt = np.mean(mid_pts, axis=0)
                _, _, vh = np.linalg.svd(mid_pts - c_pt, full_matrices=False)
                d_vec = vh[0]
                side_lines.append((c_pt, d_vec))
            else:
                valid_sides = False
                break
        if valid_sides and len(side_lines) == m:
            refined = []
            for k in range(m):
                p_a, d_a = side_lines[(k - 1) % m]
                p_b, d_b = side_lines[k]
                cross = float(d_a[0] * d_b[1] - d_a[1] * d_b[0])
                if abs(cross) > 0.15:
                    delta = p_b - p_a
                    t_a = float(delta[0] * d_b[1] - delta[1] * d_b[0]) / cross
                    inter = p_a + t_a * d_a
                    if float(np.hypot(*(inter - verts[k]))) <= 18.0 * avg_scale:
                        refined.append(inter)
                    else:
                        refined.append(verts[k])
                else:
                    refined.append(verts[k])
            verts = np.array(refined, dtype=np.float64)

    closed_poly = np.vstack([verts, verts[0:1]])
    res = _polyline_point_residuals(pts, closed_poly)
    mean_err = float(np.mean(res))
    min_ang = _min_interior_angle_deg(closed_poly, closed=True, min_seg=3.0)
    return verts, mean_err, min_ang


def _check_and_snap_rectangle(
    verts: np.ndarray, angle_tol_deg: float = 10.0, side_rel_tol: float = 0.18
) -> Tuple[bool, Dict[str, object], np.ndarray]:
    """Check if 4 polygon vertices have angles within ~10 deg of 90 deg and opposite sides roughly equal; if so, snap to a rectangle."""
    if len(verts) != 4:
        return False, {}, verts
    edges = [verts[(i + 1) % 4] - verts[i] for i in range(4)]
    lengths = [float(np.hypot(e[0], e[1])) for e in edges]
    if min(lengths) < 8.0:
        return False, {}, verts

    for i in range(4):
        v1 = -edges[(i - 1) % 4]
        v2 = edges[i]
        l1 = lengths[(i - 1) % 4]
        l2 = lengths[i]
        cos_a = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        if abs(ang - 90.0) > angle_tol_deg:
            return False, {}, verts

    rel_02 = abs(lengths[0] - lengths[2]) / max(lengths[0], lengths[2])
    rel_13 = abs(lengths[1] - lengths[3]) / max(lengths[1], lengths[3])
    if rel_02 > side_rel_tol or rel_13 > side_rel_tol:
        return False, {}, verts

    center = np.mean(verts, axis=0)
    w_avg = 0.5 * (lengths[0] + lengths[2])
    h_avg = 0.5 * (lengths[1] + lengths[3])
    u0 = edges[0] / lengths[0]
    u2 = -edges[2] / lengths[2]
    if float(np.dot(u0, u2)) < 0:
        u2 = -u2
    u_dir = u0 + u2
    u_norm = float(np.hypot(u_dir[0], u_dir[1]))
    u_dir = u_dir / u_norm if u_norm > 1e-6 else np.array([1.0, 0.0], dtype=np.float64)
    ang_deg = float(np.degrees(np.arctan2(u_dir[1], u_dir[0])))
    # Snap near-axis-aligned rectangles (within 6 deg of 0/90/180/-90) to exact axis alignment
    mod90 = ((ang_deg + 45.0) % 90.0) - 45.0
    if abs(mod90) <= 6.0:
        ang_snapped = ang_deg - mod90
        rad = np.radians(ang_snapped)
        u_dir = np.array([np.cos(rad), np.sin(rad)], dtype=np.float64)
        ang_deg = ang_snapped

    v_dir = np.array([-u_dir[1], u_dir[0]], dtype=np.float64)
    if float(np.dot(edges[1], v_dir)) < 0:
        v_dir = -v_dir

    du = 0.5 * w_avg * u_dir
    dv = 0.5 * h_avg * v_dir
    rect_verts = np.array(
        [
            center - du - dv,
            center + du - dv,
            center + du + dv,
            center - du + dv,
        ],
        dtype=np.float64,
    )
    closed_rect = np.vstack([rect_verts, rect_verts[0:1]])
    min_xy = np.min(rect_verts, axis=0)
    max_xy = np.max(rect_verts, axis=0)
    params: Dict[str, object] = {
        "x": round(float(min_xy[0]), 2),
        "y": round(float(min_xy[1]), 2),
        "width": round(float(max_xy[0] - min_xy[0]), 2),
        "height": round(float(max_xy[1] - min_xy[1]), 2),
        "angle_deg": round(float(mod90 if abs(mod90) > 6.0 else 0.0), 2),
        "vertices": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in rect_verts],
    }
    return True, params, closed_rect


def _detect_zigzag_polyline(
    pts: np.ndarray, avg_scale: float
) -> Tuple[bool, np.ndarray, float]:
    """Detect if an open chain is a sharp-cornered zigzag (repeated alternating sharp reversals) vs a smooth wave."""
    if len(pts) < 25:
        return False, pts, 1e9
    eps = 2.6 * avg_scale
    poly = (
        cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), epsilon=eps, closed=False)
        .reshape(-1, 2)
        .astype(np.float64)
    )
    m = len(poly)
    if m < 6:
        return False, poly, 1e9
    res = _polyline_point_residuals(pts, poly)
    mean_err = float(np.mean(res))
    if mean_err > 1.05 * avg_scale:
        return False, poly, mean_err

    sharp_turns = 0
    alt_signs: List[int] = []
    angles: List[float] = []
    for i in range(1, m - 1):
        v1 = poly[i - 1] - poly[i]
        v2 = poly[i + 1] - poly[i]
        l1 = float(np.hypot(v1[0], v1[1]))
        l2 = float(np.hypot(v2[0], v2[1]))
        if l1 < 4.0 * avg_scale or l2 < 4.0 * avg_scale:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        angles.append(ang)
        if ang <= 150.0:
            sharp_turns += 1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) > 1e-3:
            alt_signs.append(1 if cross > 0 else -1)

    alternations = sum(
        1 for k in range(1, len(alt_signs)) if alt_signs[k] != alt_signs[k - 1]
    )
    median_ang = float(np.median(angles)) if angles else 180.0
    if sharp_turns >= 4 and alternations >= 4 and median_ang <= 145.0:
        return True, poly, mean_err
    return False, poly, mean_err


def _is_smooth_wave(pts: np.ndarray, avg_scale: float) -> bool:
    """Detect if an open chain is a smooth oscillating wave or S-curve (direction reversals across its chord)."""
    if len(pts) < 20:
        return False
    chord = pts[-1] - pts[0]
    chord_len = float(np.hypot(chord[0], chord[1]))
    if chord_len < 15.0 * avg_scale:
        return False
    diffs = np.diff(pts, axis=0)
    tot_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    if tot_len / chord_len < 1.08:
        return False
    if tot_len < 250.0 * avg_scale and tot_len / chord_len < 1.35:
        return False
    n_vec = np.array([-chord[1], chord[0]]) / chord_len
    dists = (pts - pts[0]) @ n_vec
    th = 1.8 * avg_scale
    signs = np.zeros(len(dists), dtype=np.int8)
    signs[dists > th] = 1
    signs[dists < -th] = -1
    nonzero = signs[signs != 0]
    if len(nonzero) < 2:
        return False
    crossings = int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))
    return crossings >= 1


def _polyline_point_residuals(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Compute distance from each point in pts to the simplified polyline segments."""
    if len(poly) < 2:
        return np.zeros(len(pts), dtype=np.float64)
    min_d = np.full(len(pts), np.inf, dtype=np.float64)
    for i in range(len(poly) - 1):
        a = poly[i]
        b = poly[i + 1]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-9:
            d = np.hypot(pts[:, 0] - a[0], pts[:, 1] - a[1])
        else:
            t = np.clip(((pts - a) @ ab) / ab_len2, 0.0, 1.0)
            proj = a + np.outer(t, ab)
            d = np.hypot(pts[:, 0] - proj[:, 0], pts[:, 1] - proj[:, 1])
        min_d = np.minimum(min_d, d)
    return min_d


def _min_interior_angle_deg(poly: np.ndarray, closed: bool = False, min_seg: float = 5.0) -> float:
    """Return the sharpest interior turn angle (in degrees, 0..180) along a simplified polyline."""
    if len(poly) < 3:
        return 180.0
    pts_u = poly[:-1] if (closed and len(poly) > 3 and np.hypot(poly[-1, 0] - poly[0, 0], poly[-1, 1] - poly[0, 1]) < 3.0) else poly
    # Collapse tiny (< min_seg px) bevel micro-segments so beveled corners expose their true turn angle
    filtered = [pts_u[0]]
    for k in range(1, len(pts_u)):
        if float(np.hypot(pts_u[k, 0] - filtered[-1][0], pts_u[k, 1] - filtered[-1][1])) >= min_seg:
            filtered.append(pts_u[k])
        elif k == len(pts_u) - 1 and not closed:
            filtered[-1] = pts_u[k]
    pts_c = np.array(filtered, dtype=np.float64)
    m = len(pts_c)
    if m < 3:
        return 180.0
    min_ang = 180.0
    idx_range = range(m) if closed else range(1, m - 1)
    for i in idx_range:
        p_prev = pts_c[(i - 1) % m]
        p_cur = pts_c[i]
        p_next = pts_c[(i + 1) % m]
        v1 = p_prev - p_cur
        v2 = p_next - p_cur
        n1 = float(np.hypot(v1[0], v1[1]))
        n2 = float(np.hypot(v2[0], v2[1]))
        if n1 < 3.5 or n2 < 3.5:
            continue
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        ang = float(np.degrees(np.arccos(cos_a)))
        if ang < min_ang:
            min_ang = ang
    return min_ang


def _classify_ink_color(
    pts_work: np.ndarray,
    lab_channels: Tuple[np.ndarray, np.ndarray, np.ndarray],
    bg_a_map: np.ndarray,
    bg_b_map: np.ndarray,
    med_a: float,
    med_b: float,
) -> Tuple[str, str]:
    """Classify the stroke color along the curve into Urumi CNC layers (thru_cut, score, crease)
    using CIE LAB chromaticity shift (da*, db*) relative to the local cardboard substrate."""
    l_chan, a_chan, b_chan = lab_channels
    h, w = l_chan.shape[:2]

    # Trim junction endpoints on open strokes so touching a differently colored stroke does not bleed votes
    core_pts = pts_work
    if len(pts_work) >= 12:
        trim = max(2, min(8, len(pts_work) // 8))
        if np.hypot(pts_work[-1, 0] - pts_work[0, 0], pts_work[-1, 1] - pts_work[0, 1]) > 8.0:
            core_pts = pts_work[trim:-trim]

    step = max(1, len(core_pts) // 45)
    sample_pts = core_pts[::step]
    votes = {"red": 0, "green": 0, "blue": 0, "black": 0}

    for pt in sample_pts:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        x0, x1 = max(0, cx - 2), min(w, cx + 3)
        y0, y1 = max(0, cy - 2), min(h, cy + 3)
        patch_l = l_chan[y0:y1, x0:x1]
        if patch_l.size == 0:
            continue
        min_idx = np.unravel_index(np.argmin(patch_l), patch_l.shape)
        py, px = y0 + min_idx[0], x0 + min_idx[1]
        a_val = float(a_chan[py, px])
        b_val = float(b_chan[py, px])
        da = 0.5 * (a_val - float(bg_a_map[py, px])) + 0.5 * (a_val - med_a)
        db = 0.5 * (b_val - float(bg_b_map[py, px])) + 0.5 * (b_val - med_b)
        if da >= 5.0 and da >= abs(db) * 0.40:
            votes["red"] += 1
        elif da <= -4.5 and (db >= -5.5 or da <= db * 0.65):
            votes["green"] += 1
        elif db <= -8.0:
            votes["blue"] += 1
        else:
            votes["black"] += 1

    total = max(1, len(sample_pts))
    if votes["red"] / total >= 0.25 and votes["red"] >= votes["green"]:
        return "score", "#ef4444"
    if votes["green"] / total >= 0.25 and votes["green"] > votes["red"]:
        return "crease", "#22c55e"
    return "thru_cut", "#3b82f6"


def line_intersect(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return 0.5 * (p2 + p3)
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dtype=np.float64)


def _assemble_polygon_cycles(
    curves: List[TracedCurve], avg_scale: float = 1.0, max_corner_gap: float = 20.0
) -> List[TracedCurve]:
    """Group open line-like segments of the same color meeting at endpoints into closed polygons or rectangles."""
    open_curves = [c for c in curves if not c.is_closed and c.length_orig >= 20.0]
    other_curves = [c for c in curves if c.is_closed or c.length_orig < 20.0]
    color_groups: Dict[str, List[TracedCurve]] = {}
    for c in open_curves:
        color_groups.setdefault(c.color, []).append(c)

    assembled: List[TracedCurve] = []
    for c_name, segs in color_groups.items():
        used = set()
        m = len(segs)
        for i in range(m):
            if i in used:
                continue
            cycle = None
            for cycle_len in (4, 3):
                if cycle is not None:
                    break

                def find_cycle(path: List[Tuple[int, bool]], visited: set) -> None:
                    nonlocal cycle
                    if cycle is not None:
                        return
                    cur_idx, cur_flipped = path[-1]
                    cur_pts = segs[cur_idx].points_orig
                    cur_end = cur_pts[0] if cur_flipped else cur_pts[-1]
                    if len(path) == cycle_len:
                        start_idx, start_flipped = path[0]
                        start_pts = segs[start_idx].points_orig
                        start_pos = start_pts[-1] if start_flipped else start_pts[0]
                        if np.hypot(*(cur_end - start_pos)) <= max_corner_gap:
                            cycle = list(path)
                        return
                    for next_idx in range(m):
                        if next_idx in visited or next_idx in used:
                            continue
                        next_pts = segs[next_idx].points_orig
                        d0 = float(np.hypot(*(cur_end - next_pts[0])))
                        d1 = float(np.hypot(*(cur_end - next_pts[-1])))
                        if d0 <= max_corner_gap:
                            find_cycle(path + [(next_idx, False)], visited | {next_idx})
                        elif d1 <= max_corner_gap:
                            find_cycle(path + [(next_idx, True)], visited | {next_idx})

                find_cycle([(i, False)], {i})
                if cycle is None:
                    find_cycle([(i, True)], {i})

            if cycle is not None:
                ordered_pts = []
                corner_verts = []
                k_cyc = len(cycle)
                for step_k in range(k_cyc):
                    s_idx, s_fl = cycle[step_k]
                    s_pts = segs[s_idx].points_orig
                    if s_fl:
                        s_pts = s_pts[::-1]
                    ordered_pts.append(s_pts)
                    next_s_idx, next_s_fl = cycle[(step_k + 1) % k_cyc]
                    next_s_pts = segs[next_s_idx].points_orig
                    if next_s_fl:
                        next_s_pts = next_s_pts[::-1]
                    p1, p2 = s_pts[-min(10, len(s_pts))], s_pts[-1]
                    p3, p4 = next_s_pts[0], next_s_pts[min(10, len(next_s_pts) - 1)]
                    v_corner = line_intersect(p1, p2, p3, p4)
                    if np.hypot(*(v_corner - p2)) > max_corner_gap * 1.5:
                        v_corner = 0.5 * (p2 + p3)
                    corner_verts.append(v_corner)

                poly_verts = np.array(corner_verts, dtype=np.float64)
                closed_poly_pts = np.vstack([poly_verts, poly_verts[0:1]])
                all_pts = np.vstack(ordered_pts)
                diffs = np.diff(all_pts, axis=0)
                tot_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

                res_c = _polyline_point_residuals(all_pts, closed_poly_pts)
                if float(np.mean(res_c)) > 1.25 * avg_scale or tot_len < 160.0 * avg_scale:
                    continue

                for seg_idx, _ in cycle:
                    used.add(seg_idx)

                assembled.append(
                    TracedCurve(
                        points_working=all_pts / avg_scale,
                        simplified_working=closed_poly_pts / avg_scale,
                        points_orig=all_pts,
                        simplified_orig=closed_poly_pts,
                        is_closed=True,
                        length_working=tot_len / avg_scale,
                        length_orig=tot_len,
                        color=c_name,
                        color_hex=segs[cycle[0][0]].color_hex,
                    )
                )

        # 3-segment trapezoids
        if c_name == "green":
            max_trap_gap = max(max_corner_gap, 42.0 * avg_scale)
            for i in range(m):
                if i in used:
                    continue
                p_i = segs[i].points_orig
                for j in range(m):
                    if j == i or j in used:
                        continue
                    p_j = segs[j].points_orig
                    for k in range(m):
                        if k in (i, j) or k in used:
                            continue
                        p_k = segs[k].points_orig
                        trap_found = False
                        for f_i in (False, True):
                            pts_i = p_i[::-1] if f_i else p_i
                            for f_j in (False, True):
                                pts_j = p_j[::-1] if f_j else p_j
                                if np.hypot(*(pts_j[0] - pts_i[-1])) > max_trap_gap:
                                    continue
                                for f_k in (False, True):
                                    pts_k = p_k[::-1] if f_k else p_k
                                    if np.hypot(*(pts_k[0] - pts_j[-1])) > max_trap_gap:
                                        continue
                                    chord = pts_i[0] - pts_k[-1]
                                    l_chord = float(np.hypot(chord[0], chord[1]))
                                    base_vec = pts_j[-1] - pts_j[0]
                                    l_base = float(np.hypot(base_vec[0], base_vec[1]))
                                    if l_chord < 20.0 or l_base < 20.0:
                                        continue
                                    cos_cb = float(np.dot(chord / l_chord, base_vec / l_base))
                                    if abs(cos_cb) >= 0.75:
                                        v0 = pts_i[0]
                                        v1 = line_intersect(
                                            pts_i[-min(10, len(pts_i))],
                                            pts_i[-1],
                                            pts_j[0],
                                            pts_j[min(10, len(pts_j) - 1)],
                                        )
                                        if np.hypot(*(v1 - pts_i[-1])) > max_trap_gap * 1.5:
                                            v1 = 0.5 * (pts_i[-1] + pts_j[0])
                                        v2 = line_intersect(
                                            pts_j[-min(10, len(pts_j))],
                                            pts_j[-1],
                                            pts_k[0],
                                            pts_k[min(10, len(pts_k) - 1)],
                                        )
                                        if np.hypot(*(v2 - pts_j[-1])) > max_trap_gap * 1.5:
                                            v2 = 0.5 * (pts_j[-1] + pts_k[0])
                                        v3 = pts_k[-1]
                                        poly_verts = np.array([v0, v1, v2, v3], dtype=np.float64)
                                        closed_poly_pts = np.vstack([poly_verts, poly_verts[0:1]])
                                        all_pts = np.vstack([pts_i, pts_j, pts_k, pts_i[0:1]])
                                        res_trap = _polyline_point_residuals(all_pts, closed_poly_pts)
                                        diffs = np.diff(all_pts, axis=0)
                                        tot_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
                                        if float(np.mean(res_trap)) > 8.0 * avg_scale or tot_len < 160.0 * avg_scale:
                                            continue
                                        assembled.append(
                                            TracedCurve(
                                                points_working=all_pts / avg_scale,
                                                simplified_working=closed_poly_pts / avg_scale,
                                                points_orig=all_pts,
                                                simplified_orig=closed_poly_pts,
                                                is_closed=True,
                                                length_working=tot_len / avg_scale,
                                                length_orig=tot_len,
                                                color=c_name,
                                                color_hex=segs[i].color_hex,
                                            )
                                        )
                                        used.update([i, j, k])
                                        trap_found = True
                                        break
                                if trap_found:
                                    break
                            if trap_found:
                                break
        for i, s in enumerate(segs):
            if i not in used:
                assembled.append(s)
    assembled.extend(other_curves)
    return assembled


def stage7_and_8_fit_and_validate(
    traced_curves: List[TracedCurve],
    prep: PreprocessOutput,
    enh: EnhancementOutput,
    cfg: Config,
) -> List[FittedCurve]:
    """Stages 7 & 8: Fit candidate geometric models in strict priority order and compute confidence.

    Order (stopping at first good match):
    1. Closed chain -> circle/ellipse fit (<circle> / <ellipse>)
    2. Closed chain with ~4 corners (or 3 corners) -> snap to rectangle (<rect>) if angles ~90 deg and opposite sides equal, else closed <polygon>
    3. Open chain with low straight-line residual -> <line>
    4. Zigzag (repeated sharp direction reversals) -> corner-preserving RDP emitted as ONE <polyline>
    5. Smooth curve (the wave) -> Schneider cubic Bezier fit emitted as ONE <path>
    6. Fallback: freehand polyline with RDP (<polyline>)
    """
    orig_h, orig_w = prep.original_shape
    diag = float(np.hypot(orig_w, orig_h))
    min_r = cfg.circle_min_diag_ratio * diag
    max_r = cfg.circle_max_diag_ratio * diag

    avg_scale = 0.5 * (prep.scale_x + prep.scale_y)
    ransac_tol_orig = cfg.ransac_inlier_tol_px * max(1.0, avg_scale)
    fit_tol_orig = cfg.fit_tolerance_px * max(1.0, avg_scale * 0.85)

    # Assemble open curve segments that meet at corners into closed polygon/rect cycles
    traced_curves = _assemble_polygon_cycles(
        traced_curves, avg_scale=avg_scale, max_corner_gap=28.0 * avg_scale
    )

    bed_h, bed_w = prep.full_bed_shape
    if bed_w > 0 and prep.phys_width_mm > 0:
        dots_per_mm = float(bed_w) / float(prep.phys_width_mm)
    else:
        dots_per_mm = float(orig_w) / max(1.0, float(prep.phys_width_mm))

    rng = np.random.default_rng(42)
    results: List[FittedCurve] = []

    s_map = enh.strokeness
    sh, sw = s_map.shape
    valid_bg = ~prep.ignore_mask_working
    if np.any(valid_bg):
        med_a = float(np.median(prep.lab_channels[1][valid_bg]))
        med_b = float(np.median(prep.lab_channels[2][valid_bg]))
    else:
        med_a = float(np.median(prep.lab_channels[1]))
        med_b = float(np.median(prep.lab_channels[2]))
    bg_a_map = cv2.GaussianBlur(prep.lab_channels[1].astype(np.float32), (0, 0), 25.0)
    bg_b_map = cv2.GaussianBlur(prep.lab_channels[2].astype(np.float32), (0, 0), 25.0)

    for tc in traced_curves:
        pts = tc.points_orig
        pts_work = tc.points_working

        # Sample mean strokeness along the curve
        xs_w = np.clip(np.round(pts_work[:, 0]).astype(int), 0, sw - 1)
        ys_w = np.clip(np.round(pts_work[:, 1]).astype(int), 0, sh - 1)
        if tc.color and tc.color in enh.color_strokeness:
            s_vals = enh.color_strokeness[tc.color][ys_w, xs_w]
        else:
            s_vals = s_map[ys_w, xs_w]
        mean_contrast = float(np.clip(np.mean(s_vals), 0.0, 1.0))

        end_gap = float(np.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        mostly_closed = tc.is_closed or (
            tc.length_orig >= 160.0 * avg_scale
            and end_gap <= 22.0 * avg_scale
            and (end_gap / max(1.0, tc.length_orig)) <= 0.08
        )

        chosen_type = "freehand"
        chosen_params: Dict[str, object] = {}
        chosen_pts = tc.simplified_orig
        chosen_err = 0.0
        chosen_inlier = 1.0
        chosen_cov = 180.0

        if tc.is_closed or mostly_closed:
            # Extract clean polygon corners first to distinguish sharp 3-4 corner polygons from circles/ellipses
            poly_verts, poly_err, poly_min_ang = _extract_closed_polygon(pts, avg_scale)
            n_poly_corners = len(poly_verts)
            is_sharp_3_4_poly = (
                n_poly_corners in (3, 4)
                and poly_min_ang <= 125.0
                and poly_err <= fit_tol_orig * 1.15
            )

            # 1. Closed chain -> circle/ellipse fit
            circ_params, circ_err, circ_inlier, circ_cov, circ_pts = _fit_circle_ransac_ls(
                pts, cfg.ransac_iterations, ransac_tol_orig, min_r, max_r, rng
            )
            ell_params, ell_err, ell_inlier, ell_cov, ell_pts = _fit_ellipse_ransac_ls(
                pts, cfg.ransac_iterations, ransac_tol_orig, min_r * 0.5, max_r, rng
            )

            circle_ok = (
                bool(circ_params)
                and not (is_sharp_3_4_poly and poly_err < circ_err * 0.72)
                and circ_cov >= 270.0
                and (
                    (circ_err <= fit_tol_orig * 1.25 and circ_inlier >= 0.65)
                    or (circ_params and circ_err / max(1e-6, float(circ_params["r"])) <= 0.065)
                )
            )
            ellipse_ok = (
                bool(ell_params)
                and not (is_sharp_3_4_poly and poly_err < ell_err * 0.72)
                and ell_cov >= 280.0
                and ell_err <= fit_tol_orig * 1.15
                and ell_inlier >= 0.80
            )
            ell_aspect = (
                float(ell_params["rx"]) / max(1e-6, float(ell_params["ry"]))
                if ell_params
                else 1.0
            )

            if circle_ok and (
                not ellipse_ok
                or circ_err <= fit_tol_orig * 1.25
                or ell_aspect < 1.22
                or circ_err <= ell_err * 1.35 + 0.35
            ):
                chosen_type = "circle"
                chosen_params = {
                    "cx": circ_params["cx"],
                    "cy": circ_params["cy"],
                    "r": circ_params["r"],
                }
                chosen_pts = circ_pts
                chosen_err = circ_err
                chosen_inlier = circ_inlier
                chosen_cov = circ_cov
            elif ellipse_ok:
                chosen_type = "ellipse"
                chosen_params = ell_params
                chosen_pts = ell_pts
                chosen_err = ell_err
                chosen_inlier = ell_inlier
                chosen_cov = ell_cov
            elif 3 <= n_poly_corners <= 6:
                # 2. Closed chain with ~4 corners -> snap to rectangle if angles within ~14.5 deg of 90 deg and opposite sides equal; else closed polygon
                is_rect, rect_params, rect_pts = _check_and_snap_rectangle(
                    poly_verts, angle_tol_deg=14.5, side_rel_tol=0.25
                )
                if is_rect:
                    chosen_type = "rect"
                    chosen_params = rect_params
                    chosen_pts = rect_pts
                    rect_res = _polyline_point_residuals(pts, rect_pts)
                    chosen_err = float(np.mean(rect_res))
                    chosen_inlier = float(np.mean(rect_res <= ransac_tol_orig))
                    chosen_cov = 360.0
                else:
                    closed_poly_pts = np.vstack([poly_verts, poly_verts[0:1]])
                    chosen_type = "polygon"
                    chosen_params = {
                        "num_vertices": n_poly_corners,
                        "closed": True,
                        "vertices": [
                            [round(float(p[0]), 2), round(float(p[1]), 2)]
                            for p in poly_verts
                        ],
                    }
                    chosen_pts = closed_poly_pts
                    chosen_err = poly_err
                    chosen_inlier = float(
                        np.mean(_polyline_point_residuals(pts, closed_poly_pts) <= ransac_tol_orig)
                    )
                    chosen_cov = 360.0
            else:
                poly_res = _polyline_point_residuals(pts, tc.simplified_orig)
                chosen_type = "polygon"
                chosen_params = {
                    "num_vertices": len(tc.simplified_orig),
                    "closed": True,
                }
                chosen_pts = tc.simplified_orig
                chosen_err = float(np.mean(poly_res))
                chosen_inlier = float(np.mean(poly_res <= ransac_tol_orig))
                chosen_cov = 360.0
        else:
            # 3. Open chain with low straight-line residual -> <line>
            line_params, line_err, line_inlier, line_pts = _fit_line_ransac_ls(
                pts, cfg.ransac_iterations, ransac_tol_orig, rng
            )
            line_dir = line_pts[1] - line_pts[0]
            line_norm_len = float(np.hypot(line_dir[0], line_dir[1]))
            if line_norm_len > 1e-6:
                n_vec = np.array([-line_dir[1], line_dir[0]]) / line_norm_len
                sides = (pts - line_pts[0]) @ n_vec
                sign_changes = 0
                prev_s = 0
                for s_val in sides:
                    if abs(s_val) > 1.8 * avg_scale:
                        curr_s = 1 if s_val > 0 else -1
                        if prev_s != 0 and curr_s != prev_s:
                            sign_changes += 1
                        prev_s = curr_s
            else:
                sign_changes = 0

            if sign_changes <= 2 and line_err <= 2.6 * max(1.0, avg_scale * 0.75) and line_inlier >= 0.68:
                d0_0 = float(np.hypot(line_pts[0, 0] - pts[0, 0], line_pts[0, 1] - pts[0, 1]))
                d0_n = float(np.hypot(line_pts[0, 0] - pts[-1, 0], line_pts[0, 1] - pts[-1, 1]))
                if d0_0 <= d0_n:
                    if d0_0 < 12.0 * avg_scale:
                        line_pts[0] = pts[0]
                    if np.hypot(line_pts[1, 0] - pts[-1, 0], line_pts[1, 1] - pts[-1, 1]) < 12.0 * avg_scale:
                        line_pts[1] = pts[-1]
                else:
                    if d0_n < 12.0 * avg_scale:
                        line_pts[0] = pts[-1]
                    if np.hypot(line_pts[1, 0] - pts[0, 0], line_pts[1, 1] - pts[0, 1]) < 12.0 * avg_scale:
                        line_pts[1] = pts[0]
                line_params = {
                    "x1": round(float(line_pts[0, 0]), 2),
                    "y1": round(float(line_pts[0, 1]), 2),
                    "x2": round(float(line_pts[1, 0]), 2),
                    "y2": round(float(line_pts[1, 1]), 2),
                }
                chosen_type = "line"
                chosen_params = line_params
                chosen_pts = line_pts
                chosen_err = line_err
                chosen_inlier = line_inlier
                chosen_cov = 180.0
            else:
                # 4. Zigzag (repeated sharp direction reversals) -> RDP with corner-preserving epsilon, emitted as ONE polyline
                is_zigzag, zz_poly, zz_err = _detect_zigzag_polyline(pts, avg_scale)
                if is_zigzag:
                    bez_params, bez_err, bez_inlier, bez_pts = _fit_schneider_bezier_path(
                        pts, max_err=fit_tol_orig * 1.15
                    )
                    if bool(bez_params) and bez_err < zz_err and bez_err <= fit_tol_orig * 1.35:
                        chosen_type = "bezier"
                        chosen_params = bez_params
                        chosen_pts = bez_pts
                        chosen_err = bez_err
                        chosen_inlier = bez_inlier
                        chosen_cov = 180.0
                    else:
                        chosen_type = "polyline"
                        chosen_params = {
                            "num_vertices": len(zz_poly),
                            "closed": False,
                            "zigzag": True,
                        }
                        chosen_pts = zz_poly
                        chosen_err = zz_err
                        chosen_inlier = float(
                            np.mean(_polyline_point_residuals(pts, zz_poly) <= ransac_tol_orig)
                        )
                        chosen_cov = 180.0
                else:
                    # Check if this open chain is a sharp 2-3 segment corner polyline (e.g. arrow head V)
                    simp_v = (
                        cv2.approxPolyDP(
                            pts.astype(np.float32).reshape(-1, 1, 2),
                            epsilon=3.2 * avg_scale,
                            closed=False,
                        )
                        .reshape(-1, 2)
                        .astype(np.float64)
                    )
                    simp_res = _polyline_point_residuals(pts, simp_v)
                    simp_err = float(np.mean(simp_res))
                    min_open_ang = _min_interior_angle_deg(
                        simp_v, closed=False, min_seg=4.5 * avg_scale
                    )
                    is_sharp_open_corner = (
                        3 <= len(simp_v) <= 5
                        and min_open_ang <= 118.0
                        and simp_err <= 1.25 * avg_scale
                    )

                    circ_params, circ_err, circ_inlier, circ_cov, circ_pts = _fit_circle_ransac_ls(
                        pts, cfg.ransac_iterations, ransac_tol_orig, min_r, max_r, rng
                    )
                    arc_ok = (
                        bool(circ_params)
                        and 50.0 <= circ_cov <= 280.0
                        and float(circ_params.get("r", 0)) >= 120.0 * avg_scale
                        and tc.length_orig >= 250.0 * avg_scale
                        and circ_err <= fit_tol_orig * 1.25
                        and circ_inlier >= 0.82
                        and circ_err < line_err * 0.85
                    )

                    if arc_ok:
                        chosen_type = "arc"
                        chosen_params = circ_params
                        chosen_pts = circ_pts
                        chosen_err = circ_err
                        chosen_inlier = circ_inlier
                        chosen_cov = circ_cov
                    elif is_sharp_open_corner:
                        # 6. Fallback: sharp open corner polyline (e.g. arrow head) with RDP
                        chosen_type = "polyline"
                        chosen_params = {
                            "num_vertices": len(simp_v),
                            "closed": False,
                        }
                        chosen_pts = simp_v
                        chosen_err = simp_err
                        chosen_inlier = float(np.mean(simp_res <= ransac_tol_orig))
                        chosen_cov = 180.0
                    else:
                        is_wave = _is_smooth_wave(pts, avg_scale)
                        if is_wave:
                            # 5. Smooth curve (wave / bezier) -> Schneider cubic Bezier fit, emitted as ONE <path>
                            bez_params, bez_err, bez_inlier, bez_pts = _fit_schneider_bezier_path(
                                pts, max_err=fit_tol_orig * 1.15
                            )
                            if bool(bez_params) and bez_err <= fit_tol_orig * 1.35:
                                chosen_type = "bezier"
                                chosen_params = bez_params
                                chosen_pts = bez_pts
                                chosen_err = bez_err
                                chosen_inlier = bez_inlier
                                chosen_cov = 180.0
                            else:
                                poly_res = _polyline_point_residuals(pts, tc.simplified_orig)
                                chosen_type = "polyline"
                                chosen_params = {
                                    "num_vertices": len(tc.simplified_orig),
                                    "closed": False,
                                }
                                chosen_pts = tc.simplified_orig
                                chosen_err = float(np.mean(poly_res))
                                chosen_inlier = float(np.mean(poly_res <= ransac_tol_orig))
                                chosen_cov = 180.0
                        else:
                            # 6. Fallback: freehand polyline with RDP
                            poly_res = _polyline_point_residuals(pts, tc.simplified_orig)
                            chosen_type = "polyline"
                            chosen_params = {
                                "num_vertices": len(tc.simplified_orig),
                                "closed": False,
                            }
                            chosen_pts = tc.simplified_orig
                            chosen_err = float(np.mean(poly_res))
                            chosen_inlier = float(np.mean(poly_res <= ransac_tol_orig))
                            chosen_cov = 180.0

        # Stage 8: Confidence calculation
        cov_norm = min(chosen_cov / 180.0, 1.0)
        fit_quality = float(np.clip(1.0 - chosen_err / (fit_tol_orig * 2.5), 0.0, 1.0))

        conf = (
            cfg.weight_inlier_ratio * chosen_inlier
            + cfg.weight_coverage * cov_norm
            + cfg.weight_fit_quality * fit_quality
            + cfg.weight_stroke_contrast * mean_contrast
        )
        conf = float(np.clip(conf, 0.0, 1.0))

        if tc.color:
            c_name = tc.color
            layer = tc.color
            color_hex = tc.color_hex or "#1e88e5"
        else:
            layer, color_hex = _classify_ink_color(
                pts_work, prep.lab_channels, bg_a_map, bg_b_map, med_a, med_b
            )
            c_name = (
                "red"
                if layer == "score"
                else ("green" if layer == "crease" else "blue")
            )

        res_mm = round(float(chosen_err) / max(1e-6, dots_per_mm), 3)

        results.append(
            FittedCurve(
                curve_type=chosen_type,
                params=chosen_params,
                points=chosen_pts,
                length_px=round(tc.length_orig, 2),
                fit_error_px=round(chosen_err, 3),
                inlier_ratio=round(chosen_inlier, 3),
                angular_coverage_deg=round(chosen_cov, 1),
                mean_contrast=round(mean_contrast, 3),
                confidence=round(conf, 3),
                layer=layer,
                color_hex=color_hex,
                color=c_name,
                model=chosen_type,
                residual_mm=res_mm,
            )
        )

    # Geometric Constraint Solver: Conservative vertex snapping (<= 10 px) for touching endpoints of the SAME color only
    lines_info = []
    for i, c in enumerate(results):
        if c.curve_type in ("line", "bezier", "polyline") and len(c.points) >= 2:
            p0 = np.array(c.points[0], dtype=np.float64)
            p1 = np.array(c.points[-1], dtype=np.float64)
            d = p1 - p0
            length = float(np.hypot(d[0], d[1]))
            if length > 12.0:
                u = d / length
                lines_info.append(
                    {"idx": i, "p0": p0, "p1": p1, "u": u, "len": length, "color": c.color}
                )

    max_snap_px = 10.0

    def _update_endpoint(curve_idx: int, ep_idx: int, new_pt: np.ndarray) -> None:
        fc = results[curve_idx]
        old_other = fc.points[-1] if ep_idx == 0 else fc.points[0]
        if float(np.hypot(new_pt[0] - old_other[0], new_pt[1] - old_other[1])) < max(8.0, 0.5 * fc.length_px):
            return
        fc.points[ep_idx] = new_pt.copy()
        if fc.curve_type == "line":
            if ep_idx == 0:
                fc.params["x1"] = round(float(new_pt[0]), 2)
                fc.params["y1"] = round(float(new_pt[1]), 2)
            else:
                fc.params["x2"] = round(float(new_pt[0]), 2)
                fc.params["y2"] = round(float(new_pt[1]), 2)
        elif fc.curve_type == "bezier" and "control_points" in fc.params:
            cp = fc.params["control_points"]  # type: ignore[index]
            if ep_idx == 0:
                cp[0] = [round(float(new_pt[0]), 2), round(float(new_pt[1]), 2)]  # type: ignore[index]
            else:
                cp[3] = [round(float(new_pt[0]), 2), round(float(new_pt[1]), 2)]  # type: ignore[index]
            if "beziers" in fc.params and fc.params["beziers"]:
                bz = fc.params["beziers"]  # type: ignore[index]
                if ep_idx == 0:
                    bz[0][0] = [round(float(new_pt[0]), 2), round(float(new_pt[1]), 2)]  # type: ignore[index]
                else:
                    bz[-1][3] = [round(float(new_pt[0]), 2), round(float(new_pt[1]), 2)]  # type: ignore[index]

    line_eps = []
    for li, l_item in enumerate(lines_info):
        line_eps.append({"line_idx": li, "ep": 0, "pt": l_item["p0"], "color": l_item["color"]})
        line_eps.append({"line_idx": li, "ep": -1, "pt": l_item["p1"], "color": l_item["color"]})

    n_eps = len(line_eps)
    adj = [[] for _ in range(n_eps)]
    for a in range(n_eps):
        for b in range(a + 1, n_eps):
            if line_eps[a]["line_idx"] == line_eps[b]["line_idx"]:
                continue
            if line_eps[a]["color"] != line_eps[b]["color"]:
                continue
            d = float(np.hypot(*(line_eps[a]["pt"] - line_eps[b]["pt"])))
            if d <= max_snap_px:
                adj[a].append(b)
                adj[b].append(a)

    visited_eps = [False] * n_eps
    for start in range(n_eps):
        if visited_eps[start] or not adj[start]:
            continue
        cluster = [start]
        visited_eps[start] = True
        for v in adj[start]:
            if not visited_eps[v]:
                visited_eps[v] = True
                cluster.append(v)
        if len(cluster) < 2:
            continue

        cluster_pts = np.array([line_eps[ci]["pt"] for ci in cluster])
        best_vertex = np.mean(cluster_pts, axis=0)

        for ci in cluster:
            ec = line_eps[ci]
            sl = lines_info[ec["line_idx"]]
            _update_endpoint(sl["idx"], ec["ep"], best_vertex)
            if ec["ep"] == 0:
                sl["p0"] = best_vertex.copy()
            else:
                sl["p1"] = best_vertex.copy()

    # Merge same-color bezier segments meeting at endpoints (e.g. the two halves of the wave)
    changed = True
    while changed:
        changed = False
        n_res = len(results)
        for i in range(n_res):
            c_a = results[i]
            if c_a.curve_type != "bezier" or len(c_a.points) < 2:
                continue
            p_a0 = np.array(c_a.points[0])
            p_a1 = np.array(c_a.points[-1])
            for j in range(i + 1, n_res):
                c_b = results[j]
                if c_b.curve_type != "bezier" or c_b.color != c_a.color or len(c_b.points) < 2:
                    continue
                p_b0 = np.array(c_b.points[0])
                p_b1 = np.array(c_b.points[-1])
                d_end = min(
                    np.hypot(*(p_a1 - p_b0)),
                    np.hypot(*(p_a1 - p_b1)),
                    np.hypot(*(p_a0 - p_b0)),
                    np.hypot(*(p_a0 - p_b1)),
                )
                if d_end <= 25.0 * avg_scale:
                    pts_a = np.array(c_a.points)
                    pts_b = np.array(c_b.points)
                    if np.hypot(*(pts_a[-1] - pts_b[0])) <= 25.0 * avg_scale:
                        merged_pts = np.vstack([pts_a, pts_b])
                    elif np.hypot(*(pts_a[-1] - pts_b[-1])) <= 25.0 * avg_scale:
                        merged_pts = np.vstack([pts_a, pts_b[::-1]])
                    elif np.hypot(*(pts_a[0] - pts_b[0])) <= 25.0 * avg_scale:
                        merged_pts = np.vstack([pts_b[::-1], pts_a])
                    else:
                        merged_pts = np.vstack([pts_b, pts_a])

                    bez_params, bez_err, bez_inlier, bez_pts = _fit_schneider_bezier_path(
                        merged_pts, max_err=fit_tol_orig * 1.15
                    )
                    tot_len = float(c_a.length_px + c_b.length_px)
                    c_a.points = bez_pts
                    c_a.params = bez_params
                    c_a.length_px = round(tot_len, 2)
                    c_a.fit_error_px = round(bez_err, 3)
                    c_a.residual_mm = round(bez_err / (2.58 * avg_scale), 3)
                    results.pop(j)
                    changed = True
                    break
            if changed:
                break

    return results
