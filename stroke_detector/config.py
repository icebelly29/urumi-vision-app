"""Configuration dataclass and intermediate result structures."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ColorSpec:
    """Configuration for a single marker ink color sector in the CIE LAB (da*, db*) plane."""
    name: str
    center_deg: float
    half_width_deg: float
    min_chroma: float
    hex_color: str


def default_color_specs() -> List[ColorSpec]:
    """Return the default 3-color marker specifications tuned from sampled LAB strokes."""
    return [
        ColorSpec(name="red", center_deg=2.0, half_width_deg=52.0, min_chroma=3.8, hex_color="#e53935"),
        ColorSpec(name="green", center_deg=175.0, half_width_deg=50.0, min_chroma=3.5, hex_color="#43a047"),
        ColorSpec(name="blue", center_deg=268.0, half_width_deg=36.0, min_chroma=4.2, hex_color="#1e88e5"),
    ]


@dataclass
class Config:
    """Configuration parameters for all 9 stages of the stroke detection pipeline."""

    # Stage 0 (Optional): ArUco perspective flattening & SVG physical metrics
    enable_aruco_warp: bool = True
    aruco_output_width: int = 2000
    default_phys_width_mm: float = 775.0
    default_phys_height_mm: float = 1065.0
    aruco_frame_width_mm: float = 830.0
    aruco_frame_height_mm: float = 1130.0
    radial_undistort_k1: float = 0.0  # Optional radial barrel distortion coefficient
    auto_detect_roi: bool = True
    roi_bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) in bed pixels
    roi_shave_px: int = 16

    # Stage 1: Capture / quality gate
    min_blur_score: float = 1.5
    warn_blur_score: float = 35.0
    min_brightness: float = 30.0
    max_brightness: float = 235.0
    overlay_l_threshold: int = 235
    overlay_dilate_px: int = 15
    # Never let a crop boundary seed a ridge: this is deliberately wider than
    # the Sato kernel halo at the largest nominal scale.
    border_ignore_px: int = 12
    cardboard_safety_erosion_px: int = 16

    # Stage 2: Preprocess & Per-Color Segmentation (Physical mm parameters)
    target_long_side: int = 1000
    min_side_for_upscale: int = 500
    upscale_factor: float = 2.0
    flat_field_sigma: float = 25.0
    bilateral_d: int = 7
    bilateral_sigma_color: float = 12.0
    bilateral_sigma_space: float = 5.0
    channel_mode: str = "L"  # "L", "a", "b"
    lab_median_kernel_mm: float = 24.0  # Cardboard baseline median filter size in mm (~61 px at 2.58 px/mm)
    lab_median_ksize: int = 61
    lab_residual_sigma: float = 1.0
    chroma_smooth_sigma_mm: float = 1.2  # Normalized convolution smoothing sigma for chroma in mm
    ink_l_weight_k: float = 0.30
    k_chroma_sigma: float = 3.2  # Ink strength chroma noise floor factor (s > m_s + k_chroma * sigma_s)
    colors: List[ColorSpec] = field(default_factory=default_color_specs)

    # Stage 3: Stroke enhancement
    ridge_filter: str = "sato"  # "sato", "frangi", "meijering", "blackhat"
    ridge_sigmas_mm: Tuple[float, ...] = (0.5, 0.9)  # Physical Sato ridge filter scales in mm
    ridge_sigmas: Tuple[float, ...] = (1.5, 2.5)  # Nominal fallback in px
    black_ridges: bool = True
    min_raw_strokeness: float = 0.22  # Reject pure background noise floors

    # Stage 4: Hysteresis threshold & Speck rejection
    k_low_sigma: float = 4.0   # Hysteresis low threshold = m + k_low_sigma * sigma
    k_high_sigma: float = 8.0  # Hysteresis high threshold = m + k_high_sigma * sigma
    min_component_len_mm: float = 6.0       # Criteria (a): long stroke length threshold in mm
    min_short_stroke_len_mm: float = 2.5    # Criteria (b): short stroke length threshold in mm
    min_response_sigma_long: float = 6.0    # Criteria (a): mean ridge response in sigma units
    min_response_sigma_short: float = 9.0   # Criteria (b): mean ridge response in sigma units
    flute_reject_len_mm: float = 12.0       # Max length of flute streak to reject in mm
    flute_reject_angle_deg: float = 15.0    # Angle tolerance to dominant flute direction in deg
    flute_reject_max_width_ratio: float = 0.70  # Width / stroke_width threshold for flute rejection
    hysteresis_high_pct: float = 97.0
    hysteresis_low_pct: float = 90.0
    min_absolute_high: float = 0.20
    color_hyst_low: float = 0.14
    color_hyst_high: float = 0.26
    min_color_component_area_px: int = 40
    min_color_component_len_mm: float = 3.0
    morph_close_size: int = 3

    # Stage 5: Skeleton and perceptual grouping
    spur_prune_iters: int = 15
    spur_prune_mm: float = 1.5           # Spurs <= 1.5 mm pruned iteratively
    endpoint_gap_mm: float = 1.0         # Direct collinear gap close distance in mm
    geodesic_gap_mm: float = 4.0         # Minimal-cost path gap close distance in mm
    cross_color_gap_margin_mm: float = 1.5  # Cross-color gap <= other_width + 1.5 mm
    t_junction_snap_mm: float = 2.5
    chain_smooth_sigma_mm: float = 0.8   # Gaussian chain smoothing sigma in mm
    chain_smooth_sigma_px: float = 2.0
    min_color_run_mm: float = 3.0        # Sub-runs < 3 mm absorbed into dominant chain color
    endpoint_link_dist: float = 14.0
    endpoint_cos_sim: float = 0.85  # max ~31 degrees deviation for collinear gap bridging
    min_skeleton_length: int = 120
    filter_compact_blobs: bool = True
    min_elongation_ratio: float = 1.5

    # Stage 6: Vectorize
    rdp_epsilon: float = 2.5

    # Stage 7: Shape model selection
    ransac_iterations: int = 1500
    ransac_inlier_tol_px: float = 3.0
    fit_tolerance_px: float = 2.2  # at original resolution
    use_bic: bool = True
    circle_min_diag_ratio: float = 0.01
    circle_max_diag_ratio: float = 1.00

    # Stage 8: Validation and confidence
    weight_inlier_ratio: float = 0.30
    weight_coverage: float = 0.30
    weight_fit_quality: float = 0.20
    weight_stroke_contrast: float = 0.20

    # Stage 9: Output and fallback
    confidence_threshold: float = 0.40
    max_retries: int = 1
    svg_stroke_width_mm: float = 1.0


@dataclass
class QualityReport:
    """Output of Stage 1: Capture and quality gate."""
    blur_score: float
    brightness: float
    overlay_masked: bool
    rejected: bool
    warning: Optional[str] = None


@dataclass
class CaptureOutput:
    """Container for Stage 1 results."""
    image_bgr: np.ndarray
    ignore_mask: np.ndarray  # boolean mask, True = ignore
    quality: QualityReport
    warped_from_aruco: bool = False
    phys_width_mm: float = 775.0
    phys_height_mm: float = 1065.0
    aruco_markers_found: int = 0
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    full_bed_width: int = 0
    full_bed_height: int = 0
    warped_bed_bgr: Optional[np.ndarray] = None
    cardboard_polygon_bed: Optional[np.ndarray] = None
    px_per_mm: float = 2.5806
    zoom_factor: float = 1.0


@dataclass
class PreprocessOutput:
    """Container for Stage 2 results."""
    flat_field: np.ndarray  # float32 image at working resolution
    lab_channels: Tuple[np.ndarray, np.ndarray, np.ndarray]  # L, a, b uint8 at working res
    resized_bgr: np.ndarray  # BGR uint8 at working resolution
    ignore_mask_working: np.ndarray  # boolean mask at working resolution
    scale_x: float  # original_w / working_w
    scale_y: float  # original_h / working_h
    original_shape: Tuple[int, int]  # (H, W) of the cropped ROI (or full image)
    roi_offset_x: int = 0
    roi_offset_y: int = 0
    full_bed_shape: Tuple[int, int] = (0, 0)
    dL: Optional[np.ndarray] = None
    da: Optional[np.ndarray] = None
    db: Optional[np.ndarray] = None
    ink_strength: Optional[np.ndarray] = None
    chroma_angle_deg: Optional[np.ndarray] = None
    color_signals: Dict[str, np.ndarray] = field(default_factory=dict)
    phys_width_mm: float = 775.0
    phys_height_mm: float = 1065.0
    px_per_mm: float = 2.5806
    flute_angle_deg: float = 90.0
    smoothed_da: Optional[np.ndarray] = None
    smoothed_db: Optional[np.ndarray] = None
    ink_sigma: float = 1.0
    ink_median: float = 0.0


@dataclass
class EnhancementOutput:
    """Container for Stage 3 results."""
    strokeness: np.ndarray  # float32 [0..1]
    raw_max: float
    color_strokeness: Dict[str, np.ndarray] = field(default_factory=dict)
    color_raw_max: Dict[str, float] = field(default_factory=dict)
    color_noise_stats: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # (median, sigma)
    measured_stroke_width_mm: float = 1.6


@dataclass
class ThresholdOutput:
    """Container for Stage 4 results."""
    mask: np.ndarray  # boolean stroke mask
    high_val: float
    low_val: float
    color_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    kept_speck_mask: Optional[np.ndarray] = None
    rejected_speck_mask: Optional[np.ndarray] = None
    speck_stats: Dict[str, object] = field(default_factory=dict)


@dataclass
class SkeletonOutput:
    """Container for Stage 5 results."""
    raw_skeleton: np.ndarray  # boolean before pruning/filtering
    pruned_skeleton: np.ndarray  # boolean after spur pruning and linking
    skeleton: np.ndarray  # boolean final skeleton of long curves
    color_skeletons: Dict[str, np.ndarray] = field(default_factory=dict)
    color_chains: Dict[str, List[Tuple[np.ndarray, bool]]] = field(default_factory=dict)
    graph_nodes_vis: Optional[np.ndarray] = None


@dataclass
class TracedCurve:
    """A single vectorized curve from Stage 6."""
    points_working: np.ndarray  # (N, 2) float64 [x, y] at working resolution
    simplified_working: np.ndarray  # (M, 2) float64 [x, y] at working resolution
    points_orig: np.ndarray  # (N, 2) float64 [x, y] at original resolution
    simplified_orig: np.ndarray  # (M, 2) float64 [x, y] at original resolution
    is_closed: bool
    length_working: float
    length_orig: float
    color: str = "blue"
    color_hex: str = "#1e88e5"


@dataclass
class VectorizeOutput:
    """Container for Stage 6 results."""
    curves: List[TracedCurve]


@dataclass
class FittedCurve:
    """Output of Stage 7 & 8 for a single curve."""
    curve_type: str  # "line"|"circle"|"ellipse"|"rect"|"polygon"|"polyline"|"bezier"|"freehand"
    params: Dict[str, object]
    points: np.ndarray  # (K, 2) float64 at original resolution
    length_px: float
    fit_error_px: float
    inlier_ratio: float
    angular_coverage_deg: float
    mean_contrast: float
    confidence: float
    layer: str = "blue"
    color: str = "blue"
    model: str = "freehand"
    residual_mm: float = 0.0
    color_hex: str = "#1e88e5"
