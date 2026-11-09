# Urumi Vision: Thin Stroke Detection & Vectorization Workflow

This document explains the complete computer vision workflow for detecting thin, hand-drawn ink and pencil strokes on textured cardboard, zooming into regions of interest (ROI), removing corrugated noise, and exporting 1:1 millimeter physical SVGs.

---

## 1. System Architecture & Coordinate Spaces

The pipeline operates across three distinct coordinate frames:

1. **Full Bed Perspective Frame ($2000 \times 2748\text{ px}$)**:
   - Corrected planar view obtained via ArUco 4-corner homography.
   - Represents the calibrated cutting bed: $775.0 \times 1065.0\text{ mm}$ ($~2.58\text{ px/mm}$).
2. **Cropped ROI Frame ($(w_{\text{roi}}, h_{\text{roi}})$ at offset $(x_{\text{roi}}, y_{\text{roi}})$)**:
   - Tightly bounds the workpiece detected automatically (via OpenCV contours/YOLO) or manually selected with the Lasso tool.
   - Shaves cardboard edges inward by $20\text{--}26\text{ px}$ to eliminate physical edge shadows.
3. **Zoomed Working Frame ($1000\text{ px}$ Long Side)**:
   - High-resolution processing canvas ($w_{\text{work}}, h_{\text{work}}$) created via bicubic upsampling ($\approx 3.0\times\text{--}3.5\times$).
   - Stages 2 through 7 run exclusively in this supersampled space.
   - Coordinates map back to bed millimeters using:
     $$x_{\text{bed}} = x_{\text{work}} \cdot \frac{w_{\text{roi}}}{w_{\text{work}}} + x_{\text{roi}}, \quad y_{\text{bed}} = y_{\text{work}} \cdot \frac{h_{\text{roi}}}{h_{\text{work}}} + y_{\text{roi}}$$

---

## 2. Nine-Stage Computer Vision Pipeline

```
Raw Photo -> [Stage 0: ArUco Flatten & ROI Crop]
          -> [Stage 1: Capture & Quality Gate]
          -> [Stage 2: Bicubic Zoom & Flat-Fielding]
          -> [Stage 3: Multi-Scale Sato Hessian Ridge Enhancement]
          -> [Stage 4: Hysteresis Binarization & Morphology]
          -> [Stage 5: Skeletonization, Despeckling & Junction Spur Pruning]
          -> [Stage 6: Vectorization & Bed Coordinate Scaling]
          -> [Stage 7: Geometric Shape Model Selection]
          -> [Stage 8: Validation & Confidence Scoring]
          -> [Stage 9: Layered 1:1 mm SVG Export]
```

### Stage 0: ArUco Normalization & ROI Crop
- **ArUco Detection**: Finds markers 0, 1, 2, 3 on the outer frame. Applies homography matrix warping the board to a uniform flat bed.
- **ROI Detection**: Computes convex workpiece bounds. Applies an inward border margin (`roi_shave_px = 24`) to ignore high-contrast cardboard boundary transitions.
- **Lasso Tool**: Allows polygon or rectangle manual refinement before computation.

### Stage 1: Capture & Quality Gate
- **Laplacian Variance**: Evaluates image sharpness ($\sigma_{\text{Laplacian}}^2$). Rejects blurry images below configured threshold (`min_blur_score = 1.5`).
- **Brightness Gate**: Rejects extreme underexposure/overexposure ($30 < \mu_{\text{gray}} < 235$).
- **Overlay Ignore Mask**: Detects white UI stickers/tape ($L^* > 235$ in CIELAB, dilated by $15\text{ px}$) to prevent false ridge triggers.

### Stage 2: LAB Baseline Subtraction & Chroma Classification
- **LAB Channel Decomposition**: Converts the zoomed ROI image to CIE LAB ($L^*, a^*, b^*$).
- **Background Baseline Estimation**: Estimates local cardboard substrate levels across $L^*$, $a^*$, and $b^*$ using a 61 px median blur (`lab_median_ksize = 61`).
- **Residual Smoothing**: Subtracts the background baseline and smooths the residuals with a Gaussian filter ($\sigma = 1.0\text{ px}$) to compute $dL, da, db$, where neutral cardboard is approximately 0.
- **Ink Strength**: Evaluates overall ink response:
  $$s = \sqrt{da^2 + db^2} + k \cdot \max(-dL, 0), \quad k = 0.30$$
- **Soft Color Membership**: Classifies ink direction in the $(da^*, db^*)$ chroma plane into data-driven color sectors (`red` $\approx 2^\circ$, `green` $\approx 175^\circ$, `blue` $\approx 260^\circ$) to build soft color signals $[0.0, 1.0]$ for each marker color.

### Stage 3: Per-Color Sato Ridge Enhancement
- Identifies thin bright curvilinear ridges on each individual color signal.
- For working sigmas $\sigma \in \{1.5, 2.5\}$, evaluates scale-normalized 2D Hessian eigenvalues:
  $$\lambda_{\text{bright}} = \frac{\sqrt{(I_{xx} - I_{yy})^2 + 4 I_{xy}^2} - (I_{xx} + I_{yy})}{2}$$
- Normalizes each color response by its 99.5th percentile into $[0.0, 1.0]$.

### Stage 4: Per-Color Hysteresis & Overlap Arbitration
- Applies dual hysteresis thresholding (`color_hyst_high = 0.26`, `color_hyst_low = 0.14`) to each color map.
- Filters out components smaller than `min_color_component_area_px = 40` or shorter than `min_color_component_len_mm = 3.0`.
- Resolves overlapping mask pixels to the color with the larger soft signal.
- Downstream stages run per color, preserving stroke identity across crossings.

### Stage 5: Per-Color Topological Graph & Gaussian Smoothing
- Skeletonizes each color mask independently using Zhang-Suen thinning.
- Builds a topological junction/endpoint graph per color.
- Prunes dead-end spur branches shorter than `max_spur_len_mm = 1.5` (~4 px).
- Merges degree-2 chains into continuous strokes.
- Closes endpoint gaps under `max_endpoint_gap_mm = 1.0` and collinear gaps across crossings up to $32\text{ px}$.
- Closes 3-sided trapezoid flaps into 4-point closed polygons.
- Snaps T-junction endpoints within `t_junction_snap_mm = 2.0` onto intersecting strokes of the same color.
- Smooths each chain with a 1D Gaussian filter ($\sigma = 2.0\text{ px}$) while preserving endpoints to eliminate raster staircase noise before simplification.

### Stage 6: Vectorization & Bed Coordinate Scaling
- Decomposes skeleton graphs into ordered point chains.
- Simplifies pixel chains using corner-preserving RDP.
- Tags every curve with its detected color.
- Unscales coordinates back to full-bed physical millimeter dimensions.

### Stage 7: Strict Shape Model Selection
Evaluates candidate models in strict priority order, stopping at the first valid match:
1. **Circle / Ellipse**: Closed chain with low radial residual ($e_{\text{fit}} \le 2.2\text{ px}$, coverage $\ge 280^\circ$).
2. **Rectangle / Closed Polygon**: Closed chain with 3–4 corners. Snaps to `<rect>` if corner angles are within $10.5^\circ$ of $90^\circ$ and opposite sides match within $18\%$; otherwise emits closed `<polygon>` (covers trapezoids and triangles).
3. **Straight Line**: Open chain with low straight-line residual ($e_{\text{fit}} \le 2.0\text{ px}$).
4. **Zigzag Polyline**: Open chain with repeated alternating sharp direction reversals ($\ge 4$ sharp turns $\le 95^\circ$). Emitted as ONE `<polyline>`.
5. **Smooth Curve (Wave)**: Philip Schneider cubic Bézier fitting emitted as ONE `<path>`.
6. **Freehand Polyline**: RDP fallback polyline.

### Stage 8: Validation & Confidence Scoring
Computes composite quality metric:
$$\text{Confidence} = 0.30 \cdot R_{\text{inlier}} + 0.30 \cdot \min\left(\frac{\text{Coverage}^\circ}{180^\circ}, 1.0\right) + 0.20 \cdot Q_{\text{fit}} + 0.20 \cdot C_{\text{stroke}}$$
Calculates orthogonal fit residual in physical millimeters (`residual_mm`).

### Stage 9: 4-Layer 1:1 mm SVG Export & QC Overlay
- **Layered SVG**: Generates physical SVG (`viewBox="0 0 775 1065"`) with 4 distinct `<g>` layers in exact order:
  1. `<g id="bed" data-role="reference">`: Bed frame rectangle and cardboard sheet boundary polygon.
  2. `<g id="red">`: Red marker strokes (`#e53935`).
  3. `<g id="green">`: Green marker strokes (`#43a047`).
  4. `<g id="blue">`: Blue marker strokes (`#1e88e5`).
- **Full-Resolution QC Overlay**: Blends all 4 vector layers on top of the original flattened photo at 50% opacity at full ROI resolution (`10_qc_overlay.png`).

---

## 3. Parameter Reference

| Parameter | Default | Function |
|---|---|---|
| `lab_median_ksize` | `61` | Median blur kernel size (px) for LAB substrate baseline estimation. |
| `lab_residual_sigma` | `1.0` | Gaussian smoothing sigma for LAB residual channels `dL, da, db`. |
| `ink_l_weight_k` | `0.30` | Luminance darkening weight $k$ for ink strength calculation. |
| `ridge_sigmas` | `1.5, 2.5` | Gaussian scale factors for multi-scale Sato bright ridge filter. |
| `color_hyst_high` / `low` | `0.26 / 0.14` | Normalized threshold values for per-color hysteresis binarization. |
| `min_color_component_area_px`| `40` | Minimum component area (px) to reject cardboard fluting speckles. |
| `max_spur_len_mm` | `1.5` | Maximum dead-end spur branch length pruned at junctions. |
| `max_endpoint_gap_mm` | `1.0` | Maximum endpoint gap closed across broken same-color strokes. |
| `t_junction_snap_mm` | `2.0` | Snap radius for connecting T-junction endpoints to same-color strokes. |
| `chain_smooth_sigma_px` | `2.0` | Gaussian smoothing sigma applied to pixel chains before RDP. |

