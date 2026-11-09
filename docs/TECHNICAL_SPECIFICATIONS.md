# Urumi Vision: Complete Technical Specifications & Session History

This document consolidates all explanations, engineering decisions, algorithm designs, debugging analyses, and implementation details from the development of the 9-stage thin stroke detector pipeline.

---

## 1. Project Background & Requirements

### 1.1 The Problem
- **Input**: Phone photos (RGB, JPEG/PNG, 0.3 to 12 MP) of cardboard sheets containing thin, hand-drawn pen/pencil/marker strokes.
- **Challenges**:
  - Very low contrast: strokes are often only a few gray levels darker than the cardboard substrate.
  - Textured cardboard: dark flecks, fiber grain, and horizontal corrugation fluting.
  - JPEG compression artifacts: 8x8 DCT block boundaries.
  - Uneven illumination gradients across the cutting bed.
  - Partial frame cropping and boundary crossings.
  - Arbitrary geometries: lines, circular arcs, circles, ellipses, cubic Béziers, freehand polylines, and multi-branch arrows.
- **Strict Constraints**:
  - Pure classical computer vision: OpenCV, NumPy, SciPy, scikit-image (no deep learning for centerline stroke tracing).
  - Shape-agnostic through Stages 1–6 (only Stage 7 knows about shapes).
  - Never return a false positive: if detection confidence is below threshold, output status `"not_detected"`.
  - Maintain exact physical 1:1 millimeter alignment relative to the $775 \times 1065\text{ mm}$ cutting bed.

---

## 2. Core 9-Stage Pipeline Architecture

```
Stage 1: Capture & Quality Gate
    │  Computes Laplacian variance (blur score) and mean brightness.
    │  Detects high-luminance UI overlays (L > 235 dilated 15px) as ignore masks.
    ▼
Stage 2: Preprocess & Flat-Fielding
    │  Bicubic supersampling (long side scaled to ~1000-1500px).
    │  Converts to LAB, extracts L channel, fills ignore regions with median L.
    │  Flat-fields via large Gaussian blur subtraction (sigma ~ 25px).
    │  Edge-preserving bilateral filter (d=7-9, sigmaColor=12-16, sigmaSpace=5-7).
    ▼
Stage 3: Multi-Scale Sato Hessian Ridge Enhancement
    │  Computes 2D Hessian matrix across multiple scales (sigmas: 1.5, 2.5, 3.5).
    │  Extracts maximum positive curvature eigenvalue (lambda1) for dark ridges.
    │  Normalizes response to [0.0, 1.0].
    ▼
Stage 4: Hysteresis Thresholding & Morphology
    │  Dual-percentile thresholding (high: 96.5%, low: 91.0%).
    │  Connected component propagation from high-confidence seeds.
    │  Morphological closing (5x5 ellipse) to bridge ink gaps.
    ▼
Stage 5: Skeletonization, Despeckling & Spur Pruning
    │  Pre-filters isolated components (area < 220px, diagonal span < 95px).
    │  Zhang-Suen thinning to extract 1px centerlines.
    │  Junction-aware spur pruning (strips branches <= 32px ending at junctions).
    │  Perceptual endpoint linking (bridges gaps <= 25px with matching tangents).
    ▼
Stage 6: Vectorization & Simplification
    │  Graph decomposition into cyclic loops or branched trees.
    │  Traces ordered (x, y) coordinates.
    │  Simplifies with Ramer-Douglas-Peucker (epsilon ~ 2.5px).
    │  Maps working coordinates back to bed pixels.
    ▼
Stage 7: Shape Model Selection (Simplest Wins)
    │  Tries models in order: Line -> Circle -> Ellipse -> Arc -> Bezier -> Polyline.
    │  Evaluates residual error against tolerance (default 2px).
    ▼
Stage 8: Validation & Confidence Scoring
    │  Computes composite score: inliers, angular coverage, fit quality, contrast.
    │  Confidence = 0.30*inlier + 0.30*min(cov/180, 1) + 0.20*fit_qual + 0.20*contrast.
    ▼
Stage 9: Layer Classification & 1:1 mm SVG Export
    │  Cardboard-relative color voting (score/crease/thru_cut).
    │  Generates layered physical SVG matching ArUco bed frame.
```

---

## 3. Key Technical Challenges, Bugs & Solutions

### 3.1 Initial Resolution Downsampling vs. Sub-Pixel Strokes
- **Issue**: Warping the entire bed ($2000 \times 2748\text{ px}$) and resizing it down to $1000\text{ px}$ shrank small cardboard workpieces ($\approx 250 \times 330\text{ px}$) to just $110 \times 150\text{ px}$. This made thin pen strokes narrower than 1 pixel, causing corners of small triangles and squares to vanish.
- **Solution**: Implemented Region of Interest (ROI) supersampling. The workpiece is cropped first (via contour detection, YOLO, or interactive Lasso) and then zoomed $\approx 3.0\times\text{--}3.5\times$ via bicubic interpolation up to a $1000\text{ px}$ working frame. Detection runs on the zoomed crop, and all vector coordinates are scaled and offset back to the full bed.

### 3.2 High-Contrast Cardboard Edge Dominance
- **Issue**: The cutting bed background is black while the cardboard is light tan. The strong outer cardboard boundary produced huge Hessian ridge responses that skewed the 97th percentile hysteresis threshold, causing faint interior pen lines to be dropped.
- **Solution**: Added edge shaving (`roi_shave_px = 24`) and an inward perimeter mask (`border_ignore_px = 14`), completely excluding outer workpiece boundaries from the enhancement and thresholding stages.

### 3.3 Loss of Multi-Branch Features (Arrow Wings)
- **Issue**: Standard tree vectorizers trace only the single longest path (tree diameter), which dropped the second wing of hand-drawn arrows.
- **Solution**: Implemented branch decomposition in `_trace_component_chains`. After tracing the primary trunk, unvisited branch segments $\ge 30\text{ px}$ connected to the main trunk are extracted as secondary chains.

### 3.4 Corrugated Cardboard Flute Ridges (Specks)
- **Issue**: Cardboard has horizontal corrugation fluting. Zooming the ROI by $3\times$ stretched small grain variations into $25\text{--}85\text{ px}$ horizontal lines that passed hysteresis.
- **Solution**:
  - Increased hysteresis thresholds to $96.5\% / 91.0\%$.
  - Added a minimum bounding-box diagonal span filter (`min_span_px = 95`) and minimum area filter (`area >= 220 px`) in Stage 5, eliminating isolated grain marks while preserving valid shapes.

### 3.5 Side Whiskers at Stroke Junctions
- **Issue**: Horizontal cardboard flutes touching drawn shapes created short $5\text{--}30\text{ px}$ whisker-like spurs branching off triangles, squares, circles, and arrow shafts.
- **Solution**: Developed **Junction-Aware Spur Pruning** (`pruneSkeletonSpurs`). It walks inward from degree-1 endpoints along degree-2 pixels up to $32\text{ px}$. If a path encounters a branch junction (degree $\ge 3$) within $32\text{ px}$, the entire whisker branch is deleted. Branches longer than $32\text{ px}$ (like arrow wings $\approx 75\text{ px}$) remain untouched.

### 3.6 Color Classification Bias on Brown Cardboard
- **Issue**: Natural cardboard has higher red reflectance than blue ($R > G > B$). Evaluating raw RGB pixel values misclassified dark pencil and black ballpoint strokes as `red` (`score`).
- **Solution**: Implemented background-relative CIELAB chromaticity evaluation:
  $$\Delta a^* = a^* - \text{med}_{a^*}, \quad \Delta b^* = b^* - \text{med}_{b^*}$$
  - Strokes with $\sqrt{(\Delta a^*)^2 + (\Delta b^*)^2} < 14$ match the cardboard hue and are classified as `black` (`thru_cut`).
  - True red ink requires $\Delta a^* \ge +12$.
  - True green ink requires $\Delta a^* \le -10$.
  - True blue ink requires $\Delta b^* \le -10$.

### 3.8 Multi-Color Intersecting Stroke Segmentation & Classification
- **Issue**: Previously, color was discarded during stages 2–5 and inferred only after vectorization in stage 6. Strokes of different colors that touched or intersected (e.g. green trapezoid meeting a red square, or a green line crossing red ticks) were merged into a single skeleton component, causing color bleeding and fragmentation.
- **Solution**:
  - Implemented per-color segmentation in Stages 2–4. Baseline cardboard color is estimated per channel ($L^*, a^*, b^*$) via large median blur ($61\text{ px}$) and subtracted to yield residuals $dL, da, db$.
  - Ink strength $s = \sqrt{da^2 + db^2} + 0.30 \cdot \max(-dL, 0)$ is weighted by soft angular chroma membership into data-driven color signals (`red` $\approx 2^\circ$, `green` $\approx 175^\circ$, `blue` $\approx 260^\circ$).
  - Overlapping pixels between color masks are assigned to the color with the larger soft signal.
  - Skeletons and graphs are built and tracked per color through all downstream stages.

### 3.9 Skeleton Staircase Noise and Fragmentation
- **Issue**: Digital raster skeletonization produces jagged 1-px staircase noise. Running Ramer-Douglas-Peucker directly preserved excess vertices or generated wobbly strokes. Furthermore, multi-edge junctions fragmented coherent shapes into dozens of small segments.
- **Solution**:
  - Filtered pixel chains with a 1D Gaussian filter ($\sigma = 2\text{ px}$) while preserving endpoints prior to geometric simplification.
  - Built topological junction graphs per color, pruned dead-end spurs ($\le 1.5\text{ mm}$), merged chains through degree-2 nodes, bridged small endpoint gaps ($\le 1.0\text{ mm}$) and collinear crossings ($\le 32\text{ px}$), closed 3-sided trapezoid flaps, and snapped T-junction endpoints ($\le 2.0\text{ mm}$) onto strokes of the same color.

### 3.10 Strict Multi-Primitive Shape Fitting & 4-Layer Output
- **Issue**: Circles were well-fit, but squares, trapezoids, and zigzags fragmented into disconnected lines.
- **Solution**:
  - Enforced a strict evaluation hierarchy:
    1. Closed chain $\to$ Circle / Ellipse
    2. Closed chain with 3–4 corners $\to$ Snapped rectangle ($90^\circ \pm 10.5^\circ$) or closed polygon (trapezoids/triangles)
    3. Open chain with straight residual $\to$ `<line>`
    4. Repeated alternating sharp turns $\to$ Single `<polyline>` (zigzag)
    5. Smooth open curve $\to$ Single Philip Schneider cubic Bézier `<path>` (wave)
    6. Fallback $\to$ Freehand `<polyline>`
  - Generated SVG with 4 ordered `<g>` layers: `<g id="bed" data-role="reference">`, `<g id="red">`, `<g id="green">`, and `<g id="blue">`.
  - Added a full-resolution 50% opacity QC overlay blending the SVG vector output over the camera photo.

---

## 4. Performance Optimizations

1. **Fast 2D Sato Hessian Filter**: Replaced slow `skimage.filters.sato` with an OpenCV C++ implementation using `cv2.GaussianBlur` and `cv2.Sobel`, reducing enhancement time on 1 MP images from $365\text{ ms}$ to $45\text{ ms}$.
2. **OpenCV Connected-Component Hysteresis**: Replaced SciPy Python-level hysteresis labeling with `cv2.connectedComponents`, accelerating binarization from $110\text{ ms}$ to $3\text{ ms}$.
3. **Bounding-Box Skeletonization & Downsampled Baseline Estimation**: Subsampled large median blurs on high-resolution ROIs, reducing preprocessing time while maintaining sub-millimeter stroke accuracy.

---

## 5. Artifacts and Source Deliverables

- **HTML Dry Run Application**: [`pipeline_dryrun_stroke_detector.html`](file:///c:/Users/nikhil/Coding/urumi-vision-app/pipeline_dryrun_stroke_detector.html)
- **Python Module**:
  - Configuration: [`stroke_detector/config.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/config.py)
  - Processing Stages: [`stroke_detector/stages.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/stages.py)
  - Model Fitting: [`stroke_detector/fitting.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/fitting.py)
  - Orchestration Pipeline: [`stroke_detector/pipeline.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/pipeline.py)
  - Visualization & QC Overlay: [`stroke_detector/viz.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/viz.py)
  - CLI: [`stroke_detector/cli.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/cli.py)
- **Test Suites**:
  - [`tests/test_stroke_detector.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/tests/test_stroke_detector.py) (9 unit tests).
  - [`tests/verify_pipeline.py`](file:///c:/Users/nikhil/Coding/urumi-vision-app/tests/verify_pipeline.py) (full test-image acceptance verification and synthetic blue-marker verification).
- **Documentation**:
  - [`docs/CV_PIPELINE_ARCHITECTURE.md`](file:///c:/Users/nikhil/Coding/urumi-vision-app/docs/CV_PIPELINE_ARCHITECTURE.md)
  - [`docs/WORKFLOW.md`](file:///c:/Users/nikhil/Coding/urumi-vision-app/docs/WORKFLOW.md)
  - [`stroke_detector/README.md`](file:///c:/Users/nikhil/Coding/urumi-vision-app/stroke_detector/README.md)

