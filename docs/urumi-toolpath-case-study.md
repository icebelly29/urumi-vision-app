# Technical Case Study: Urumi Vision Computer Vision Pipeline

## Executive Summary and Architecture

The Urumi vision pipeline converts physical hand-drawn sketches inside an ArUco calibration frame into scale-accurate, closed-loop SVG toolpaths. The pipeline executes completely in client-side web browsers. It outputs vector data ready for CNC routers, laser cutters, and vinyl blade cutters without a server backend.

### Core Specifications
- **Target Medium:** Cardboard substrate and white paper sheets.
- **Dimensional Fidelity:** Scale-accurate mapping ($1\text{ pixel} = X\text{ mm}$) calculated from known ArUco frame geometry.
- **Output:** Closed-loop, topologically continuous SVG paths (`thru_cut`, `score`, `crease`).
- **Runtime Environment:** Client-side WebAssembly and HTML5 Canvas.

### Technology Stack
- **OpenCV.js (WebAssembly):** Image processing, matrix calculations, thresholding, and contour extraction.
- **HTML5 Canvas:** High-resolution image capture and pixel manipulation.
- **JavaScript (ES6+) / React:** Pipeline orchestration and user interface controls.

### Vision Pipeline Flow

```
+---------------------------+
| ArUco Calibration Frame   |
+-------------+-------------+
              |
              v
+---------------------------+
| Perspective Warp (1:1 mm) |
+-------------+-------------+
              |
              v
+---------------------------+
| Bilateral Filter Denoise  |
+-------------+-------------+
              |
              v
+---------------------------+
| Adaptive Binarization     |
+-------------+-------------+
              |
              v
+---------------------------+
| Morphological Closing     |
+-------------+-------------+
              |
              v
+---------------------------+
| Contour Extraction        |
+-------------+-------------+
              |
              v
+---------------------------+
| Douglas-Peucker Smoothing |
+-------------+-------------+
              |
              v
+---------------------------+
| Closed-Loop SVG Toolpath  |
+---------------------------+
```

1. **ArUco Perspective Warp:** Detects four ArUco markers to compute a homography transformation matrix. Warps the drawing area into an orthorectified image with a defined millimeter-per-pixel ratio.
2. **Bilateral Denoising:** Smooths flat substrate textures while preserving sharp ink stroke boundaries.
3. **Adaptive Binarization:** Segments ink strokes from varying substrate backgrounds using local neighborhood statistics.
4. **Morphological Closing:** Bridges micro-discontinuities caused by pen velocity variations or uneven surface fibers.
5. **Contour Extraction:** Executes `cv.findContours` to retrieve outer stroke boundaries and interior cutout shapes.
6. **Douglas-Peucker Vectorization:** Reduces dense pixel chains into simplified, clean SVG toolpath vectors.

---

## The Failure Mode: White Paper vs. Cardboard

### Initial Implementation: Global Otsu Thresholding
The initial prototype used global Otsu thresholding (`cv.threshold` with `cv.THRESH_OTSU`). Otsu calculates a single global threshold value that minimizes intra-class intensity variance across the full image.

### Substrate Comparison

| Criterion | White Drawing Paper | Brown Corrugated Cardboard |
| :--- | :--- | :--- |
| **Substrate Color** | Clean white ($L^* \approx 90\text{--}98$) | Medium brown ($L^* \approx 45\text{--}60$) |
| **Surface Consistency** | Smooth, uniform bleached fibers | Wood pulp flakes, grain, ridges |
| **Histogram Distribution** | Bimodal (distinct foreground/background peaks) | Unimodal or overlapping broad distributions |
| **Lighting Sensitivity** | Resilient to minor shadow gradients | Highly sensitive to ambient shadow falls |
| **Otsu Segment Performance** | Clean, uninterrupted stroke segmentation | Catastrophic failure across wood grain |

### Mechanism of Failure
1. **Low Contrast Range:** Brown wood pulp absorbs more light. The intensity delta between ballpoint ink and dark pulp fibers is small.
2. **Surface Grain and Micro-Texture:** Corrugated cardboard contains uneven pulp ridges. Global Otsu classifies dark fiber ridges and ambient falloff shadows as foreground ink strokes.
3. **Uneven Lighting:** Variations in ambient room lighting shift local background levels below the global threshold.

### Failure Symptoms
- **Disconnected Micro-Contours:** Hundreds of isolated false-positive contours appear along cardboard flutes and pulp grain.
- **Ragged Stroke Edges:** Legitimate pen strokes degrade into fragmented, jagged paths.
- **Unclosed Toolpaths:** Toolpaths contain open loops and broken lines that prevent toolpath generation for CNC cutting tools.

---

## Engineering Fix and Implementation

### The Corrected Pipeline
To resolve substrate grain artifacts, the pipeline uses a three-stage segmentation strategy:

1. **Bilateral Filtering (`cv.bilateralFilter`):** Replaces basic Gaussian blur. Gaussian blur softens crisp line edges. Bilateral filtering smooths small wood-grain textures while preserving sharp step edges of ink lines.
2. **Local Adaptive Thresholding (`cv.adaptiveThreshold`):** Computes individual threshold values for each local pixel window ($41 \times 41$). This neutralizes ambient lighting gradients and cardboard tonal variations.
3. **Morphological Closing (`cv.morphologyEx` with `cv.MORPH_CLOSE`):** Applies dilation followed by erosion with an elliptical structuring element ($3 \times 3$). This bridges small ink gaps without merging adjacent parallel markings.

### Implementation Code

```javascript
/**
 * Processes warped substrate image into clean binary stroke mask.
 * Mitigates cardboard pulp noise while preserving continuous ink lines.
 * 
 * @param {cv.Mat} warpedRgba - Orthorectified input image (RGBA).
 * @returns {cv.Mat} binaryMask - Clean binary mask (CV_8UC1).
 */
function extractCleanInkMask(warpedRgba) {
  let gray = new cv.Mat();
  let blurred = new cv.Mat();
  let adaptiveMask = new cv.Mat();
  let darkGate = new cv.Mat();
  let binaryMask = new cv.Mat();

  try {
    // Convert source to single-channel 8-bit grayscale
    cv.cvtColor(warpedRgba, gray, cv.COLOR_RGBA2GRAY);

    // 1. Bilateral filter: edge-preserving smoothing
    // Diameter = 7, SigmaColor = 12, SigmaSpace = 5
    cv.bilateralFilter(gray, blurred, 7, 12, 5);

    // 2. Local adaptive binarization (inverted: ink = 255, substrate = 0)
    // Uses adaptive mean over 41x41 window with constant offset C = 4
    cv.adaptiveThreshold(
      blurred,
      adaptiveMask,
      255,
      cv.ADAPTIVE_THRESH_MEAN_C,
      cv.THRESH_BINARY_INV,
      41,
      4
    );

    // 3. Absolute darkness gate to filter faint cardboard ridges
    cv.threshold(gray, darkGate, 175, 255, cv.THRESH_BINARY_INV);
    cv.bitwise_and(adaptiveMask, darkGate, binaryMask);

    // 4. Morphological closing to seal small ink gaps
    let kernel = cv.getStructuringElement(
      cv.MORPH_ELLIPSE,
      new cv.Size(3, 3)
    );
    cv.morphologyEx(binaryMask, binaryMask, cv.MORPH_CLOSE, kernel);
    kernel.delete();

    return binaryMask.clone();
  } finally {
    // Release WebAssembly memory buffers
    gray.delete();
    blurred.delete();
    adaptiveMask.delete();
    darkGate.delete();
    binaryMask.delete();
  }
}
```

---

## Verification and Proof

### 1. Programmatic Verification
Continuous validation checks evaluate all generated vector paths before toolpath export:

- **Loop Closure Assertion:** Every path must form a closed loop. The distance between start and end vertices must satisfy:
  $$\Delta d = \sqrt{(x_{\text{end}} - x_{\text{start}})^2 + (y_{\text{end}} - y_{\text{start}})^2} < 0.5\text{ mm}$$
- **Minimum Enclosed Area Check:** Filters out noise paths with an area below $3.0\text{ mm}^2$.
- **Douglas-Peucker Epsilon Calibration:** Optimizes point counts with $\epsilon = 0.75\text{ mm}$. This reduces path complexity while maintaining dimensional fidelity.

### 2. Physical Dry-Run Verification
- **Target Hardware:** Vinyl blade plotters and CNC flatbed cutters.
- **Zero-Power Dry-Run Procedure:** The machine traces paths above the substrate with the tool disengaged.
- **Test Criteria:** 
  - The tool follows continuous cutting loops without stop-start jumps.
  - No spindle hesitations occur from micro-segmentation.
  - The vector paths align with the original pen markings within $\pm 0.5\text{ mm}$ tolerance.

