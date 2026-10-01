# Urumi Vision Pipeline — Deep Dive Technical Reference

Every function is covered. Parameters, return values, and the exact reason each operation exists are explained.

---

## Module 1: `WarpEngine`

**File:** `pipeline/WarpEngine.js`  
**Purpose:** Takes a raw, skewed phone photo and outputs a geometrically corrected, top-down flat image.

---

### `constructor(outputWidth)`

```js
this.OUTPUT_WIDTH = outputWidth;   // 2000px
this.currentFrameConfig = null;    // set after warp, read by SvgGenerator
this.borderSizeMm = 0;
```

`OUTPUT_WIDTH` is the fixed pixel width of every output image. The height is calculated to match the physical aspect ratio of the frame. This means a 775mm × 1065mm frame always becomes `2000 × 2748px` regardless of the input photo resolution.

---

### `normalizePerspective(src, srcMask)`

**Parameters:**
- `src` — `cv.Mat` (RGBA). The raw photo from the phone.
- `srcMask` — `cv.Mat` or `null`. An optional binary mask (from Lasso tool) that gets warped alongside the image.

**Returns:** `{ image: cv.Mat, mask: cv.Mat|null }` or `null` on failure.

**What it does internally:**

#### 1. Grayscale + ArUco detection
```js
cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
let detector = new cv.aruco_ArucoDetector(dictionary, parameters, refineParameters);
detector.detectMarkers(gray, corners, ids);
```
ArUco detection only works on single-channel (grayscale) images. The detector uses `DICT_4X4_50` — a dictionary of 50 unique 4×4 bit patterns. Each marker has a unique integer ID. The `corners` output is a vector of 4-point sets (one per marker), and `ids` is the list of which ID each marker is.

`aruco_RefineParameters(10, 3, true)` — refines detected marker corners by searching within a 10px window, doing 3 refinement iterations. This improves sub-pixel accuracy of the corner locations.

#### 2. Center point calculation
```js
let cx = (corner.data32F[0] + corner.data32F[2] + corner.data32F[4] + corner.data32F[6]) / 4;
let cy = (corner.data32F[1] + corner.data32F[3] + corner.data32F[5] + corner.data32F[7]) / 4;
```
Each marker has 4 corners in the `data32F` float array, stored as `[x0, y0, x1, y1, x2, y2, x3, y3]`. The center is the average of all 4 corners.

#### 3. ID-based corner assignment
```js
let detectedIds = Object.keys(markerCenters).map(Number).sort((a, b) => a - b);
const tl = markerCenters[detectedIds[0]]; // lowest ID = Top-Left
const tr = markerCenters[detectedIds[1]];
const br = markerCenters[detectedIds[2]];
const bl = markerCenters[detectedIds[3]]; // highest ID = Bottom-Left
```
**Why sort by ID instead of by pixel position?** If you sort by pixel X/Y, a rotated photo can swap TL and TR. The physical frame always prints IDs in a fixed clockwise order (e.g. 12=TL, 13=TR, 14=BR, 15=BL), so sorting by ID number is rotation-invariant.

#### 4. Physical dimension lookup
```js
cropPhysW = 775.0;   // mm — inner drawing area width
cropPhysH = 1065.0;  // mm — inner drawing area height
arucoMarginX = (physWidth - cropPhysW) / 2.0;  // = (830 - 775) / 2 = 27.5mm
arucoMarginY = (physHeight - cropPhysH) / 2.0; // = (1130 - 1065) / 2 = 32.5mm
```
The ArUco markers are printed at the corners of the full 830×1130mm frame. But the actual cutting area is 775×1065mm (the inner region). `arucoMarginX/Y` is the physical offset from the marker center to the crop boundary.

#### 5. Destination point calculation
```js
const dotsPerMm = this.OUTPUT_WIDTH / cropPhysW;  // 2000 / 775 = 2.581 px/mm
const outW = 2000;
const outH = Math.round(dotsPerMm * cropPhysH);   // 2000/775 * 1065 = 2748px

const marginXPx = arucoMarginX * dotsPerMm;  // 27.5mm * 2.581 = 71px
const marginYPx = arucoMarginY * dotsPerMm;  // 32.5mm * 2.581 = 83.9px
```
The destination points push the ArUco centers **outside** the output canvas by `marginXPx/marginYPx`. This means the crop boundary maps exactly to the canvas edges:
```js
dstTri = [
    (-marginXPx, -marginYPx),          // TL marker maps to above-left of canvas
    (outW + marginXPx, -marginYPx),    // TR marker maps to above-right
    (outW + marginXPx, outH + marginYPx), // BR
    (-marginXPx, outH + marginYPx)     // BL
]
```

#### 6. Perspective transform
```js
let M = cv.getPerspectiveTransform(srcTri, dstTri);
cv.warpPerspective(src, warped, M, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255,255,255,255));
```
`getPerspectiveTransform` computes a 3×3 homography matrix `M` from the 4 source-to-destination point pairs. `warpPerspective` applies that matrix to every pixel using bilinear interpolation (`INTER_LINEAR`). Pixels outside the source image are filled with white (`255,255,255,255`).

If a mask was provided, it gets warped with `INTER_NEAREST` (no blending — binary values must stay 0 or 255).

---

## Module 2: `InkExtractor`

**File:** `pipeline/InkExtractor.js`  
**Purpose:** Finds all ink strokes on the warped image, removes noise, converts strokes to polylines, and classifies each polyline by color (blue/red/green/black).

---

### `extractColorPaths(img, frameConfig, borderSizeMm, warpedMask)`

**Parameters:**
- `img` — `cv.Mat` (RGBA). The warped flat image from WarpEngine.
- `frameConfig` — `{ cropPhysW, cropPhysH }`. Physical dimensions, used by SvgGenerator later.
- `borderSizeMm` — currently unused (reserved).
- `warpedMask` — `cv.Mat` (GRAY, binary) or `null`. The Lasso selection mask.

**Returns:** `{ thru_cut: { paths, color }, score: { paths, color }, crease: { paths, color } }`

---

#### Sub-step A: Color space preparation
```js
cv.cvtColor(img, hsv, cv.COLOR_RGBA2RGB);
cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);
```
Two representations are maintained in parallel:
- **HSV** — for color range detection (hue is invariant to brightness).
- **Gray** — for adaptive thresholding (finds locally dark pixels regardless of color).

---

#### Sub-step B: Adaptive threshold
```js
cv.bilateralFilter(gray, blurred, 9, 75, 75);
cv.adaptiveThreshold(blurred, adaptiveMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 10);
```
**`bilateralFilter(gray, blurred, 9, 75, 75)`**
- Diameter: 9px neighborhood
- `sigmaColor=75` — pixels within 75 gray levels of center are averaged
- `sigmaSpace=75` — pixels within 75px spatial distance are averaged

This is **edge-preserving smoothing**. Normal Gaussian blur also blurs ink edges, making them harder to threshold. Bilateral filter keeps ink edges sharp while smoothing paper texture noise.

**`adaptiveThreshold(..., 51, 10)`**
- Block size: 51px — the local neighborhood to compute the threshold
- C=10 — a pixel must be 10 gray levels darker than its 51px neighborhood average to be considered ink

This adapts to uneven lighting. A shadow on the paper raises the local average; the ink still gets detected because it is darker than its surroundings, not just dark in absolute terms.

---

#### Sub-step C: Absolute darkness gate
```js
cv.threshold(gray, darkGate, 150, 255, cv.THRESH_BINARY_INV);
cv.bitwise_and(adaptiveMask, darkGate, darkInk);
```
`THRESH_BINARY_INV`: pixels with gray < 150 → 255, others → 0.

**Why combine both?** Adaptive threshold alone picks up fold lines and cardboard texture (locally dark but not actually ink). The 150 gate alone misses ink under shadows (dark, but not dark enough absolutely). The intersection (`AND`) requires a pixel to be BOTH locally dark AND absolutely dark — this is specific to actual ink.

---

#### Sub-step D: HSV color masks
```js
// Red (wraps around in HSV — needs two ranges)
cv.inRange(hsv, Scalar(0,60,40), Scalar(12,255,255), redMask1);   // lower red
cv.inRange(hsv, Scalar(168,60,40), Scalar(179,255,255), redMask2); // upper red

// Green
cv.inRange(hsv, Scalar(36,60,40), Scalar(90,255,255), greenMask);

// Blue
cv.inRange(hsv, Scalar(100,80,40), Scalar(140,255,255), blueMask);
```
**Why two ranges for red?** In HSV, hue is a 0–179 circle. Red sits at both hue=0 and hue=179 — it wraps around. You need two separate `inRange` calls that are then OR'd together.

**`S >= 60, V >= 40`** — minimum saturation and brightness. This prevents gray cardboard (low saturation) and very dark black ink (low value) from matching the colored ranges.

The `coloredInkMask` is the OR of all three color ranges. This catches ink that is too light or thin to pass the darkness gate alone (e.g. a faint green marker).

---

#### Sub-step E: Combine and clean
```js
cv.bitwise_or(darkInk, coloredInkMask, allStrokes);
let closeK = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
cv.morphologyEx(allStrokes, allStrokes, cv.MORPH_CLOSE, closeK);
```
**Morphological Close = Dilation followed by Erosion**
- Dilation bridges thin gaps and connects wobbly segments.
- Erosion shrinks the boundaries back to their original size.
- Net effect: small gaps inside strokes are filled without eroding thin ink strokes.

**Border exclusion:** 4 `cv.rectangle` calls zero out a thin strip around all 4 edges. Warping always creates artifacts at the periphery because bilinear interpolation pulls in border pixels.

---

### `_processMaskToPaths(imgWidth, imgHeight, rawMask)`

**Purpose:** Converts the binary ink mask into a list of geometric shapes (circles, polygons, polylines).

#### Connected component filtering
```js
let numLabels = cv.connectedComponentsWithStats(preClosedStrokes, labels, stats, centroids, 8, cv.CV_32S);
let areaFloor = Math.max(100, Math.floor(0.00005 * imgWidth * imgHeight));
let spanFloor = Math.max(20, Math.floor(0.01 * Math.min(imgWidth, imgHeight)));
```
`connectedComponentsWithStats` labels each separate blob with a unique integer. It also outputs `stats` — an array where each row is `[x, y, width, height, area]` for that blob.

**`areaFloor`** — at 2000×2748px, this = `Math.max(100, 275)` = 275px. Any blob smaller than 275px² is noise.  
**`spanFloor`** — at 2000px wide, this = 20px. A blob narrower than 20px in both directions is noise. This exists to catch thin long lines that have small area but are real strokes.

The system skips geometric shape detection (which would ruin wobbly hand-drawn circles) and feeds the entire mask directly into the skeletonizer.

### `_vectorizeAndSkeletonize(maskMat)`

```js
let boolArray = new Array(maskMat.cols * maskMat.rows);
for (let i = 0; i < data.length; i++) boolArray[i] = data[i] > 128 ? 1 : 0;
return TraceSkeleton.fromBoolArray(boolArray, maskMat.cols, maskMat.rows).polylines;
```
**TraceSkeleton** is a fast topology-preserving thinning algorithm. It reduces every blob to its 1-pixel-wide medial axis (skeleton), then traces that skeleton into a list of polylines. Each polyline is an array of `[x, y]` points.

Without this step, a 5mm thick ink stroke would vectorize as a filled rectangle, not a centerline path.

---

### `_simplifyPath(path, epsilon)` — Douglas-Peucker

```js
let d = this._perpendicularDistance(path[i], path[0], path[end]);
if (dmax > epsilon) { /* recurse */ }
return [path[0], path[end]];
```
**Algorithm:**
1. Draw a line from the first to the last point.
2. Find the point in between with the greatest perpendicular distance from that line.
3. If that distance > epsilon, keep that point and recurse on both halves.
4. If all distances < epsilon, discard all middle points — they fit on a straight line.

Epsilon values used:
- `1.5px` for small paths and color-classified segments
- `3.0px` for large paths (arrows, long lines)

A skeletonized line might have 500 points. After D-P with epsilon=3, it becomes 2–10 points.

---

### `_smoothPath(path, iterations)` — Moving Average

```js
s.push([cur[i-1][0] * 0.25 + cur[i][0] * 0.5 + cur[i+1][0] * 0.25, ...]);
```
Weighted moving average: each point is replaced by 25% of the previous point + 50% of itself + 25% of the next point. Applied only to small paths (diagonal < 80px) to reduce jitter. Not applied to large paths because it rounds sharp corners like arrowheads.

---

### Color Classification Loop

For each path, up to 30 sample points are taken at equal arc-length intervals. At each point:

#### Darkest neighbor search
```js
for (let wy = -1; wy <= 1; wy++) {
    for (let wx = -1; wx <= 1; wx++) {
        let cg = gray.data[sy * gray.cols + sx];
        if (cg < bestGv) { bestGv = cg; /* store this pixel's RGB */ }
    }
}
```
Reads a 3×3 neighborhood and picks the darkest pixel. This finds the actual ink pixel at the center of the skeleton, not a half-transparent edge pixel.

#### High-Saturation Fast Path
```js
if (s_val > 80 && v_val > 40) {
    if (h_val <= 10 || h_val >= 170) { colorVotes.red++; continue; }
    // ...
}
```
If the pixel is vibrant (`Saturation > 80`, `Value > 40`), the hue directly determines the color (Red: <10 or >170, Green: 35–85, Blue: 95–135).

#### Von Kries white balance fallback
```js
let bgAvg = (bgR + bgG + bgB) / 3;
let adjR = bestR * (bgAvg / Math.max(1, bgR));
```
For faint or dark pixels, a background pixel is found by scanning around the ink point. That background color is scaled to neutral gray by finding `bgAvg = (R+G+B)/3` and multiplying each channel by `bgAvg / bgChannel`. The ink pixel gets the same multipliers applied. This removes the warm color cast of cardboard from the ink's perceived color.

#### Whole-path Classification
Instead of fracturing a single stroke into multiple colors based on a sliding window, the system determines the dominant color for the **entire path** using majority vote across all sampled points. This ensures continuous strokes are never broken into pieces.

---

## Module 3: `SvgGenerator`

**File:** `pipeline/SvgGenerator.js`  
**Purpose:** Converts pixel-coordinate paths into a millimeter-unit SVG file.

---

### `generateLayeredSvg(layersData, width, height, frameConfig)`

**Parameters:**
- `layersData` — the classified path data from InkExtractor.
- `width`, `height` — pixel dimensions of the warped image (e.g. 2000×2748).
- `frameConfig` — `{ cropPhysW, cropPhysH }` — physical dimensions in mm.

**Returns:** `{ svg: string, meta: { dots_per_mm, physical_width, physical_height } }`

#### Coordinate conversion
```js
const dotsPerMm = width / physWidth;   // 2000 / 775 = 2.581 px/mm
const toMm = (px) => (px / dotsPerMm);
```
Every path coordinate is divided by `dotsPerMm` to go from pixels to mm. The SVG `viewBox` is `"0 0 775 1065"` — one SVG unit = one millimeter.

This is critical because the downstream CNC/laser machine reads the SVG and moves in real physical distances. If coordinates were in pixels, the machine would move the wrong distance.

#### Layer structure
```xml
<g id="layer_bed_frame"> <!-- magenta boundary rectangle --> </g>
<g id="thru_cut">        <!-- blue paths --> </g>
<g id="score">           <!-- red paths --> </g>
<g id="crease">          <!-- green paths --> </g>
```
Each `<g>` has a unique `id` so post-processing software can select and act on specific cut operations independently.

#### Path data format
```js
const d = `M ${shape.points.map(pt => `${toMmX(pt[0])},${toMmY(pt[1])}`).join(' L ')}`;
```
Straight-line segments (`L`) only — no cubic Bezier curves. This is intentional: CNC G-code generators expect linear segments. `stroke-linejoin: round` and `stroke-linecap: round` in CSS handle the visual smoothness at corners.

---

## Module 4: `Communicator`

**File:** `Communication.js`  
**Purpose:** Manages the WebRTC peer-to-peer data channel between the mobile app and the desktop receiver.

---

### `_getPeerIdFromUrl()`
```js
const urlParams = new URLSearchParams(window.location.search);
return urlParams.get('peerId');
```
Reads `?peerId=main-ui-test-77777` from the URL. The QR code that the desktop generates encodes this URL. When the phone opens it, the mobile app knows exactly which desktop peer to connect to.

### `init()`
```js
this.peer = new Peer();  // random auto-assigned ID for the mobile
this.peer.on('open', (id) => { this._connectToMainUi(); });
```
Creates a PeerJS instance with a random ID (the mobile doesn't need a fixed ID — only the desktop does, since the phone always initiates the connection).

### `_connectToMainUi()`
```js
this.conn = this.peer.connect(this.mainUiPeerId, { reliable: true });
this.conn.on('open', () => {
    this.conn.send({ type: 'MOBILE_APP_READY' });
});
```
`reliable: true` uses SCTP under the hood — the same protocol as TCP. This guarantees that the SVG payload (which can be several MB) arrives in order and without data loss.

### `sendPayload(svgContent, imageUrl, meta)`
```js
this.conn.send({
    type: 'PROCESSING_COMPLETE',
    payload: { svg, image, dots_per_mm, physical_width, physical_height }
});
```
The entire payload is sent as a single JSON object over the data channel. PeerJS handles chunking if the payload exceeds the WebRTC MTU (usually ~64KB per chunk).

---

## Module 5: `ImageProcessor` (orchestrator)

**File:** `ImageProcessor.js` (local) and `svg-skeletonization/src/ImageProcessor.js` (NPM package, runs in a Web Worker)

The local file is not used directly by `app.js`. `app.js` uses `new SvgSkeletonization.ImageProcessor()` — the compiled NPM package version which offloads all OpenCV work to a **Web Worker thread** to prevent blocking the UI.

### `flatten(imageElement)`
Runs only `WarpEngine.normalizePerspective`. Stops before InkExtractor. Returns a canvas. Used to show the user the flattened image in the Lasso modal.

### `processFlattened(flattenedCanvas, maskCanvas)`
Skips perspective warp (already done). Reads the mask canvas, runs InkExtractor and SvgGenerator. Used after the user confirms their Lasso selection.

### `process(imageElement, maskCanvas)`
The full one-shot pipeline: warp + extract + SVG. Used when Magic Pen is off.

---

## Data Flow Summary

```
Phone Camera Photo (RGBA, any resolution)
    │
    ▼
WarpEngine.normalizePerspective()
    │   cv.cvtColor → gray
    │   aruco_ArucoDetector.detectMarkers → 4 corner centers
    │   getPerspectiveTransform → 3×3 homography matrix M
    │   warpPerspective → flat RGBA image
    ▼
warpedMat (RGBA, 2000 × ~2748px)
    │
    ▼
InkExtractor.extractColorPaths()
    │   bilateralFilter → blur without edge destruction
    │   adaptiveThreshold → local darkness map
    │   threshold(120) → absolute darkness gate
    │   inRange(HSV) × 3 → colored ink masks
    │   bitwise_or → allStrokes
    │   morphologyEx(OPEN) → remove specks
    │   connectedComponentsWithStats → filter small blobs
    │   convexHull + approxPolyDP → detect perfect circles/polygons
    │   TraceSkeleton.fromBoolArray → 1px center lines → polylines
    │   Douglas-Peucker → simplify point count
    │   Von Kries white balance + sliding window vote → color label
    ▼
layersData { thru_cut: [paths], score: [paths], crease: [paths] }
    │
    ▼
SvgGenerator.generateLayeredSvg()
    │   px ÷ dotsPerMm → mm coordinates
    │   <g id="thru_cut"> blue paths
    │   <g id="score"> red paths
    │   <g id="crease"> green paths
    ▼
{ svg: string, image: dataURL, meta: { dots_per_mm, physical_width, physical_height } }
    │
    ▼
Communicator.sendPayload() → WebRTC data channel → Desktop Receiver
```
