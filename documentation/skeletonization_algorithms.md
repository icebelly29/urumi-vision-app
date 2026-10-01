# SVG Skeletonization — Algorithm Flowchart

This document shows the algorithms and data flow inside `svg-skeletonization/src`.

---

## File Inventory

| File | Role | Lines |
|---|---|---|
| `index.js` | Public entry point — re-exports all modules | 6 |
| `ImageProcessor.js` | Main-thread proxy. Sends `ImageData` to the Worker, returns SVG + image. | 130 |
| `pipeline.worker.js` | Web Worker. Loads OpenCV, runs the three core engines in sequence. | 159 |
| `WarpEngine.js` | ArUco marker detection + perspective warp to a flat, scaled canvas. | 156 |
| `InkExtractor.js` | Ink mask extraction, geometric shape detection, skeletonization, color classification, and path fracturing. | 467 |
| `trace_skeleton.js` | Minified TraceSkeleton library: Zhang-Suen thinning + recursive divide-and-conquer polyline tracing. | 4 (minified) |
| `SvgGenerator.js` | Converts pixel-coordinate paths into a layered SVG with mm-scale coordinates. | 62 |

---

## 1. High-Level Pipeline

This chart shows the complete data flow from a camera image to the final SVG string.

```mermaid
flowchart TD
    A["User captures photo<br/>(HTML Image / Canvas)"] --> B["ImageProcessor.js<br/>Main-Thread Proxy"]
    B -->|"_extractImageData()<br/>Draws element onto a hidden canvas,<br/>reads back raw RGBA pixels"| C["ImageData (RGBA pixels)"]
    C -->|"postMessage() via Workaer"| D["pipeline.worker.js<br/>Web Worker"]
    
    D --> E{"Action?"}
    
    E -->|"flatten"| F["WarpEngine.normalizePerspective()<br/>Detects 4 ArUco markers, computes<br/>a homography, warps the photo flat"]
    F --> F1["Return warped ImageData"]
    
    E -->|"process"| G["WarpEngine.normalizePerspective()<br/>Detects ArUco markers, warps photo flat"]
    G --> H["InkExtractor.extractColorPaths()<br/>Builds ink mask, detects shapes,<br/>skeletonizes strokes, classifies color"]
    H --> I["SvgGenerator.generateLayeredSvg()<br/>Converts pixel paths to mm-scale<br/>layered SVG string"]
    I --> J["Return SVG + meta + ImageData"]
    
    E -->|"processFlattened"| H2["InkExtractor.extractColorPaths()<br/>Builds ink mask, detects shapes,<br/>skeletonizes strokes, classifies color"]
    H2 --> I2["SvgGenerator.generateLayeredSvg()<br/>Converts pixel paths to mm-scale<br/>layered SVG string"]
    I2 --> J2["Return SVG + meta + ImageData"]
    
    J --> K["ImageProcessor receives result"]
    J2 --> K
    F1 --> K
    K --> L["Caller gets {svg, meta, image}"]

    style A fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style D fill:#1e293b,color:#e2e8f0,stroke:#ef4444
    style L fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

> **Important:** The `process` action runs the full pipeline (warp + extract + SVG). The `flatten` action runs only the warp step. The `processFlattened` action skips the warp and runs extract + SVG on a pre-warped image.

---

## 2. WarpEngine — Perspective Normalization

**Purpose:** Detect 4 ArUco markers in the photo. Apply a perspective transform to produce a flat, axis-aligned image at a known physical scale.

```mermaid
flowchart TD
    W1["Input: RGBA Mat (camera photo)"] --> W2["Convert to grayscale"]
    W2 --> W3["ArUco detector<br/>DICT_4X4_50"]
    W3 --> W4{"4 markers found?"}
    W4 -->|"No"| W5["Return null<br/>(abort pipeline)"]
    W4 -->|"Yes"| W6["Compute center of each marker"]
    
    W6 --> W7["Sort markers by ArUco ID<br/>(lowest = TL, highest = BL)"]
    W7 --> W8{"ID range?"}
    
    W8 -->|"IDs 12-15<br/>Custom Frame"| W9["Read w,h from URL params<br/>Default: 830x1130mm<br/>Crop: 775x1065mm"]
    W8 -->|"IDs 0-3"| W10["Small: 150x230mm<br/>Margin: 11mm"]
    W8 -->|"IDs 4-7"| W11["Medium: 210x290mm<br/>Margin: 17mm"]
    W8 -->|"IDs 8-11"| W12["Large: 270x350mm<br/>Margin: 16mm"]
    
    W9 --> W13["Compute cropPhysW, cropPhysH<br/>and pixel margins"]
    W10 --> W13
    W11 --> W13
    W12 --> W13
    
    W13 --> W14["dotsPerMm = OUTPUT_WIDTH / cropPhysW<br/>outH = dotsPerMm * cropPhysH"]
    W14 --> W15["Build src quad (marker centers)<br/>Build dst quad (output corners + margins)"]
    W15 --> W16["cv.getPerspectiveTransform<br/>Takes the 4 source points and 4 destination<br/>points, solves an 8-parameter equation to<br/>produce a 3x3 transformation matrix M"]
    W16 --> W17["cv.warpPerspective<br/>Applies matrix M to every pixel in the image.<br/>Remaps each pixel to its new position using<br/>bilinear interpolation. Fills empty borders white."]
    W17 --> W18["Store frameConfig<br/>{cropPhysW, cropPhysH}"]
    W18 --> W19["Return {image: warped Mat,<br/>mask: warped mask or null}"]

    style W1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style W19 fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

### Key Algorithm Details

| Concept | Method |
|---|---|
| **Marker detection** | OpenCV `aruco_ArucoDetector` with `DICT_4X4_50` dictionary |
| **Corner assignment** | Sort detected IDs numerically — the frame generator always assigns IDs clockwise from top-left |
| **Perspective transform** | 4-point homography via `getPerspectiveTransform` then `warpPerspective` |
| **Output scale** | Fixed width of 2000 px. Height is calculated to maintain the physical aspect ratio. |
| **Mask warping** | If a lasso mask exists, it is warped with `INTER_NEAREST` to maintain sharp binary edges |

---

## 3. InkExtractor — Core Extraction Pipeline

This is the largest and most complex module. It has two major stages:

1. **Mask Generation** — find all ink pixels in the warped image.
2. **Path Extraction + Color Classification** — convert the mask into classified vector paths.

### 3A. Ink Mask Generation (Steps 1-4)

```mermaid
flowchart TD
    I1["Input: warped RGBA Mat"] --> I2["Convert to HSV and Grayscale"]
    
    I2 --> I3["STEP 1: Adaptive Threshold<br/>bilateralFilter: blurs the image but keeps<br/>sharp edges intact by only averaging pixels<br/>that are similar in brightness. d=9 sigma=75.<br/>adaptiveThreshold: for each pixel, computes<br/>the mean brightness in a 51x51 block around it.<br/>If the pixel is 10+ darker than that mean,<br/>mark it as ink. This adapts to local lighting."]
    
    I2 --> I4["STEP 2: Absolute Darkness Gate<br/>cv.threshold: compares every pixel to a fixed<br/>value of 120. If brightness is below 120, mark<br/>as ink. If above, mark as background.<br/>This hard cutoff rejects light-colored paper."]
    
    I3 --> I5["AND: adaptiveMask AND darkGate<br/>= darkInk mask"]
    I4 --> I5
    
    I2 --> I6["STEP 3: Colored Ink Masks via HSV ranges<br/>cv.inRange: for each pixel, checks if the<br/>Hue, Saturation, and Value all fall inside<br/>a given min-max box. Outputs a binary mask<br/>where 255 = inside the range, 0 = outside."]
    I6 --> I6a["Red: H 0-12 or H 168-179, S 60+, V 40+<br/>Two ranges because red wraps around<br/>the 0/180 boundary on the hue wheel"]
    I6 --> I6b["Green: H 36-90, S 60+, V 40+<br/>Covers yellow-green through cyan-green"]
    I6 --> I6c["Blue: H 100-140, S 80+, V 40+<br/>Higher saturation floor than others<br/>to avoid false positives from shadows"]
    I6a --> I7["cv.bitwise_or: combines all color masks<br/>by ORing them pixel-by-pixel into one mask.<br/>Any pixel that matches ANY color = ink."]
    I6b --> I7
    I6c --> I7
    
    I5 --> I8["STEP 4: OR darkInk + coloredInkMask<br/>= allStrokes"]
    I7 --> I8
    
    I8 --> I9["cv.morphologyEx MORPH_OPEN, 3x3 ellipse<br/>First erodes: shrinks all white regions by 1px,<br/>which kills tiny 1-2px noise specks entirely.<br/>Then dilates: grows everything back by 1px,<br/>restoring real ink strokes to original size."]
    I9 --> I10{"Lasso mask exists?"}
    I10 -->|"Yes"| I11["AND allStrokes with lasso mask"]
    I10 -->|"No"| I12["Skip"]
    I11 --> I13["STEP 4b: Border Exclusion<br/>Erase 1 percent border strip on all 4 edges<br/>minimum 8px"]
    I12 --> I13

    style I1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style I13 fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

> **Note:** The dual-gate approach (adaptive threshold AND absolute darkness) prevents the system from detecting paper texture as ink. The separate colored-ink HSV masks catch faint pen strokes that do not pass the darkness gate.

### 3B. Mask-to-Paths Conversion — `_processMaskToPaths`

```mermaid
flowchart TD
    P1["Input: allStrokes binary mask"] --> P2["cv.morphologyEx MORPH_CLOSE 3x3 rect<br/>First dilates then erodes. This fills small<br/>1-2px gaps inside ink strokes without<br/>changing the overall stroke size."]
    P2 --> P3["cv.connectedComponentsWithStats<br/>Flood-fills the mask to find isolated blobs.<br/>Assigns a unique numeric label to each blob.<br/>Also computes area, bounding box width,<br/>bounding box height for every blob."]
    P3 --> P4["Filter blobs:<br/>Keep if area >= 0.005 percent of image<br/>OR max span >= 1 percent of min dimension.<br/>Removes tiny dust specks while keeping<br/>thin but long ink strokes."]
    P4 --> P5["cv.morphologyEx MORPH_CLOSE 3x3 rect<br/>Second close pass on the filtered mask<br/>to reconnect any strokes that lost pixels<br/>during the blob-filtering step."]
    
    P5 --> P6["GEOMETRIC SHAPE DETECTION"]
    P6 --> P6a["Heavy MORPH_CLOSE 15x15 ellipse<br/>Very aggressive close that merges nearby<br/>ink regions. Bridges gaps in hand-drawn<br/>circles and triangles so they look solid."]
    P6a --> P6b["cv.findContours RETR_EXTERNAL<br/>Walks the border of each white region<br/>and records the boundary as an ordered<br/>list of x,y points. RETR_EXTERNAL means<br/>only outermost boundaries, no holes."]
    P6b --> P6c{"For each contour"}
    
    P6c --> P6d["cv.convexHull: wraps a rubber band<br/>around all contour points to get the<br/>smallest convex shape that contains them.<br/>solidity = contour area / hull area.<br/>High solidity = compact convex shape."]
    P6d --> P6e{"solidity > 0.75?"}
    P6e -->|"No"| P6f["Leave for skeletonization"]
    P6e -->|"Yes"| P6g["cv.approxPolyDP: walks along the contour<br/>and drops points that deviate less than<br/>5 percent of the perimeter from a straight<br/>line. Reduces a wobbly hand-drawn shape<br/>to its essential vertices."]
    P6g --> P6h{"circularity > 0.75?<br/>circularity = 4 * pi * area / perimeter^2.<br/>A perfect circle = 1.0, a square = 0.78."}
    P6h -->|"Yes"| P6i["cv.minEnclosingCircle: finds the<br/>smallest circle that fully contains<br/>all contour points. Returns center and radius."]
    P6h -->|"No"| P6j{"3-8 vertices?"}
    P6j -->|"Yes"| P6k["Snap to polygon: take the vertices<br/>from approxPolyDP and connect them<br/>as straight line segments."]
    P6j -->|"No"| P6f
    
    P6i --> P6l["Erase blob from finalMask<br/>filled or thick brush"]
    P6k --> P6l
    
    P6l --> P7["SKELETONIZATION"]
    P6f --> P7
    P7 --> P7a["_vectorizeAndSkeletonize<br/>Converts OpenCV mask to a flat bool array,<br/>then calls TraceSkeleton"]
    P7a --> P7b["Convert mask to bool array<br/>Each pixel becomes 1 for ink or 0 for empty"]
    P7b --> P7c["TraceSkeleton.fromBoolArray<br/>Runs Zhang-Suen thinning to get a<br/>1px skeleton, then traces it into polylines"]
    P7c --> P7d["Returns raw polylines"]
    
    P7d --> P8["POST-PROCESSING"]
    P8 --> P8a["Filter oversized paths<br/>more than 45 percent of image dimension"]
    P8a --> P8b["Filter hair-thin elongated noise<br/>aspect ratio more than 12:1 and long"]
    P8b --> P8c{"Path diagonal less than 80px?"}
    P8c -->|"Yes"| P8d["_smoothPath: weighted moving-average<br/>on each interior point, 2 iterations<br/>then _simplifyPath: Douglas-Peucker<br/>removes redundant points, epsilon=1.5"]
    P8c -->|"No"| P8e["_simplifyPath: Douglas-Peucker only<br/>epsilon=3.0, no moving-average<br/>Keeps sharp corners like arrowheads"]
    
    P8d --> P9["Merge: perfectShapes + smoothedPaths"]
    P8e --> P9

    style P1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style P9 fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

### 3C. Color Classification + Path Fracturing (Steps 5-6)

```mermaid
flowchart TD
    C1["Input: array of path shapes"] --> C2{"For each shape"}
    C2 --> C3["Sample 2-30 points along the path<br/>evenly spaced by arc length"]
    
    C3 --> C4["For each sample point:<br/>Read 3x3 neighborhood in original image"]
    C4 --> C5["Find darkest pixel in neighborhood<br/>best RGB candidate"]
    C5 --> C6["Find local background color<br/>scan 5-20px to the right<br/>for first non-ink pixel"]
    C6 --> C7["Von Kries White Balance<br/>adjR = bestR x bgAvg / bgR<br/>adjG = bestG x bgAvg / bgG<br/>adjB = bestB x bgAvg / bgB"]
    
    C7 --> C8["Hybrid Color Decision"]
    C8 --> C8a["Green: adjG > adjR+5 AND bestG > bestB+2<br/>Von Kries for R/G, raw for G/B"]
    C8 --> C8b["Blue: bestB > bestG+10 AND bestB > bestR+5<br/>raw values only"]
    C8 --> C8c["Red: adjR > adjG+15 AND adjR > adjB+15<br/>Von Kries only"]
    C8 --> C8d["Black: none of the above"]
    
    C8a --> C9["Sliding Window Majority Vote<br/>window = 7 samples each side<br/>Smooths out noise in color labels"]
    C8b --> C9
    C8c --> C9
    C8d --> C9
    
    C9 --> C10["PATH FRACTURING<br/>Split path at color boundaries"]
    C10 --> C11["Each segment assigned to a layer:<br/>black/blue = thru_cut<br/>red = score<br/>green = crease"]
    C11 --> C12["Re-simplify each segment<br/>_simplifyPath epsilon=1.5"]
    C12 --> C13["Output: layersData<br/>thru_cut, score, crease<br/>each with paths and color"]

    style C1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style C13 fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

> **Important — Why the hybrid color approach?** Cardboard has a strong red bias (R=180, G=140, B=100). Von Kries white balance removes this bias and correctly separates green ink from the red background. However, the low blue of cardboard causes the Von Kries blue multiplier to inflate camera noise. Therefore, blue detection uses raw RGB values to prevent false positives.

---

## 4. TraceSkeleton — Thinning and Tracing

This is the minified `trace_skeleton.js`. The algorithm has two phases:

```mermaid
flowchart TD
    T1["Input: boolean pixel array<br/>W x H"] --> T2["PHASE 1: Zhang-Suen Thinning"]
    T2 --> T2a["Iterate until no pixels change"]
    T2a --> T2b["Sub-iteration 0:<br/>Mark pixel for removal if<br/>2-6 neighbors<br/>exactly 1 zero-to-one transition<br/>top AND right AND bottom = 0<br/>OR right AND bottom AND left = 0"]
    T2b --> T2c["Sub-iteration 1:<br/>Same conditions but<br/>top AND right AND left = 0<br/>OR top AND bottom AND left = 0"]
    T2c --> T2d["Remove all marked pixels"]
    T2d --> T2a
    T2a -->|"Converged"| T3["1-pixel-wide skeleton"]
    
    T3 --> T4["PHASE 2: Recursive Divide-and-Conquer Tracing"]
    T4 --> T4a["traceSkeleton region"]
    T4a --> T4b{"Region small enough?<br/>both dims <= chunkSize=10"}
    T4b -->|"Yes"| T4c["Walk the boundary<br/>Find entry/exit points of skeleton<br/>Connect through center<br/>Return polyline segments"]
    T4b -->|"No"| T4d["Find optimal split line<br/>row or column with minimum crossings"]
    T4d --> T4e["Split region into two halves"]
    T4e --> T4f["Recurse on each half"]
    T4f --> T4g["Merge polylines at split boundary<br/>stitch endpoints that are close"]
    T4g --> T4h["Return merged polylines"]
    
    style T1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style T4h fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

### Zhang-Suen Thinning — Conditions Summary

| Check | Purpose |
|---|---|
| **2 to 6 neighbors** | Keep endpoints (1 neighbor) and prevent erosion of thick regions |
| **Exactly 1 transition (0 to 1)** | Preserve connectivity — only thin at edges, not at junctions |
| **Product checks** | Make sure the pixel is on the correct side of the stroke. Alternates between sub-iterations to prevent directional bias. |

---

## 5. SvgGenerator — Pixel-to-MM SVG Output

```mermaid
flowchart TD
    S1["Input: layersData, image width/height,<br/>frameConfig cropPhysW cropPhysH"] --> S2["Compute dotsPerMm = width / cropPhysW"]
    S2 --> S3["toMm px = px / dotsPerMm"]
    
    S3 --> S4["Create SVG header<br/>viewBox = 0 0 physWidth physHeight<br/>1 SVG unit = 1 mm"]
    S4 --> S5["Layer: bed_frame<br/>Rectangle at 0,0 to physW x physH<br/>Color: magenta"]
    
    S5 --> S6{"For each layer<br/>thru_cut, score, crease"}
    S6 --> S7{"For each shape"}
    S7 -->|"circle"| S8["SVG circle element<br/>cx, cy, r in mm"]
    S7 -->|"polygon"| S9["SVG polygon element<br/>points in mm"]
    S7 -->|"path"| S10["SVG path element<br/>M ... L ... in mm"]
    
    S8 --> S11["Close layer group"]
    S9 --> S11
    S10 --> S11
    S11 --> S12["Return svg string and meta<br/>dots_per_mm, physW, physH"]

    style S1 fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style S12 fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```

> **Note:** The SVG uses **top-left origin** with Y increasing downward — same as the warped image. Any Y-axis flip for G-code or CNC machines must occur in a separate post-processor, not here.

---

## 6. Utility Algorithms

### Douglas-Peucker Path Simplification — `_simplifyPath`

- **Input:** A polyline (ordered list of x,y points) and an epsilon tolerance value.
- **How it works:**
  1. Draw a straight line from the first point to the last point of the polyline.
  2. For every point between them, measure the perpendicular distance from that point to the line. This measurement is done by `_perpendicularDistance`, which uses the standard point-to-line-distance formula.
  3. Find the point with the maximum distance (`dmax`).
  4. If `dmax` is larger than epsilon, the curve bends too much to be a straight line. Split the polyline at that point and recursively simplify each half.
  5. If `dmax` is smaller than epsilon, the curve is close enough to straight. Replace all intermediate points with a single straight line.
- **Effect:** Reduces point count while it preserves shape fidelity within the epsilon tolerance. A larger epsilon produces fewer points but loses fine detail.

### Perpendicular Distance — `_perpendicularDistance`

- **Input:** A point and a line defined by two endpoints.
- **How it works:** Calculates the shortest distance from the point to the infinite line through the two endpoints. Uses the formula: `|((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)| / sqrt((y2-y1)^2 + (x2-x1)^2)`. If the two endpoints are the same, it returns the straight-line distance between them.
- **Usage:** Called only by `_simplifyPath` to decide which points to keep and which to discard.

### Moving-Average Smoothing — `_smoothPath`

- **Input:** A polyline and an iteration count.
- **How it works:**
  1. Walk through every interior point (skip the first and last).
  2. Replace each interior point with a weighted average of itself and its two neighbors: `new_x = 0.25 * prev_x + 0.50 * current_x + 0.25 * next_x`. Same for y.
  3. The first and last points never change, so the path endpoints stay fixed.
  4. Repeat the entire pass for N iterations. More iterations = smoother curve.
- **Usage:** Applied only to small paths (diagonal less than 80px) to reduce jitter. Large paths skip this to preserve sharp corners such as arrowheads.

---

## 7. Complete End-to-End Data Transform

```mermaid
flowchart LR
    A["Camera RGBA Image"] -->|"WarpEngine"| B["Flat, Scaled RGBA Image<br/>2000px wide"]
    B -->|"InkExtractor<br/>Steps 1-4"| C["Binary Ink Mask"]
    C -->|"Geometric Detection"| D["Perfect Shapes<br/>circles, polygons"]
    C -->|"TraceSkeleton"| E["Skeleton Polylines"]
    D --> F["Merged Path List"]
    E --> F
    F -->|"Color Classification<br/>+ Path Fracturing"| G["Layered Paths<br/>thru_cut, score, crease"]
    G -->|"SvgGenerator"| H["Layered SVG String<br/>mm coordinates"]

    style A fill:#1e293b,color:#e2e8f0,stroke:#3b82f6
    style H fill:#1e293b,color:#e2e8f0,stroke:#22c55e
```
