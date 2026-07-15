# Urumi Vision App & Magic Pen Pipeline

A high-performance vision processing application designed for a 113x83cm ArUco-based cutting bed. It leverages OpenCV.js and WebRTC (PeerJS) to capture images, correct perspective distortion, extract multi-colored hand-drawn ink strokes, and generate layered SVG files ready for CNC/G-code translation.

## Core Capabilities

- **ArUco Marker Detection:** Automatically detects markers at the corners of the cutting bed and applies homography to flatten the image.
- **Intelligent Ink Extraction:** Robustly separates ink from cardboard backgrounds, despite intense shadows and lighting gradients.
- **Dynamic Geometric Snapping:** Converts hand-drawn circles and rectangles into precise geometric SVG elements.
- **Multi-Color Segmentation:** Skeletons are mathematically fractured at color boundaries, allowing users to draw complex, intersecting shapes with multiple pens.
- **Layered CNC Output:** Generates a 1:1 scale millimeter-accurate SVG with distinct layers mapped to specific CNC operations.

---

## Project Architecture & Pipeline

The core computer vision logic is modularized into three primary engines located in the `/pipeline/` directory.

### 1. Warp Engine (`WarpEngine.js`)
- **Marker Detection:** Utilizes OpenCV's ArUco module to detect the four corner markers defining the physical boundary of the cutting bed.
- **Perspective Correction:** Calculates a transformation matrix (`cv.getPerspectiveTransform`) and warps the skewed camera image (`cv.warpPerspective`) into a perfectly flat, top-down projection scaled exactly to the physical bed dimensions.

### 2. Ink Extraction & Vectorization (`InkExtractor.js`)
This module is the heart of the "Magic Pen" feature, employing advanced computer vision techniques to convert messy analog drawings into crisp digital vectors.

* **Mask Generation:** Uses bilateral filtering and blackhat morphology to isolate ink from the textured cardboard background.
* **Geometric Feature Extraction:** 
  - Extracts contours (`cv.findContours`) and analyzes shape solidity and circularity (`cv.convexHull`).
  - Snaps high-solidity structures into perfect geometric polygons (`cv.approxPolyDP`) and circles.
  - **Edge Shattering:** Multi-vertex polygons are dynamically shattered into discrete, independent line segments. This allows a single closed geometric shape to have different colors on different sides.
* **Skeletonization (`TraceSkeleton`):** 
  - For non-geometric hand-drawn strokes, the mask is thinned down to a precise 1-pixel centerline using a Zhang-Suen thinning algorithm.
  - This prevents double-line tracing and minimizes G-code bloat.
* **Path Smoothing:** Applies the Douglas-Peucker simplification algorithm to optimize nodes for CNC machines, preserving sharp corners while removing hand-jitter.

### 3. Advanced Color Classification Engine
The color classification logic in `InkExtractor.js` was custom-built to solve extreme environmental challenges (shadows, cardboard hue, camera noise, and intersecting colors).

* **Point-by-Point Path Fracturing:** 
  Standard skeletonization blindly groups touching strokes into a single continuous path. To support intersecting multi-color lines, our engine casts a localized 3x3 search window at every point along the skeleton to sample the darkest ink core. It applies a sliding-window majority vote (to eliminate micro-noise) and actively fractures the vector path wherever a color transition occurs.
* **Local Von Kries White Balancing:** 
  Global color thresholds fail under uneven lighting. Instead, the algorithm dynamically samples the cardboard background locally for every pixel. It applies a localized Von Kries multiplier to neutralize lighting gradients and the cardboard's inherent yellow tint before evaluating the ink.
* **Hybrid Color Dominance Model:** 
  - *Red vs. Black:* Uses the Von Kries-adjusted RGB ratios. Since black ink on yellow cardboard naturally registers as red to a camera, this white-balancing step is strictly required to distinguish true red ink from black ink.
  - *Green vs. Blue:* Uses **raw** sensor RGB dominance. This hybrid bypass intentionally skips Von Kries for the blue/green spectrum because smartphone cameras have high noise floors in the blue channel under low light. Multiplying that noise floor artificially causes dark green ink to bloom and falsely classify as blue.
* **Cross-Contamination Prevention:** The sampling window is tightly restricted to 3x3 pixels to prevent the algorithm from accidentally sampling adjacent, unconnected ink strokes (e.g., a blue tick mark drawn millimeters away from a green line).

### 4. SVG Generation (`SvgGenerator.js`)
- Takes the fully classified and fractured vector segments and assemblies a valid XML SVG string.
- Automatically maps colors to their designated CNC operations:
  - **Red Ink** ➔ `score`
  - **Green Ink** ➔ `crease`
  - **Blue / Black Ink** ➔ `thru_cut`
- Outputs paths that are scaled perfectly to physical millimeters (e.g., 1 SVG unit = 1 mm) for seamless CAM integration.

---

## Usage & Local Development

You can serve this directory using any static file server.

```bash
# Using Python
python -m http.server 5500

# Using Node.js
npx http-server -p 5500
```

1. Open `http://localhost:5500/test_receiver.html` on your desktop to launch the receiver UI.
2. Scan the QR code or open the URL on your mobile device (devices must be on the same network).
3. Tap the capture button on your phone, line up the ArUco bed, and upload.
4. The processed, CNC-ready SVG will instantly appear on your desktop receiver via PeerJS WebRTC.
