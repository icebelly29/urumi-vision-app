# Urumi Vision App

A high-performance vision processing application designed for a 113x83cm ArUco-based cutting bed. It leverages OpenCV.js and WebRTC (PeerJS) to capture images, correct perspective distortion, extract handwritten ink strokes, and generate layered SVG files ready for CNC/G-code translation.

## Features

- **ArUco Marker Detection & Perspective Warping:** Automatically detects ArUco markers at the corners of the cutting bed and applies perspective correction to flatten the image.
- **Dynamic Bed Frame Support:** Supports dynamic coordinate mapping based on URL parameters or JSON configuration, effectively cropping out checkerboard borders.
- **Color-Based Stroke Extraction:** Uses blackhat morphology and HSV saturation to accurately extract and separate red, green, and blue/black ink.
- **Geometric Snapping:** Converts hand-drawn circles and polygons into precise geometric SVG elements.
- **Path Smoothing & Simplification:** Uses Douglas-Peucker simplification and iterative Gaussian smoothing to optimize paths for CNC machines while reducing G-code bloat.
- **Layered SVG Generation:** Outputs a 1:1 scale millimeter-accurate SVG with distinct layers for `thru_cut`, `score`, and `crease`.
- **WebRTC Integration:** Directly streams processing results to a desktop receiver UI via PeerJS.

## Project Architecture

The core image processing logic has been modularized into a dedicated pipeline for better maintainability:

- `index.html` - The main mobile-facing UI for capturing or uploading images.
- `app.js` - Handles UI interactions, file uploading, and initiates processing.
- `Communication.js` - Manages the PeerJS connection to send data to the desktop receiver.
- `test_receiver.html` - A test desktop receiver UI to view the generated SVGs.
- `ImageProcessor.js` - Orchestrator for the vision pipeline.

### Pipeline Modules (`/pipeline/`)

1. **`WarpEngine.js`**
   - Handles the OpenCV ArUco dictionary initialization and detection.
   - Calculates the target bounding box and perspective transform matrix.
   - Outputs the flattened and cropped `cv.Mat` image.

2. **`InkExtractor.js`**
   - Core extraction logic using bilateral filtering and blackhat morphology.
   - Isolates the ink from the background paper.
   - Detects contours, simplifies them, and applies skeletonization (`TraceSkeleton`).
   - Uses a dynamic point-by-point evaluation and sliding-window majority vote to shatter intersecting skeletons at color boundaries.
   - Employs a hybrid Raw/Von-Kries color classification model to handle lighting gradients and eliminate camera noise-floor bloat.

3. **`SvgGenerator.js`**
   - Takes the extracted paths and shapes to assemble a valid XML SVG string.
   - Ensures correct physical dimensions and millimeter-to-pixel ratios.
   - Organizes elements by CNC operations (`layer_bed_frame`, `thru_cut`, `score`, `crease`).

## Usage & Setup

### 1. Local Development

You can serve this directory using any static file server.

```bash
# Using Python
python -m http.server 5500

# Using Node.js (http-server)
npx http-server -p 5500
```

### 2. Testing the Pipeline

1. Open `http://localhost:5500/test_receiver.html` on your desktop.
2. A QR code and URL will appear.
3. Open the URL on your mobile device (must be on the same network or use a tunnel like Ngrok).
4. Tap the capture button on your phone and upload a photo of the cutting bed.
5. The processed SVG and image will appear on your desktop receiver.

## Technical Details

- **Dependencies:** OpenCV.js (v4.8.0), PeerJS (v1.3.2), Lucide Icons.
- **Coordinate System:** The resulting SVG maps 1 SVG unit to 1 physical millimeter.
- **Ink Classification:** 
  - Red (Score)
  - Green (Crease)
  - Blue/Black (Thru-cut)

## Deep Dive: Magic Pen Vision Algorithms

Over the course of the project, we encountered and solved numerous complex physical and environmental issues inherent in smartphone-based computer vision. To ensure 100% vector accuracy for CNC routing, the `InkExtractor.js` engine relies on several advanced algorithms:

### 1. Dynamic Edge Shattering (Geometric Snapping)
Standard geometric detection (like `cv.approxPolyDP`) forces closed hand-drawn loops into a monolithic polygon (e.g., a perfect square). Because SVG polygons only support a single stroke color, drawing a square with 4 different colored pens would cause the entire shape to inherit the dominant color.
* **The Solution:** We explicitly "shatter" all mathematically extracted polygons into discrete 2-point line segments (`<polyline>`). This allows a single snapped geometric square to preserve the unique colors of each of its physical edges.

### 2. Point-by-Point Path Fracturing
When the pipeline falls back to `TraceSkeleton` for non-geometric, messy lines, Zhang-Suen skeletonization inherently groups intersecting strokes into a single continuous path. Previously, assigning a single color to an entire path caused minority colors to be swallowed if a blue line intersected a red line.
* **The Solution:** The engine evaluates the color at every single pixel along the skeletonized line. It uses a **3x3 dark-pixel search window** to find the ink core, preventing cross-contamination from adjacent strokes. It then applies a **7-point sliding-window majority vote** to eliminate camera noise. When the algorithm detects a true shift in color, it dynamically fractures the SVG path in half exactly at the color boundary.

### 3. Hybrid Color Dominance Model
Distinguishing dark ink on a yellow cardboard bed under uneven lighting proved highly challenging.
* **Red vs. Black (Von Kries White Balance):** Because yellow cardboard reflects high amounts of red and green light, neutral black ink naturally reads as "Red" to a camera sensor. To solve this, the pipeline samples the localized background and applies a **Von Kries White Balance multiplier** to artificially neutralize the cardboard's yellow tint, allowing the algorithm to mathematically differentiate true Red ink from Black ink.
* **Green vs. Blue (Raw Dominance):** Smartphone cameras have notoriously high noise floors in the blue channel in low light. Multiplying that noise floor by the Von Kries multiplier artificially inflated the blue channel, causing dark green ink to register as blue. To solve this, the pipeline bypasses the Von Kries multiplier for the green/blue spectrum and strictly evaluates the **Raw Sensor RGB**. Since raw green ink is physically greener than it is blue, it easily overrides the camera's blue noise floor.
