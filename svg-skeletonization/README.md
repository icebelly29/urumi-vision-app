# SVG Skeletonization

A highly specialized computer vision pipeline for extracting, skeletonizing, and classifying hand-drawn or printed paths from images, specifically designed for CNC machines, pen plotters, and digital fabrication workflows. 

It handles perspective distortion (via ArUco markers), ink color classification, geometric shape extraction, and outputs layered, machine-ready SVG files.

## Features

- **Perspective Warping**: Automatically corrects perspective distortion if 4 ArUco markers are detected on the physical machine bed.
- **Color Classification**: Identifies and categorizes ink colors (e.g., Red for score, Green for crease, Black for through-cut).
- **Vectorization**: Transforms raw pixel masks into clean `TraceSkeleton` center-lines or outline contours.
- **Shape Snapping**: Automatically snaps hand-drawn circles and polygons to perfect geometric SVG primitives.
- **SVG Generation**: Compiles the extracted paths into a layered, scaled SVG file ready for CNC execution.

## Installation

You can install the package via npm:

```bash
npm install svg-skeletonization
```

### Important: OpenCV.js Peer Dependency

`svg-skeletonization` heavily relies on OpenCV for image processing. To keep the bundle size small and prevent WebAssembly loading issues in modern bundlers, OpenCV is **not** bundled directly into this package. 

You must load `opencv.js` globally in your application. The easiest way to do this is by adding the following script tag to your HTML file:

```html
<!-- Add this to your index.html -->
<script async src="https://docs.opencv.org/4.8.0/opencv.js" type="text/javascript"></script>
```

You must ensure that `cv` is fully initialized before attempting to run the pipeline.

## Usage

Here is a basic example of how to use the pipeline to process an image.

```javascript
import { ImageProcessor } from 'svg-skeletonization';

async function processImage(imageElement) {
    // 1. Ensure OpenCV is fully loaded
    if (!window.cv || typeof window.cv.Mat !== 'function') {
        throw new Error("OpenCV is not loaded yet.");
    }

    // 2. Instantiate the processor
    const processor = new ImageProcessor();

    try {
        // 3. Run the pipeline (One-step execution)
        const result = await processor.process(imageElement);
        
        console.log("Processed SVG String:", result.svg);
        console.log("Processed Image Data URL:", result.image);
        console.log("Metadata:", result.meta);
        
        // Example: Inject SVG into the DOM
        document.getElementById('svg-container').innerHTML = result.svg;
        
    } catch (error) {
        console.error("Pipeline failed:", error);
    }
}
```

### Two-Step Pipeline Execution (e.g. for Lasso Selection on flattened view)
If you want to display the flattened image to a user (e.g. so they can draw a lasso mask on the aligned/distortion-free view) before final processing, you can split the pipeline execution:

```javascript
// 1. Warp and flatten the image first
const flattenedCanvas = await processor.flatten(imageElement);

// 2. [Optional] Show flattenedCanvas in a Lasso UI and get a maskCanvas

// 3. Process the flattened image with the mask
const result = await processor.processFlattened(flattenedCanvas, maskCanvas);
```

## API Reference

### `ImageProcessor`
The primary orchestrator class that runs the full vision pipeline.

- `process(imageElement, maskCanvas = null)`: 
  - `imageElement`: An `<img>` or `<canvas>` element containing the source image.
  - `maskCanvas` (optional): A canvas containing a user-drawn lasso mask aligned to the original image to exclude unwanted background noise.
  - Returns a Promise resolving to `{ svg: string, image: string, meta: object }`.

- `flatten(imageElement)`:
  - `imageElement`: An `<img>` or `<canvas>` element containing the source image.
  - Runs ArUco detection and perspective warping.
  - Returns a Promise resolving to an HTML `<canvas>` element containing the flattened image.

- `processFlattened(flattenedCanvas, maskCanvas = null)`:
  - `flattenedCanvas`: The HTML `<canvas>` returned by `flatten()`.
  - `maskCanvas` (optional): A canvas containing a user-drawn lasso mask aligned with the *flattened* image.
  - Runs color/ink extraction, skeletonization, and SVG generation.
  - Returns a Promise resolving to `{ svg: string, image: string, meta: object }`.

### `WarpEngine`
Handles the detection of ArUco markers and homography perspective transformation.
- Automatically calculates scaling factors based on real-world bed dimensions.

### `InkExtractor`
Handles the core pixel-level logic.
- Adaptive thresholding and color gating.
- Uses Von Kries white balancing to adapt to uneven lighting on cardboard.
- Bridges gaps in paths and simplifies complex vector arrays using Douglas-Peucker algorithms.

### `SvgGenerator`
Takes the raw mathematical paths and geometric primitives and writes them to a formatted XML/SVG string grouped by color layers.

## Development

If you want to contribute or build the package locally:

1. Clone the repository and navigate to the package directory.
2. Run `npm install` to install dependencies (Vite).
3. Run `npm run build` to compile the ES Modules into the `dist/` directory.

To test the package locally, you can spin up the Vite server:
```bash
npm run dev
# Then open http://localhost:5173/test.html (or whatever port Vite assigns)
```

## License

ISC License.
