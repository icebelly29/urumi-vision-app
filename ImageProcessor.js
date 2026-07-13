/**
 * ImageProcessor.js
 * Orchestrates OpenCV ArUco detection, perspective warping, vectorization,
 * skeletonization, path simplification, and layered SVG generation.
 */

class ImageProcessor {
    constructor() {
        this.OUTPUT_WIDTH = 2000;
        this.colorProfiles = {
            thru_cut: { layer: 'thru_cut', color: '#3b82f6' },
            score: { layer: 'score', color: '#ef4444' },
            crease: { layer: 'crease', color: '#22c55e' }
        };
        
        this.warpEngine = new WarpEngine(this.OUTPUT_WIDTH);
        this.inkExtractor = new InkExtractor(this.colorProfiles);
    }

    async process(imageElement, vectorizationMode = 'skeleton') {
        if (!cv || !cv.Mat) {
            throw new Error("OpenCV is not initialized yet.");
        }

        console.log("Starting Image Processing Pipeline...");

        let src = cv.imread(imageElement);
        let dst = new cv.Mat();

        // 1. Detect ArUco & Normalize Perspective
        const warpedMat = this.warpEngine.normalizePerspective(src);
        if (!warpedMat) {
            src.delete();
            dst.delete();
            throw new Error("Could not detect 4 ArUco markers for bed rectangle.");
        }

        // 2. Extract paths by color
        const layersData = this.inkExtractor.extractColorPaths(warpedMat, this.warpEngine.currentFrameConfig, this.warpEngine.borderSizeMm, vectorizationMode);

        // 3. Generate Layered SVG
        const resultLayered = SvgGenerator.generateLayeredSvg(layersData, warpedMat.cols, warpedMat.rows, this.warpEngine.currentFrameConfig);
        const svgContent = resultLayered.svg;
        const metaContent = resultLayered.meta;

        // Prepare return data (SVG text + Image Data URL)
        const canvas = document.createElement('canvas');
        cv.imshow(canvas, warpedMat);
        const imageUrl = canvas.toDataURL('image/jpeg', 0.8);

        // Cleanup OpenCV mats
        src.delete();
        dst.delete();
        warpedMat.delete();

        console.log("Pipeline complete.");

        return {
            svg: svgContent,
            image: imageUrl,
            meta: metaContent
        };
    }
}

window.ImageProcessor = ImageProcessor;
