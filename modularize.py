import os
import re

with open('ImageProcessor.js', 'r', encoding='utf-8') as f:
    content = f.read()

# We will just write new files
warp_engine = '''class WarpEngine {
    constructor(outputWidth) {
        this.OUTPUT_WIDTH = outputWidth;
        this.currentFrameConfig = null;
        this.borderSizeMm = 0;
    }
'''

# Extract normalizePerspective
match = re.search(r'    _normalizePerspective\(src\) \{.*?(?=    _extractColorPaths)', content, re.DOTALL)
if match:
    func = match.group(0).replace('    _normalizePerspective', '    normalizePerspective')
    warp_engine += func
    warp_engine += '}\n'
else:
    print("Failed to find normalizePerspective")

with open('pipeline/WarpEngine.js', 'w', encoding='utf-8') as f:
    f.write(warp_engine)

ink_extractor = '''class InkExtractor {
    constructor(colorProfiles) {
        this.colorProfiles = colorProfiles;
    }
'''

# Extract extractColorPaths
match = re.search(r'    _extractColorPaths\(img\) \{.*?(?=    _vectorizeAndSkeletonize)', content, re.DOTALL)
if match:
    func = match.group(0).replace('    _extractColorPaths', '    extractColorPaths')
    
    # Need to replace this.currentFrameConfig and this.borderSizeMm with arguments
    func = func.replace('extractColorPaths(img) {', 'extractColorPaths(img, frameConfig, borderSizeMm) {')
    func = func.replace('this.currentFrameConfig', 'frameConfig')
    func = func.replace('this.borderSizeMm', 'borderSizeMm')
    
    ink_extractor += func
else:
    print("Failed to find _extractColorPaths")

# Extract the rest of the private functions before _generateLayeredSvg
match = re.search(r'    _vectorizeAndSkeletonize\(maskMat\) \{.*?(?=    _generateLayeredSvg)', content, re.DOTALL)
if match:
    ink_extractor += match.group(0)
    ink_extractor += '}\n'
else:
    print("Failed to find _vectorizeAndSkeletonize")

with open('pipeline/InkExtractor.js', 'w', encoding='utf-8') as f:
    f.write(ink_extractor)

svg_generator = '''class SvgGenerator {
'''

# Extract _generateLayeredSvg
match = re.search(r'    _generateLayeredSvg\(layersData, width, height\) \{.*?\n    \}\n', content, re.DOTALL)
if match:
    func = match.group(0).replace('    _generateLayeredSvg', '    static generateLayeredSvg(layersData, width, height, frameConfig) {')
    func = func.replace('this.currentFrameConfig', 'frameConfig')
    # Fix the method definition since we added frameConfig
    func = re.sub(r'static generateLayeredSvg\(layersData, width, height\).*?\{', 'static generateLayeredSvg(layersData, width, height, frameConfig) {', func, count=1)
    
    svg_generator += func
    svg_generator += '}\n'
else:
    print("Failed to find _generateLayeredSvg")

with open('pipeline/SvgGenerator.js', 'w', encoding='utf-8') as f:
    f.write(svg_generator)

image_processor = '''/**
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

    async process(imageElement) {
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
        const layersData = this.inkExtractor.extractColorPaths(warpedMat, this.warpEngine.currentFrameConfig, this.warpEngine.borderSizeMm);

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
'''

with open('ImageProcessor.js', 'w', encoding='utf-8') as f:
    f.write(image_processor)
