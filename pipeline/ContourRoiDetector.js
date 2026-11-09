/**
 * ContourRoiDetector.js
 * Robust workpiece & ROI detector for the Urumi Vision pipeline.
 * Uses cardboard chromatic signature + morphological bridging to find the exact workpiece sheet.
 */
class ContourRoiDetector {
    /**
     * Detects workpiece ROI on warpedMat.
     * @param {cv.Mat} warpedMat - Perspective corrected bed image (RGBA)
     * @param {Object} options - Custom padding and frame margin
     * @returns {Object|null} - { x, y, width, height, confidence, type }
     */
    static detect(warpedMat, options = {}) {
        if (!cv || !cv.Mat) {
            throw new Error("OpenCV is not initialized.");
        }

        const padding = options.padding !== undefined ? options.padding : 15;
        const frameMargin = options.frameMargin !== undefined ? options.frameMargin : 80;

        const rows = warpedMat.rows;
        const cols = warpedMat.cols;

        // Convert to HSV
        let rgbMat = new cv.Mat();
        let hsvMat = new cv.Mat();
        cv.cvtColor(warpedMat, rgbMat, cv.COLOR_RGBA2RGB);
        cv.cvtColor(rgbMat, hsvMat, cv.COLOR_RGB2HSV);
        rgbMat.delete();

        // 1. CARDBOARD CHROMATIC FILTER
        // Brown/tan kraft cardboard has Hue ~8..42 and Saturation > 25
        let lower = new cv.Mat(rows, cols, hsvMat.type(), [7, 25, 35, 0]);
        let upper = new cv.Mat(rows, cols, hsvMat.type(), [42, 255, 255, 255]);
        let cardboardMask = new cv.Mat();
        cv.inRange(hsvMat, lower, upper, cardboardMask);
        lower.delete();
        upper.delete();
        hsvMat.delete();

        // Mask out the ArUco frame perimeter
        const mTop = Math.min(frameMargin, rows);
        const mBot = Math.max(0, rows - frameMargin);
        const mLeft = Math.min(frameMargin, cols);
        const mRight = Math.max(0, cols - frameMargin);

        let roiTop = cardboardMask.roi(new cv.Rect(0, 0, cols, mTop));
        roiTop.setTo(new cv.Scalar(0));
        roiTop.delete();

        let roiBot = cardboardMask.roi(new cv.Rect(0, mBot, cols, rows - mBot));
        roiBot.setTo(new cv.Scalar(0));
        roiBot.delete();

        let roiLeft = cardboardMask.roi(new cv.Rect(0, 0, mLeft, rows));
        roiLeft.setTo(new cv.Scalar(0));
        roiLeft.delete();

        let roiRight = cardboardMask.roi(new cv.Rect(mRight, 0, cols - mRight, rows));
        roiRight.setTo(new cv.Scalar(0));
        roiRight.delete();

        // Morphological CLOSE to bridge drawing lines on the cardboard
        let kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(25, 25));
        let closedMask = new cv.Mat();
        cv.morphologyEx(cardboardMask, closedMask, cv.MORPH_CLOSE, kernel);
        kernel.delete();
        cardboardMask.delete();

        // Find contours of the cardboard sheet
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(closedMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let bestBox = null;
        let maxArea = 0;

        for (let i = 0; i < contours.size(); i++) {
            const cnt = contours.get(i);
            const area = cv.contourArea(cnt);
            // Must be a significant physical workpiece (> 8000 px^2 on 2000px canvas)
            if (area > 8000 && area > maxArea) {
                maxArea = area;
                const rect = cv.boundingRect(cnt);
                bestBox = rect;
            }
            cnt.delete();
        }

        contours.delete();
        hierarchy.delete();
        closedMask.delete();

        if (bestBox) {
            const boxX = Math.max(0, bestBox.x - padding);
            const boxY = Math.max(0, bestBox.y - padding);
            const boxW = Math.min(cols - boxX, bestBox.width + (padding * 2));
            const boxH = Math.min(rows - boxY, bestBox.height + (padding * 2));

            return {
                x: boxX,
                y: boxY,
                width: boxW,
                height: boxH,
                confidence: 0.99,
                type: 'cardboard_contour'
            };
        }

        // 2. FALLBACK: Ink cluster detection (for white paper or direct drawing)
        return this.detectInkCluster(warpedMat, padding, frameMargin);
    }

    static detectInkCluster(warpedMat, padding, frameMargin) {
        const rows = warpedMat.rows;
        const cols = warpedMat.cols;

        let gray = new cv.Mat();
        cv.cvtColor(warpedMat, gray, cv.COLOR_RGBA2GRAY);

        let blurred = new cv.Mat();
        cv.GaussianBlur(gray, blurred, new cv.Size(9, 9), 0);
        gray.delete();

        let thresh = new cv.Mat();
        cv.adaptiveThreshold(blurred, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 10);
        blurred.delete();

        // Clear perimeter
        let rTop = thresh.roi(new cv.Rect(0, 0, cols, Math.min(frameMargin, rows)));
        rTop.setTo(new cv.Scalar(0));
        rTop.delete();

        let rBot = thresh.roi(new cv.Rect(0, Math.max(0, rows - frameMargin), cols, Math.min(frameMargin, rows)));
        rBot.setTo(new cv.Scalar(0));
        rBot.delete();

        let rLeft = thresh.roi(new cv.Rect(0, 0, Math.min(frameMargin, cols), rows));
        rLeft.setTo(new cv.Scalar(0));
        rLeft.delete();

        let rRight = thresh.roi(new cv.Rect(Math.max(0, cols - frameMargin), 0, Math.min(frameMargin, cols), rows));
        rRight.setTo(new cv.Scalar(0));
        rRight.delete();

        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let minX = cols, minY = rows, maxX = 0, maxY = 0;
        let foundAny = false;

        // Only count substantial ink contours (area > 200) to ignore noise
        for (let i = 0; i < contours.size(); i++) {
            const cnt = contours.get(i);
            const area = cv.contourArea(cnt);
            if (area > 200) {
                const rect = cv.boundingRect(cnt);
                minX = Math.min(minX, rect.x);
                minY = Math.min(minY, rect.y);
                maxX = Math.max(maxX, rect.x + rect.width);
                maxY = Math.max(maxY, rect.y + rect.height);
                foundAny = true;
            }
            cnt.delete();
        }

        thresh.delete();
        contours.delete();
        hierarchy.delete();

        if (!foundAny) return null;

        const boxX = Math.max(0, minX - padding);
        const boxY = Math.max(0, minY - padding);
        const boxW = Math.min(cols - boxX, (maxX - minX) + (padding * 2));
        const boxH = Math.min(rows - boxY, (maxY - minY) + (padding * 2));

        return {
            x: boxX,
            y: boxY,
            width: boxW,
            height: boxH,
            confidence: 0.95,
            type: 'ink_contour'
        };
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ContourRoiDetector;
}
