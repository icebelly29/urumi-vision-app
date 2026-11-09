/**
 * ColorSpectrumAnalyzer.js
 * Advanced CIELAB color space analysis with targeted thin ink stroke sampling.
 */
class ColorSpectrumAnalyzer {
    /**
     * Analyzes CIELAB color space across both the full canvas and thin ink strokes.
     * @param {cv.Mat} warpedMat - Perspective corrected bed image (RGBA)
     * @param {cv.Mat|null} userMaskMat - Optional ROI mask
     */
    static analyze(warpedMat, userMaskMat = null) {
        if (!cv || !cv.Mat) {
            throw new Error("OpenCV is not initialized.");
        }

        const rows = warpedMat.rows;
        const cols = warpedMat.cols;

        // 1. Convert RGBA -> RGB -> Lab
        let rgbMat = new cv.Mat();
        let labMat = new cv.Mat();
        cv.cvtColor(warpedMat, rgbMat, cv.COLOR_RGBA2RGB);
        cv.cvtColor(rgbMat, labMat, cv.COLOR_RGB2Lab);

        // 2. Extract thin ink stroke mask using adaptive threshold
        let grayMat = new cv.Mat();
        let blurred = new cv.Mat();
        cv.cvtColor(warpedMat, grayMat, cv.COLOR_RGBA2GRAY);
        cv.GaussianBlur(grayMat, blurred, new cv.Size(9, 9), 0);

        let inkThresh = new cv.Mat();
        cv.adaptiveThreshold(blurred, inkThresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 51, 8);

        // Darkness gate to eliminate light texture variations
        let darkGate = new cv.Mat();
        cv.threshold(grayMat, darkGate, 160, 255, cv.THRESH_BINARY_INV);
        cv.bitwise_and(inkThresh, darkGate, inkThresh);
        darkGate.delete();
        grayMat.delete();
        blurred.delete();

        // Clear perimeter margin (80px) to ignore ArUco frame edges
        const frameMargin = 80;
        let rTop = inkThresh.roi(new cv.Rect(0, 0, cols, Math.min(frameMargin, rows)));
        rTop.setTo(new cv.Scalar(0)); rTop.delete();
        let rBot = inkThresh.roi(new cv.Rect(0, Math.max(0, rows - frameMargin), cols, Math.min(frameMargin, rows)));
        rBot.setTo(new cv.Scalar(0)); rBot.delete();
        let rLeft = inkThresh.roi(new cv.Rect(0, 0, Math.min(frameMargin, cols), rows));
        rLeft.setTo(new cv.Scalar(0)); rLeft.delete();
        let rRight = inkThresh.roi(new cv.Rect(Math.max(0, cols - frameMargin), 0, Math.min(frameMargin, cols), rows));
        rRight.setTo(new cv.Scalar(0)); rRight.delete();

        if (userMaskMat && !userMaskMat.empty()) {
            cv.bitwise_and(inkThresh, userMaskMat, inkThresh);
        }

        // Split channels L, a, b
        let labPlanes = new cv.MatVector();
        cv.split(labMat, labPlanes);
        let lPlane = labPlanes.get(0);
        let aPlane = labPlanes.get(1);
        let bPlane = labPlanes.get(2);

        // 3. Sample Full Canvas Pixels
        const step = Math.max(3, Math.floor(Math.min(cols, rows) / 250));
        const fullSamples = [];
        const fullAbHistogram = new Array(64).fill(0).map(() => new Array(64).fill(0));

        const labData = labMat.data;
        const rgbData = rgbMat.data;
        const inkData = inkThresh.data;

        for (let y = 0; y < rows; y += step) {
            for (let x = 0; x < cols; x += step) {
                const idx = (y * cols + x) * 3;

                const L_raw = labData[idx];
                const a_raw = labData[idx + 1];
                const b_raw = labData[idx + 2];

                const R = rgbData[idx];
                const G = rgbData[idx + 1];
                const B = rgbData[idx + 2];

                const L_std = (L_raw * 100 / 255);
                const a_std = a_raw - 128;
                const b_std = b_raw - 128;

                const sampleObj = { R, G, B, L: L_std, a: a_std, b: b_std, L_raw, a_raw, b_raw };
                fullSamples.push(sampleObj);

                const binA = Math.max(0, Math.min(63, Math.floor(a_raw / 4)));
                const binB = Math.max(0, Math.min(63, Math.floor(b_raw / 4)));
                fullAbHistogram[binB][binA]++;
            }
        }

        // 4. Sample Thin Ink Stroke Pixels densely (dense scan ensures thin 1-2px lines are not missed)
        const inkSamples = [];
        const inkAbHistogram = new Array(64).fill(0).map(() => new Array(64).fill(0));
        const inkScanStep = Math.max(1, Math.floor(Math.min(cols, rows) / 600));

        for (let y = 0; y < rows; y += inkScanStep) {
            const rowOffset = y * cols;
            for (let x = 0; x < cols; x += inkScanStep) {
                if (inkData[rowOffset + x] > 0) {
                    const idx = (rowOffset + x) * 3;

                    const L_raw = labData[idx];
                    const a_raw = labData[idx + 1];
                    const b_raw = labData[idx + 2];

                    const R = rgbData[idx];
                    const G = rgbData[idx + 1];
                    const B = rgbData[idx + 2];

                    const L_std = (L_raw * 100 / 255);
                    const a_std = a_raw - 128;
                    const b_std = b_raw - 128;

                    const sampleObj = { R, G, B, L: L_std, a: a_std, b: b_std, L_raw, a_raw, b_raw };
                    inkSamples.push(sampleObj);

                    const binA = Math.max(0, Math.min(63, Math.floor(a_raw / 4)));
                    const binB = Math.max(0, Math.min(63, Math.floor(b_raw / 4)));
                    inkAbHistogram[binB][binA]++;
                }
            }
        }

        rgbMat.delete();
        inkThresh.delete();

        return {
            labMat,
            lPlane,
            aPlane,
            bPlane,
            fullSamples,
            inkSamples,
            fullAbHistogram,
            inkAbHistogram,
            samples: fullSamples,
            abHistogram: fullAbHistogram,
            width: cols,
            height: rows
        };
    }

    /**
     * Renders a 2D Chromaticity Spectrum (a* vs b*) heatmap onto a canvas.
     */
    static renderChromaticitySpectrum(abHistogram, canvasWidth = 260, canvasHeight = 260, title = "") {
        const canvas = document.createElement('canvas');
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        const ctx = canvas.getContext('2d');

        // Dark background
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        let maxDensity = 1;
        for (let b = 0; b < 64; b++) {
            for (let a = 0; a < 64; a++) {
                if (abHistogram[b][a] > maxDensity) maxDensity = abHistogram[b][a];
            }
        }

        const cellW = canvasWidth / 64;
        const cellH = canvasHeight / 64;

        for (let b = 0; b < 64; b++) {
            for (let a = 0; a < 64; a++) {
                const count = abHistogram[b][a];
                if (count === 0) continue;

                const intensity = Math.min(1, Math.log(count + 1) / Math.log(maxDensity + 1));
                const aVal = a * 4 - 128;
                const bVal = b * 4 - 128;

                let cr = Math.min(255, Math.max(20, 128 + aVal * 1.6));
                let cg = Math.min(255, Math.max(20, 128 - aVal * 0.9 + bVal * 0.3));
                let cb = Math.min(255, Math.max(20, 128 - bVal * 1.6));

                ctx.fillStyle = `rgba(${Math.round(cr)}, ${Math.round(cg)}, ${Math.round(cb)}, ${intensity * 0.9 + 0.15})`;
                const py = canvasHeight - (b + 1) * cellH;
                const px = a * cellW;
                ctx.fillRect(px, py, cellW + 0.5, cellH + 0.5);
            }
        }

        // Draw crosshair axes
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(canvasWidth / 2, 0);
        ctx.lineTo(canvasWidth / 2, canvasHeight);
        ctx.moveTo(0, canvasHeight / 2);
        ctx.lineTo(canvasWidth, canvasHeight / 2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Axis annotations
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px monospace';
        ctx.fillText('← Green (-a*)', 6, canvasHeight / 2 - 4);
        ctx.fillText('Red (+a*) →', canvasWidth - 78, canvasHeight / 2 - 4);
        ctx.fillText('↑ Yellow (+b*)', canvasWidth / 2 + 4, 12);
        ctx.fillText('↓ Blue (-b*)', canvasWidth / 2 + 4, canvasHeight - 6);

        if (title) {
            ctx.fillStyle = '#38bdf8';
            ctx.font = 'bold 11px sans-serif';
            ctx.fillText(title, 8, 14);
        }

        return canvas;
    }

    /**
     * Extracts dominant color clusters from samples.
     */
    static getDominantPalette(samples, numClusters = 5, isInkOnly = false) {
        if (!samples || samples.length === 0) return [];

        const quantizeStep = isInkOnly ? 16 : 22;
        const bins = {};

        for (const s of samples) {
            const rQ = Math.floor(s.R / quantizeStep) * quantizeStep;
            const gQ = Math.floor(s.G / quantizeStep) * quantizeStep;
            const bQ = Math.floor(s.B / quantizeStep) * quantizeStep;
            const key = `${rQ}_${gQ}_${bQ}`;

            if (!bins[key]) {
                bins[key] = { count: 0, rSum: 0, gSum: 0, bSum: 0, lSum: 0, aSum: 0, bLabSum: 0 };
            }
            bins[key].count++;
            bins[key].rSum += s.R;
            bins[key].gSum += s.G;
            bins[key].bSum += s.B;
            bins[key].lSum += s.L;
            bins[key].aSum += s.a;
            bins[key].bLabSum += s.b;
        }

        const sorted = Object.values(bins).sort((a, b) => b.count - a.count).slice(0, numClusters);
        const totalSamples = samples.length;

        return sorted.map((c, i) => {
            const avgR = Math.round(c.rSum / c.count);
            const avgG = Math.round(c.gSum / c.count);
            const avgB = Math.round(c.bSum / c.count);
            const avgL = (c.lSum / c.count).toFixed(1);
            const avgA = (c.aSum / c.count).toFixed(1);
            const avgBLab = (c.bLabSum / c.count).toFixed(1);
            const pct = ((c.count / totalSamples) * 100).toFixed(1);

            let hex = '#' + ((1 << 24) + (avgR << 16) + (avgG << 8) + avgB).toString(16).slice(1);

            let label = isInkOnly ? `Ink Pen Color ${i + 1}` : `Surface Color ${i + 1}`;

            const aNum = parseFloat(avgA);
            const bNum = parseFloat(avgBLab);
            const lNum = parseFloat(avgL);

            if (isInkOnly) {
                if (bNum < -6 || (avgB > avgR + 8 && avgB > avgG - 8)) {
                    label = "Blue Pen Ink (thru_cut)";
                } else if (aNum < -6 || (avgG > avgB + 8 && avgG > avgR - 5)) {
                    label = "Green Pen Ink (crease)";
                } else if (aNum > 10 || (avgR > avgG + 15 && avgR > avgB + 15)) {
                    label = "Red Pen Ink (score)";
                } else if (lNum < 32) {
                    label = "Black Pen Ink (thru_cut)";
                } else {
                    label = `Stroke Edge Tone ${i + 1}`;
                }
            } else {
                if (lNum < 28) label = "Black / Shadow";
                else if (bNum > 12 && aNum > -3) label = "Cardboard Surface";
                else if (aNum < -8) label = "Green Tint";
                else if (aNum > 14) label = "Red Tint";
                else if (bNum < -8) label = "Blue Tint";
                else if (lNum > 78) label = "Bed / Frame Highlight";
            }

            return {
                hex,
                rgb: `rgb(${avgR}, ${avgG}, ${avgB})`,
                r: avgR, g: avgG, b: avgB,
                L: avgL, a: avgA, bLab: avgBLab,
                percentage: pct,
                label
            };
        });
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = ColorSpectrumAnalyzer;
}
