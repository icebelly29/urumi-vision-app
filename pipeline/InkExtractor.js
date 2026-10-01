class InkExtractor {
    constructor(colorProfiles) {
        this.colorProfiles = colorProfiles;
    }

    extractColorPaths(img, frameConfig, borderSizeMm, warpedMask = null) {
        let hsv = new cv.Mat();
        let gray = new cv.Mat();
        cv.cvtColor(img, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);

        // STEP 1: Adaptive threshold to find locally-dark pixels
        let blurred = new cv.Mat();
        cv.bilateralFilter(gray, blurred, 9, 75, 75);
        let adaptiveMask = new cv.Mat();
        cv.adaptiveThreshold(blurred, adaptiveMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 4);
        blurred.delete();

        // STEP 2: Absolute darkness gate — keep faint thin ink pixels (gray < 150)
        let darkGate = new cv.Mat();
        cv.threshold(gray, darkGate, 150, 255, cv.THRESH_BINARY_INV);
        let darkInk = new cv.Mat();
        cv.bitwise_and(adaptiveMask, darkGate, darkInk);
        adaptiveMask.delete(); darkGate.delete();

        // STEP 3: Colored ink masks for faint pens that may not pass the darkness gate
        let coloredInkMask = new cv.Mat();
        let redMask1 = new cv.Mat();
        let r1L = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(0, 60, 40, 0));
        let r1H = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(12, 255, 255, 0));
        cv.inRange(hsv, r1L, r1H, redMask1);
        let redMask2 = new cv.Mat();
        let r2L = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(168, 60, 40, 0));
        let r2H = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(179, 255, 255, 0));
        cv.inRange(hsv, r2L, r2H, redMask2);
        let greenMask = new cv.Mat();
        let gL = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(36, 60, 40, 0));
        let gH = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(90, 255, 255, 0));
        cv.inRange(hsv, gL, gH, greenMask);
        let blueMask = new cv.Mat();
        let bL = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(100, 80, 40, 0));
        let bH = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(140, 255, 255, 0));
        cv.inRange(hsv, bL, bH, blueMask);

        cv.bitwise_or(redMask1, redMask2, coloredInkMask);
        cv.bitwise_or(coloredInkMask, greenMask, coloredInkMask);
        cv.bitwise_or(coloredInkMask, blueMask, coloredInkMask);
        redMask1.delete(); redMask2.delete(); greenMask.delete(); blueMask.delete();
        r1L.delete(); r1H.delete(); r2L.delete(); r2H.delete();
        gL.delete(); gH.delete(); bL.delete(); bH.delete();

        // STEP 4: Combine dark ink + colored ink
        let allStrokes = new cv.Mat();
        cv.bitwise_or(darkInk, coloredInkMask, allStrokes);
        darkInk.delete(); coloredInkMask.delete();

        // Morphological close to bridge thin gaps without eroding thin strokes
        let closeK = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(2, 2));
        cv.morphologyEx(allStrokes, allStrokes, cv.MORPH_CLOSE, closeK);
        closeK.delete();

        // 4c: Apply user Lasso Mask (if provided)
        if (warpedMask && !warpedMask.empty()) {
            cv.bitwise_and(allStrokes, warpedMask, allStrokes);
        }

        // STEP 4b: Border exclusion — erase a thin strip around the image edge.
        // Cardboard edges and warping artifacts always appear at the periphery.
        let borderPx = Math.max(8, Math.floor(0.01 * Math.min(img.cols, img.rows)));
        cv.rectangle(allStrokes, new cv.Point(0, 0), new cv.Point(img.cols, borderPx), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(0, img.rows - borderPx), new cv.Point(img.cols, img.rows), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(0, 0), new cv.Point(borderPx, img.rows), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(img.cols - borderPx, 0), new cv.Point(img.cols, img.rows), new cv.Scalar(0), cv.FILLED);

        console.log(`[Ink] Mask pixels: ${cv.countNonZero(allStrokes)}, image: ${img.cols}x${img.rows}`);

        // STEP 5: Extract paths
        let allPaths = this._processMaskToPaths(img.cols, img.rows, allStrokes);
        console.log(`[Ink] Total paths: ${allPaths.length}`);

        // STEP 6: Classify color by reading original image at path points.
        // Each skeleton path is one continuous stroke — classify the ENTIRE path
        // with a single color via majority vote. No fracturing.
        let results = {
            thru_cut: { paths: [], color: this.colorProfiles.thru_cut.color },
            score: { paths: [], color: this.colorProfiles.score.color },
            crease: { paths: [], color: this.colorProfiles.crease.color }
        };

        for (let i = 0; i < allPaths.length; i++) {
            let shape = allPaths[i];
            if (!shape.points || shape.points.length < 2) continue;

            // Sample points along the path for color classification
            let pathLen = 0;
            let segments = [];
            for (let k = 1; k < shape.points.length; k++) {
                let dx = shape.points[k][0] - shape.points[k - 1][0];
                let dy = shape.points[k][1] - shape.points[k - 1][1];
                let dist = Math.sqrt(dx * dx + dy * dy);
                pathLen += dist;
                segments.push({ dist, dx, dy, start: shape.points[k - 1] });
            }
            if (pathLen < 10) continue;

            let sampleCount = Math.min(50, Math.max(5, Math.floor(pathLen / 3)));
            let step = pathLen / Math.max(1, sampleCount - 1);
            let colorVotes = { red: 0, green: 0, blue: 0, black: 0 };
            let currentSegDist = 0;
            let segIdx = 0;

            for (let j = 0; j < sampleCount; j++) {
                let targetDist = j * step;
                while (segIdx < segments.length && currentSegDist + segments[segIdx].dist < targetDist - 0.001) {
                    currentSegDist += segments[segIdx].dist;
                    segIdx++;
                }
                let seg = segments[Math.min(segIdx, segments.length - 1)];
                let t = seg.dist > 0 ? (targetDist - currentSegDist) / seg.dist : 0;
                t = Math.max(0, Math.min(1, t));
                let cpx = Math.round(seg.start[0] + seg.dx * t);
                let cpy = Math.round(seg.start[1] + seg.dy * t);

                // Search a SMALL radius (3px) for the nearest ink pixel.
                // Small radius prevents bleeding into adjacent shapes.
                let bestR = 0, bestG = 0, bestB = 0, bestGv = 255;
                let bestX = cpx, bestY = cpy;
                for (let wy = -3; wy <= 3; wy++) {
                    for (let wx = -3; wx <= 3; wx++) {
                        let sx = Math.max(0, Math.min(cpx + wx, img.cols - 1));
                        let sy = Math.max(0, Math.min(cpy + wy, img.rows - 1));
                        if (allStrokes.data[sy * allStrokes.cols + sx] === 0) continue;
                        let cg = gray.data[sy * gray.cols + sx];
                        if (cg < bestGv) {
                            bestGv = cg;
                            bestX = sx;
                            bestY = sy;
                            let offImg = (sy * img.cols + sx) * 4;
                            bestR = img.data[offImg];
                            bestG = img.data[offImg + 1];
                            bestB = img.data[offImg + 2];
                        }
                    }
                }
                if (bestGv === 255) continue; // No ink found near this point

                // Primary: HSV classification.
                // Only trust hue if saturation is clearly above the cardboard background (S~100-130).
                let hOff = (bestY * hsv.cols + bestX) * 3;
                let h_val = hsv.data[hOff];
                let s_val = hsv.data[hOff + 1];
                let v_val = hsv.data[hOff + 2];

                if (s_val > 80 && v_val > 40) {
                    if (h_val <= 10 || h_val >= 170) { colorVotes.red++; continue; }
                    else if (h_val >= 35 && h_val <= 85) { colorVotes.green++; continue; }
                    else if (h_val >= 95 && h_val <= 135) { colorVotes.blue++; continue; }
                }

                // Fallback: Von Kries white balance for faint/dark ink
                let bgR = 180, bgG = 140, bgB = 100;
                let foundBg = false;
                for (let r = 5; r <= 25 && !foundBg; r += 5) {
                    let dirs = [[cpx+r,cpy],[cpx-r,cpy],[cpx,cpy+r],[cpx,cpy-r]];
                    for (let d of dirs) {
                        let bx = Math.max(0, Math.min(d[0], img.cols - 1));
                        let by = Math.max(0, Math.min(d[1], img.rows - 1));
                        if (allStrokes.data[by * allStrokes.cols + bx] === 0) {
                            let off = (by * img.cols + bx) * 4;
                            bgR = img.data[off]; bgG = img.data[off+1]; bgB = img.data[off+2];
                            foundBg = true; break;
                        }
                    }
                }

                let bgAvg = (bgR + bgG + bgB) / 3;
                let adjR = bestR * (bgAvg / Math.max(1, bgR));
                let adjG = bestG * (bgAvg / Math.max(1, bgG));
                let adjB = bestB * (bgAvg / Math.max(1, bgB));

                if ((adjG > adjR) && (bestG > bestB - 5)) colorVotes.green++;
                else if ((bestB > bestG + 10) && (bestB > bestR + 5)) colorVotes.blue++;
                else if ((adjR > adjG + 5) && (adjR > adjB + 5)) colorVotes.red++;
                else colorVotes.black++;
            }

            // Determine the dominant color for this entire path
            let dominantColor = 'black';
            let maxVotes = colorVotes.black;
            if (colorVotes.red > maxVotes) { dominantColor = 'red'; maxVotes = colorVotes.red; }
            if (colorVotes.green > maxVotes) { dominantColor = 'green'; maxVotes = colorVotes.green; }
            if (colorVotes.blue > maxVotes) { dominantColor = 'blue'; maxVotes = colorVotes.blue; }

            let layer = 'thru_cut';
            if (dominantColor === 'red') layer = 'score';
            else if (dominantColor === 'green') layer = 'crease';

            // Push the ORIGINAL shape with its original points — preserves shape accuracy
            results[layer].paths.push(shape);
        }

        hsv.delete(); gray.delete(); allStrokes.delete();
        return results;
    }

    _processMaskToPaths(imgWidth, imgHeight, rawMask) {
        let preClosedStrokes = new cv.Mat();
        let preCloseKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(rawMask, preClosedStrokes, cv.MORPH_CLOSE, preCloseKernel);
        preCloseKernel.delete();

        let labels = new cv.Mat();
        let stats = new cv.Mat();
        let centroids = new cv.Mat();
        let numLabels = cv.connectedComponentsWithStats(preClosedStrokes, labels, stats, centroids, 8, cv.CV_32S);

        let areaFloor = Math.max(100, Math.floor(0.00005 * imgWidth * imgHeight));
        let spanFloor = Math.max(20, Math.floor(0.01 * Math.min(imgWidth, imgHeight)));

        let filteredMask = cv.Mat.zeros(preClosedStrokes.rows, preClosedStrokes.cols, cv.CV_8UC1);
        let labelsData = labels.data32S;
        let outData = filteredMask.data;
        let keepLabel = new Uint8Array(numLabels);

        for (let i = 1; i < numLabels; i++) {
            let area = stats.intPtr(i, cv.CC_STAT_AREA)[0];
            let compW = stats.intPtr(i, cv.CC_STAT_WIDTH)[0];
            let compH = stats.intPtr(i, cv.CC_STAT_HEIGHT)[0];
            // Keep anything that is EITHER reasonably large (area) 
            // OR reasonably long (span) to capture thin ink strokes.
            if (area >= areaFloor || Math.max(compW, compH) >= spanFloor) {
                keepLabel[i] = 255;
            }
        }
        for (let i = 0; i < labelsData.length; i++) {
            outData[i] = keepLabel[labelsData[i]] || 0;
        }

        preClosedStrokes.delete(); labels.delete(); stats.delete(); centroids.delete();

        let finalMask = new cv.Mat();
        let closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(filteredMask, finalMask, cv.MORPH_CLOSE, closeKernel);
        closeKernel.delete();
        filteredMask.delete();

        // Skeletonize the entire mask directly — no geometric shape detection.
        // The skeletonizer traces the center-line of every ink stroke faithfully.
        let rawPaths = this._vectorizeAndSkeletonize(finalMask);
        finalMask.delete();
        console.log(`[Ink] Raw skeleton paths: ${rawPaths.length}`);

        let smoothedPaths = rawPaths.map(path => {
            if (path.length < 2) return null;

            let diagonal = 0;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let pt of path) {
                if (pt[0] < minX) minX = pt[0]; if (pt[0] > maxX) maxX = pt[0];
                if (pt[1] < minY) minY = pt[1]; if (pt[1] > maxY) maxY = pt[1];
            }
            let width = maxX - minX, height = maxY - minY;
            diagonal = Math.sqrt(width * width + height * height);

            // For skeleton mode, use sharp lines with Douglas-Peucker simplification.
            if (diagonal < 80) {
                // Small paths (text, details) - minimal smoothing to prevent jitter
                return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 2), 1.5) };
            }
            // Large paths (arrows, long lines) - no moving-average smoothing, just crisp simplification
            return { type: 'path', points: this._simplifyPath(path, 2.0) };
        }).filter(p => p !== null);

        return smoothedPaths;
    }

    _vectorizeAndSkeletonize(maskMat) {
        let data = maskMat.data;
        let boolArray = new Array(maskMat.cols * maskMat.rows);
        for (let i = 0; i < data.length; i++) boolArray[i] = data[i] > 128 ? 1 : 0;
        return TraceSkeleton.fromBoolArray(boolArray, maskMat.cols, maskMat.rows).polylines;
    }

    _simplifyPath(path, epsilon) {
        if (path.length <= 2) return path;
        let dmax = 0, index = 0;
        const end = path.length - 1;
        for (let i = 1; i < end; i++) {
            let d = this._perpendicularDistance(path[i], path[0], path[end]);
            if (d > dmax) { index = i; dmax = d; }
        }
        if (dmax > epsilon) {
            let r1 = this._simplifyPath(path.slice(0, index + 1), epsilon);
            let r2 = this._simplifyPath(path.slice(index, end + 1), epsilon);
            return r1.slice(0, r1.length - 1).concat(r2);
        }
        return [path[0], path[end]];
    }

    _smoothPath(path, iterations = 5) {
        if (path.length <= 2) return path;
        let cur = path;
        for (let iter = 0; iter < iterations; iter++) {
            let s = [cur[0]];
            for (let i = 1; i < cur.length - 1; i++) {
                s.push([cur[i - 1][0] * 0.25 + cur[i][0] * 0.5 + cur[i + 1][0] * 0.25, cur[i - 1][1] * 0.25 + cur[i][1] * 0.5 + cur[i + 1][1] * 0.25]);
            }
            s.push(cur[cur.length - 1]);
            cur = s;
        }
        return cur;
    }

    _perpendicularDistance(pt, lineStart, lineEnd) {
        let x0 = pt[0], y0 = pt[1], x1 = lineStart[0], y1 = lineStart[1], x2 = lineEnd[0], y2 = lineEnd[1];
        let den = Math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2);
        if (den === 0) return Math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2);
        return Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / den;
    }
}
