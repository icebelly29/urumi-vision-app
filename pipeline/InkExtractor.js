class InkExtractor {
    constructor(colorProfiles) {
        this.colorProfiles = colorProfiles;
    }

    extractColorPaths(img, frameConfig, borderSizeMm, vectorizationMode = 'skeleton', warpedMask = null) {
        let hsv = new cv.Mat();
        let gray = new cv.Mat();
        cv.cvtColor(img, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);

        // STEP 1: Adaptive threshold to find locally-dark pixels
        let blurred = new cv.Mat();
        cv.bilateralFilter(gray, blurred, 9, 75, 75);
        let adaptiveMask = new cv.Mat();
        cv.adaptiveThreshold(blurred, adaptiveMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 51, 10);
        blurred.delete();

        // STEP 2: Absolute darkness gate — only keep pixels with gray < 120.
        // The Lasso tool now handles workpiece edges, so we can be more permissive
        // to ensure black ink on white paper isn't fragmented.
        let darkGate = new cv.Mat();
        cv.threshold(gray, darkGate, 120, 255, cv.THRESH_BINARY_INV);
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

        // Morphological open to kill texture specks
        let openK = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
        cv.morphologyEx(allStrokes, allStrokes, cv.MORPH_OPEN, openK);
        openK.delete();

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
        let allPaths = this._processMaskToPaths(img.cols, img.rows, allStrokes, vectorizationMode);
        console.log(`[Ink] Total paths: ${allPaths.length}`);

        // STEP 6: Classify color by reading original HSV at path points
        let results = {
            thru_cut: { paths: [], color: this.colorProfiles.thru_cut.color },
            score: { paths: [], color: this.colorProfiles.score.color },
            crease: { paths: [], color: this.colorProfiles.crease.color }
        };

        for (let i = 0; i < allPaths.length; i++) {
            let shape = allPaths[i];
            let evalPoints = [];

            if (shape.type === 'circle') {
                for (let a = 0; a < 30; a++) {
                    let angle = (a / 30) * Math.PI * 2;
                    evalPoints.push([
                        Math.round(shape.cx + shape.r * Math.cos(angle)),
                        Math.round(shape.cy + shape.r * Math.sin(angle))
                    ]);
                }
            } else if (shape.points && shape.points.length >= 2) {
                let pathLen = 0;
                let segments = [];
                for (let k = 1; k < shape.points.length; k++) {
                    let dx = shape.points[k][0] - shape.points[k - 1][0];
                    let dy = shape.points[k][1] - shape.points[k - 1][1];
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    pathLen += dist;
                    segments.push({ dist, dx, dy, start: shape.points[k - 1] });
                }

                if (shape.type === 'path' && pathLen < 20) continue;

                let sampleCount = Math.min(30, Math.max(2, Math.floor(pathLen / 5)));
                let step = pathLen / Math.max(1, sampleCount - 1);
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
                    evalPoints.push([
                        Math.round(seg.start[0] + seg.dx * t),
                        Math.round(seg.start[1] + seg.dy * t)
                    ]);
                }
            }

            if (evalPoints.length < 2) continue;

            let pointColors = [];
            let debugHues = [];

            for (let j = 0; j < evalPoints.length; j++) {
                let cpx = evalPoints[j][0];
                let cpy = evalPoints[j][1];

                let bestR = 0, bestG = 0, bestB = 0, bestGv = 255;
                for (let wy = -1; wy <= 1; wy++) {
                    for (let wx = -1; wx <= 1; wx++) {
                        let sx = Math.max(0, Math.min(cpx + wx, img.cols - 1));
                        let sy = Math.max(0, Math.min(cpy + wy, img.rows - 1));
                        if (allStrokes.data[sy * allStrokes.cols + sx] === 0) continue;
                        let cg = gray.data[sy * gray.cols + sx];

                        if (cg < bestGv) {
                            bestGv = cg;
                            let offImg = (sy * img.cols + sx) * 4;
                            bestR = img.data[offImg];
                            bestG = img.data[offImg + 1];
                            bestB = img.data[offImg + 2];
                        }
                    }
                }
                if (bestGv === 255) continue;

                // Find local background color to adapt to shadows/lighting gradients
                let bgR = 180, bgG = 140, bgB = 100;
                for (let wx = 5; wx <= 20; wx++) {
                    let bx = Math.min(cpx + wx, img.cols - 1);
                    if (allStrokes.data[cpy * allStrokes.cols + bx] === 0) {
                        let offBg = (cpy * img.cols + bx) * 4;
                        bgR = img.data[offBg];
                        bgG = img.data[offBg + 1];
                        bgB = img.data[offBg + 2];
                        break;
                    }
                }

                // Von Kries White Balance: Equalize the local background to pure gray
                let bgAvg = (bgR + bgG + bgB) / 3;
                let multR = bgAvg / Math.max(1, bgR);
                let multG = bgAvg / Math.max(1, bgG);
                let multB = bgAvg / Math.max(1, bgB);

                let adjR = bestR * multR;
                let adjG = bestG * multG;
                let adjB = bestB * multB;

                debugHues.push(`aR=${adjR.toFixed(0)} aG=${adjG.toFixed(0)} aB=${adjB.toFixed(0)}`);

                // Hybrid Classification:
                // Cardboard is yellowish (high R, med G, low B). 
                // Black ink will naturally have raw R > G > B, so raw dominance falsely classifies Black as Red. 
                // We MUST use Von Kries `adjR` to correctly distinguish true Red from Black.
                // However, dark Green ink absorbs heavily, and the camera noise floor artificially inflates raw Blue.
                // Von Kries `multB` (e.g. 1.4x) explodes this blue noise, causing dark Green ink to falsely vote Blue.
                // Since Black ink inherently has raw G > B (because cardboard has G > B), 
                // we can safely use RAW G and RAW B to distinguish Green and Blue from Black without false positives!
                
                if (bestG > bestR && bestG > bestB && (bestG - Math.max(bestR, bestB) > 2)) {
                    pointColors.push('green');
                } else if (bestB > bestG && bestB > bestR && (bestB - Math.max(bestR, bestG) > 5)) {
                    pointColors.push('blue');
                } else if (adjR > adjG && adjR > adjB && (adjR - Math.max(adjG, adjB) > 15)) {
                    pointColors.push('red');
                } else {
                    pointColors.push('black');
                }
            }

            // Sliding window majority vote to smooth out color noise/artifacts
            let smoothedColors = [];
            const WINDOW_SIZE = 7; 
            for (let j = 0; j < pointColors.length; j++) {
                let votes = { red: 0, green: 0, blue: 0, black: 0 };
                for (let w = -WINDOW_SIZE; w <= WINDOW_SIZE; w++) {
                    let idx = Math.max(0, Math.min(pointColors.length - 1, j + w));
                    votes[pointColors[idx]]++;
                }
                let dom = 'black', mx = votes.black;
                if (votes.red > mx) { dom = 'red'; mx = votes.red; }
                if (votes.green > mx) { dom = 'green'; mx = votes.green; }
                if (votes.blue > mx) { dom = 'blue'; mx = votes.blue; }
                smoothedColors.push(dom);
            }

            // Fracture the path at color boundaries
            let currentColor = smoothedColors[0];
            let currentSegment = [evalPoints[0]];

            const pushSegment = (color, segment) => {
                if (segment.length < 5) return; // Ignore microscopic noise
                
                let layer = 'thru_cut';
                if (color === 'red') layer = 'score';
                else if (color === 'green') layer = 'crease';
                
                // Re-simplify the segment so we don't output thousands of dense evalPoints to SVG
                let simplified = this._simplifyPath(segment, 1.5);
                
                if (shape.type === 'circle') {
                    // A circle broken by colors must be drawn as paths, not <circle>
                    results[layer].paths.push({ type: 'path', points: simplified });
                } else {
                    results[layer].paths.push({ type: 'path', points: simplified });
                }
            };

            for (let j = 1; j < evalPoints.length; j++) {
                if (smoothedColors[j] === currentColor) {
                    currentSegment.push(evalPoints[j]);
                } else {
                    pushSegment(currentColor, currentSegment);
                    currentColor = smoothedColors[j];
                    currentSegment = [evalPoints[j]]; // Start new segment
                }
            }
            pushSegment(currentColor, currentSegment);
        }

        hsv.delete(); gray.delete(); allStrokes.delete();
        return results;
    }

    _processMaskToPaths(imgWidth, imgHeight, rawMask, vectorizationMode) {
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
            // We want to keep anything that is EITHER reasonably large (area) 
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

        let perfectShapes = [];
        let rawPaths = [];

        if (vectorizationMode === 'skeleton') {
            // Heavily close the mask just for geometric detection to bridge gaps in hand-drawn shapes
            let shapeMask = new cv.Mat();
            let shapeKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(15, 15));
            cv.morphologyEx(finalMask, shapeMask, cv.MORPH_CLOSE, shapeKernel);
            shapeKernel.delete();

            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(shapeMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
            shapeMask.delete();

            for (let i = 0; i < contours.size(); ++i) {
                let contour = contours.get(i);
                let area = cv.contourArea(contour);
                if (area < 100) { contour.delete(); continue; }

                // Use the convex hull to smooth out all hand-drawn wiggles
                let hull = new cv.Mat();
                cv.convexHull(contour, hull, false, true);
                let hullArea = cv.contourArea(hull);
                let solidity = area / (hullArea || 1);

                let isGeometric = false;

                // Only attempt to snap to perfect geometry if the shape is mostly convex.
                // A hollow triangle or circle has high solidity (~0.9+).
                // An arrow or text has very low solidity (< 0.4) and will be left as a path.
                if (solidity > 0.75) {
                    let hullPerimeter = cv.arcLength(hull, true);
                    let approx = new cv.Mat();
                    // 0.05 is highly forgiving for snapping to perfect corners
                    cv.approxPolyDP(hull, approx, 0.05 * hullPerimeter, true);

                    let vertices = approx.rows;
                    let circularity = 4 * Math.PI * (hullArea / (hullPerimeter * hullPerimeter));

                    // 0.75 allows for slightly squashed hand-drawn circles
                    if (circularity > 0.75) {
                        let circle = cv.minEnclosingCircle(contour);
                        perfectShapes.push({ type: 'circle', cx: circle.center.x, cy: circle.center.y, r: circle.radius });
                        isGeometric = true;
                    } else if (vertices >= 3 && vertices <= 8) {
                        for (let j = 0; j < vertices; j++) {
                            let p1 = [approx.data32S[j * 2], approx.data32S[j * 2 + 1]];
                            let p2 = (j === vertices - 1) 
                                ? [approx.data32S[0], approx.data32S[1]] 
                                : [approx.data32S[(j + 1) * 2], approx.data32S[(j + 1) * 2 + 1]];
                            
                            perfectShapes.push({ type: 'path', points: [p1, p2] });
                        }
                        isGeometric = true;
                    }
                    approx.delete();
                }

                if (isGeometric) {
                    let blobMask = cv.Mat.zeros(finalMask.rows, finalMask.cols, cv.CV_8U);
                    cv.drawContours(blobMask, contours, i, new cv.Scalar(255), cv.FILLED);
                    let maskROI = new cv.Mat();
                    cv.bitwise_and(finalMask, blobMask, maskROI);
                    let inkArea = cv.countNonZero(maskROI);

                    // If it's a solid shape, fill it to erase it. If hollow, draw over the stroke to erase it.
                    if (area > 0 && inkArea / area > 0.6) {
                        cv.drawContours(finalMask, contours, i, new cv.Scalar(0), cv.FILLED);
                    } else {
                        // Use a thick brush to erase the stroke so it doesn't get skeletonized
                        cv.drawContours(finalMask, contours, i, new cv.Scalar(0), 30);
                    }
                    blobMask.delete(); maskROI.delete();
                }

                hull.delete();
                contour.delete();
            }
            contours.delete();
            hierarchy.delete();
        }

        if (vectorizationMode === 'contour') {
            let rc = new cv.MatVector();
            let rh = new cv.Mat();
            cv.findContours(finalMask, rc, rh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
            for (let i = 0; i < rc.size(); ++i) {
                let contour = rc.get(i);
                if (cv.contourArea(contour) < 20) { contour.delete(); continue; }
                let pts = [];
                for (let j = 0; j < contour.rows; j++) {
                    pts.push([contour.data32S[j * 2], contour.data32S[j * 2 + 1]]);
                }
                if (pts.length > 0) pts.push([...pts[0]]);
                rawPaths.push(pts);
                contour.delete();
            }
            rc.delete(); rh.delete();
        } else {
            rawPaths = this._vectorizeAndSkeletonize(finalMask);
        }
        finalMask.delete();

        let smoothedPaths = rawPaths.map(path => {
            if (path.length < 2) return null;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let pt of path) {
                if (pt[0] < minX) minX = pt[0]; if (pt[0] > maxX) maxX = pt[0];
                if (pt[1] < minY) minY = pt[1]; if (pt[1] > maxY) maxY = pt[1];
            }
            let width = maxX - minX, height = maxY - minY;
            if (width > imgWidth * 0.45 || height > imgHeight * 0.45) return null;
            let maxDim = Math.max(width, height), minDim = Math.max(1, Math.min(width, height));
            if (maxDim > Math.min(imgWidth, imgHeight) * 0.15 && (maxDim / minDim) > 12) return null;

            let diagonal = Math.sqrt(width * width + height * height);

            // For contour mode, use minimal simplification to preserve the exact ink border
            if (vectorizationMode === 'contour') {
                return { type: 'path', points: this._simplifyPath(path, 1.0) };
            }

            // For skeleton mode, we want sharp lines. 
            // We use standard Douglas-Peucker simplification which naturally creates straight lines and sharp corners.
            // We remove the heavy _smoothPath moving-average because it destroys acute angles (like arrowheads).
            if (diagonal < 80) {
                // Small paths (text, details) - minimal smoothing to prevent jitter
                return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 2), 1.5) };
            }
            // Large paths (arrows, long lines) - no moving-average smoothing, just crisp simplification
            return { type: 'path', points: this._simplifyPath(path, 3.0) };
        }).filter(p => p !== null);

        return perfectShapes.concat(smoothedPaths);
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
