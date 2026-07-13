class InkExtractor {
    constructor(colorProfiles) {
        this.colorProfiles = colorProfiles;
    }

    extractColorPaths(img, frameConfig, borderSizeMm, vectorizationMode = 'skeleton') {
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

        // STEP 2: Absolute darkness gate — only keep pixels with gray < 160.
        // This kills cardboard edges (gray~170) and paper shadows (gray~200)
        // while keeping all real ink (gray < 130 on any surface).
        let darkGate = new cv.Mat();
        cv.threshold(gray, darkGate, 160, 255, cv.THRESH_BINARY_INV);
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
            let samplePoints;
            if (shape.type === 'circle') {
                samplePoints = [];
                for (let a = 0; a < 12; a++) {
                    let angle = (a / 12) * Math.PI * 2;
                    samplePoints.push([
                        Math.round(shape.cx + shape.r * Math.cos(angle)),
                        Math.round(shape.cy + shape.r * Math.sin(angle))
                    ]);
                }
            } else {
                samplePoints = shape.points;
            }
            if (!samplePoints || samplePoints.length < 2) continue;

            if (shape.type === 'path') {
                let pathLen = 0;
                for (let k = 1; k < samplePoints.length; k++) {
                    let dx = samplePoints[k][0] - samplePoints[k - 1][0];
                    let dy = samplePoints[k][1] - samplePoints[k - 1][1];
                    pathLen += Math.sqrt(dx * dx + dy * dy);
                }
                if (pathLen < 20) continue;
            }

            let votes = { red: 0, green: 0, blue: 0, black: 0 };
            let debugHues = [];
            let sampleCount = Math.min(30, samplePoints.length);
            for (let j = 0; j < sampleCount; j++) {
                let idx = Math.floor(j * (samplePoints.length - 1) / Math.max(1, sampleCount - 1));
                let cpx = Math.round(samplePoints[idx][0]);
                let cpy = Math.round(samplePoints[idx][1]);

                // Search a 5x5 window but ONLY consider confirmed ink pixels (in allStrokes mask).
                let bestS = -1, bestH = 0, bestV = 0, bestGv = 255;
                for (let wy = -2; wy <= 2; wy++) {
                    for (let wx = -2; wx <= 2; wx++) {
                        let sx = Math.max(0, Math.min(cpx + wx, hsv.cols - 1));
                        let sy = Math.max(0, Math.min(cpy + wy, hsv.rows - 1));
                        if (allStrokes.data[sy * allStrokes.cols + sx] === 0) continue;
                        let off = (sy * hsv.cols + sx) * 3;
                        let cs = hsv.data[off + 1];
                        let cg = gray.data[sy * gray.cols + sx];
                        if (cs > bestS) {
                            bestS = cs; bestH = hsv.data[off]; bestV = hsv.data[off + 2]; bestGv = cg;
                        }
                    }
                }
                if (bestS < 0) continue;

                debugHues.push(`H=${bestH} S=${bestS} V=${bestV} gv=${bestGv}`);

                // Classification based on actual measured values:
                // Red arrow: H=4-9, S=200+, gv=45-67
                // Green triangle on cardboard: H=27-43, S=94-126, gv=56-75
                // Black circle: H=15-17, S=135-181, gv=26-37
                // Cardboard boundary: H=15-19, S=68-134, gv=104-130
                
                if (bestS > 25) {
                    if (bestH < 12 || bestH >= 168) {
                        votes.red++;
                    } else if (bestH >= 12 && bestH < 36) {
                        // Brown/orange zone: could be dark green ink OR cardboard edge
                        // Dark green ink: gv < 90 (it's genuinely dark)
                        // Cardboard edge: gv > 100 (it's lighter)
                        // Black circle on cardboard: gv < 40 (very dark, low S)
                        if (bestGv < 40) {
                            votes.black++;
                        } else if (bestGv < 90) {
                            votes.green++; // Dark green ink appearing as olive/brown
                        }
                        // else: gv >= 90 = cardboard surface, skip entirely
                    } else if (bestH >= 36 && bestH <= 115) {
                        if (bestH >= 105 && bestS > 100 && bestV > 120) {
                            votes.blue++;
                        } else {
                            votes.green++;
                        }
                    } else if (bestH > 115 && bestH <= 145) {
                        votes.blue++;
                    } else if (bestGv < 100) {
                        votes.black++;
                    }
                } else if (bestGv < 100) {
                    // Low saturation + very dark = true black ink
                    votes.black++;
                }
            }

            let total = votes.red + votes.green + votes.blue + votes.black;
            if (total < 2) continue;

            let dom = 'black', mx = votes.black;
            if (votes.red > mx) { dom = 'red'; mx = votes.red; }
            if (votes.green > mx) { dom = 'green'; mx = votes.green; }
            if (votes.blue > mx) { dom = 'blue'; mx = votes.blue; }

            console.log(`[Ink] Path ${i} (${shape.type}): ${dom} | votes: R=${votes.red} G=${votes.green} B=${votes.blue} K=${votes.black} | samples: ${debugHues.slice(0,5).join(', ')}`);

            if (dom === 'red') results.score.paths.push(shape);
            else if (dom === 'green') results.crease.paths.push(shape);
            else results.thru_cut.paths.push(shape);
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

        let areaFloor = Math.max(4, Math.floor(0.000002 * imgWidth * imgHeight));
        let spanFloor = Math.max(8, Math.floor(0.004 * Math.min(imgWidth, imgHeight)));

        let filteredMask = cv.Mat.zeros(preClosedStrokes.rows, preClosedStrokes.cols, cv.CV_8UC1);
        let labelsData = labels.data32S;
        let outData = filteredMask.data;
        let keepLabel = new Uint8Array(numLabels);

        for (let i = 1; i < numLabels; i++) {
            let area = stats.intPtr(i, cv.CC_STAT_AREA)[0];
            let compW = stats.intPtr(i, cv.CC_STAT_WIDTH)[0];
            let compH = stats.intPtr(i, cv.CC_STAT_HEIGHT)[0];
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
            let contours = new cv.MatVector();
            let hierarchy = new cv.Mat();
            cv.findContours(finalMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

            for (let i = 0; i < contours.size(); ++i) {
                let contour = contours.get(i);
                let area = cv.contourArea(contour);
                if (area < 100) { contour.delete(); continue; }

                let perimeter = cv.arcLength(contour, true);
                let approx = new cv.Mat();
                cv.approxPolyDP(contour, approx, 0.04 * perimeter, true);

                let isGeometric = false;
                let vertices = approx.rows;

                if (vertices === 3) {
                    let pts = [];
                    for (let j = 0; j < 3; j++) pts.push([approx.data32S[j * 2], approx.data32S[j * 2 + 1]]);
                    pts.push([...pts[0]]);
                    perfectShapes.push({ type: 'polygon', points: pts });
                    isGeometric = true;
                } else if (vertices === 4) {
                    let pts = [];
                    for (let j = 0; j < 4; j++) pts.push([approx.data32S[j * 2], approx.data32S[j * 2 + 1]]);
                    pts.push([...pts[0]]);
                    perfectShapes.push({ type: 'polygon', points: pts });
                    isGeometric = true;
                } else {
                    let circularity = 4 * Math.PI * (area / (perimeter * perimeter));
                    if (circularity > 0.8) {
                        let circle = cv.minEnclosingCircle(contour);
                        perfectShapes.push({ type: 'circle', cx: circle.center.x, cy: circle.center.y, r: circle.radius });
                        isGeometric = true;
                    }
                }

                if (isGeometric) {
                    let blobMask = cv.Mat.zeros(finalMask.rows, finalMask.cols, cv.CV_8U);
                    cv.drawContours(blobMask, contours, i, new cv.Scalar(255), cv.FILLED);
                    let maskROI = new cv.Mat();
                    cv.bitwise_and(finalMask, blobMask, maskROI);
                    let inkArea = cv.countNonZero(maskROI);
                    if (area > 0 && inkArea / area > 0.8) {
                        cv.drawContours(finalMask, contours, i, new cv.Scalar(0), cv.FILLED);
                    } else {
                        cv.drawContours(finalMask, contours, i, new cv.Scalar(0), 15);
                    }
                    blobMask.delete(); maskROI.delete();
                }
                approx.delete();
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
            if (diagonal < 80) {
                if (vectorizationMode === 'contour') return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 2), 0.5) };
                return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 8), 1.2) };
            }
            if (vectorizationMode === 'contour') return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 2), 1.0) };
            return { type: 'path', points: this._simplifyPath(this._smoothPath(path, 15), 2.5) };
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
                s.push([cur[i-1][0]*0.25 + cur[i][0]*0.5 + cur[i+1][0]*0.25, cur[i-1][1]*0.25 + cur[i][1]*0.5 + cur[i+1][1]*0.25]);
            }
            s.push(cur[cur.length - 1]);
            cur = s;
        }
        return cur;
    }

    _perpendicularDistance(pt, lineStart, lineEnd) {
        let x0 = pt[0], y0 = pt[1], x1 = lineStart[0], y1 = lineStart[1], x2 = lineEnd[0], y2 = lineEnd[1];
        let den = Math.sqrt((y2-y1)**2 + (x2-x1)**2);
        if (den === 0) return Math.sqrt((x0-x1)**2 + (y0-y1)**2);
        return Math.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) / den;
    }
}
