import TraceSkeleton from './trace_skeleton.js';

export class InkExtractor {
    constructor(colorProfiles) {
        this.colorProfiles = colorProfiles || {
            thru_cut: { layer: 'thru_cut', color: '#3b82f6' },
            score: { layer: 'score', color: '#ef4444' },
            crease: { layer: 'crease', color: '#22c55e' }
        };
    }

    extractColorPaths(img, frameConfig, borderSizeMm, warpedMask = null, enabledColors = null) {
        const activePens = enabledColors || { blue: true, red: true, green: true, black: true };

        let rgb = new cv.Mat();
        let lab = new cv.Mat();
        let hsv = new cv.Mat();
        let gray = new cv.Mat();
        cv.cvtColor(img, rgb, cv.COLOR_RGBA2RGB);
        cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);
        cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
        cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);

        let labPlanes = new cv.MatVector();
        cv.split(lab, labPlanes);
        let lChan = labPlanes.get(0);
        let aChan = labPlanes.get(1);
        let bChan = labPlanes.get(2);

        // STEP 1: Adaptive threshold + flat-field difference to capture both dark and faint ballpoint strokes
        let blurred = new cv.Mat();
        cv.bilateralFilter(gray, blurred, 7, 12, 5);
        let adaptiveMask = new cv.Mat();
        cv.adaptiveThreshold(blurred, adaptiveMask, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 41, 4);
        blurred.delete();

        // STEP 2: Darkness gate relative to cardboard background (keep faint pen lines)
        let darkGate = new cv.Mat();
        cv.threshold(gray, darkGate, 175, 255, cv.THRESH_BINARY_INV);
        let darkInk = new cv.Mat();
        cv.bitwise_and(adaptiveMask, darkGate, darkInk);
        adaptiveMask.delete(); darkGate.delete();

        // STEP 3: Colored ink masks for red, green, blue pens
        let coloredInkMask = cv.Mat.zeros(hsv.rows, hsv.cols, cv.CV_8UC1);
        if (activePens.red) {
            let redMask1 = new cv.Mat(), redMask2 = new cv.Mat();
            let r1L = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(0, 50, 40, 0));
            let r1H = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(12, 255, 255, 0));
            cv.inRange(hsv, r1L, r1H, redMask1);
            let r2L = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(168, 50, 40, 0));
            let r2H = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(179, 255, 255, 0));
            cv.inRange(hsv, r2L, r2H, redMask2);
            cv.bitwise_or(coloredInkMask, redMask1, coloredInkMask);
            cv.bitwise_or(coloredInkMask, redMask2, coloredInkMask);
            redMask1.delete(); redMask2.delete(); r1L.delete(); r1H.delete(); r2L.delete(); r2H.delete();
        }
        if (activePens.green) {
            let greenMask = new cv.Mat();
            let gL = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(35, 45, 40, 0));
            let gH = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(90, 255, 255, 0));
            cv.inRange(hsv, gL, gH, greenMask);
            cv.bitwise_or(coloredInkMask, greenMask, coloredInkMask);
            greenMask.delete(); gL.delete(); gH.delete();
        }
        if (activePens.blue) {
            let blueMask = new cv.Mat();
            let bL = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(95, 65, 40, 0));
            let bH = new cv.Mat(hsv.rows, hsv.cols, hsv.type(), new cv.Scalar(145, 255, 255, 0));
            cv.inRange(hsv, bL, bH, blueMask);
            cv.bitwise_or(coloredInkMask, blueMask, coloredInkMask);
            blueMask.delete(); bL.delete(); bH.delete();
        }

        // STEP 4: Combine dark ink + colored ink
        let allStrokes = new cv.Mat();
        if (activePens.black) {
            cv.bitwise_or(darkInk, coloredInkMask, allStrokes);
        } else {
            allStrokes = coloredInkMask.clone();
        }
        darkInk.delete(); coloredInkMask.delete();

        // 3x3 Morphological close to bridge 1-px micro-gaps without merging adjacent tick marks
        let closeK = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3));
        cv.morphologyEx(allStrokes, allStrokes, cv.MORPH_CLOSE, closeK);
        closeK.delete();

        // Apply user Lasso / ROI Mask (if provided)
        if (warpedMask && !warpedMask.empty()) {
            let maskEr = new cv.Mat();
            let erK = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(15, 15));
            cv.erode(warpedMask, maskEr, erK);
            cv.bitwise_and(allStrokes, maskEr, allStrokes);
            maskEr.delete(); erK.delete();
        }

        // Border exclusion
        let borderPx = Math.max(8, Math.floor(0.01 * Math.min(img.cols, img.rows)));
        cv.rectangle(allStrokes, new cv.Point(0, 0), new cv.Point(img.cols, borderPx), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(0, img.rows - borderPx), new cv.Point(img.cols, img.rows), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(0, 0), new cv.Point(borderPx, img.rows), new cv.Scalar(0), cv.FILLED);
        cv.rectangle(allStrokes, new cv.Point(img.cols - borderPx, 0), new cv.Point(img.cols, img.rows), new cv.Scalar(0), cv.FILLED);

        // Estimate background CIE LAB median and local 25-px Gaussian background maps
        const validA = [], validB = [];
        for (let i = 0; i < lChan.data.length; i += 16) {
            if (!warpedMask || warpedMask.empty() || warpedMask.data[i] > 0) {
                validA.push(aChan.data[i]);
                validB.push(bChan.data[i]);
            }
        }
        validA.sort((x, y) => x - y);
        validB.sort((x, y) => x - y);
        const medA = validA.length ? validA[Math.floor(validA.length / 2)] : 134;
        const medB = validB.length ? validB[Math.floor(validB.length / 2)] : 150;

        let bgAMat = new cv.Mat(), bgBMat = new cv.Mat();
        cv.GaussianBlur(aChan, bgAMat, new cv.Size(0, 0), 25, 25);
        cv.GaussianBlur(bChan, bgBMat, new cv.Size(0, 0), 25, 25);

        // STEP 5: Extract paths via topology-preserving skeletonization and corner-aware simplification
        let allPaths = this._processMaskToPaths(img.cols, img.rows, allStrokes);

        // STEP 6: Classify color by reading CIE LAB chromaticity shift relative to local cardboard background
        let results = {
            thru_cut: { paths: [], color: this.colorProfiles.thru_cut.color },
            score: { paths: [], color: this.colorProfiles.score.color },
            crease: { paths: [], color: this.colorProfiles.crease.color }
        };

        for (let i = 0; i < allPaths.length; i++) {
            let shape = allPaths[i];
            let evalPoints = [];

            if (shape.type === 'circle') {
                for (let a = 0; a < 36; a++) {
                    let angle = (a / 36) * Math.PI * 2;
                    evalPoints.push([
                        Math.round(shape.cx + shape.r * Math.cos(angle)),
                        Math.round(shape.cy + shape.r * Math.sin(angle))
                    ]);
                }
            } else if (shape.points && shape.points.length >= 2) {
                let pathLen = 0;
                for (let k = 1; k < shape.points.length; k++) {
                    pathLen += Math.hypot(shape.points[k][0] - shape.points[k - 1][0], shape.points[k][1] - shape.points[k - 1][1]);
                }
                if (pathLen < 16) continue;
                evalPoints = this._densifyPath(shape.points, 3.0);
            }

            if (evalPoints.length < 2) continue;

            // Trim junction endpoints on open paths so touching strokes of different colors do not bleed votes
            let corePts = evalPoints;
            const endGap = Math.hypot(evalPoints[evalPoints.length - 1][0] - evalPoints[0][0], evalPoints[evalPoints.length - 1][1] - evalPoints[0][1]);
            if (shape.type !== 'circle' && endGap > 10.0 && evalPoints.length >= 12) {
                const trim = Math.max(2, Math.min(8, Math.floor(evalPoints.length / 8)));
                if (evalPoints.length > 2 * trim + 2) {
                    corePts = evalPoints.slice(trim, evalPoints.length - trim);
                }
            }

            const votes = { red: 0, green: 0, blue: 0, black: 0 };
            const lData = lChan.data, aData = aChan.data, bData = bChan.data;
            const bgAData = bgAMat.data, bgBData = bgBMat.data;
            const cols = img.cols, rows = img.rows;
            let nSamples = 0;

            for (let j = 0; j < corePts.length; j++) {
                const cpx = Math.min(cols - 1, Math.max(0, Math.round(corePts[j][0])));
                const cpy = Math.min(rows - 1, Math.max(0, Math.round(corePts[j][1])));
                let minL = 999, bestDa = 0, bestDb = 0;
                for (let wy = -2; wy <= 2; wy++) {
                    const sy = Math.min(rows - 1, Math.max(0, cpy + wy));
                    for (let wx = -2; wx <= 2; wx++) {
                        const sx = Math.min(cols - 1, Math.max(0, cpx + wx));
                        const idx = sy * cols + sx;
                        const lv = lData[idx];
                        if (lv < minL) {
                            minL = lv;
                            const aVal = aData[idx], bVal = bData[idx];
                            bestDa = 0.5 * (aVal - bgAData[idx]) + 0.5 * (aVal - medA);
                            bestDb = 0.5 * (bVal - bgBData[idx]) + 0.5 * (bVal - medB);
                        }
                    }
                }
                nSamples++;
                if (bestDa >= 5.0 && bestDa >= Math.abs(bestDb) * 0.40) {
                    votes.red++;
                } else if (bestDa <= -4.5 && (bestDb >= -5.5 || bestDa <= bestDb * 0.65)) {
                    votes.green++;
                } else if (bestDb <= -8.0) {
                    votes.blue++;
                } else {
                    votes.black++;
                }
            }

            const total = Math.max(1, nSamples);
            let dominant = 'black';
            if (votes.red / total >= 0.25 && votes.red >= votes.green && votes.red >= votes.blue) dominant = 'red';
            else if (votes.green / total >= 0.25 && votes.green > votes.red && votes.green >= votes.blue) dominant = 'green';
            else if (votes.blue / total >= 0.25) dominant = 'blue';

            if (!activePens[dominant]) continue;

            let layer = 'thru_cut';
            if (dominant === 'red') layer = 'score';
            else if (dominant === 'green') layer = 'crease';

            results[layer].paths.push(shape);
        }

        rgb.delete(); lab.delete(); hsv.delete(); gray.delete();
        lChan.delete(); aChan.delete(); bChan.delete(); labPlanes.delete();
        bgAMat.delete(); bgBMat.delete(); allStrokes.delete();
        return results;
    }

    _processMaskToPaths(imgWidth, imgHeight, rawMask) {
        let labels = new cv.Mat();
        let stats = new cv.Mat();
        let centroids = new cv.Mat();
        let numLabels = cv.connectedComponentsWithStats(rawMask, labels, stats, centroids, 8, cv.CV_32S);

        let areaFloor = 24;
        let spanFloor = 18;

        let filteredMask = cv.Mat.zeros(rawMask.rows, rawMask.cols, cv.CV_8UC1);
        let labelsData = labels.data32S;
        let outData = filteredMask.data;
        let keepLabel = new Uint8Array(numLabels);

        for (let i = 1; i < numLabels; i++) {
            let area = stats.intPtr(i, cv.CC_STAT_AREA)[0];
            let compW = stats.intPtr(i, cv.CC_STAT_WIDTH)[0];
            let compH = stats.intPtr(i, cv.CC_STAT_HEIGHT)[0];
            let span = Math.hypot(compW, compH);
            if (area >= areaFloor && span >= spanFloor) {
                keepLabel[i] = 255;
            }
        }
        for (let i = 0; i < labelsData.length; i++) {
            outData[i] = keepLabel[labelsData[i]] || 0;
        }
        labels.delete(); stats.delete(); centroids.delete();

        // Fill small interior pinholes (< 180 px^2) at thick stroke junctions so skeletonization does not create tiny loops
        let inv = new cv.Mat();
        cv.bitwise_not(filteredMask, inv);
        let hLbl = new cv.Mat(), hStats = new cv.Mat(), hCent = new cv.Mat();
        let nHoles = cv.connectedComponentsWithStats(inv, hLbl, hStats, hCent, 4, cv.CV_32S);
        let fillHole = new Uint8Array(nHoles);
        for (let h = 1; h < nHoles; h++) {
            const hArea = hStats.intPtr(h, cv.CC_STAT_AREA)[0];
            const hLeft = hStats.intPtr(h, cv.CC_STAT_LEFT)[0];
            const hTop = hStats.intPtr(h, cv.CC_STAT_TOP)[0];
            const hW = hStats.intPtr(h, cv.CC_STAT_WIDTH)[0];
            const hH = hStats.intPtr(h, cv.CC_STAT_HEIGHT)[0];
            const touchesBorder = (hLeft <= 0 || hTop <= 0 || (hLeft + hW) >= imgWidth || (hTop + hH) >= imgHeight);
            if (!touchesBorder && hArea < 180) fillHole[h] = 1;
        }
        let hLblData = hLbl.data32S;
        for (let i = 0; i < hLblData.length; i++) {
            if (fillHole[hLblData[i]]) outData[i] = 255;
        }
        inv.delete(); hLbl.delete(); hStats.delete(); hCent.delete();

        let rawPaths = this._vectorizeAndSkeletonize(filteredMask);
        filteredMask.delete();

        let outShapes = [];
        for (let path of rawPaths) {
            if (!path || path.length < 2) continue;
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            let pathLen = 0;
            for (let i = 0; i < path.length; i++) {
                const pt = path[i];
                if (pt[0] < minX) minX = pt[0];
                if (pt[0] > maxX) maxX = pt[0];
                if (pt[1] < minY) minY = pt[1];
                if (pt[1] > maxY) maxY = pt[1];
                if (i > 0) pathLen += Math.hypot(pt[0] - path[i - 1][0], pt[1] - path[i - 1][1]);
            }
            const width = maxX - minX, height = maxY - minY;
            if (width > imgWidth * 0.65 || height > imgHeight * 0.65) continue;
            if (pathLen < 16) continue;

            const endGap = Math.hypot(path[path.length - 1][0] - path[0][0], path[path.length - 1][1] - path[0][1]);
            const isClosed = endGap <= 10.0 && pathLen >= 45.0;

            // Check if closed loop is a true circle (without turning sharp triangles/squares into circles)
            const simp = this._simplifyPath(path, isClosed ? 2.0 : 1.2);
            if (isClosed && simp.length >= 6) {
                let cx = 0, cy = 0;
                for (let pt of path) { cx += pt[0]; cy += pt[1]; }
                cx /= path.length; cy /= path.length;
                let rSum = 0;
                for (let pt of path) rSum += Math.hypot(pt[0] - cx, pt[1] - cy);
                const rMean = rSum / path.length;
                let errSum = 0;
                for (let pt of path) errSum += Math.abs(Math.hypot(pt[0] - cx, pt[1] - cy) - rMean);
                const meanErr = errSum / path.length;
                if (rMean >= 10.0 && meanErr / rMean < 0.065) {
                    outShapes.push({ type: 'circle', cx: +cx.toFixed(2), cy: +cy.toFixed(2), r: +rMean.toFixed(2) });
                    continue;
                }
            }

            if (isClosed && simp.length >= 3) {
                const f = simp[0], l = simp[simp.length - 1];
                if (Math.hypot(f[0] - l[0], f[1] - l[1]) > 1.0) {
                    simp.push([f[0], f[1]]);
                }
            }
            outShapes.push({ type: 'path', points: simp });
        }

        return outShapes;
    }

    _vectorizeAndSkeletonize(maskMat) {
        const w = maskMat.cols, h = maskMat.rows;
        const data = maskMat.data;
        const boolArray = new Array(w * h);
        for (let i = 0; i < data.length; i++) boolArray[i] = data[i] > 128 ? 1 : 0;

        // 1. Thin to 1-px skeleton using Zhang-Suen
        TraceSkeleton.thinningZS(boolArray, w, h);

        // 2. Split at junctions (degree >= 3) and trace clean chains without spiderweb shortcuts
        return this._extractCleanChains(boolArray, w, h);
    }

    _extractCleanChains(skel, W, H) {
        const dirs = [-W - 1, -W, -W + 1, -1, 1, W - 1, W, W + 1];
        const deg = new Uint8Array(W * H);

        for (let y = 1; y < H - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                const idx = y * W + x;
                if (!skel[idx]) continue;
                let count = 0;
                for (let d = 0; d < 8; d++) {
                    if (skel[idx + dirs[d]]) count++;
                }
                deg[idx] = count;
            }
        }

        const work = new Uint8Array(W * H);
        const juncMask = new Uint8Array(W * H);
        for (let i = 0; i < skel.length; i++) {
            if (skel[i]) {
                if (deg[i] >= 3) juncMask[i] = 1;
                else work[i] = 1;
            }
        }

        const visited = new Uint8Array(W * H);
        const chains = [];
        const n8 = [-W, 1, W, -1, -W - 1, -W + 1, W + 1, W - 1];

        const findJunctionNeighbor = (idx) => {
            for (let d = 0; d < 8; d++) {
                const nb = idx + n8[d];
                if (nb >= 0 && nb < W * H && juncMask[nb]) {
                    return [nb % W, Math.floor(nb / W)];
                }
            }
            return null;
        };

        const traceFrom = (startIdx) => {
            const chain = [];
            const startJunc = findJunctionNeighbor(startIdx);
            if (startJunc) chain.push(startJunc);

            let curr = startIdx;
            let lastIdx = startIdx;
            while (curr !== -1) {
                visited[curr] = 1;
                lastIdx = curr;
                chain.push([curr % W, Math.floor(curr / W)]);
                let next = -1;
                for (let d = 0; d < 8; d++) {
                    const nb = curr + n8[d];
                    if (nb >= 0 && nb < W * H && work[nb] && !visited[nb]) {
                        next = nb;
                        break;
                    }
                }
                curr = next;
            }
            if (lastIdx !== startIdx) {
                const endJunc = findJunctionNeighbor(lastIdx);
                if (endJunc) chain.push(endJunc);
            }
            return chain;
        };

        for (let y = 1; y < H - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                const idx = y * W + x;
                if (!work[idx] || visited[idx]) continue;
                let nbCount = 0;
                for (let d = 0; d < 8; d++) if (work[idx + dirs[d]]) nbCount++;
                if (nbCount <= 1) {
                    const c = traceFrom(idx);
                    if (c.length >= 6) chains.push(c);
                }
            }
        }

        for (let y = 1; y < H - 1; y++) {
            for (let x = 1; x < W - 1; x++) {
                const idx = y * W + x;
                if (work[idx] && !visited[idx]) {
                    const c = traceFrom(idx);
                    if (c.length >= 10) {
                        c.push([c[0][0], c[0][1]]);
                        chains.push(c);
                    }
                }
            }
        }

        return chains;
    }

    _densifyPath(pts, stepPx = 3.0) {
        if (pts.length < 2) return pts.slice();
        const out = [pts[0]];
        for (let i = 1; i < pts.length; i++) {
            const x0 = pts[i - 1][0], y0 = pts[i - 1][1];
            const x1 = pts[i][0], y1 = pts[i][1];
            const segLen = Math.hypot(x1 - x0, y1 - y0);
            const steps = Math.max(1, Math.round(segLen / stepPx));
            for (let s = 1; s <= steps; s++) {
                const t = s / steps;
                out.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]);
            }
        }
        return out;
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

    _perpendicularDistance(pt, lineStart, lineEnd) {
        let x0 = pt[0], y0 = pt[1], x1 = lineStart[0], y1 = lineStart[1], x2 = lineEnd[0], y2 = lineEnd[1];
        let den = Math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2);
        if (den === 0) return Math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2);
        return Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / den;
    }
}
