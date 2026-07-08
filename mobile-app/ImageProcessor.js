/**
 * ImageProcessor.js
 * Handles OpenCV ArUco detection, perspective warping, vectorization,
 * skeletonization, path simplification, and layered SVG generation.
 */

class ImageProcessor {
    constructor() {
        // Physical dimensions of the bed in some units, or arbitrary ratio.
        // We normalize to a standard width of 2000px for high-fidelity processing.
        // 1000px was too low and caused sub-pixel resolution loss for thin ballpoint pens.
        this.OUTPUT_WIDTH = 2000;

        // Define color ranges in HSV for classification
        // OpenCV HSV ranges: H: 0-179, S: 0-255, V: 0-255
        this.colorProfiles = {
            thru_cut: {
                layer: 'thru_cut',
                color: '#3b82f6' // UrumiCutter Standard Blue (Default Cut)
            },
            score: {
                layer: 'score',
                color: '#ef4444' // UrumiCutter Standard Red (Score)
            },
            crease: {
                layer: 'crease',
                color: '#22c55e' // UrumiCutter Standard Green (Crease)
            }
        };
    }

    /**
     * Main entry point for the pipeline
     */
    async process(imageElement) {
        if (!cv || !cv.Mat) {
            throw new Error("OpenCV is not initialized yet.");
        }

        console.log("Starting Image Processing Pipeline...");

        let src = cv.imread(imageElement);
        let dst = new cv.Mat();

        // 1. Detect ArUco & Normalize Perspective
        const warpedMat = this._normalizePerspective(src);
        if (!warpedMat) {
            src.delete();
            dst.delete();
            throw new Error("Could not detect 4 ArUco markers for bed rectangle.");
        }

        // 2. Extract paths by color
        const layersData = this._extractColorPaths(warpedMat);

        // 3. Generate Layered SVG
        const resultLayered = this._generateLayeredSvg(layersData, warpedMat.cols, warpedMat.rows);
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

    _normalizePerspective(src) {
        let gray = new cv.Mat();
        cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

        let dictionary = cv.getPredefinedDictionary(cv.DICT_4X4_50);
        let corners = new cv.MatVector();
        let ids = new cv.Mat();
        let parameters = new cv.aruco_DetectorParameters();
        let refineParameters = new cv.aruco_RefineParameters(10, 3, true);

        let detector = new cv.aruco_ArucoDetector(dictionary, parameters, refineParameters);
        detector.detectMarkers(gray, corners, ids);

        if (ids.rows < 4) {
            console.warn(`Only found ${ids.rows} markers. Need 4.`);
            gray.delete(); corners.delete(); ids.delete(); dictionary.delete(); parameters.delete(); refineParameters.delete(); detector.delete();
            return null;
        }

        // Map centers
        let markerCenters = {};
        for (let i = 0; i < ids.rows; i++) {
            let id = ids.data32S[i];
            let corner = corners.get(i);
            let cx = (corner.data32F[0] + corner.data32F[2] + corner.data32F[4] + corner.data32F[6]) / 4;
            let cy = (corner.data32F[1] + corner.data32F[3] + corner.data32F[5] + corner.data32F[7]) / 4;
            markerCenters[id] = { x: cx, y: cy };
        }

        // Instead of sorting by visual x/y (which breaks if the phone is rotated),
        // we strictly sort by the ArUco ID values. The frame generator ALWAYS issues IDs 
        // clockwise starting from Top-Left. Example (12=TL, 13=TR, 14=BR, 15=BL).
        let detectedIds = Object.keys(markerCenters).map(Number).sort((a, b) => a - b);

        if (detectedIds.length !== 4) {
            gray.delete(); corners.delete(); ids.delete(); dictionary.delete(); parameters.delete(); refineParameters.delete(); detector.delete();
            throw new Error(`Expected exactly 4 markers, but found ${detectedIds.length}.`);
        }

        const tl = markerCenters[detectedIds[0]];
        const tr = markerCenters[detectedIds[1]];
        const br = markerCenters[detectedIds[2]];
        const bl = markerCenters[detectedIds[3]];

        if (!tl || !tr || !br || !bl) {
            gray.delete(); corners.delete(); ids.delete(); dictionary.delete(); parameters.delete(); refineParameters.delete(); detector.delete();
            throw new Error("Could not map 4 corner markers by ID.");
        }

        // --- ASPECT RATIO & CROPPING FIX ---
        // Determine physical dimensions based on detected ArUco IDs
        let physWidth = 210.0;
        let physHeight = 290.0;
        let innerMargin = 17.0;

        // We already have detectedIds from the sorting logic above
        if (detectedIds.includes(12) || detectedIds.includes(13) || detectedIds.includes(14) || detectedIds.includes(15)) {
            // Custom frame (IDs 12-15) - dynamic from URL QR code
            const urlParams = new URLSearchParams(window.location.search);
            const customW = parseFloat(urlParams.get('w'));
            const customH = parseFloat(urlParams.get('h'));

            physWidth = !isNaN(customW) ? customW : 475.0;
            physHeight = !isNaN(customH) ? customH : 475.0;
            // The inner margin (padding inside checkerboard) for the Frame Generator methodology
            // is (arucoSize / 2) + 1. Assuming default arucoSize=50, it is 26.
            innerMargin = 26.0;
        } else if (detectedIds.includes(8) || detectedIds.includes(9) || detectedIds.includes(10) || detectedIds.includes(11)) {
            // Large frame (IDs 8-11)
            physWidth = 270.0;
            physHeight = 350.0;
            innerMargin = 16.0;
        } else if (detectedIds.includes(4) || detectedIds.includes(5) || detectedIds.includes(6) || detectedIds.includes(7)) {
            // Medium frame (IDs 4-7)
            physWidth = 210.0;
            physHeight = 290.0;
            innerMargin = 17.0;
        } else if (detectedIds.includes(0) || detectedIds.includes(1) || detectedIds.includes(2) || detectedIds.includes(3)) {
            // Small frame (IDs 0-3)
            physWidth = 150.0;
            physHeight = 230.0;
            innerMargin = 11.0;
        }

        const cropPhysW = physWidth - (2 * innerMargin);
        const cropPhysH = physHeight - (2 * innerMargin);

        // Store for SVG generation later
        this.currentFrameConfig = { cropPhysW, cropPhysH };

        const dotsPerMm = this.OUTPUT_WIDTH / cropPhysW;
        const outW = this.OUTPUT_WIDTH;
        const outH = Math.round(dotsPerMm * cropPhysH);

        const marginPx = innerMargin * dotsPerMm;

        let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            tl.x, tl.y,
            tr.x, tr.y,
            br.x, br.y,
            bl.x, bl.y
        ]);

        let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            -marginPx, -marginPx,
            outW + marginPx, -marginPx,
            outW + marginPx, outH + marginPx,
            -marginPx, outH + marginPx
        ]);

        let M = cv.getPerspectiveTransform(srcTri, dstTri);
        let warped = new cv.Mat();
        let size = new cv.Size(outW, outH);
        cv.warpPerspective(src, warped, M, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));

        gray.delete(); corners.delete(); ids.delete(); dictionary.delete(); parameters.delete(); refineParameters.delete(); detector.delete();
        srcTri.delete(); dstTri.delete(); M.delete();

        return warped;
    }

    _extractColorPaths(img) {
        let hsv = new cv.Mat();
        let gray = new cv.Mat();
        cv.cvtColor(img, hsv, cv.COLOR_RGBA2RGB);
        cv.cvtColor(hsv, hsv, cv.COLOR_RGB2HSV);
        cv.cvtColor(img, gray, cv.COLOR_RGBA2GRAY);

        // 1. Bilateral Filter to smooth noise while preserving edges
        let blurred = new cv.Mat();
        cv.bilateralFilter(gray, blurred, 9, 75, 75);

        // 2. Isolate the white paper from the grey felt using Otsu.
        let paperMask = new cv.Mat();
        cv.threshold(blurred, paperMask, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU);

        // Find the largest contour (the paper) and fill it solid
        let contours = new cv.MatVector();
        let hierarchy = new cv.Mat();
        cv.findContours(paperMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

        let solidPaperMask = cv.Mat.zeros(paperMask.rows, paperMask.cols, cv.CV_8U);
        let maxArea = 0;
        let maxContourIdx = -1;
        for (let i = 0; i < contours.size(); i++) {
            let area = cv.contourArea(contours.get(i));
            if (area > maxArea) { maxArea = area; maxContourIdx = i; }
        }
        if (maxContourIdx !== -1) {
            cv.drawContours(solidPaperMask, contours, maxContourIdx, new cv.Scalar(255, 255, 255, 255), cv.FILLED);
        }
        paperMask.delete(); contours.delete(); hierarchy.delete();

        // 3. Blackhat for Dark Lines (Matches Python Pipeline)
        let blackhat = new cv.Mat();
        let bhKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(15, 15));
        cv.morphologyEx(blurred, blackhat, cv.MORPH_BLACKHAT, bhKernel);
        bhKernel.delete();
        blurred.delete();

        // In Python: tunedThresh = max(1, round(otsu * 0.9)).
        // In OpenCV.js (WebAssembly), Otsu on mostly-zero blackhat images can return 0 or 1,
        // which threshold at 1 and causes massive noise that fuses letters like 'T' and 'E'.
        // Blackhat normalizes illumination, so a fixed threshold of 15 safely drops camera noise and extracts pure dark strokes.
        let tunedThresh = 15;
        let darkStrokes = new cv.Mat();
        cv.threshold(blackhat, darkStrokes, tunedThresh, 255, cv.THRESH_BINARY);
        blackhat.delete();

        // 4. Extract colored ink via HSV Saturation (for faint ballpoint pens)
        let hsvForMask = new cv.Mat();
        let rgbForMask = new cv.Mat();
        cv.cvtColor(img, rgbForMask, cv.COLOR_RGBA2RGB);
        cv.cvtColor(rgbForMask, hsvForMask, cv.COLOR_RGB2HSV);

        let coloredInkMask = new cv.Mat();
        let lowScalar = new cv.Scalar(0, 38, 0, 0);
        let highScalar = new cv.Scalar(179, 255, 248, 0);
        let low = new cv.Mat(hsvForMask.rows, hsvForMask.cols, hsvForMask.type(), lowScalar);
        let high = new cv.Mat(hsvForMask.rows, hsvForMask.cols, hsvForMask.type(), highScalar);
        cv.inRange(hsvForMask, low, high, coloredInkMask);

        rgbForMask.delete(); hsvForMask.delete(); low.delete(); high.delete();

        // 5. Combine and Mask with Paper
        let allStrokes = new cv.Mat();
        cv.bitwise_or(darkStrokes, coloredInkMask, allStrokes);
        darkStrokes.delete(); coloredInkMask.delete();

        cv.bitwise_and(allStrokes, solidPaperMask, allStrokes);
        solidPaperMask.delete();

        // 5.5 Pre-Close: Bridge jagged/dotted lines from the raw HSV mask 
        // BEFORE connected components filtering. Otherwise, dotted lines (like faint ballpoint) get deleted!
        // We use 3x3 to prevent fusing of tiny text elements like 'E' and arrow heads.
        // CRITICAL: Out-of-place to avoid WebAssembly memory wipe bug.
        let preClosedStrokes = new cv.Mat();
        let preCloseKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(allStrokes, preClosedStrokes, cv.MORPH_CLOSE, preCloseKernel);
        preCloseKernel.delete();
        allStrokes.delete();

        // 6. Connected Components Filtering (Remove tiny isolated noise specks)
        let labels = new cv.Mat();
        let stats = new cv.Mat();
        let centroids = new cv.Mat();
        let numLabels = cv.connectedComponentsWithStats(preClosedStrokes, labels, stats, centroids, 8, cv.CV_32S);

        let w_px = img.cols;
        let h_px = img.rows;
        // Significantly reduced area and span floors to ensure faint dotted strokes survive,
        // since we are only using a 3x3 pre-close which preserves tiny details but leaves some dots.
        let areaFloor = Math.max(4, Math.floor(0.000002 * w_px * h_px));
        let spanFloor = Math.max(8, Math.floor(0.004 * Math.min(w_px, h_px)));

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

        // 7. Final Morph Close (3x3 Rect)
        // 3x3 preserves legibility of small handwriting and sharp corners perfectly.
        let finalMask = new cv.Mat();
        let closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
        cv.morphologyEx(filteredMask, finalMask, cv.MORPH_CLOSE, closeKernel);
        closeKernel.delete();
        filteredMask.delete();

        // 8. Vectorize & Skeletonize ALL paths at once
        let rawPaths = this._vectorizeAndSkeletonize(finalMask);
        finalMask.delete();
        // Magic Pen: Snap hand-drawn strokes to perfect geometric shapes,
        // and apply Moving Average smoothing to raw handwriting.
        let paths = this._snapToPerfectShapes(rawPaths);

        gray.delete();

        // 4. Classify each path by sampling its HSV pixels
        let results = {
            thru_cut: { paths: [], color: this.colorProfiles.thru_cut.color },
            score: { paths: [], color: this.colorProfiles.score.color },
            crease: { paths: [], color: this.colorProfiles.crease.color }
        };

        for (let i = 0; i < paths.length; i++) {
            let path = paths[i];
            if (path.length < 2) continue;

            // Filter out tiny noise / crumpled paper shadows
            let pathLen = 0;
            for (let k = 1; k < path.length; k++) {
                let dx = path[k][0] - path[k - 1][0];
                let dy = path[k][1] - path[k - 1][1];
                pathLen += Math.sqrt(dx * dx + dy * dy);
            }
            if (pathLen < 20) continue; // Skip artifacts shorter than 20 pixels (~3.5mm)

            let votes = { red: 0, blue: 0, green: 0, neutral: 0 };
            let neutralV = 0; // Sum of V for neutral points to detect shadows

            // Sample up to 28 points along the path
            let sampleCount = Math.min(28, path.length);
            for (let j = 0; j < sampleCount; j++) {
                let idx = Math.floor(j * (path.length - 1) / Math.max(1, sampleCount - 1));
                let px = Math.round(path[idx][0]);
                let py = Math.round(path[idx][1]);

                let bestScore = -1;
                let bestH = 0, bestS = 0, bestV = 0;

                // Check 5x5 window around the point to find the most saturated ink pixel
                for (let dy = -2; dy <= 2; dy++) {
                    for (let dx = -2; dx <= 2; dx++) {
                        let sx = Math.max(0, Math.min(px + dx, hsv.cols - 1));
                        let sy = Math.max(0, Math.min(py + dy, hsv.rows - 1));

                        // hsv data is interleaved CV_8UC3: [H, S, V, H, S, V, ...]
                        let offset = (sy * hsv.cols + sx) * 3;
                        let h = hsv.data[offset];
                        let s = hsv.data[offset + 1];
                        let v = hsv.data[offset + 2];

                        let score = s * 2 + Math.max(0, 245 - v);
                        if (score > bestScore) {
                            bestScore = score;
                            bestH = h; bestS = s; bestV = v;
                        }
                    }
                }

                if (bestS < 35 || bestV > 250) {
                    votes.neutral++;
                    neutralV += bestV;
                } else if (bestH < 10 || bestH >= 165) {
                    votes.red++;
                    // Expanded Green all the way to 115. Dark green ink in cool lighting shifts heavily towards cyan/blue.
                } else if (bestH >= 25 && bestH <= 115) {
                    votes.green++;
                } else if (bestH > 115 && bestH <= 150) {
                    votes.blue++;
                } else {
                    votes.neutral++;
                }
            }

            let dominant = 'neutral';
            let maxVotes = votes.neutral;
            if (votes.blue > maxVotes) { dominant = 'blue'; maxVotes = votes.blue; }
            if (votes.red > maxVotes) { dominant = 'red'; maxVotes = votes.red; }
            if (votes.green > maxVotes) { dominant = 'green'; maxVotes = votes.green; }

            if (dominant === 'red') {
                results.score.paths.push(path);
            } else if (dominant === 'green') {
                results.crease.paths.push(path);
            } else if (dominant === 'blue') {
                results.thru_cut.paths.push(path);
            } else {
                // If it's neutral, it might be a shadow or black ink.
                // Black ink has a very low V (Value/Brightness). Shadows have a higher V.
                let avgV = votes.neutral > 0 ? neutralV / votes.neutral : 255;
                if (avgV < 130) {
                    // True black marker -> treat as cut
                    results.thru_cut.paths.push(path);
                }
                // Else: It's a shadow or paper wrinkle. Ignore it.
            }
        }

        hsv.delete();
        return results;
    }

    _vectorizeAndSkeletonize(maskMat) {
        // TraceSkeleton expects a flat array of 0s and 1s
        let data = maskMat.data;
        let boolArray = new Array(maskMat.cols * maskMat.rows);
        for (let i = 0; i < data.length; i++) {
            boolArray[i] = data[i] > 128 ? 1 : 0;
        }

        let trace = TraceSkeleton.fromBoolArray(boolArray, maskMat.cols, maskMat.rows);
        return trace.polylines;
    }

    _simplifyPath(path, epsilon) {
        // Douglas-Peucker path simplification
        if (path.length <= 2) return path;

        let dmax = 0;
        let index = 0;
        const end = path.length - 1;

        for (let i = 1; i < end; i++) {
            let d = this._perpendicularDistance(path[i], path[0], path[end]);
            if (d > dmax) {
                index = i;
                dmax = d;
            }
        }

        if (dmax > epsilon) {
            let recResults1 = this._simplifyPath(path.slice(0, index + 1), epsilon);
            let recResults2 = this._simplifyPath(path.slice(index, end + 1), epsilon);
            return recResults1.slice(0, recResults1.length - 1).concat(recResults2);
        } else {
            return [path[0], path[end]];
        }
    }

    _smoothPath(path, iterations = 5) {
        if (path.length <= 2) return path;
        let currentPath = path;
        // 1D Gaussian smoothing [0.25, 0.5, 0.25] applied iteratively yields beautiful spline-like organic curves
        for (let iter = 0; iter < iterations; iter++) {
            let smoothed = [];
            smoothed.push(currentPath[0]); // Keep endpoints fixed
            for (let i = 1; i < currentPath.length - 1; i++) {
                let pPrev = currentPath[i - 1];
                let p = currentPath[i];
                let pNext = currentPath[i + 1];
                smoothed.push([
                    pPrev[0] * 0.25 + p[0] * 0.5 + pNext[0] * 0.25,
                    pPrev[1] * 0.25 + p[1] * 0.5 + pNext[1] * 0.25
                ]);
            }
            smoothed.push(currentPath[currentPath.length - 1]);
            currentPath = smoothed;
        }
        return currentPath;
    }

    _snapToPerfectShapes(paths) {
        return paths.map(path => {
            if (path.length < 2) return path;

            // 1. Calculate Bounding Box and Center
            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            for (let pt of path) {
                if (pt[0] < minX) minX = pt[0];
                if (pt[0] > maxX) maxX = pt[0];
                if (pt[1] < minY) minY = pt[1];
                if (pt[1] > maxY) maxY = pt[1];
            }

            let width = maxX - minX;
            let height = maxY - minY;
            let diagonal = Math.sqrt(width * width + height * height);
            let cx = minX + width / 2;
            let cy = minY + height / 2;

            // Ignore very small paths (likely text like "TEST" or "OK", or arrow heads)
            // Apply 5 iterations of Gaussian smoothing to make tiny text look like clean ink, not jittery pixels
            if (diagonal < 40) {
                return this._simplifyPath(this._smoothPath(path, 5), 0.5);
            }

            // 2. Check if shape is closed
            let pFirst = path[0];
            let pLast = path[path.length - 1];
            let distClose = Math.sqrt(Math.pow(pFirst[0] - pLast[0], 2) + Math.pow(pFirst[1] - pLast[1], 2));
            // A shape is closed if its ends are very close relative to its overall size
            let isClosed = distClose < Math.max(20, diagonal * 0.1);

            if (isClosed) {
                // Try Circle Match
                let sumR = 0;
                let distances = [];
                for (let pt of path) {
                    let d = Math.sqrt(Math.pow(pt[0] - cx, 2) + Math.pow(pt[1] - cy, 2));
                    distances.push(d);
                    sumR += d;
                }
                let avgR = sumR / path.length;

                // Max deviation from radius
                let maxDev = 0;
                for (let d of distances) {
                    if (Math.abs(d - avgR) > maxDev) maxDev = Math.abs(d - avgR);
                }

                // If max deviation is small, it's roughly circular
                if (maxDev < avgR * 0.25) {
                    // Generate perfect circle
                    let circlePath = [];
                    const numPoints = 72; // 5-degree increments
                    for (let i = 0; i <= numPoints; i++) {
                        let angle = (i / numPoints) * Math.PI * 2;
                        circlePath.push([cx + avgR * Math.cos(angle), cy + avgR * Math.sin(angle)]);
                    }
                    return circlePath;
                }

                // Try Rectangle/Quad Match
                // Use heavy Douglas-Peucker to simplify to base vertices
                let heavySimplify = this._simplifyPath(path, Math.max(15.0, diagonal * 0.05));
                // A closed quad usually has 5 points (start point repeated at end) or 4 points
                if (heavySimplify.length === 5 || heavySimplify.length === 4) {
                    // Ensure it is explicitly closed 5 points (4 edges)
                    if (heavySimplify.length === 4) {
                        heavySimplify.push([heavySimplify[0][0], heavySimplify[0][1]]);
                    } else {
                        heavySimplify[4] = [heavySimplify[0][0], heavySimplify[0][1]];
                    }

                    // Check angles to ensure it's roughly rectangular (~90 degrees)
                    let isRect = true;
                    for (let i = 0; i < 4; i++) {
                        let p0 = heavySimplify[i];
                        let p1 = heavySimplify[i + 1];
                        let p2 = heavySimplify[(i + 2) % 4];
                        // Vector p1->p0 and p1->p2
                        let v1 = [p0[0] - p1[0], p0[1] - p1[1]];
                        let v2 = [p2[0] - p1[0], p2[1] - p1[1]];
                        // Dot product
                        let dot = v1[0] * v2[0] + v1[1] * v2[1];
                        let mag1 = Math.sqrt(v1[0] * v1[0] + v1[1] * v1[1]);
                        let mag2 = Math.sqrt(v2[0] * v2[0] + v2[1] * v2[1]);
                        if (mag1 === 0 || mag2 === 0) continue;
                        let angle = Math.acos(dot / (mag1 * mag2)) * (180 / Math.PI);
                        // Must be between 65 and 115 degrees to be a strict rectangle
                        if (Math.abs(angle - 90) > 25) {
                            isRect = false;
                            break;
                        }
                    }
                    if (isRect) return heavySimplify;
                }
            } else {
                // Open shape: DO NOT snap to straight lines. 
                // Snapping open shapes to 2-point lines causes intersections (like arrow heads and 'K' diagonals) 
                // to detach from their main shafts because the skeletonizer naturally retracts slightly at junctions.
                // We will fall through to the smoothing logic below.
            }

            // Fallback: If no perfect geometry matched, it is arbitrary handwriting (or rejected shapes like arrows).
            // Apply Gaussian smoothing to iron out the human wobble and skeleton staircases
            let smoothed = this._smoothPath(path, 5);
            // Finally, simplify with a very low epsilon to remove redundant collinear points without adding blockiness
            return this._simplifyPath(smoothed, 0.5);
        });
    }

    _perpendicularDistance(pt, lineStart, lineEnd) {
        let x0 = pt[0], y0 = pt[1];
        let x1 = lineStart[0], y1 = lineStart[1];
        let x2 = lineEnd[0], y2 = lineEnd[1];

        let den = Math.sqrt(Math.pow(y2 - y1, 2) + Math.pow(x2 - x1, 2));
        if (den === 0) {
            // It's a point or perfectly closed loop. Distance is Euclidean distance to that point.
            return Math.sqrt(Math.pow(x0 - x1, 2) + Math.pow(y0 - y1, 2));
        }

        let num = Math.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1);
        return num / den;
    }

    _generateLayeredSvg(layersData, width, height) {
        let physWidth = 176;
        let physHeight = 256;

        if (this.currentFrameConfig) {
            physWidth = this.currentFrameConfig.cropPhysW;
            physHeight = this.currentFrameConfig.cropPhysH;
        }

        const dotsPerMm = width / physWidth;

        // Use 100% width/height so it scales perfectly in web previews
        let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="100%" height="100%">\n`;
        svg += `  <style>path { fill: none; stroke-width: 1px; vector-effect: non-scaling-stroke; stroke-linejoin: round; stroke-linecap: round; }</style>\n`;

        // 1. Bed Frame Layer
        svg += `  <g id="layer_bed_frame">\n`;
        svg += `    <rect x="0" y="0" width="${width}" height="${height}" fill="none" stroke="#FF00FF" stroke-width="2"/>\n`;
        svg += `  </g>\n`;

        // 2. Data Layers
        for (const [layerId, data] of Object.entries(layersData)) {
            if (data.paths.length === 0) continue;

            svg += `  <g id="${layerId}">\n`;
            data.paths.forEach(path => {
                if (path.length < 2) return;
                const d = `M ${path.map(pt => `${pt[0]},${pt[1]}`).join(' L ')}`;
                svg += `    <path d="${d}" fill="none" stroke="${data.color}" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/>\n`;
            });
            svg += `  </g>\n`;
        }

        svg += `</svg>`;
        return {
            svg: svg,
            meta: {
                dots_per_mm: dotsPerMm,
                physical_width: physWidth,
                physical_height: physHeight
            }
        };
    }
}

window.ImageProcessor = ImageProcessor;
