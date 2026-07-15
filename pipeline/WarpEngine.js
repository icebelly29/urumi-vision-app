class WarpEngine {
    constructor(outputWidth) {
        this.OUTPUT_WIDTH = outputWidth;
        this.currentFrameConfig = null;
        this.borderSizeMm = 0;
    }
    normalizePerspective(src, srcMask = null) {
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
        // We need to calculate two critical physical dimensions:
        // 1. cropPhysW, cropPhysH: The physical dimensions of the FINAL cropped drawing area (in mm).
        // 2. arucoToCropMargin: The physical distance from the ArUco centers to the crop boundary (in mm).

        let cropPhysW, cropPhysH, arucoMarginX, arucoMarginY;

        if (detectedIds.includes(12) || detectedIds.includes(13) || detectedIds.includes(14) || detectedIds.includes(15)) {
            // Custom Frame Generator (IDs 12-15)
            const urlParams = new URLSearchParams(window.location.search);
            const customW = parseFloat(urlParams.get('w'));
            const customH = parseFloat(urlParams.get('h'));
            const physWidth = !isNaN(customW) ? customW : 830.0;
            const physHeight = !isNaN(customH) ? customH : 1130.0;

            // aruco_pos in the JSON places ArUco centers at the paper corners [0,0] etc.
            // The user measured the actual inner drawing area as 780mm x 1070mm
            // So we explicitly set the crop area to these dimensions.
            cropPhysW = 775.0;
            cropPhysH = 1065.0;

            // The margin is half the difference between the ArUco frame size and the crop size.
            // This properly handles rectangular margins (e.g. 25mm on sides, 30mm on top/bottom).
            arucoMarginX = (physWidth - cropPhysW) / 2.0;
            arucoMarginY = (physHeight - cropPhysH) / 2.0;
        } else {
            // Legacy Fixed Frames
            let physWidth = 210.0;
            let physHeight = 290.0;
            let legacyMargin = 17.0;

            if (detectedIds.includes(8) || detectedIds.includes(9) || detectedIds.includes(10) || detectedIds.includes(11)) {
                physWidth = 270.0; physHeight = 350.0; legacyMargin = 16.0;
            } else if (detectedIds.includes(4) || detectedIds.includes(5) || detectedIds.includes(6) || detectedIds.includes(7)) {
                physWidth = 210.0; physHeight = 290.0; legacyMargin = 17.0;
            } else if (detectedIds.includes(0) || detectedIds.includes(1) || detectedIds.includes(2) || detectedIds.includes(3)) {
                physWidth = 150.0; physHeight = 230.0; legacyMargin = 11.0;
            }

            // For legacy frames, the math historically used legacyMargin for both physical reduction and pixel mapping.
            // We maintain this exact behavior to avoid breaking old printed frames.
            cropPhysW = physWidth - (2 * legacyMargin);
            cropPhysH = physHeight - (2 * legacyMargin);
            arucoMarginX = legacyMargin;
            arucoMarginY = legacyMargin;
        }

        // Store for SVG generation later
        this.currentFrameConfig = { cropPhysW, cropPhysH };

        const dotsPerMm = this.OUTPUT_WIDTH / cropPhysW;
        const outW = this.OUTPUT_WIDTH;
        const outH = Math.round(dotsPerMm * cropPhysH);

        const marginXPx = arucoMarginX * dotsPerMm;
        const marginYPx = arucoMarginY * dotsPerMm;

        let srcTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            tl.x, tl.y,
            tr.x, tr.y,
            br.x, br.y,
            bl.x, bl.y
        ]);

        let dstTri = cv.matFromArray(4, 1, cv.CV_32FC2, [
            -marginXPx, -marginYPx,
            outW + marginXPx, -marginYPx,
            outW + marginXPx, outH + marginYPx,
            -marginXPx, outH + marginYPx
        ]);

        console.log(`[Warp] cropPhysW=${cropPhysW}mm, cropPhysH=${cropPhysH}mm, marginX=${arucoMarginX}mm, marginY=${arucoMarginY}mm`);
        console.log(`[Warp] Output canvas: ${outW}x${outH}px, dotsPerMm=${dotsPerMm.toFixed(3)}`);
        console.log(`[Warp] TL marker center in photo: (${tl.x.toFixed(1)}, ${tl.y.toFixed(1)})`);
        console.log(`[Warp] TR marker center in photo: (${tr.x.toFixed(1)}, ${tr.y.toFixed(1)})`);
        console.log(`[Warp] BR marker center in photo: (${br.x.toFixed(1)}, ${br.y.toFixed(1)})`);
        console.log(`[Warp] BL marker center in photo: (${bl.x.toFixed(1)}, ${bl.y.toFixed(1)})`);
        console.log(`[Warp] marginPx: X=${marginXPx.toFixed(1)}, Y=${marginYPx.toFixed(1)}`);
        console.log(`[Warp] Dst TL=(${-marginXPx},${-marginYPx}), TR=(${outW + marginXPx},${-marginYPx}), BR=(${outW + marginXPx},${outH + marginYPx}), BL=(${-marginXPx},${outH + marginYPx})`);

        let M = cv.getPerspectiveTransform(srcTri, dstTri);
        let warped = new cv.Mat();
        let size = new cv.Size(outW, outH);
        cv.warpPerspective(src, warped, M, size, cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(255, 255, 255, 255));
        
        let warpedMask = null;
        if (srcMask) {
            warpedMask = new cv.Mat();
            cv.warpPerspective(srcMask, warpedMask, M, size, cv.INTER_NEAREST, cv.BORDER_CONSTANT, new cv.Scalar(0));
        }

        console.log(`[Warp] Done. Output mat: ${warped.cols}x${warped.rows}, channels=${warped.channels()}`);

        gray.delete(); corners.delete(); ids.delete(); dictionary.delete(); parameters.delete(); refineParameters.delete(); detector.delete();
        srcTri.delete(); dstTri.delete(); M.delete();

        return { image: warped, mask: warpedMask };
    }

}
