/**
 * CardboardDetector.js
 * In-browser workpiece ROI detection using ONNX Runtime Web and YOLOv8.
 */
class CardboardDetector {
    constructor(modelPath = 'cardboard_detector.onnx') {
        this.modelPath = modelPath;
        this.session = null;
        this.inputSize = 640;
        this.classNames = ['CARDBOARD'];
        this.cardboardClassIdx = 0;
    }

    async init() {
        if (this.session) return;
        if (typeof ort === 'undefined') {
            throw new Error("ONNX Runtime Web (ort) is not loaded. Include ort.min.js.");
        }
        console.log("Loading CardboardDetector model:", this.modelPath);
        this.session = await ort.InferenceSession.create(this.modelPath, {
            executionProviders: ['wasm']
        });
        console.log("CardboardDetector model ready.");
    }

    preprocess(sourceCanvas) {
        const tensorCanvas = document.createElement('canvas');
        tensorCanvas.width = this.inputSize;
        tensorCanvas.height = this.inputSize;
        const ctx = tensorCanvas.getContext('2d');
        ctx.drawImage(sourceCanvas, 0, 0, this.inputSize, this.inputSize);
        const imgData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
        const { data } = imgData;

        const numPixels = this.inputSize * this.inputSize;
        const floatData = new Float32Array(3 * numPixels);

        for (let i = 0; i < numPixels; i++) {
            floatData[i] = data[i * 4] / 255.0;                  // R
            floatData[numPixels + i] = data[i * 4 + 1] / 255.0;  // G
            floatData[2 * numPixels + i] = data[i * 4 + 2] / 255.0; // B
        }

        return new ort.Tensor('float32', floatData, [1, 3, this.inputSize, this.inputSize]);
    }

    async detect(sourceCanvasOrMat, confidenceThreshold = 0.25, shavePadding = 0) {
        await this.init();

        let sourceCanvas;
        let origW, origH;

        if (sourceCanvasOrMat instanceof HTMLCanvasElement) {
            sourceCanvas = sourceCanvasOrMat;
            origW = sourceCanvas.width;
            origH = sourceCanvas.height;
        } else if (sourceCanvasOrMat && sourceCanvasOrMat.cols) {
            sourceCanvas = document.createElement('canvas');
            cv.imshow(sourceCanvas, sourceCanvasOrMat);
            origW = sourceCanvasOrMat.cols;
            origH = sourceCanvasOrMat.rows;
        } else {
            throw new Error("Invalid input: Expected HTMLCanvasElement or cv.Mat");
        }

        const inputTensor = this.preprocess(sourceCanvas);
        const feeds = { images: inputTensor };
        const results = await this.session.run(feeds);

        const outputTensor = results[this.session.outputNames[0]];
        const dims = outputTensor.dims; // e.g. [1, 5, 8400] for 640x640 single-class
        const numChannels = dims ? dims[1] : 5;
        const numAnchors = dims ? dims[2] : 8400;
        const data = outputTensor.data;

        // If single-class (channels=5: cx, cy, w, h, score), score row is 4.
        // If 5-class (channels=9: cx, cy, w, h, class0..4), cardboard row is 5.
        const scoreRow = numChannels === 5 ? 4 : (numChannels > 5 ? 5 : 4);

        let bestBox = null;
        let maxScore = -1;

        for (let anchor = 0; anchor < numAnchors; anchor++) {
            const cardboardScore = data[scoreRow * numAnchors + anchor];

            if (cardboardScore > confidenceThreshold && cardboardScore > maxScore) {
                maxScore = cardboardScore;
                const cx = data[0 * numAnchors + anchor];
                const cy = data[1 * numAnchors + anchor];
                const w = data[2 * numAnchors + anchor];
                const h = data[3 * numAnchors + anchor];

                const scaleX = origW / this.inputSize;
                const scaleY = origH / this.inputSize;

                const boxX = Math.max(0, (cx - w / 2) * scaleX);
                const boxY = Math.max(0, (cy - h / 2) * scaleY);
                const boxW = Math.min(origW - boxX, w * scaleX);
                const boxH = Math.min(origH - boxY, h * scaleY);

                const insetX = Math.min(shavePadding, Math.floor(boxW * 0.4));
                const insetY = Math.min(shavePadding, Math.floor(boxH * 0.4));

                bestBox = {
                    x: Math.round(boxX + insetX),
                    y: Math.round(boxY + insetY),
                    width: Math.round(boxW - 2 * insetX),
                    height: Math.round(boxH - 2 * insetY),
                    confidence: maxScore,
                    label: 'CARDBOARD'
                };
            }
        }

        return bestBox;
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = CardboardDetector;
}
