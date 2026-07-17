/**
 * app.js
 * Main UI Controller for the Mobile Web App
 */

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressFill = document.getElementById('progressFill');
    const progressStatus = document.getElementById('progressStatus');
    const statusCard = document.getElementById('statusCard');
    const statusTitle = document.getElementById('statusTitle');
    const statusMessage = document.getElementById('statusMessage');
    const previewImg = document.getElementById('previewImg');
    const previewContainer = document.getElementById('previewContainer');
    const svgOverlay = document.getElementById('svgOverlay');
    const btnOverlayTest = document.getElementById('btnOverlayTest');
    const canvas = document.getElementById('canvas');

    let overlayActive = false;
    let currentSvgData = null;

    btnOverlayTest.addEventListener('click', () => {
        overlayActive = !overlayActive;
        if (overlayActive) {
            previewImg.style.opacity = '0.5';
            svgOverlay.style.display = 'block';
            svgOverlay.innerHTML = currentSvgData;
            
            const svgEl = svgOverlay.querySelector('svg');
            if (svgEl) {
                svgEl.style.width = '100%';
                svgEl.style.height = '100%';
                svgEl.style.position = 'absolute';
                svgEl.style.top = '0';
                svgEl.style.left = '0';
            }
        } else {
            previewImg.style.opacity = '1';
            svgOverlay.style.display = 'none';
            svgOverlay.innerHTML = '';
        }
    });

    let opencvReady = false;
    let processor = new ImageProcessor();
    let communicator = new Communicator();

    // Disable upload area until OpenCV loads
    uploadArea.classList.add('disabled');

    // Setup Communicator UI updates
    communicator.onStatusChange((status, message) => {
        const dot = document.getElementById('conn-status-dot');
        const text = document.getElementById('conn-status-text');
        
        dot.className = 'status-dot ' + status;
        text.textContent = message;
    });

    // Initialize WebRTC Communicator
    communicator.init();

    // Check OpenCV Ready state
    const checkOpenCvReady = setInterval(() => {
        if (typeof cv !== 'undefined' && cv.Mat) {
            opencvReady = true;
            uploadArea.classList.remove('disabled');
            clearInterval(checkOpenCvReady);
        }
    }, 500);

    // Bind Upload Area
    uploadArea.addEventListener('click', () => {
        if (opencvReady) {
            fileInput.click();
        }
    });

    // Handle File Selection
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // Reset UI
        statusCard.className = 'status-card';
        statusCard.style.display = 'none';
        previewContainer.style.display = 'none';
        progressContainer.style.display = 'block';
        progressFill.style.width = '30%';
        progressStatus.innerHTML = '<i data-lucide="loader-2" class="animate-spin" style="width:14px; height:14px; vertical-align:middle; margin-right:4px;"></i> Reading file...';
        lucide.createIcons();

        try {
            // Handle SVG Direct Upload
            if (file.type === 'image/svg+xml' || file.name.toLowerCase().endsWith('.svg')) {
                progressFill.style.width = '70%';
                progressStatus.innerHTML = 'Sending SVG Payload...';
                
                const svgText = await file.text();
                
                communicator.sendPayload(svgText, null);
                
                progressFill.style.width = '100%';
                showSuccess("Direct SVG sent successfully to Main UI.", null, svgText);
            } 
            // Handle Image Processing (JPG/PNG/Camera)
            else if (file.type.startsWith('image/')) {
                progressFill.style.width = '50%';
                progressStatus.innerHTML = 'Processing Image...';

                const imgUrl = URL.createObjectURL(file);
                const img = new Image();
                
                img.onload = async () => {
                    URL.revokeObjectURL(imgUrl);
                    
                    // Draw to hidden canvas
                    canvas.width = img.width;
                    canvas.height = img.height;
                    const ctx = canvas.getContext('2d');
                    ctx.drawImage(img, 0, 0, img.width, img.height);

                    try {
                        const magicPenEnabled = document.getElementById('magicPenToggle').checked;
                        const vectorizationMode = document.getElementById('vectorizationMode').value;
                        
                        if (magicPenEnabled) {
                            openLassoTool(img, canvas, vectorizationMode);
                        } else {
                            runPipeline(canvas, vectorizationMode, null);
                        }
                    } catch (err) {
                        showError("Processing failed: " + err.message);
                    }
                };

                img.onerror = () => {
                    URL.revokeObjectURL(imgUrl);
                    showError("Failed to load image data.");
                };
                
                img.src = imgUrl;
            } else {
                showError("Unsupported file format. Please upload an image or SVG.");
            }
        } catch (error) {
            console.error(error);
            showError("Error: " + error.message);
        }
    });
    // Pipeline Execution Logic
    async function runPipeline(imgCanvas, vectorizationMode, maskCanvas) {
        progressContainer.style.display = 'block';
        progressFill.style.width = '70%';
        progressStatus.innerHTML = 'Normalizing & Vectorizing...';
        
        try {
            const result = await processor.process(imgCanvas, vectorizationMode, maskCanvas);

            progressFill.style.width = '90%';
            progressStatus.innerHTML = 'Sending via WebRTC...';

            communicator.sendPayload(result.svg, result.image, result.meta);

            // MOCK LOCAL STORAGE BRIDGE
            try {
                localStorage.setItem('urumi_payload', JSON.stringify({
                    type: 'PROCESSING_COMPLETE',
                    payload: {
                        svg: result.svg,
                        image: result.image,
                        dots_per_mm: result.meta?.dots_per_mm,
                        physical_width: result.meta?.physical_width,
                        physical_height: result.meta?.physical_height
                    }
                }));
                localStorage.removeItem('urumi_payload');
            } catch (e) {
                console.warn("LocalStorage bridge full or disabled.");
            }

            progressFill.style.width = '100%';
            showSuccess("Processed Image sent successfully.", result.image, result.svg);
        } catch (err) {
            showError("Processing failed: " + err.message);
        }
    }

    // --- LASSO TOOL LOGIC ---
    function openLassoTool(img, srcCanvas, vectorizationMode) {
        const lassoModal = document.getElementById('lassoModal');
        const lassoCanvas = document.getElementById('lassoCanvas');
        const btnPoly = document.getElementById('btnLassoPoly');
        const btnRect = document.getElementById('btnLassoRect');
        const btnClear = document.getElementById('btnLassoClear');
        const btnConfirm = document.getElementById('btnLassoConfirm');
        
        lassoModal.style.display = 'flex';
        progressContainer.style.display = 'none';

        const ctx = lassoCanvas.getContext('2d');
        
        // Scale canvas for display
        const displayWidth = window.innerWidth * 0.9;
        const displayHeight = window.innerHeight * 0.7;
        const scale = Math.min(displayWidth / img.width, displayHeight / img.height);
        
        lassoCanvas.width = img.width * scale;
        lassoCanvas.height = img.height * scale;
        
        let currentMode = 'poly';
        let points = [];
        let isDrawing = false;
        let rectStart = null;
        let rectEnd = null;

        function render() {
            ctx.clearRect(0, 0, lassoCanvas.width, lassoCanvas.height);
            ctx.drawImage(img, 0, 0, lassoCanvas.width, lassoCanvas.height);
            
            // Dim background outside mask
            if (points.length > 2 || (rectStart && rectEnd)) {
                ctx.fillStyle = 'rgba(0,0,0,0.5)';
                ctx.fillRect(0, 0, lassoCanvas.width, lassoCanvas.height);
                
                ctx.save();
                ctx.beginPath();
                if (currentMode === 'poly' && points.length > 0) {
                    ctx.moveTo(points[0].x, points[0].y);
                    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
                    ctx.closePath();
                } else if (currentMode === 'rect' && rectStart && rectEnd) {
                    ctx.rect(rectStart.x, rectStart.y, rectEnd.x - rectStart.x, rectEnd.y - rectStart.y);
                }
                ctx.clip();
                ctx.drawImage(img, 0, 0, lassoCanvas.width, lassoCanvas.height);
                ctx.restore();
            }

            // Draw guidelines
            ctx.strokeStyle = '#3B82F6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            if (currentMode === 'poly' && points.length > 0) {
                ctx.moveTo(points[0].x, points[0].y);
                for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
                if (points.length > 2) ctx.closePath();
                ctx.stroke();
            } else if (currentMode === 'rect' && rectStart && rectEnd) {
                ctx.strokeRect(rectStart.x, rectStart.y, rectEnd.x - rectStart.x, rectEnd.y - rectStart.y);
            }
        }

        render();

        // Mode toggles
        const setMode = (mode) => {
            currentMode = mode;
            points = [];
            rectStart = null;
            rectEnd = null;
            btnPoly.style.borderColor = mode === 'poly' ? '#3B82F6' : '#444';
            btnRect.style.borderColor = mode === 'rect' ? '#3B82F6' : '#444';
            render();
        };
        btnPoly.onclick = () => setMode('poly');
        btnRect.onclick = () => setMode('rect');
        setMode('poly');

        btnClear.onclick = () => {
            points = [];
            rectStart = null;
            rectEnd = null;
            render();
        };

        const getPos = (e) => {
            const rect = lassoCanvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const clientY = e.touches ? e.touches[0].clientY : e.clientY;
            return {
                x: clientX - rect.left,
                y: clientY - rect.top
            };
        };

        lassoCanvas.onmousedown = lassoCanvas.ontouchstart = (e) => {
            e.preventDefault();
            const pos = getPos(e);
            if (currentMode === 'poly') {
                points.push(pos);
            } else {
                isDrawing = true;
                rectStart = pos;
                rectEnd = pos;
            }
            render();
        };

        lassoCanvas.onmousemove = lassoCanvas.ontouchmove = (e) => {
            e.preventDefault();
            if (!isDrawing || currentMode !== 'rect') return;
            rectEnd = getPos(e);
            render();
        };

        lassoCanvas.onmouseup = lassoCanvas.ontouchend = () => {
            isDrawing = false;
        };

        btnConfirm.onclick = () => {
            // Generate full res binary mask
            const maskCanvas = document.getElementById('maskCanvas');
            maskCanvas.width = img.width;
            maskCanvas.height = img.height;
            const mCtx = maskCanvas.getContext('2d');
            
            mCtx.fillStyle = 'black';
            mCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            
            mCtx.fillStyle = 'white';
            mCtx.beginPath();
            
            const scaleX = img.width / lassoCanvas.width;
            const scaleY = img.height / lassoCanvas.height;

            if (currentMode === 'poly' && points.length > 2) {
                mCtx.moveTo(points[0].x * scaleX, points[0].y * scaleY);
                for (let i = 1; i < points.length; i++) {
                    mCtx.lineTo(points[i].x * scaleX, points[i].y * scaleY);
                }
                mCtx.closePath();
                mCtx.fill();
            } else if (currentMode === 'rect' && rectStart && rectEnd) {
                mCtx.fillRect(
                    rectStart.x * scaleX, 
                    rectStart.y * scaleY, 
                    (rectEnd.x - rectStart.x) * scaleX, 
                    (rectEnd.y - rectStart.y) * scaleY
                );
            } else {
                // No valid mask drawn, just fill white (keep everything)
                mCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            }

            lassoModal.style.display = 'none';
            
            // Clean up event listeners so they don't stack
            btnPoly.onclick = btnRect.onclick = btnClear.onclick = btnConfirm.onclick = null;
            lassoCanvas.onmousedown = lassoCanvas.onmousemove = lassoCanvas.onmouseup = null;
            lassoCanvas.ontouchstart = lassoCanvas.ontouchmove = lassoCanvas.ontouchend = null;

            runPipeline(srcCanvas, vectorizationMode, maskCanvas);
        };
    }

    function showSuccess(msg, imageUrl = null, svgData = null) {
        progressContainer.style.display = 'none';
        statusCard.className = 'status-card success';
        statusCard.style.display = 'block';
        statusTitle.innerHTML = '<span><i data-lucide="check-circle"></i></span> Sent Successfully!';
        statusMessage.innerText = msg;
        
        currentSvgData = svgData;
        overlayActive = false;
        previewImg.style.opacity = '1';
        svgOverlay.style.display = 'none';
        svgOverlay.innerHTML = '';
        
        if (imageUrl) {
            previewImg.src = imageUrl;
            previewContainer.style.display = 'block';
        } else {
            previewContainer.style.display = 'none';
        }

        if (imageUrl && svgData) {
            btnOverlayTest.style.display = 'inline-block';
        } else {
            btnOverlayTest.style.display = 'none';
        }

        lucide.createIcons();
    }

    function showError(msg) {
        progressContainer.style.display = 'none';
        statusCard.className = 'status-card error';
        statusCard.style.display = 'block';
        statusTitle.innerHTML = '<span><i data-lucide="alert-triangle"></i></span> Error';
        statusMessage.innerText = msg;
        previewContainer.style.display = 'none';
        btnOverlayTest.style.display = 'none';
        lucide.createIcons();
    }
});
