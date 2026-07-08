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
    const canvas = document.getElementById('canvas');

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
        previewImg.style.display = 'none';
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
                showSuccess("Direct SVG sent successfully to Main UI.");
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
                        progressStatus.innerHTML = 'Normalizing & Vectorizing...';
                        const result = await processor.process(canvas);

                        progressFill.style.width = '90%';
                        progressStatus.innerHTML = 'Sending via WebRTC...';

                        communicator.sendPayload(result.svg, result.image, result.meta);

                        progressFill.style.width = '100%';
                        showSuccess("Processed Image sent successfully.", result.image);
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

    function showSuccess(msg, imageUrl = null) {
        progressContainer.style.display = 'none';
        statusCard.className = 'status-card success';
        statusCard.style.display = 'block';
        statusTitle.innerHTML = '<span><i data-lucide="check-circle"></i></span> Sent Successfully!';
        statusMessage.innerText = msg;
        
        if (imageUrl) {
            previewImg.src = imageUrl;
            previewImg.style.display = 'block';
        } else {
            previewImg.style.display = 'none';
        }
        lucide.createIcons();
    }

    function showError(msg) {
        progressContainer.style.display = 'none';
        statusCard.className = 'status-card error';
        statusCard.style.display = 'block';
        statusTitle.innerHTML = '<span><i data-lucide="alert-triangle"></i></span> Error';
        statusMessage.innerText = msg;
        previewImg.style.display = 'none';
        lucide.createIcons();
    }
});
