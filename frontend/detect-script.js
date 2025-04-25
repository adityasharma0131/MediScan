// MediScan - Frontend Script
document.addEventListener('DOMContentLoaded', function() {
    // Backend API URL - update this to match your server address
    const API_URL = 'http://localhost:8000';
    
    // Elements
    const dropArea = document.getElementById('drop-area');
    const browseBtn = document.getElementById('browse-btn');
    const fileInput = document.getElementById('file-input');
    const filePreview = document.getElementById('file-preview');
    const previewImage = document.getElementById('preview-image');
    const detectBtn = document.getElementById('detect-button');
    const loadingEl = document.getElementById('loading');
    const progressValue = document.getElementById('progress-value');
    const progressInfo = document.getElementById('progress-info');
    const resultsContainer = document.getElementById('results-container');
    const originalImage = document.getElementById('original-image');
    const resultImage = document.getElementById('result-image');
    const gradcamImage = document.getElementById('gradcam-image');
    const detectionList = document.getElementById('detection-list');

    // Track currently uploaded file
    let currentFile = null;

    // Initialize drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Add highlighting for drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('highlight');
        }, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('highlight');
        }, false);
    });
    
    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFiles(files);
        }
    }
    
    // Handle browse button click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    // Handle file selection via browse
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });
    
    // Process selected files
    function handleFiles(files) {
        currentFile = files[0];
        if (isValidImageFile(currentFile)) {
            displayPreview(currentFile);
        } else {
            alert('Please upload a valid image file (JPG or PNG).');
        }
    }
    
    // Check if file is a valid image type
    function isValidImageFile(file) {
        const acceptedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        return acceptedTypes.includes(file.type);
    }
    
    // Display image preview
    function displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            filePreview.style.display = 'flex';
            dropArea.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
    
    // Handle detection button click
    detectBtn.addEventListener('click', () => {
        if (currentFile) {
            uploadAndDetect(currentFile);
        } else {
            alert('Please select an image first.');
        }
    });
    
    // Upload image and run detection
    function uploadAndDetect(file) {
        // Show loading screen
        filePreview.style.display = 'none';
        loadingEl.style.display = 'block';
        progressValue.style.width = '0%';
        progressInfo.textContent = 'Uploading...';
        
        // Create form data for upload
        const formData = new FormData();
        formData.append('file', file);
        
        // Simulated progress for better UX
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += Math.random() * 5;
            if (progress > 95) {
                progress = 95;
                clearInterval(progressInterval);
            }
            progressValue.style.width = `${progress}%`;
            progressInfo.textContent = `Processing... ${Math.round(progress)}%`;
        }, 200);
        
        console.log("Sending detection request to:", `${API_URL}/detect`);
        
        // Send to API
        fetch(`${API_URL}/detect`, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log("Response status:", response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            clearInterval(progressInterval);
            
            // Log the response for debugging
            console.log("Detection response:", data);
            
            // Check if the response contains the required data
            if (!data) {
                throw new Error('Empty response from server');
            }
            
            // Start multi-stage loading process with synced progress bar
            // Stage 1: Initial processing (20%)
            progressValue.style.width = '20%';
            progressInfo.textContent = 'Analysis complete! Preparing results...';
            
            setTimeout(() => {
                // Stage 2: Processing image analysis (40%)
                progressValue.style.width = '40%';
                progressInfo.textContent = 'Processing image analysis...';
                
                setTimeout(() => {
                    // Stage 3: Identifying fracture patterns (60%)
                    progressValue.style.width = '60%';
                    progressInfo.textContent = 'Identifying fracture patterns...';
                    
                    setTimeout(() => {
                        // Stage 4: Generating visualization (80%)
                        progressValue.style.width = '80%';
                        progressInfo.textContent = 'Generating visualization...';
                        
                        setTimeout(() => {
                            // Stage 5: Finalizing report (100%)
                            progressValue.style.width = '100%';
                            progressInfo.textContent = 'Finalizing report...';
                            
                            setTimeout(() => {
                                displayResults(data, file);
                            }, 2000);
                        }, 2000);
                    }, 2000);
                }, 2000);
            }, 2000);
        })
        .catch(error => {
            clearInterval(progressInterval);
            console.error('Detection error:', error);
            progressInfo.textContent = 'Error occurred. Please try again.';
            // Return to preview
            loadingEl.style.display = 'none';
            filePreview.style.display = 'flex';
        });
    }
    
    // Display detection results
    function displayResults(data, file) {
        try {
            console.log("Displaying results:", data);
            
            // Hide loading screen
            loadingEl.style.display = 'none';
            
            // Display results container
            resultsContainer.style.display = 'block';
            resultsContainer.style.opacity = '0';
            setTimeout(() => {
                resultsContainer.style.opacity = '1';
            }, 100);
            
            // Set original image
            console.log("Setting original image from file");
            const originalImageURL = URL.createObjectURL(file);
            originalImage.src = originalImageURL;
            
            // Set result image from API response
            if (data.result_image) {
                console.log("Setting result image:", `${API_URL}${data.result_image}`);
                resultImage.src = `${API_URL}${data.result_image}`;
            } else {
                console.warn("No result image in response");
            }
            
            // Set Grad-CAM image from API response if available
            if (data.gradcam_image) {
                console.log("Setting gradcam image:", `${API_URL}${data.gradcam_image}`);
                gradcamImage.src = `${API_URL}${data.gradcam_image}`;
                document.getElementById('gradcam-container').style.display = 'block';
            } else {
                console.log("No gradcam image, hiding container");
                document.getElementById('gradcam-container').style.display = 'none';
            }
            
            // Display detection results
            console.log("Processing detections:", data.detections);
            displayDetections(data.detections || []);
            
        } catch (error) {
            console.error('Error displaying results:', error);
            console.error('Error occurred at:', error.stack);
            // Return to preview
            resultsContainer.style.display = 'none';
            filePreview.style.display = 'flex';
        }
    }
    
    // Display detection items
    function displayDetections(detections) {
        detectionList.innerHTML = '';
        
        if (!detections || detections.length === 0) {
            detectionList.innerHTML = '<div class="detection-item"><div class="detection-content"><h4>No fractures detected</h4><p>The AI did not detect any fractures in this image.</p></div></div>';
            return;
        }
        
        detections.forEach((detection, index) => {
            const confidencePercent = Math.round(detection.confidence * 100);
            let confidenceClass = 'low';
            if (confidencePercent >= 80) {
                confidenceClass = 'high';
            } else if (confidencePercent >= 50) {
                confidenceClass = 'medium';
            }
            
            const detectionItem = document.createElement('div');
            detectionItem.className = 'detection-item';
            detectionItem.innerHTML = `
                <div class="detection-icon">
                    <i class="fas fa-bone"></i>
                </div>
                <div class="detection-content">
                    <h4>${detection.class} #${index + 1}</h4>
                    <div class="confidence ${confidenceClass}">
                        <span>Confidence: ${confidencePercent}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-value" style="width: ${confidencePercent}%"></div>
                        </div>
                    </div>
                    <div class="location-info">
                        <i class="fas fa-map-marker-alt"></i>
                        <span>Location: x: ${Math.round(detection.box.x1)}-${Math.round(detection.box.x2)}, y: ${Math.round(detection.box.y1)}-${Math.round(detection.box.y2)}</span>
                    </div>
                </div>
            `;
            detectionList.appendChild(detectionItem);
        });
    }
    
    // Add a reset button to start over
    const resetButtons = document.querySelectorAll('.reset-button');
    resetButtons.forEach(button => {
        button.addEventListener('click', resetDetector);
    });
    
    function resetDetector() {
        // Reset UI state
        resultsContainer.style.display = 'none';
        filePreview.style.display = 'none';
        dropArea.style.display = 'block';
        currentFile = null;
        fileInput.value = '';
    }

    // Check server status on page load
    checkServerStatus();
    
    function checkServerStatus() {
        fetch(`${API_URL}/status`)
            .then(response => response.json())
            .then(data => {
                console.log('Server status:', data.message);
            })
            .catch(error => {
                console.error('Server connection error:', error);
                alert('Warning: Could not connect to detection server. Please make sure the backend server is running.');
            });
    }
});