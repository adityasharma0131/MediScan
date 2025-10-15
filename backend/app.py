# Import necessary FastAPI modules
from fastapi import FastAPI, File, UploadFile, HTTPException
# PyTorch (used internally by YOLO for model inference)
import torch
# For enabling Cross-Origin Resource Sharing (important for frontend-backend communication)
from fastapi.middleware.cors import CORSMiddleware
# Used for sending files (e.g., returning images)
from fastapi.responses import FileResponse
# Standard Python modules for system tasks
import os
from uuid import uuid4  # For generating unique IDs
from ultralytics import YOLO  # YOLOv8 model class
import logging  # For structured logging of events and errors
import glob
import ultralytics
from PIL import Image  # Image loading/manipulation
import numpy as np  # Numerical operations
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing
from fastapi.staticfiles import StaticFiles  # Serve static files (images, results)
import traceback  # For detailed error stack traces

# Initialize the FastAPI app
app = FastAPI()

# 🧾 Configure logging to show timestamps, log level, and messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🌐 Define allowed origins for CORS (frontends that can access this API)
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://localhost:5500",      # VS Code Live Server default
    "http://127.0.0.1:5500",
    "*"  # Wildcard for testing (any domain) – not recommended for production
]

# ⚙️ Add middleware to enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,        # which domains can call the API
    allow_credentials=True,
    allow_methods=["*"],          # allow all HTTP methods
    allow_headers=["*"],          # allow all headers
    expose_headers=["*"]          # expose all headers in responses
)

# 📁 Ensure necessary folders exist (created automatically if missing)
os.makedirs("uploads", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("results/explanations", exist_ok=True)
os.makedirs("results/gradcam", exist_ok=True)

# 🧠 Load YOLOv8 model
try:
    logger.info("Loading YOLOv8 model...")
    logger.info(f"Ultralytics version: {ultralytics.__version__}")
    # Load pre-trained YOLOv8 model from 'models/model.pt'
    model = YOLO('models/model.pt')
    logger.info("Model loaded successfully.")
except Exception as e:
    # Log and raise if model loading fails
    logger.error(f"Error loading model: {e}")
    logger.error(traceback.format_exc())
    raise e

# 📂 Mount static directories so their files are directly accessible via URL
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ✅ Root route (homepage)
@app.get("/")
async def root():
    return {"message": "Welcome to the YOLOv8 detection API!"}

# ✅ Basic server status check
@app.get("/status")
async def get_status():
    return {"message": "Server is running", "status": "success"}

# ✅ Health-check endpoint (used for monitoring)
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running properly"}

# 🚀 Main detection endpoint — accepts an uploaded image
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 1️⃣ Validate file type (must be an image)
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    
    # 2️⃣ Create a unique detection ID for saving files
    detection_id = str(uuid4())
    
    # 3️⃣ Define file save paths
    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_extension = os.path.splitext(file.filename)[1]
    input_file_path = f"{upload_folder}/{detection_id}{file_extension}"
    
    # 4️⃣ Save uploaded file locally
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        # 5️⃣ Load the image using PIL and convert to RGB
        image = Image.open(input_file_path).convert("RGB")
        image_array = np.array(image)

        # 6️⃣ Run YOLOv8 inference on the image
        results = model(image_array)
        
        # 7️⃣ Save detection visualization output
        output_file_path = f"results/{detection_id}_result.jpg"
        results_plotted = results[0].plot()  # overlay detections on image
        cv2.imwrite(output_file_path, results_plotted)
        
        # 8️⃣ Parse YOLO detection results (boxes, classes, confidence)
        detections = []
        detection_boxes = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]        # bounding box coordinates
                confidence = box.conf[0]            # confidence score
                class_id = int(box.cls[0])          # detected class ID
                class_name = result.names[class_id] # class label
                
                detection_boxes.append([x1, y1, x2, y2])
                detections.append({
                    "id": i,
                    "class": class_name,
                    "confidence": float(confidence),
                    "box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2)
                    }
                })
        
        # 9️⃣ Generate an explanation image (draw boxes + highlight)
        explanation_file_path = None
        if len(detection_boxes) > 0:
            explanation_img = image_array.copy()
            
            for box in detection_boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                # Draw green rectangle around detection
                cv2.rectangle(explanation_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create a semi-transparent red highlight overlay
                highlight = np.zeros_like(explanation_img, dtype=np.uint8)
                pad = 10
                cv2.rectangle(
                    highlight,
                    (max(0, x1-pad), max(0, y1-pad)),
                    (min(explanation_img.shape[1], x2+pad), min(explanation_img.shape[0], y2+pad)),
                    (0, 0, 255),
                    -1
                )
                explanation_img = cv2.addWeighted(explanation_img, 1, highlight, 0.3, 0)
            
            explanation_file_path = f"results/explanations/{detection_id}_explanation.jpg"
            cv2.imwrite(explanation_file_path, explanation_img)
        
        # 🔥 Generate a Grad-CAM style heatmap for visual explanation
        gradcam_file_path = None
        if len(detection_boxes) > 0:
            gradcam_img = generate_gradcam(image_array, detection_boxes, results[0])
            gradcam_file_path = f"results/gradcam/{detection_id}_gradcam.jpg"
            cv2.imwrite(gradcam_file_path, gradcam_img)
        
        # 🔚 Return response to frontend
        return {
            "detection_id": detection_id,
            "message": "Detection completed successfully",
            "result_image": f"/results/{detection_id}_result.jpg",
            "explanation_image": f"/results/explanations/{detection_id}_explanation.jpg" if explanation_file_path else None,
            "gradcam_image": f"/results/gradcam/{detection_id}_gradcam.jpg" if gradcam_file_path else None,
            "detections": detections
        }
    
    except Exception as e:
        # Log and return any errors during detection
        logger.error(f"Error during detection: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error during detection: {str(e)}")


# 🎨 Function to generate a simulated Grad-CAM visualization
def generate_gradcam(image, boxes, result):
    """
    Creates a heatmap overlay around detected regions.
    This is a simplified (non-gradient) version of Grad-CAM.
    """
    vis_img = image.copy()
    
    # Ensure image is RGB
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    elif vis_img.shape[2] == 1:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)
    
    height, width = vis_img.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Add gaussian blobs at each detection box center
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_width = x2 - x1
        box_height = y2 - y1
        
        y, x = np.ogrid[:height, :width]
        sigma_x = max(box_width / 6, 10)
        sigma_y = max(box_height / 6, 10)
        
        gaussian = np.exp(-(
            ((x - center_x) ** 2) / (2 * sigma_x ** 2) + 
            ((y - center_y) ** 2) / (2 * sigma_y ** 2)
        ))
        heatmap = np.maximum(heatmap, gaussian)
    
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    alpha = 0.4
    gradcam_visualization = cv2.addWeighted(vis_img, 1 - alpha, heatmap_colored, alpha, 0)
    
    # Draw bounding boxes on top
    for box in boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(gradcam_visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return gradcam_visualization


# 📸 Endpoint to retrieve Grad-CAM visualization for a previous detection
@app.get("/gradcam/{image_id}")
async def get_gradcam(image_id: str):
    """
    Returns a Grad-CAM image if it exists for the given detection ID.
    """
    result_path = f"results/{image_id}_result.jpg"
    if not os.path.exists(result_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    gradcam_path = f"results/gradcam/{image_id}_gradcam.jpg"
    
    if os.path.exists(gradcam_path):
        return FileResponse(gradcam_path)
    
    # If no gradcam image exists yet
    raise HTTPException(status_code=404, detail="Grad-CAM not available for this image")
