from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import cv2
import torch
import time
import uuid
import io
import os
import base64
import logging
from PIL import Image

from inference import model, grad_cam, preprocess, DEVICE, IMAGE_SIZE

# =========================================================
# CONFIG & LOGGING
# =========================================================

logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NeuroSeg AI API")

# =========================================================
# SESSION CACHE
# =========================================================

SESSIONS = {}
MAX_SESSIONS = 50

def cleanup_sessions():
    if len(SESSIONS) > MAX_SESSIONS:
        # Evict the oldest sessions
        sorted_sessions = sorted(SESSIONS.items(), key=lambda x: x[1]["timestamp"])
        evict_count = len(sorted_sessions) // 5
        for i in range(evict_count):
            SESSIONS.pop(sorted_sessions[i][0])
        logger.info(f"Evicted {evict_count} oldest sessions from cache.")

# =========================================================
# METRIC FUNCTIONS
# =========================================================

def dice_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    return (2. * intersection + 1e-8) / (np.sum(pred) + np.sum(gt) + 1e-8)


def iou_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    intersection = np.sum(pred * gt)
    union = np.sum((pred + gt) > 0)
    return (intersection + 1e-8) / (union + 1e-8)


def sensitivity_score(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    TP = np.sum((pred == 1) & (gt == 1))
    FN = np.sum((pred == 0) & (gt == 1))
    return (TP + 1e-8) / (TP + FN + 1e-8)

# =========================================================
# HELPERS
# =========================================================

def load_mri_image(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Failed to decode image file.")
    return image


def load_tif_mask(contents: bytes) -> np.ndarray:
    # Decode raw file bytes directly using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("Failed to decode mask file.")
    mask = (mask > 0).astype(np.uint8)
    return mask


def to_base64_png(image: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"


def compute_segmentation_results(image_np, pred_384, threshold, overlay_alpha, gt_mask=None):
    # Apply threshold to the 384x384 prediction
    mask_384 = (pred_384 > threshold).astype(np.uint8)
    
    # Resize mask to original image size
    mask_resized = cv2.resize(
        mask_384,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )
    
    # Statistics
    tumor_area = int(np.sum(mask_resized))
    tumor_percentage = (tumor_area / image_np.size) * 100
    
    if tumor_area > 0:
        confidence = float(pred_384[mask_384 == 1].mean())
    else:
        confidence = 0.0
        
    # Metrics
    dice = 0.0
    iou = 0.0
    sensitivity = 0.0
    
    if gt_mask is not None:
        gt_resized = cv2.resize(
            gt_mask,
            (mask_resized.shape[1], mask_resized.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        dice = dice_score(mask_resized, gt_resized)
        iou = iou_score(mask_resized, gt_resized)
        sensitivity = sensitivity_score(mask_resized, gt_resized)
        
    # Generate Overlay Image (BGR)
    overlay = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    
    # Red mask for tumor
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 2] = mask_resized * 255
    
    # Blend overlay with transparent red mask
    overlay = cv2.addWeighted(overlay, 1, red_mask, overlay_alpha, 0)
    
    # Draw Bounding Box (Green)
    contours, _ = cv2.findContours(
        mask_resized,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    # Draw Centroid (Yellow)
    M = cv2.moments(mask_resized)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 0), -1)
        
    return overlay, {
        "tumor_area": tumor_area,
        "tumor_percentage": round(tumor_percentage, 2),
        "confidence": round(confidence, 4),
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "sensitivity": round(sensitivity, 4)
    }

# =========================================================
# REQUEST SCHEMAS
# =========================================================

class UpdateRequest(BaseModel):
    session_id: str
    threshold: float
    overlay_alpha: float
    show_gradcam: bool

# =========================================================
# API ENDPOINTS
# =========================================================

@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    gt_file: UploadFile = File(None)
):
    try:
        logger.info(f"Received file for segmentation: {file.filename}")
        
        # Load files
        contents = await file.read()
        image_np = load_mri_image(contents)
        
        gt_mask = None
        if gt_file is not None:
            logger.info(f"Received ground truth mask: {gt_file.filename}")
            gt_contents = await gt_file.read()
            gt_mask = load_tif_mask(gt_contents)
            
        # Run Model Inference (resizes to 384x384 under the hood)
        input_tensor = preprocess(image_np)
        
        start_time = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output)
            pred_384 = prob.squeeze().cpu().numpy()
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Generate Grad-CAM Map (requires backward gradient computation)
        cam_384 = grad_cam.generate(input_tensor)
        
        # Save session to cache
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {
            "image_np": image_np,
            "pred": pred_384,
            "cam": cam_384,
            "gt_mask": gt_mask,
            "timestamp": time.time()
        }
        cleanup_sessions()
        
        # Compute default overlay & metrics (Threshold = 0.5, Alpha = 0.5)
        overlay, stats = compute_segmentation_results(
            image_np, pred_384, 0.5, 0.5, gt_mask
        )
        stats["inference_time"] = round(inference_time, 2)
        
        # Generate Grad-CAM blended image
        cam_resized = cv2.resize(cam_384, (image_np.shape[1], image_np.shape[0]))
        cam_resized = np.clip(cam_resized, 0, 1)
        cam_uint8 = np.uint8(cam_resized * 255)
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        original_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
        blended = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
        
        # Base64 encode images for immediate transmission
        original_b64 = to_base64_png(original_bgr)
        overlay_b64 = to_base64_png(overlay)
        gradcam_b64 = to_base64_png(blended)
        
        return {
            "session_id": session_id,
            "original": original_b64,
            "overlay": overlay_b64,
            "gradcam": gradcam_b64,
            "stats": stats,
            "device": "GPU" if "cuda" in str(DEVICE).lower() else "CPU"
        }
        
    except Exception as e:
        logger.error(f"Error in /api/upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/update")
async def api_update(req: UpdateRequest):
    try:
        session_id = req.session_id
        if session_id not in SESSIONS:
            logger.warning(f"Session not found or expired: {session_id}")
            raise HTTPException(status_code=404, detail="Session expired or not found. Please upload the image again.")
            
        session = SESSIONS[session_id]
        session["timestamp"] = time.time()  # Keep alive
        
        image_np = session["image_np"]
        pred_384 = session["pred"]
        cam_384 = session["cam"]
        gt_mask = session["gt_mask"]
        
        # Re-compute segmentations at new threshold / opacity
        overlay, stats = compute_segmentation_results(
            image_np, pred_384, req.threshold, req.overlay_alpha, gt_mask
        )
        
        overlay_b64 = to_base64_png(overlay)
        
        # Generate Grad-CAM base64 if toggled
        gradcam_b64 = None
        if req.show_gradcam:
            cam_resized = cv2.resize(cam_384, (image_np.shape[1], image_np.shape[0]))
            cam_resized = np.clip(cam_resized, 0, 1)
            cam_uint8 = np.uint8(cam_resized * 255)
            heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
            original_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
            heatmap = cv2.resize(heatmap, (original_bgr.shape[1], original_bgr.shape[0]))
            blended = cv2.addWeighted(original_bgr, 0.6, heatmap, 0.4, 0)
            gradcam_b64 = to_base64_png(blended)
            
        return {
            "overlay": overlay_b64,
            "gradcam": gradcam_b64,
            "stats": stats
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in /api/update: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session update failed: {str(e)}")


# =========================================================
# STATIC FILE ROUTING
# =========================================================

# Serve files under frontend/ at /static
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Serve index.html at root URL
@app.get("/")
async def read_index():
    index_path = os.path.join("frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")

