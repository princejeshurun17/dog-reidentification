"""
Inference service for dog detection and re-identification.
Loads YOLO for detection and re-ID model for embeddings.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict

import io
import base64
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db import DogDatabase
from faiss_store import FAISSStore


# Configuration
YOLO_MODEL_PATH = "models/yolo.pt"
REID_MODEL_PATH = "models/dog.pt"
CROP_SIZE = (224, 224)

# YOLO Detection Guardrails
YOLO_CONFIDENCE_THRESHOLD = 0.30  # Minimum confidence for dog face detection
MIN_BBOX_SIZE = 50  # Minimum width/height in pixels (filters tiny detections)
MAX_BBOX_SIZE = 2000  # Maximum width/height (filters unrealistic detections)
MIN_ASPECT_RATIO = 0.5  # Min width/height ratio (prevents extreme rectangles)
MAX_ASPECT_RATIO = 2.0  # Max width/height ratio

# Re-ID Guardrails
SIMILARITY_THRESHOLD = 0.50  # High precision threshold to minimize false positives
MIN_MARGIN = 0.05  # Minimum difference between top 2 matches for confident decision
LOW_CONFIDENCE_THRESHOLD = 0.60  # Warn user if match is below this
EMBEDDING_DIM = 2048  # ResNet50 layer4 output dimension


# Initialize FastAPI app
app = FastAPI(title="Dog Re-ID Inference Service")

# Global model instances
yolo_model = None
reid_model = None
device = None
db = None
faiss_store = None


class EnrollRequest(BaseModel):
    """Request model for enrolling a new dog."""
    dog_id: Optional[int] = None
    name: str
    contact_info: str = ""
    notes: str = ""
    image_path: str


class DetectionResult(BaseModel):
    """Model for detection results."""
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str


class MatchResult(BaseModel):
    """Model for similarity match results."""
    dog_id: int
    name: str
    similarity: float
    contact_info: str = ""
    notes: str = ""
    image_path: str = ""


@app.on_event("startup")
async def startup_event():
    """Initialize models and database on startup."""
    global yolo_model, reid_model, device, db, faiss_store
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load YOLO model
    if not os.path.exists(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found at {YOLO_MODEL_PATH}")
    
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    yolo_model.to(device)
    
    # Load Re-ID model
    if not os.path.exists(REID_MODEL_PATH):
        raise FileNotFoundError(f"Re-ID model not found at {REID_MODEL_PATH}")
    
    print(f"Loading Re-ID model from {REID_MODEL_PATH}...")
    checkpoint = torch.load(REID_MODEL_PATH, map_location=device, weights_only=False)
    
    # Create ResNet50 backbone and use full layer4 features
    reid_model = models.resnet50(pretrained=False)
    
    # Load the backbone weights first
    if isinstance(checkpoint, dict) and 'state_dict_backbone' in checkpoint:
        reid_model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)
        print("✓ Loaded ResNet50 backbone weights")
    else:
        raise ValueError("Unexpected model format. Expected dict with 'state_dict_backbone' key.")
    
    # Remove only the final FC layer to get 2048-dim features from layer4
    reid_model.fc = nn.Identity()
    
    reid_model.to(device)
    reid_model.eval()
    print("✓ Using layer4 features (2048-dim) - 84% accuracy with 40% separation")
    
    # Initialize database and FAISS
    print("Initializing database and FAISS...")
    db = DogDatabase()
    faiss_store = FAISSStore(embedding_dim=EMBEDDING_DIM)
    
    # Load dog IDs mapping
    print("Loading dog IDs mapping...")
    faiss_store.load_dog_ids()
    
    # Rebuild FAISS index if needed
    dogs = db.get_all_dogs()
    print(f"Database has {len(dogs)} dogs")
    print(f"FAISS index has {faiss_store.index.ntotal} embeddings")
    print(f"FAISS dog_ids has {len(faiss_store.dog_ids)} mappings")
    
    if len(dogs) != faiss_store.index.ntotal:
        print(f"⚠ Mismatch detected! Rebuilding FAISS index from database...")
        dogs_data = [(dog['dog_id'], dog['embedding']) for dog in dogs]
        faiss_store.rebuild_from_database(dogs_data)
    else:
        print("✓ FAISS index is in sync with database")
    
    # Verify after rebuild
    print(f"Final: FAISS has {faiss_store.index.ntotal} embeddings, {len(faiss_store.dog_ids)} ID mappings")
    
    print("Inference service ready!")


def detect_dog_faces(image: Image.Image) -> List[Dict[str, Any]]:
    """
    Detect dog faces using YOLO model with validation guardrails.
    
    Filters out false positives by checking:
    - Detection confidence threshold
    - Bounding box size (too small = noise, too large = unrealistic)
    - Aspect ratio (prevents extreme rectangles)
    
    Args:
        image: PIL Image
        
    Returns:
        List of validated detection dictionaries with bbox, confidence, class_id
    """
    results = yolo_model(image, verbose=False)
    
    detections = []
    rejected_count = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            
            # Only process face detections (class 1)
            if class_id == 1:
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                confidence = float(box.conf[0])
                
                # Guardrail 1: Confidence threshold
                if confidence < YOLO_CONFIDENCE_THRESHOLD:
                    print(f"[YOLO REJECTED] Low confidence: {confidence:.3f} < {YOLO_CONFIDENCE_THRESHOLD}")
                    rejected_count += 1
                    continue
                
                # Calculate bbox dimensions
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Guardrail 2: Minimum size (filter noise)
                if width < MIN_BBOX_SIZE or height < MIN_BBOX_SIZE:
                    print(f"[YOLO REJECTED] Too small: {width:.0f}x{height:.0f} < {MIN_BBOX_SIZE}")
                    rejected_count += 1
                    continue
                
                # Guardrail 3: Maximum size (filter unrealistic)
                if width > MAX_BBOX_SIZE or height > MAX_BBOX_SIZE:
                    print(f"[YOLO REJECTED] Too large: {width:.0f}x{height:.0f} > {MAX_BBOX_SIZE}")
                    rejected_count += 1
                    continue
                
                # Guardrail 4: Aspect ratio (filter extreme rectangles)
                if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
                    print(f"[YOLO REJECTED] Invalid aspect ratio: {aspect_ratio:.2f} (valid: {MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO})")
                    rejected_count += 1
                    continue
                
                # Passed all guardrails
                print(f"[YOLO ACCEPTED] Confidence: {confidence:.3f}, Size: {width:.0f}x{height:.0f}, Ratio: {aspect_ratio:.2f}")
                detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": "face",
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio
                })
    
    if rejected_count > 0:
        print(f"[YOLO SUMMARY] Accepted: {len(detections)}, Rejected: {rejected_count}")
    
    return detections


def crop_and_resize(image: Image.Image, bbox: List[float]) -> Image.Image:
    """
    Crop and resize image to fixed size.
    
    Args:
        image: PIL Image
        bbox: Bounding box [x1, y1, x2, y2]
        
    Returns:
        Cropped and resized image
    """
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image.crop((x1, y1, x2, y2))
    resized = cropped.resize(CROP_SIZE, Image.LANCZOS)
    return resized


def generate_embedding(image: Image.Image) -> np.ndarray:
    """
    Generate embedding vector from cropped face image.
    
    Args:
        image: PIL Image (224x224)
        
    Returns:
        Embedding vector as numpy array (2048-dim from layer4)
    """
    # ImageNet normalization (standard for ResNet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Convert to tensor and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply normalization
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(device)
    
    # Generate embedding
    with torch.no_grad():
        features = reid_model(img_tensor)
    
    # Layer4 output is (batch, 2048, 1, 1) - squeeze to (2048,)
    if len(features.shape) == 4:
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    
    embedding = features.squeeze()
    
    # Convert to numpy
    embedding_np = embedding.cpu().numpy().flatten()
    
    # Check normalization
    norm = np.linalg.norm(embedding_np)
    print(f"[EMBEDDING] Generated embedding: shape={embedding_np.shape}, norm={norm:.4f}")
    print(f"[EMBEDDING] Stats - Min: {embedding_np.min():.3f}, Max: {embedding_np.max():.3f}, Mean: {embedding_np.mean():.3f}, Std: {embedding_np.std():.3f}")
    
    return embedding_np


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Dog Re-ID Inference Service",
        "status": "running",
        "device": str(device),
        "dogs_in_db": db.get_dog_count() if db else 0
    }


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Main inference endpoint: detect faces, generate embeddings, find matches.
    
    Returns:
        JSON with detections and matches
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Detect dog faces
        detections = detect_dog_faces(image)
        
        if not detections:
            return JSONResponse({
                "success": True,
                "message": "No dog faces detected",
                "detections": []
            })
        
        # Process each detection
        results = []
        for detection in detections:
            # Crop and resize
            cropped = crop_and_resize(image, detection["bbox"])
            
            # Encode cropped image to base64 for display
            buffered = io.BytesIO()
            cropped.save(buffered, format="JPEG")
            cropped_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Generate embedding
            embedding = generate_embedding(cropped)
            
            # Debug: print embedding statistics
            print(f"[EMBEDDING] Stats - min: {embedding.min():.4f}, max: {embedding.max():.4f}, mean: {embedding.mean():.4f}, std: {embedding.std():.4f}")
            print(f"[EMBEDDING] First 10 values: {embedding[:10]}")
            
            # Search for matches
            matches = faiss_store.search(embedding, k=3)  # Get top 3 for debugging
            
            print(f"[SEARCH] Top matches for detection:")
            for i, (dog_id, similarity) in enumerate(matches):
                dog = db.get_dog(dog_id)
                dog_name = dog['name'] if dog else 'Unknown'
                print(f"  {i+1}. Dog ID {dog_id} ({dog_name}): similarity = {similarity:.4f}")
            
            detection_result = {
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "embedding_dim": len(embedding),
                "cropped_image": f"data:image/jpeg;base64,{cropped_base64}",
                "matches": [],
                "top_scores": [{"dog_id": m[0], "similarity": float(m[1])} for m in matches[:3]]  # Debug info
            }
            
            # Guardrails: Check threshold, margin, and confidence
            if matches and len(matches) > 0:
                top_similarity = matches[0][1]
                second_similarity = matches[1][1] if len(matches) > 1 else 0.0
                margin = top_similarity - second_similarity
                
                # Check if match meets all guardrails
                meets_threshold = top_similarity >= SIMILARITY_THRESHOLD
                meets_margin = margin >= MIN_MARGIN or len(matches) == 1
                is_high_confidence = top_similarity >= LOW_CONFIDENCE_THRESHOLD
                
                print(f"[GUARDRAILS] Similarity: {top_similarity:.4f}, Margin: {margin:.4f}")
                print(f"[GUARDRAILS] Threshold check: {meets_threshold}, Margin check: {meets_margin}")
                
                if meets_threshold and meets_margin:
                    dog_id, similarity = matches[0]
                    dog = db.get_dog(dog_id)
                    
                    if dog:
                        # Determine confidence level
                        if is_high_confidence and margin >= 0.10:
                            confidence_level = "high"
                            confidence_text = "High confidence match"
                        elif is_high_confidence:
                            confidence_level = "medium-high"
                            confidence_text = "Good match"
                        else:
                            confidence_level = "medium"
                            confidence_text = "Moderate confidence - please verify"
                        
                        print(f"[MATCH] {confidence_text}: {dog['name']} (similarity: {similarity:.4f}, margin: {margin:.4f})")
                        
                        match_info = {
                            "dog_id": dog_id,
                            "name": dog["name"],
                            "similarity": similarity,
                            "confidence_level": confidence_level,
                            "confidence_text": confidence_text,
                            "margin": margin,
                            "contact_info": dog.get("contact_info", ""),
                            "notes": dog.get("notes", ""),
                            "image_path": dog.get("image_path", "")
                        }
                        detection_result["matches"].append(match_info)
                        
                        # Log identification
                        db.log_identification(dog_id, file.filename, similarity)
                elif meets_threshold and not meets_margin:
                    print(f"[REJECTED] Similarity {top_similarity:.4f} above threshold but margin {margin:.4f} too small (ambiguous match)")
                else:
                    print(f"[NO MATCH] Best similarity {top_similarity:.4f} below threshold {SIMILARITY_THRESHOLD}")
            else:
                print(f"[NO MATCH] No dogs in database")
            
            # Store embedding for potential enrollment (temporary)
            detection_result["embedding"] = embedding.tolist()
            
            results.append(detection_result)
        
        return JSONResponse({
            "success": True,
            "detections": results,
            "total_faces": len(results)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/enroll")
async def enroll(request: EnrollRequest):
    """
    Enroll a new dog or update existing one.
    Expects embedding to be provided from previous inference.
    """
    try:
        # This endpoint expects the embedding to be passed or retrieved
        # For simplicity, we'll accept image_path and re-process
        # In production, you might cache embeddings temporarily
        
        if not request.name:
            raise HTTPException(status_code=400, detail="Dog name is required")
        
        # Note: In a real implementation, you'd retrieve the cached embedding
        # from the previous inference call. For now, return success message.
        
        return JSONResponse({
            "success": True,
            "message": "Enrollment endpoint - implement caching or pass embedding",
            "dog_id": None
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dogs")
async def list_dogs():
    """List all enrolled dogs."""
    try:
        dogs = db.get_all_dogs()
        
        # Remove embedding from response (too large)
        dogs_info = []
        for dog in dogs:
            dog_info = {
                "dog_id": dog["dog_id"],
                "name": dog["name"],
                "contact_info": dog.get("contact_info", ""),
                "notes": dog.get("notes", ""),
                "image_path": dog.get("image_path", ""),
                "created_at": dog.get("created_at", "")
            }
            dogs_info.append(dog_info)
        
        return JSONResponse({
            "success": True,
            "dogs": dogs_info,
            "total": len(dogs_info)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        return JSONResponse({
            "success": True,
            "database": {
                "total_dogs": db.get_dog_count()
            },
            "faiss": faiss_store.get_stats(),
            "device": str(device),
            "similarity_threshold": SIMILARITY_THRESHOLD
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history(limit: int = 10):
    """Get recent identification history."""
    try:
        history = db.get_recent_identifications(limit)
        
        return JSONResponse({
            "success": True,
            "history": history,
            "count": len(history)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_faiss():
    """Reload FAISS index from disk to pick up new enrollments."""
    global faiss_store
    
    try:
        print("Reloading FAISS index from disk...")
        
        # Reinitialize FAISS store (loads from disk)
        faiss_store = FAISSStore(embedding_dim=EMBEDDING_DIM)
        faiss_store.load_dog_ids()
        
        # Verify sync with database
        dogs = db.get_all_dogs()
        db_count = len(dogs)
        faiss_count = faiss_store.index.ntotal
        
        print(f"✓ FAISS reloaded: {faiss_count} embeddings, DB has {db_count} dogs")
        
        if db_count != faiss_count:
            print("⚠ Warning: FAISS and DB out of sync. Rebuilding...")
            dogs_data = [(dog['dog_id'], dog['embedding']) for dog in dogs]
            faiss_store.rebuild_from_database(dogs_data)
        
        return JSONResponse({
            "success": True,
            "message": "FAISS index reloaded",
            "faiss_embeddings": faiss_store.index.ntotal,
            "database_dogs": db_count
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
