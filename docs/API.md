# API Documentation

## Inference Service API (Port 8000)

Base URL: `http://127.0.0.1:8000`

### Endpoints

---

#### `GET /`

**Description**: Health check and service status

**Response**:
```json
{
  "service": "Dog Re-ID Inference Service",
  "status": "running",
  "device": "cuda",
  "dogs_in_db": 15
}
```

---

#### `POST /infer`

**Description**: Detect dog faces and identify them

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**:
```json
{
  "success": true,
  "detections": [
    {
      "bbox": [100, 150, 300, 350],
      "confidence": 0.95,
      "embedding_dim": 2048,
      "cropped_image": "base64_encoded_string",
      "matches": [
        {
          "dog_id": 3,
          "name": "Max",
          "similarity": 0.87,
          "margin": 0.12,
          "confidence_level": "high",
          "confidence_text": "High confidence match (87.0% similarity)",
          "contact_info": "+1234567890",
          "notes": "Golden Retriever, 3 years old",
          "image_path": "/uploads/20231115_140523_max.jpg"
        }
      ],
      "embedding": [0.123, -0.456, ...]
    }
  ],
  "total_faces": 1
}
```

**Enhanced Fields**:
- `cropped_image`: Base64-encoded JPEG of detected dog face (for display)
- `margin`: Difference between top match and second-best match
- `confidence_level`: "high", "medium-high", or "medium" (based on threshold and margin)
- `confidence_text`: Human-readable confidence explanation
- `embedding_dim`: Always 2048 (ResNet50 Layer4)

**Confidence Levels**:
- **High**: Similarity ≥ 0.70, Margin ≥ 0.05
- **Medium-High**: Similarity ≥ 0.60, Margin ≥ 0.05
- **Medium**: Similarity ≥ 0.60, Margin < 0.05 (requires verification)

**YOLO Detection Guardrails**:
- Minimum confidence: 0.45
- Bounding box size: 50-2000 pixels
- Aspect ratio: 0.5-2.0
- Detections failing these checks are filtered out
```

**Error Response**:
```json
{
  "detail": "Error message"
}
```

---

#### `POST /enroll`

**Description**: Enroll a new dog (placeholder - see UI enroll endpoint)

**Request**:
```json
{
  "name": "Buddy",
  "contact_info": "+1234567890",
  "notes": "Labrador, friendly",
  "image_path": "/uploads/buddy.jpg"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Enrollment endpoint - implement caching or pass embedding",
  "dog_id": null
}
```

---

#### `GET /dogs`

**Description**: List all enrolled dogs

**Response**:
```json
{
  "success": true,
  "dogs": [
    {
      "dog_id": 1,
      "name": "Max",
      "contact_info": "+1234567890",
      "notes": "Golden Retriever",
      "image_path": "/uploads/max.jpg",
      "created_at": "2023-11-15 14:05:23"
    }
  ],
  "total": 1
}
```

---

#### `GET /stats`

**Description**: Get system statistics

**Response**:
```json
{
  "success": true,
  "database": {
    "total_dogs": 15
  },
  "faiss": {
    "total_embeddings": 15,
    "dimension": 512,
    "dog_ids_count": 15
  },
  "device": "cuda",
  "similarity_threshold": 0.7
}
```

---

#### `GET /history?limit=10`

**Description**: Get recent identification history

**Query Parameters**:
- `limit` (optional): Number of records to return (default: 10)

**Response**:
```json
{
  "success": true,
  "history": [
    {
      "id": 1,
      "dog_id": 3,
      "name": "Max",
      "image_path": "20231115_140523_dog.jpg",
      "similarity_score": 0.87,
      "identified_at": "2023-11-15 14:05:23"
    }
  ],
  "count": 1
}
```

---

## UI Service API (Port 5000)

Base URL: `http://127.0.0.1:5000`

### Endpoints

---

#### `GET /`

**Description**: Main web interface (HTML page)

**Response**: HTML page for upload mode

---

#### `GET /live`

**Description**: Live video detection interface (HTML page)

**Response**: HTML page with:
- Real-time video streaming from webcam
- Bounding box overlay on detected dogs
- Two modes: Live Stream (continuous) and Capture Photo (single frame)
- Camera switching for multi-camera devices
- Frame rate control (1-10 FPS)
- Instant registration for unknown dogs

**Features**:
- MediaDevices API for camera access
- Canvas overlay for real-time bounding boxes
- Color-coded boxes: Green (high confidence), Blue (medium), Orange (unknown)
- Mobile-optimized with front/back camera switching
- Comprehensive error handling for camera access

---

#### `POST /api/process`

**Description**: Upload and process an image (proxies to inference service)

**Usage**: Works for both upload mode and live video mode (frame-by-frame processing)

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file or video frame)

**Response**: Same as inference service `/infer` with additional fields:
```json
{
  "success": true,
  "uploaded_file": "20231115_140523_dog.jpg",
  "upload_path": "/uploads/20231115_140523_dog.jpg",
  "detections": [
    {
      "bbox": [100, 150, 300, 350],
      "confidence": 0.95,
      "cropped_image": "base64_encoded_jpeg",
      "embedding": [0.123, -0.456, ...],
      "matches": [
        {
          "dog_id": 3,
          "name": "Max",
          "similarity": 0.87,
          "margin": 0.12,
          "confidence_level": "high",
          "confidence_text": "High confidence match"
        }
      ]
    }
  ],
  "total_faces": 1
}
```

**Live Video Usage**:
- Frame rate: 1-10 FPS (default: 3 FPS)
- Processing latency: ~170ms average (45ms detection + 123ms embedding + 2ms search)
- Recommended: 3 FPS for smooth performance with 333ms per frame buffer

**Error Responses**:
```json
// No file provided
{
  "success": false,
  "error": "No file provided"
}

// Invalid file type
{
  "success": false,
  "error": "Invalid file type"
}

// Inference service unavailable
{
  "success": false,
  "error": "Cannot connect to inference service. Please ensure it is running."
}
```

---

#### `POST /api/enroll`

**Description**: Enroll a new dog with metadata and embedding

**Important**: This endpoint expects the embedding array from a detection result, not a file upload.

**Request**:
```json
{
  "name": "Buddy",
  "contact_info": "+1234567890",
  "notes": "Labrador, 2 years old",
  "embedding": [0.123, -0.456, ...],  // 2048-dim array from detection
  "image_path": "/uploads/buddy.jpg"
}
```

**Response**:
```json
{
  "success": true,
  "message": "Successfully enrolled Buddy",
  "dog_id": 16
}
```

**Workflow**:
1. Upload image via `/api/process` → Get detection with embedding
2. Extract `embedding` from detection result
3. Send enrollment request with embedding array
4. Backend adds to database and FAISS index
5. Inference service automatically reloads FAISS

**Live Mode Usage**:
- Capture photo in Capture Photo mode
- Fill in name/owner in registration overlay
- Embedding automatically extracted from detection
- Registration happens instantly without page reload
```

**Error Response**:
```json
{
  "success": false,
  "error": "Dog name is required"
}
```

---

#### `GET /api/dogs`

**Description**: Fetch list of all enrolled dogs

**Response**: Proxied from inference service `/dogs`

---

#### `GET /api/history?limit=10`

**Description**: Fetch identification history

**Query Parameters**:
- `limit` (optional): Number of records (default: 10)

**Response**: Proxied from inference service `/history`

---

#### `GET /api/stats`

**Description**: Fetch system statistics

**Response**: Proxied from inference service `/stats`

---

#### `GET /uploads/<filename>`

**Description**: Serve uploaded images

**Response**: Image file

---

#### `GET /health`

**Description**: Health check for both services

**Response**:
```json
{
  "ui_service": "running",
  "inference_service": "running"
}
```

---

## Data Models

### Detection Object
```typescript
{
  bbox: [number, number, number, number],  // [x1, y1, x2, y2]
  confidence: number,                       // 0-1
  embedding_dim: number,                    // Usually 512
  matches: Match[],                         // Array of matches
  embedding: number[]                       // Embedding vector
}
```

### Match Object
```typescript
{
  dog_id: number,
  name: string,
  similarity: number,        // 0-1 (cosine similarity)
  contact_info: string,
  notes: string,
  image_path: string
}
```

### Dog Object
```typescript
{
  dog_id: number,
  name: string,
  contact_info: string,
  notes: string,
  image_path: string,
  created_at: string,       // ISO timestamp
  updated_at: string        // ISO timestamp
}
```

---

## Error Codes

| Status Code | Description |
|------------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid input) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (inference service down) |

---

## Usage Examples

### Using cURL (Windows PowerShell)

**Inference:**
```powershell
curl -X POST -F "file=@dog.jpg" http://127.0.0.1:8000/infer
```

**Get Statistics:**
```powershell
curl http://127.0.0.1:8000/stats
```

**List Dogs:**
```powershell
curl http://127.0.0.1:8000/dogs
```

### Using Python Requests

```python
import requests

# Process image
with open('dog.jpg', 'rb') as f:
    response = requests.post(
        'http://127.0.0.1:5000/api/process',
        files={'file': f}
    )
    result = response.json()
    print(result)

# Enroll new dog
data = {
    'name': 'Buddy',
    'contact_info': '+1234567890',
    'notes': 'Friendly dog',
    'embedding': result['detections'][0]['embedding'],
    'image_path': result['upload_path']
}
response = requests.post(
    'http://127.0.0.1:5000/api/enroll',
    json=data
)
print(response.json())
```

### Using JavaScript (Fetch API)

#### Upload Mode
```javascript
// Upload and process image
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://127.0.0.1:5000/api/process', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Detections:', data.detections);
    // Display cropped faces
    data.detections.forEach(det => {
        if (det.cropped_image) {
            const img = document.createElement('img');
            img.src = `data:image/jpeg;base64,${det.cropped_image}`;
            document.body.appendChild(img);
        }
    });
});

// Enroll dog
fetch('http://127.0.0.1:5000/api/enroll', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        name: 'Buddy',
        contact_info: '+1234567890',
        notes: 'Labrador',
        embedding: detectionResult.embedding,
        image_path: uploadedPath
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

#### Live Video Mode
```javascript
// Start camera
const video = document.getElementById('videoElement');
const stream = await navigator.mediaDevices.getUserMedia({
    video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'  // or 'environment' for back camera
    }
});
video.srcObject = stream;

// Capture and process frame
async function processFrame() {
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    const blob = await new Promise(resolve => 
        canvas.toBlob(resolve, 'image/jpeg', 0.8)
    );
    
    const formData = new FormData();
    formData.append('file', blob, 'frame.jpg');
    
    const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    // Draw bounding boxes
    result.detections.forEach(det => {
        const [x, y, width, height] = det.bbox;
        ctx.strokeStyle = det.matches.length > 0 ? '#10b981' : '#f59e0b';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);
        
        if (det.matches.length > 0) {
            const match = det.matches[0];
            ctx.fillStyle = '#10b981';
            ctx.fillText(`${match.name} (${(match.similarity * 100).toFixed(0)}%)`, x, y - 10);
        }
    });
}

// Run at 3 FPS
setInterval(() => {
    if (!isProcessing) {
        processFrame();
    }
}, 333);  // 333ms = 3 FPS
```

---

## Performance Metrics

### Upload Mode
- Processing time: 2-5 seconds (CPU), 0.5-1 second (GPU)
- FAISS search: <100ms for 10,000+ dogs

### Live Video Mode
- Frame processing: ~170ms average
  - YOLO detection: 45ms
  - Embedding generation: 123ms
  - FAISS search: 2ms
- Recommended frame rate: 3 FPS (333ms per frame)
- Max theoretical FPS: ~6 FPS (limited by processing time)
- Network overhead: +10-50ms depending on connection
