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
      "embedding_dim": 512,
      "matches": [
        {
          "dog_id": 3,
          "name": "Max",
          "similarity": 0.87,
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

**Response**: HTML page

---

#### `POST /api/process`

**Description**: Upload and process an image (proxies to inference service)

**Request**:
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response**: Same as inference service `/infer` with additional fields:
```json
{
  "success": true,
  "uploaded_file": "20231115_140523_dog.jpg",
  "upload_path": "/uploads/20231115_140523_dog.jpg",
  "detections": [...],
  "total_faces": 1
}
```

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

**Request**:
```json
{
  "name": "Buddy",
  "contact_info": "+1234567890",
  "notes": "Labrador, 2 years old",
  "embedding": [0.123, -0.456, ...],
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
