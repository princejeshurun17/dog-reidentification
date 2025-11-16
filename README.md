# Dog Re-Identification System

A Windows-friendly local application for dog face detection and re-identification using YOLO and deep learning embeddings with FAISS similarity search.

## System Overview

This system consists of two separate services:

1. **Flask UI Server** (`frontend/`) - Web interface for uploading images and viewing results
2. **FastAPI Inference Service** (`backend/`) - ML model inference and FAISS search

### Features

- 🔍 Dog face detection using YOLO
- 🐕 Face cropping and normalization (224x224)
- 🧠 Deep learning embeddings (2048-dim Layer4 features) for re-identification
- 📊 FAISS-based similarity search with cosine similarity
- 💾 SQLite database for metadata storage
- 🖼️ Cropped face display in detection cards
- 🎯 60% similarity threshold with multi-level guardrails
- 🛡️ Confidence indicators (High/Medium-High/Medium)
- ⚡ GPU acceleration with CPU fallback
- 📹 **Live video detection** with real-time streaming and bounding boxes
- 🔄 **Camera switching** for mobile devices (front/back camera)
- 📸 **Capture mode** with instant registration overlay
- 📱 **Mobile-optimized** with comprehensive camera error handling
- 📈 **84.1% accuracy** on PetFace dataset

## Prerequisites

- Python 3.8 or higher
- Windows OS
- 4GB+ RAM recommended
- (Optional) NVIDIA GPU with CUDA for faster inference

## Installation

### 1. Clone or Download Repository

Ensure you have the following structure:
```
d:\FYP\frontend\
├── models/                 # Model files
│   ├── yolo.pt            # YOLO detection model
│   └── dog.pt             # ResNet50 re-ID model
├── requirements.txt
├── backend/               # FastAPI inference service
│   ├── inference_service.py
│   ├── db.py
│   └── faiss_store.py
├── frontend/              # Flask UI server
│   ├── app.py
│   └── templates/
│       ├── index.html         # Upload mode
│       └── live.html          # Live video detection mode
├── tests/                 # Test scripts
│   ├── test_system.py
│   ├── quick_test.py
│   ├── compare_layers.py
│   ├── enhanced_inference.py
│   └── evaluate_reid.py
├── docs/                  # Documentation
│   ├── API.md
│   ├── architecture.md
│   └── plan-dogReid.prompt.md
├── logs/                  # Test results
├── data/                  # Created automatically
│   ├── uploads/
│   ├── dogs.db
│   ├── faiss.index
│   └── faiss.index.ids.npy
├── scripts/
│   ├── manage_db.py
│   └── test_integration.py
├── README.md
├── QUICKSTART.md
└── SESSION_SUMMARY.md     # Development log
```
    └── faiss.index
```

### 2. Create Virtual Environment

```powershell
cd d:\FYP\frontend
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

#### For CPU-only (recommended for testing):
```powershell
pip install -r requirements.txt
```

#### For GPU (CUDA 11.8):
```powershell
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install flask fastapi uvicorn[standard] python-multipart ultralytics Pillow faiss-cpu sqlalchemy numpy requests python-dotenv pytest pytest-asyncio
```

### 4. Verify Model Files

Ensure `yolo.pt` and `dog.pt` are in the root directory:
```powershell
ls yolo.pt, dog.pt
```

## Running the System

### Step 1: Start the Inference Service

Open a terminal and run:
```powershell
cd d:\FYP\frontend
.\venv\Scripts\Activate.ps1
python backend\inference_service.py
```

You should see:
```
Using device: cuda  # or cpu
Loading YOLO model from yolo.pt...
Loading Re-ID model from dog.pt...
Inference service ready!
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Start the Flask UI Server

Open a **new** terminal window and run:
```powershell
cd d:\FYP\frontend
.\venv\Scripts\Activate.ps1
python frontend\app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### Step 3: Access the Web Interface

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## Usage

### Upload Mode (Traditional)

1. Open browser: `http://localhost:5000`
2. Click "Upload Image" and select a dog photo
3. View detection results:
   - If dog is registered: See match with confidence score
   - If dog is unknown: Option to register with name and owner info
4. Registered dogs are added to database for future recognition

### Live Video Detection Mode (New!)

1. Navigate to: `http://localhost:5000/live`
2. **Stream Mode**:
   - Click "Start Camera" to begin live detection
   - Real-time bounding boxes appear over detected dogs
   - Color-coded boxes: Green (match), Orange (unknown), Blue (medium confidence)
   - Adjust frame rate (1-10 FPS) based on performance
   - Toggle bounding boxes and labels on/off
3. **Capture Mode**:
   - Switch to "Capture Photo" mode
   - Click "Take Photo" to capture current frame
   - View detailed results in overlay
   - Register unknown dogs instantly with name/owner form
4. **Camera Switching**:
   - Click "Switch Camera" to toggle between cameras
   - Mobile: Switches between front/back cameras
   - Desktop: Cycles through all available webcams

### Camera Troubleshooting

**Permission Denied:**
- **Browser**: Click the camera icon in address bar → Allow
- **iOS Safari**: Tap "Allow" when prompted (required each session)
- **Android Chrome**: Settings → Site Settings → Camera → Allow

**Camera In Use:**
- Close other apps using the camera (Instagram, Zoom, etc.)
- On mobile: Force close background apps

**HTTPS Required:**
- Some browsers require HTTPS for camera access
- Use `localhost` for local testing (HTTPS not required)
- For remote access, set up HTTPS or use ngrok/similar

**No Camera Detected:**
- Check device has working camera
- Try different browser (Chrome recommended)
- Check system permissions (Windows Settings → Privacy → Camera)

### Performance Tips

- **Frame Rate**: Start with 3 FPS (default), increase if system handles well
- **Latency**: ~170ms average processing time per frame
- **Max Theoretical FPS**: ~6 FPS (based on processing latency)
- **Mobile**: Use lower frame rates (1-3 FPS) for better battery life

## Usage

### Identifying a Dog

1. Click "Choose Image" or drag-and-drop a dog photo
2. The system will:
   - Detect dog faces using YOLO
   - Crop and resize faces to 224x224
   - Generate embeddings using the re-ID model
   - Search FAISS index for similar faces
3. View results:
   - **Match Found**: Shows dog name, similarity score, and metadata
   - **Unknown Dog**: Offers to register the new dog

### Registering a New Dog

1. When an unknown dog is detected, click "Register This Dog"
2. Fill in the form:
   - **Name** (required)
   - **Contact Information** (optional)
   - **Notes** (optional - breed, age, etc.)
3. Click "Register"
4. The dog is added to both SQLite and FAISS index

### View Statistics

The dashboard shows:
- Total registered dogs
- Processing device (CPU/CUDA)
- System status

## API Documentation

### Inference Service (Port 8000)

#### `GET /`
Health check and status

#### `POST /infer`
- **Input**: Image file (multipart/form-data)
- **Output**: JSON with detections, embeddings, and matches
```json
{
  "success": true,
  "detections": [{
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.95,
    "matches": [{
      "dog_id": 1,
      "name": "Max",
      "similarity": 0.87
    }],
    "embedding": [...]
  }]
}
```

#### `GET /dogs`
List all registered dogs

#### `GET /stats`
System statistics (dog count, FAISS stats, device info)

#### `GET /history?limit=10`
Recent identification history

### UI Service (Port 5000)

#### `GET /`
Main web interface

#### `POST /api/process`
Upload and process image (proxies to inference service)

#### `POST /api/enroll`
Register new dog with metadata

#### `GET /api/dogs`
Fetch all dogs

#### `GET /api/history`
Fetch identification history

## Configuration

Edit the following constants in `backend/inference_service.py`:

```python
YOLO_MODEL_PATH = "yolo.pt"          # Path to YOLO model
REID_MODEL_PATH = "dog.pt"           # Path to re-ID model
CROP_SIZE = (224, 224)               # Face crop size
SIMILARITY_THRESHOLD = 0.70          # Matching threshold (0-1)
EMBEDDING_DIM = 512                  # Re-ID embedding dimension
```

## CLI Utilities

### Rebuild FAISS Index

If the FAISS index becomes corrupted or out of sync:

```powershell
python -c "
import sys
sys.path.insert(0, 'backend')
from db import DogDatabase
from faiss_store import FAISSStore

db = DogDatabase()
faiss = FAISSStore()
dogs = db.get_all_dogs()
dogs_data = [(d['dog_id'], d['embedding']) for d in dogs]
faiss.rebuild_from_database(dogs_data)
print('FAISS index rebuilt successfully')
"
```

### List All Dogs

```powershell
python -c "
import sys
sys.path.insert(0, 'backend')
from db import DogDatabase

db = DogDatabase()
dogs = db.get_all_dogs()
for dog in dogs:
    print(f'{dog[\"dog_id\"]}: {dog[\"name\"]} - Created: {dog[\"created_at\"]}')
"
```

### Clear Database (⚠️ Destructive)

```powershell
Remove-Item data\dogs.db, data\faiss.index, data\faiss.index.ids.npy -Force
```

## Troubleshooting

### "Cannot connect to inference service"

**Solution**: Ensure the inference service is running on port 8000:
```powershell
netstat -ano | findstr :8000
```

### "CUDA out of memory"

**Solution**: Use CPU mode or reduce batch size. Edit `inference_service.py`:
```python
device = torch.device("cpu")  # Force CPU
```

### "No module named 'faiss'"

**Solution**: Install FAISS:
```powershell
pip install faiss-cpu
```

### Models not loading

**Solution**: Verify model files exist and are not corrupted:
```powershell
ls -l yolo.pt, dog.pt
```

### Port already in use

**Solution**: Kill the process or change ports:
```powershell
# Find process using port 5000
netstat -ano | findstr :5000
# Kill process (replace PID)
taskkill /PID <PID> /F
```

### Camera access denied/blocked

**Solution**: Check browser permissions:
```
Chrome: Settings → Privacy and Security → Site Settings → Camera
Firefox: Preferences → Privacy & Security → Permissions → Camera
Safari: Preferences → Websites → Camera
```

**Mobile specific**:
- **iOS**: Settings → Safari → Camera (must be "Ask")
- **Android**: Settings → Apps → Chrome → Permissions → Camera

### Camera "already in use" error

**Solution**: Close other applications using camera:
```powershell
# Windows: Check camera processes
Get-Process | Where-Object {$_.ProcessName -match "zoom|teams|skype"}
```

### Bounding boxes not appearing

**Solution**: Check settings panel toggles:
- Ensure "Show Bounding Boxes" is checked
- Verify "Show Labels" is enabled if you want labels
- Check browser console for JavaScript errors

### High latency/low FPS in live mode

**Solution**: Reduce processing load:
- Lower frame rate to 1-2 FPS
- Close other applications
- Use GPU if available (check inference_service.py logs)
- Reduce camera resolution (fallback constraints activate automatically)

## Testing

### Unit Tests

```powershell
pytest backend/tests/ -v
```

### Integration Test

```powershell
# Ensure both services are running, then:
curl -X POST -F "file=@test_image.jpg" http://127.0.0.1:8000/infer
```

### Manual QA Checklist

#### Upload Mode
- [ ] Upload image with known dog → Match found
- [ ] Upload image with unknown dog → Registration flow works
- [ ] Register new dog → Appears in database
- [ ] Upload same dog again → Match found with high similarity
- [ ] Stats page shows correct counts
- [ ] History page shows recent identifications

#### Live Video Mode
- [ ] Camera starts successfully on desktop
- [ ] Camera starts successfully on mobile (iOS/Android)
- [ ] Bounding boxes appear over detected dogs
- [ ] Color coding works (green=match, orange=unknown)
- [ ] Frame rate adjustable (1-10 FPS)
- [ ] Camera switching works (front/back on mobile)
- [ ] Capture photo opens overlay with results
- [ ] Registration form works in capture overlay
- [ ] Unknown dogs can be registered instantly
- [ ] Error messages display for camera issues

## Performance Notes

### Upload Mode
- **CPU Mode**: ~2-5 seconds per image
- **GPU Mode (CUDA)**: ~0.5-1 second per image
- **FAISS Search**: <0.1 seconds for databases under 10,000 entries
- **Memory Usage**: ~1-2GB (CPU), ~3-4GB (GPU)

### Live Video Mode
- **Processing Latency**: ~170ms per frame average
  - Detection: 45ms
  - Embedding: 123ms
  - Search: 2ms
- **Recommended Frame Rate**: 3 FPS (333ms per frame)
- **Max Theoretical FPS**: ~6 FPS (limited by processing time)
- **Mobile Performance**: 1-3 FPS recommended for battery life
- **Network Latency**: Additional 10-50ms depending on connection

## Future Enhancements

- [x] Live video detection with real-time streaming (COMPLETED)
- [x] Camera switching for mobile devices (COMPLETED)
- [x] Instant registration from capture overlay (COMPLETED)
- [ ] Recording/playback of detection sessions
- [ ] Multi-person collaborative monitoring
- [ ] Alert system for specific dogs detected
- [ ] Background retraining with new samples
- [ ] Export/import database functionality
- [ ] REST API authentication
- [ ] Docker containerization
- [ ] Batch processing mode
- [ ] Web-based admin panel for managing dogs
- [ ] Integration with IP cameras/RTSP streams

## License

This project is for educational purposes. Ensure you have appropriate licenses for the YOLO and re-ID models.

## Support

For issues or questions, please check:
1. Logs in console output
2. `data/` directory permissions
3. Model file integrity
4. Python version compatibility

---

**Built with ❤️ for dog lovers and computer vision enthusiasts**
