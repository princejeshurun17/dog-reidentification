# Dog Re-Identification System

A cross-platform application for dog face detection and re-identification using YOLO and deep learning embeddings with FAISS similarity search. Deployable on Windows, Linux, and Raspberry Pi with Docker support.

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
- 🐳 **Docker deployment** for easy setup on any platform
- 🥧 **Raspberry Pi support** with optimized performance settings
- 📈 **84.1% accuracy** on PetFace dataset

## Prerequisites

### For Native Installation:
- Python 3.8 or higher (Python 3.11 recommended for Raspberry Pi)
- Windows, Linux, or Raspberry Pi OS
- 4GB+ RAM recommended (8GB for Raspberry Pi)
- (Optional) NVIDIA GPU with CUDA for faster inference

### For Docker Deployment:
- Docker and Docker Compose installed
- 4GB+ RAM (8GB recommended for Raspberry Pi)
- No Python installation required

## Quick Start

### Option 1: Docker Deployment (Recommended)

See **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** for complete Docker setup instructions.

```bash
# Install Docker, clone repo, upload models
docker compose up -d

# Access at http://localhost:5000
```

### Option 2: Native Installation

See **[QUICKSTART.md](QUICKSTART.md)** for detailed step-by-step setup.

### Option 3: Raspberry Pi Setup

See **[RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)** for SSH deployment guide.

---

## Installation

## Project Structure

```
d:\FYP\frontend\
├── models/                 # Model files
│   ├── yolo.pt            # YOLO detection model
│   └── dog.pt             # ResNet50 re-ID model
├── requirements.txt
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Multi-container orchestration
├── .dockerignore          # Docker build exclusions
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
├── QUICKSTART.md          # Step-by-step native setup
├── DOCKER_DEPLOYMENT.md   # Docker deployment guide
├── RASPBERRY_PI_SETUP.md  # Raspberry Pi SSH setup
└── SESSION_SUMMARY.md     # Development log
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

## Performance Comparison

| Platform | Deployment | Single Image | Live FPS | Memory | Setup Time |
|----------|-----------|--------------|----------|--------|------------|
| Windows PC | Native | 2-5 sec | 3-6 FPS | 1-2 GB | 20 min |
| Windows PC | Docker | 2-5 sec | 3-6 FPS | 1.5-2.5 GB | 30 min |
| Raspberry Pi 4 | Native | 3-8 sec | 1-2 FPS | 1.5-2 GB | 60 min |
| Raspberry Pi 4 | Docker | 3-8 sec | 1-2 FPS | 1.7-2.5 GB | 40 min |

**Recommended:**
- **Windows Development**: Native (faster setup)
- **Production/Raspberry Pi**: Docker (easier management, auto-restart)

## Configuration

### Backend Settings (`backend/inference_service.py`)

```python
YOLO_MODEL_PATH = "yolo.pt"          # Path to YOLO model
REID_MODEL_PATH = "dog.pt"           # Path to re-ID model
CROP_SIZE = (224, 224)               # Face crop size
SIMILARITY_THRESHOLD = 0.60          # Matching threshold (0-1)
EMBEDDING_DIM = 2048                 # Re-ID embedding dimension (Layer4)
```

### Docker Environment Variables

```yaml
# docker-compose.yml
environment:
  - BACKEND_URL=http://backend:8000  # Frontend → Backend communication
  - PYTHONUNBUFFERED=1               # Real-time logging
```

### Timeout Configuration

For slower devices (Raspberry Pi), timeouts are pre-configured:
- **Backend API**: 60s inference, 20s queries
- **Frontend fetch**: 90s with AbortController

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
- **CPU Mode**: ~2-5 seconds per image (Windows), 3-8 seconds (Raspberry Pi)
- **GPU Mode (CUDA)**: ~0.5-1 second per image
- **FAISS Search**: <0.1 seconds for databases under 10,000 entries
- **Memory Usage**: ~1-2GB (CPU), ~3-4GB (GPU)

### Live Video Mode
- **Processing Latency**: ~170ms per frame average (Windows)
  - Detection: 45ms
  - Embedding: 123ms
  - Search: 2ms
- **Raspberry Pi Latency**: ~500-1000ms per frame
- **Recommended Frame Rate**: 
  - Windows: 3 FPS (333ms per frame)
  - Raspberry Pi: 1-2 FPS
- **Max Theoretical FPS**: ~6 FPS (Windows), ~2 FPS (Raspberry Pi)
- **Mobile Performance**: 1-3 FPS recommended for battery life
- **Network Latency**: Additional 10-50ms depending on connection

### Docker Overhead
- **Container Start**: 10-20 seconds (includes model loading)
- **Memory**: +200MB compared to native
- **Processing**: Similar to native (negligible overhead)
- **Build Time**: 5-10 minutes (Windows), 15-30 minutes (Raspberry Pi)

## Deployment Options

### Development
- **Recommended**: Native Python installation (QUICKSTART.md)
- **Fastest setup**: 20 minutes on Windows
- **Best for**: Testing, debugging, development

### Production
- **Recommended**: Docker deployment (DOCKER_DEPLOYMENT.md)
- **Benefits**: Auto-restart, easy updates, consistent environment
- **Best for**: Server deployment, always-on systems

### Raspberry Pi
- **Option 1**: Docker (DOCKER_DEPLOYMENT.md) - Easier setup (40 min)
- **Option 2**: Native (RASPBERRY_PI_SETUP.md) - Better performance insight (60 min)
- **Best for**: Edge deployment, portable systems

## Future Enhancements

- [x] Live video detection with real-time streaming (COMPLETED)
- [x] Camera switching for mobile devices (COMPLETED)
- [x] Docker containerization (COMPLETED)
- [x] Raspberry Pi support (COMPLETED)
- [x] Instant registration from capture overlay (COMPLETED)
- [ ] Recording/playback of detection sessions
- [ ] Multi-person collaborative monitoring
- [ ] Alert system for specific dogs detected
- [ ] Background retraining with new samples
- [ ] Export/import database functionality
- [ ] REST API authentication
- [ ] Batch processing mode
- [ ] Web-based admin panel for managing dogs
- [ ] Integration with IP cameras/RTSP streams
- [ ] Cloudflare Tunnel setup for remote access

## Documentation

- **[README.md](README.md)** - Overview and setup guide (this file)
- **[QUICKSTART.md](QUICKSTART.md)** - Step-by-step native installation
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Docker deployment guide
- **[RASPBERRY_PI_SETUP.md](RASPBERRY_PI_SETUP.md)** - Raspberry Pi SSH setup
- **[SESSION_SUMMARY.md](SESSION_SUMMARY.md)** - Development log and technical details
- **[docs/API.md](docs/API.md)** - API endpoint reference
- **[docs/architecture.md](docs/architecture.md)** - System architecture

## License

This project is for educational purposes. Ensure you have appropriate licenses for the YOLO and re-ID models.

## Support

For issues or questions:
1. **GitHub Issues**: [https://github.com/princejeshurun17/dog-reidentification/issues](https://github.com/princejeshurun17/dog-reidentification/issues)
2. **Check Logs**: Console output for errors
3. **Troubleshooting Guides**: See deployment documentation
4. **Common Issues**: 
   - `data/` directory permissions
   - Model file integrity
   - Python/Docker version compatibility
   - Camera access permissions (live mode)

---

**Version**: 3.1  
**Last Updated**: November 16, 2025  
**Built with ❤️ for dog lovers and computer vision enthusiasts**
