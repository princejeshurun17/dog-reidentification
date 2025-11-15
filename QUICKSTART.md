# Quick Start Guide

## Easy Launch (Recommended) 🚀

### Windows PowerShell
```powershell
.\start.ps1
```

### Windows Command Prompt
```cmd
start.bat
```

Both scripts will:
1. ✅ Check virtual environment
2. ✅ Start backend (port 8000)
3. ✅ Start frontend (port 5000)
4. ✅ Open in separate windows
5. ✅ Wait for you to press a key to stop

**Then open:** http://localhost:5000

---

## Prerequisites
- Python 3.8+
- Model files in `models/` directory (`yolo.pt` and `dog.pt`)
- Virtual environment with dependencies installed

## Installation

```powershell
# Navigate to project
cd d:\FYP\frontend

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Running the System

### Option 1: Automated (Recommended) 🚀
```powershell
.\start.ps1
```
This starts both services and keeps them running until you press a key.

### Option 2: Manual

#### Terminal 1 - Start Inference Service
```powershell
.\venv\Scripts\Activate.ps1
python backend\inference_service.py
```

Wait for: `Inference service ready!`

#### Terminal 2 - Start UI Server
```powershell
.\venv\Scripts\Activate.ps1
python frontend\app.py
```

Wait for: `Running on http://127.0.0.1:5000`

### Access Web Interface
Open browser: http://127.0.0.1:5000

---

## Stopping Services

### Using Script
```powershell
.\stop.ps1
```

### Manual
Press `Ctrl+C` in each terminal window

## Usage

1. **Upload Image**: Click "Choose Image" or drag-and-drop
2. **View Results**: See detected faces and matches
3. **Register New Dog**: Click "Register This Dog" for unknowns
4. **Monitor Stats**: Check dashboard for total dogs and device info

## Testing

Run integration tests:
```powershell
python scripts\test_integration.py path\to\dog_image.jpg
```

## Management

List all dogs:
```powershell
python scripts\manage_db.py --list
```

View statistics:
```powershell
python scripts\manage_db.py --stats
```

Rebuild FAISS index:
```powershell
python scripts\manage_db.py --rebuild
```

## Troubleshooting

**Cannot connect to inference service**
- Ensure inference service is running on port 8000
- Check for firewall blocking

**CUDA errors**
- System will auto-fallback to CPU
- Or force CPU: Edit `inference_service.py`, set `device = torch.device("cpu")`

**Models not found**
- Place `yolo.pt` and `dog.pt` in `d:\FYP\frontend\`

## Documentation

- Full guide: `README.md`
- API docs: `docs\API.md`
- Architecture: `docs\architecture.md`
