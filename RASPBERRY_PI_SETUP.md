# Raspberry Pi Deployment Guide

Complete step-by-step guide to deploy the Dog Re-Identification System on Raspberry Pi via SSH.

## Prerequisites

### Hardware Requirements
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Raspberry Pi 3B+** also works but slower
- SD card: 32GB+ recommended
- Power supply: Official 5V 3A recommended
- (Optional) Raspberry Pi Camera Module or USB webcam for live detection

### What You Need Before Starting
- Raspberry Pi with Raspberry Pi OS (64-bit recommended) installed
- SSH enabled on Raspberry Pi
- Raspberry Pi connected to network (WiFi or Ethernet)
- Your GitHub repository URL: `https://github.com/princejeshurun17/dog-reidentification.git`
- Model files (`yolo.pt` and `dog.pt`) accessible for upload

---

## Part 1: Initial Setup & SSH Connection

### Step 1: Find Your Raspberry Pi's IP Address

**Option A - From Raspberry Pi Desktop:**
```bash
hostname -I
```

**Option B - From Your Router:**
- Log into your router admin panel
- Look for connected devices
- Find device named "raspberrypi" or similar

**Option C - Network Scan (from your Windows PC):**
```powershell
# Install network scanner
# Or use: arp -a | findstr "b8-27-eb"
```

### Step 2: Connect via SSH

**From Windows PowerShell:**
```powershell
ssh pi@<RASPBERRY_PI_IP>
# Example: ssh pi@192.168.1.100
```

**Default credentials:**
- Username: `pi`
- Password: `raspberry` (change this after first login!)

**First Login - Change Password:**
```bash
passwd
# Enter new password twice
```

---

## Part 2: System Preparation

### Step 3: Update System
```bash
# Update package lists
sudo apt update

# Upgrade installed packages (this may take 10-20 minutes)
sudo apt upgrade -y

# Reboot to apply updates
sudo reboot
```

Wait 1-2 minutes, then reconnect via SSH.

### Step 4: Install Required System Packages
```bash
# Install Python 3.11 (recommended for compatibility)
# Python 3.13 has compatibility issues with many ML packages
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Set Python 3.11 as default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip for Python 3.11
sudo apt install -y python3-pip

# Install system libraries for image processing
sudo apt install -y libatlas-base-dev libopenblas-dev libjpeg-dev zlib1g-dev

# Install OpenGL libraries (required for OpenCV)
sudo apt install -y libgl1 libglib2.0-0

# Install build tools (needed for some Python packages)
sudo apt install -y build-essential cmake git

# Install libhdf5 (for some ML libraries)
sudo apt install -y libhdf5-dev libhdf5-103

# Install video libraries (for camera support)
sudo apt install -y libv4l-dev v4l-utils

# Verify Python installation
python3 --version  # Should show Python 3.11.x
```

### Step 5: Increase Swap Space (Important for 4GB Pi)
```bash
# Stop swap
sudo dphys-swapfile swapoff

# Edit swap config
sudo nano /etc/dphys-swapfile

# Find line: CONF_SWAPSIZE=100
# Change to: CONF_SWAPSIZE=2048

# Save: Ctrl+O, Enter, Ctrl+X

# Restart swap service
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Verify new swap size
free -h  # Should show ~2GB swap
```

---

## Part 3: Clone Repository & Setup

### Step 6: Create Project Directory
```bash
# Create projects directory
mkdir -p ~/projects
cd ~/projects

# Clone your repository
git clone https://github.com/princejeshurun17/dog-reidentification.git

# Navigate to project
cd dog-reidentification

# Verify files
ls -la
```

### Step 7: Create Python Virtual Environment
```bash
# Make sure you're in the project directory
cd ~/projects/dog-reidentification

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# IMPORTANT: Your prompt should now show (venv) at the start
# Example: (venv) admin@raspberrypi:~/projects/dog-reidentification $

# If you don't see (venv), the activation failed. Check:
which python  # Should show: ~/projects/dog-reidentification/venv/bin/python

# Upgrade pip in the virtual environment
pip install --upgrade pip
```

**Troubleshooting activation:**
```bash
# If source command doesn't work, try:
. venv/bin/activate

# Or use full path:
source ~/projects/dog-reidentification/venv/bin/activate

# Verify you're in venv:
which pip  # Should show: ~/projects/dog-reidentification/venv/bin/pip
```

### Step 8: Install Python Dependencies (This takes 20-40 minutes)

**Important:** Install in this specific order for Raspberry Pi compatibility.

```bash
# Install NumPy 1.x with pre-built wheel (FASTER - avoids compilation)
# Use piwheels repository for ARM-optimized packages
pip install numpy==1.26.4 --index-url https://www.piwheels.org/simple

# Install core dependencies
pip install Pillow==10.0.0

# Install PyTorch for ARM (CPU only, no CUDA on Pi)
# For Raspberry Pi OS 64-bit:
pip install torch>=2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# Install web frameworks
pip install flask==3.0.0
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6

# Install YOLO (will take time)
pip install ultralytics==8.0.200

# Install database and search
pip install sqlalchemy==2.0.23
pip install faiss-cpu>=1.12.0

# Install utilities
pip install requests==2.31.0
pip install python-dotenv==1.0.0

# Verify installations
pip list
```

**If installation fails:**
```bash
# Try installing problematic packages individually
# Or use: pip install <package> --no-cache-dir
```

---

## Part 4: Upload Model Files

### Step 9: Transfer Model Files to Raspberry Pi

**Option A - Using SCP (from Windows PowerShell):**
```powershell
# Navigate to where your models are stored
cd d:\FYP\frontend

# Create models directory on Pi
ssh pi@<PI_IP> "mkdir -p ~/projects/dog-reidentification/models"

# Upload YOLO model
scp models\yolo.pt pi@<PI_IP>:~/projects/dog-reidentification/models/

# Upload ReID model
scp models\dog.pt pi@<PI_IP>:~/projects/dog-reidentification/models/

# Example:
# scp models\yolo.pt pi@192.168.1.100:~/projects/dog-reidentification/models/
```

**Option B - Using SFTP:**
```powershell
# Connect with FileZilla or WinSCP
# Host: sftp://<PI_IP>
# Username: pi
# Password: <your_password>
# Navigate to: /home/pi/projects/dog-reidentification/models/
# Upload yolo.pt and dog.pt
```

**Option C - Using wget (if models are on cloud storage):**
```bash
# On Raspberry Pi
cd ~/projects/dog-reidentification/models/

# Download from cloud (example URLs)
wget https://your-storage-url/yolo.pt
wget https://your-storage-url/dog.pt
```

### Step 10: Verify Model Files
```bash
cd ~/projects/dog-reidentification
ls -lh models/

# Should show:
# yolo.pt (size should be ~6-10MB)
# dog.pt (size should be ~80-100MB)
```

---

## Part 5: Configure for Raspberry Pi

### Step 11: Create Required Directories
```bash
cd ~/projects/dog-reidentification

# Create data directories
mkdir -p data/uploads
mkdir -p logs

# Set permissions
chmod 755 data
chmod 755 data/uploads
chmod 755 logs
```

### Step 12: Optimize Configuration for Raspberry Pi

**Edit inference service for CPU-only mode:**
```bash
nano backend/inference_service.py
```

Find the device configuration (around line 20-30) and ensure it says:
```python
device = torch.device("cpu")  # Raspberry Pi uses CPU only
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

**Optional - Reduce YOLO image size for faster processing:**
```bash
nano backend/inference_service.py
```

Find `imgsz` parameter and change from 640 to 416 for faster processing:
```python
results = model(image, imgsz=416, verbose=False)  # Reduced from 640
```

---

## Part 6: Test the System

### Step 13: Test Backend Service

```bash
# Ensure virtual environment is activated
source ~/projects/dog-reidentification/venv/bin/activate

# Start inference service
cd ~/projects/dog-reidentification
python backend/inference_service.py
```

**Expected output:**
```
Using device: cpu
Loading YOLO model from models/yolo.pt...
Loading ReID model from models/dog.pt...
FAISS index initialized with dimension 2048
Inference service ready!
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Test from another SSH session or your Windows PC:**
```bash
# Open new terminal
curl http://<PI_IP>:8000/

# Should return JSON with service status
```

Press `Ctrl+C` to stop the service.

### Step 14: Test Frontend Service

**Start frontend (in new terminal):**
```bash
# SSH into Pi again
ssh pi@<PI_IP>

# Activate environment
cd ~/projects/dog-reidentification
source venv/bin/activate

# Start Flask app
python frontend/app.py
```

**Expected output:**
```
 * Running on http://0.0.0.0:5000
 * Running on http://192.168.1.100:5000
```

**Access from your Windows browser:**
```
http://<PI_IP>:5000
```

You should see the upload interface!

---

## Part 7: Running Both Services (Production Setup)

### Step 15: Using Screen for Background Processes

**Install screen:**
```bash
sudo apt install -y screen
```

**Start backend in screen:**
```bash
cd ~/projects/dog-reidentification
source venv/bin/activate

# Create screen session for backend
screen -S backend

# Inside screen: start inference service
python backend/inference_service.py

# Detach from screen: Ctrl+A, then D
```

**Start frontend in screen:**
```bash
# Create screen session for frontend
screen -S frontend

# Inside screen: activate venv and start frontend
cd ~/projects/dog-reidentification
source venv/bin/activate
python frontend/app.py

# Detach from screen: Ctrl+A, then D
```

**Useful screen commands:**
```bash
screen -ls              # List all screen sessions
screen -r backend       # Reattach to backend
screen -r frontend      # Reattach to frontend
# Inside screen: Ctrl+A then D to detach
# Inside screen: Ctrl+C to stop service
```

---

## Part 8: Create Startup Scripts (Optional)

### Step 16: Auto-start Services on Boot

**Create systemd service for backend:**
```bash
sudo nano /etc/systemd/system/dog-reid-backend.service
```

**Paste this content:**
```ini
[Unit]
Description=Dog Re-ID Backend Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/projects/dog-reidentification
Environment="PATH=/home/pi/projects/dog-reidentification/venv/bin"
ExecStart=/home/pi/projects/dog-reidentification/venv/bin/python backend/inference_service.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Create systemd service for frontend:**
```bash
sudo nano /etc/systemd/system/dog-reid-frontend.service
```

**Paste this content:**
```ini
[Unit]
Description=Dog Re-ID Frontend Service
After=network.target dog-reid-backend.service

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/projects/dog-reidentification
Environment="PATH=/home/pi/projects/dog-reidentification/venv/bin"
ExecStart=/home/pi/projects/dog-reidentification/venv/bin/python frontend/app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start services:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services (start on boot)
sudo systemctl enable dog-reid-backend
sudo systemctl enable dog-reid-frontend

# Start services now
sudo systemctl start dog-reid-backend
sudo systemctl start dog-reid-frontend

# Check status
sudo systemctl status dog-reid-backend
sudo systemctl status dog-reid-frontend
```

**Service management commands:**
```bash
sudo systemctl stop dog-reid-backend      # Stop service
sudo systemctl restart dog-reid-backend   # Restart service
sudo systemctl disable dog-reid-backend   # Disable auto-start
sudo journalctl -u dog-reid-backend -f    # View logs
```

---

## Part 9: Camera Setup (For Live Detection)

### Step 17: Enable Camera

**For Raspberry Pi Camera Module:**
```bash
# Enable camera interface
sudo raspi-config

# Navigate: Interface Options -> Camera -> Enable
# Reboot: sudo reboot
```

**For USB Webcam:**
```bash
# Check if camera is detected
ls /dev/video*

# Should show: /dev/video0

# Test camera
v4l2-ctl --list-devices
```

### Step 18: Test Live Detection

**Access from browser:**
```
http://<PI_IP>:5000/live
```

**Camera permissions:**
- Browser will ask for camera permission
- On mobile: Ensure you're accessing via `http://` or `https://`
- Local network access should work fine

---

## Part 10: Performance Optimization

### Step 19: Optimize for Raspberry Pi Performance

**Create optimized config file:**
```bash
cd ~/projects/dog-reidentification
nano config_pi.py
```

**Add these optimizations:**
```python
# Raspberry Pi Optimizations
YOLO_IMAGE_SIZE = 416  # Reduced from 640
YOLO_CONFIDENCE = 0.50  # Slightly higher to reduce false positives
MAX_CONCURRENT_REQUESTS = 1  # Process one at a time
ENABLE_GPU = False  # No GPU on Pi
BATCH_SIZE = 1  # Process single images

# Frame rate limits for live detection
DEFAULT_FPS = 1  # Start with 1 FPS on Pi
MAX_FPS = 3  # Max 3 FPS on Pi (vs 10 on PC)
```

**Expected Performance on Raspberry Pi 4:**
- Single image processing: 3-8 seconds
- Live detection: 1-2 FPS recommended
- YOLO detection: ~1-2 seconds
- Embedding generation: ~2-4 seconds
- Database search: <100ms

---

## Part 11: Troubleshooting

### Common Issues & Solutions

**1. "Out of memory" during pip install:**
```bash
# Increase swap space (see Step 5)
# Or install packages one at a time:
pip install --no-cache-dir <package-name>
```

**1a. NumPy installation stuck at "Preparing metadata":**
```bash
# Stop the stuck installation: Ctrl+C

# Use piwheels repository for pre-built ARM wheels (MUCH FASTER)
pip install numpy==1.26.4 --index-url https://www.piwheels.org/simple

# If piwheels fails, try forcing binary wheel only (no source compilation):
pip install numpy==1.26.4 --only-binary=:all:

# Alternative: Install from Debian packages (system-wide, not recommended in venv)
# sudo apt install python3-numpy
```

**1b. NumPy 2.x compatibility errors:**
```bash
# Downgrade to NumPy 1.x (compatible with most ML packages)
pip uninstall numpy -y
pip install "numpy<2" --no-cache-dir

# If you have Python 3.13, consider using Python 3.11 instead:
# Recreate virtual environment with Python 3.11
deactivate
cd ~/projects/dog-reidentification
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate
# Then reinstall all packages
```

**1c. "externally-managed-environment" error:**
```bash
# This means you're NOT in the virtual environment
# Make sure you see (venv) in your prompt

# Activate the virtual environment:
cd ~/projects/dog-reidentification
source venv/bin/activate

# Verify you're in venv:
which pip  # Should show path ending in /venv/bin/pip

# If venv doesn't exist, create it:
python3 -m venv venv
source venv/bin/activate

# NEVER use --break-system-packages (it can break your OS)
```

**2. "No module named 'cv2'" or OpenCV errors:**
```bash
# If you get "ImportError: libGL.so.1: cannot open shared object file"
# Install missing OpenGL libraries:
sudo apt install -y libgl1 libglib2.0-0

# Or better for Pi - use headless OpenCV (recommended):
pip uninstall opencv-python -y
pip install opencv-python-headless==4.8.1.78
```

**3. "Port already in use":**
```bash
# Find and kill process
sudo lsof -i :8000
sudo kill -9 <PID>
```

**4. "Permission denied" for camera:**
```bash
# Add user to video group
sudo usermod -a -G video pi
# Logout and login again
```

**5. Services not starting on boot:**
```bash
# Check logs
sudo journalctl -u dog-reid-backend -n 50
sudo journalctl -u dog-reid-frontend -n 50

# Check service status
sudo systemctl status dog-reid-backend
```

**6. Slow performance:**
```bash
# Reduce YOLO image size to 320 or 416
# Lower frame rate to 1 FPS
# Close unnecessary services:
sudo systemctl stop bluetooth
sudo systemctl stop cups  # Printing service
```

**7. Cannot access from other devices:**
```bash
# Check firewall (usually not enabled by default)
sudo ufw status

# If enabled, allow ports:
sudo ufw allow 5000
sudo ufw allow 8000
```

---

## Part 12: Accessing from Outside Your Network

### Step 20: Port Forwarding (Optional)

**Router Configuration:**
1. Log into your router admin panel
2. Find "Port Forwarding" or "Virtual Server" settings
3. Forward external port 5000 → Pi IP:5000
4. Forward external port 8000 → Pi IP:8000

**Security Warning:** 
- Use strong passwords
- Consider VPN instead of direct port forwarding
- Add authentication to the app if exposing publicly

**Alternative - Using ngrok:**
```bash
# Install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.tgz
tar -xvzf ngrok-v3-stable-linux-arm64.tgz
sudo mv ngrok /usr/local/bin/

# Setup ngrok account at ngrok.com
ngrok config add-authtoken <YOUR_TOKEN>

# Tunnel to frontend
ngrok http 5000

# You'll get a public URL like: https://abc123.ngrok.io
```

---

## Part 13: Maintenance & Updates

### Step 21: Updating the Code

**Pull latest changes from GitHub:**
```bash
cd ~/projects/dog-reidentification

# Stop services
sudo systemctl stop dog-reid-backend
sudo systemctl stop dog-reid-frontend

# Pull updates
git pull origin main

# Activate venv and update dependencies if needed
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl start dog-reid-backend
sudo systemctl start dog-reid-frontend
```

### Step 22: Backup Database

```bash
# Create backup directory
mkdir -p ~/backups

# Backup database
cp ~/projects/dog-reidentification/data/dogs.db ~/backups/dogs.db.$(date +%Y%m%d)

# Backup FAISS index
cp ~/projects/dog-reidentification/data/faiss.index* ~/backups/
```

### Step 23: Monitor System Resources

```bash
# Check CPU and memory usage
htop  # Install: sudo apt install htop

# Check temperature
vcgencmd measure_temp

# Check disk space
df -h

# Check running processes
ps aux | grep python
```

---

## Quick Reference

### Start Services Manually
```bash
# Terminal 1 - Backend
cd ~/projects/dog-reidentification
source venv/bin/activate
python backend/inference_service.py

# Terminal 2 - Frontend
cd ~/projects/dog-reidentification
source venv/bin/activate
python frontend/app.py
```

### Start Services with systemd
```bash
sudo systemctl start dog-reid-backend
sudo systemctl start dog-reid-frontend
```

### Access URLs
- **Upload Mode:** `http://<PI_IP>:5000`
- **Live Mode:** `http://<PI_IP>:5000/live`
- **API Status:** `http://<PI_IP>:8000`

### Check Logs
```bash
# Service logs
sudo journalctl -u dog-reid-backend -f
sudo journalctl -u dog-reid-frontend -f

# Or if using screen:
screen -r backend  # View backend console
screen -r frontend  # View frontend console
```

---

## Performance Expectations

| Operation | Raspberry Pi 4 (4GB) | Desktop PC (CPU) | Desktop PC (GPU) |
|-----------|---------------------|------------------|------------------|
| Single Image | 3-8 seconds | 2-5 seconds | 0.5-1 second |
| YOLO Detection | 1-2 seconds | 0.5-1 second | 0.05-0.1 second |
| Embedding | 2-4 seconds | 1-3 seconds | 0.3-0.5 second |
| Live FPS | 1-2 FPS | 3-6 FPS | 10+ FPS |
| Memory Usage | 1.5-2GB | 1-2GB | 3-4GB |

---

## Support & Resources

- **Documentation:** Check README.md and SESSION_SUMMARY.md in project
- **API Reference:** docs/API.md
- **GitHub Issues:** https://github.com/princejeshurun17/dog-reidentification/issues
- **Raspberry Pi Forums:** https://forums.raspberrypi.com/

---

**Deployment Guide Version:** 1.0  
**Last Updated:** November 16, 2025  
**Tested On:** Raspberry Pi 4 (4GB), Raspberry Pi OS 64-bit (Bookworm)
