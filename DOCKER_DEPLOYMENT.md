# Docker Deployment Guide for Raspberry Pi

Complete guide to deploy the Dog Re-Identification System using Docker on Raspberry Pi.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup](#detailed-setup)
- [Managing Containers](#managing-containers)
- [Troubleshooting](#troubleshooting)
- [Performance](#performance)

---

## Prerequisites

### Hardware Requirements
- **Raspberry Pi 4** (4GB RAM minimum, 8GB recommended)
- **Raspberry Pi 3B+** also works but slower
- SD card: 32GB+ recommended
- Raspberry Pi OS (64-bit recommended) installed
- Network connection (WiFi or Ethernet)

### What You Need
- Docker and Docker Compose installed on Raspberry Pi
- SSH access to your Raspberry Pi
- Model files: `yolo.pt` and `dog.pt`
- GitHub repository access

---

## Quick Start

### For Impatient Users (5 Commands)

```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

# 2. Clone repository
git clone https://github.com/princejeshurun17/dog-reidentification.git
cd dog-reidentification

# 3. Upload models to models/ directory

# 4. Build and run
docker compose up -d

# 5. Access application
# Open browser: http://<RASPBERRY_PI_IP>:5000
```

---

## Detailed Setup

### Step 1: Connect to Raspberry Pi

```bash
# From your Windows PC
ssh pi@<RASPBERRY_PI_IP>

# Example: ssh pi@192.168.1.100
```

### Step 2: Install Docker

```bash
# Download Docker installation script
curl -fsSL https://get.docker.com -o get-docker.sh

# Install Docker
sudo sh get-docker.sh

# Add your user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER

# Logout and login for group change to take effect
logout
```

**Reconnect via SSH:**
```bash
ssh pi@<RASPBERRY_PI_IP>
```

**Verify Docker installation:**
```bash
docker --version
# Should show: Docker version 24.x.x or higher

docker compose version
# Should show: Docker Compose version v2.x.x
```

### Step 3: Install Docker Compose (if not included)

```bash
# If docker compose command doesn't work, install plugin
sudo apt-get update
sudo apt-get install -y docker-compose-plugin

# Verify
docker compose version
```

### Step 4: Clone Repository

```bash
# Navigate to home directory
cd ~

# Clone repository
git clone https://github.com/princejeshurun17/dog-reidentification.git

# Navigate to project
cd dog-reidentification

# Verify files
ls -la
```

### Step 5: Upload Model Files

**Option A - Using SCP (from Windows PowerShell):**
```powershell
# Navigate to your models directory
cd d:\FYP\frontend

# Upload models to Raspberry Pi
scp models\yolo.pt pi@<PI_IP>:~/dog-reidentification/models/
scp models\dog.pt pi@<PI_IP>:~/dog-reidentification/models/

# Example:
# scp models\yolo.pt pi@192.168.1.100:~/dog-reidentification/models/
```

**Option B - Using SFTP/WinSCP:**
- Connect to `sftp://<PI_IP>`
- Navigate to `/home/pi/dog-reidentification/models/`
- Upload `yolo.pt` and `dog.pt`

**Verify files (on Raspberry Pi):**
```bash
cd ~/dog-reidentification
ls -lh models/

# Should show:
# yolo.pt (~6-10 MB)
# dog.pt (~80-100 MB)
```

### Step 6: Create Required Directories

```bash
cd ~/dog-reidentification

# Create data directories
mkdir -p data/uploads
mkdir -p logs

# Set permissions
chmod -R 755 data logs
```

### Step 7: Build Docker Images

```bash
cd ~/dog-reidentification

# Build images (this will take 15-30 minutes on Raspberry Pi)
docker compose build

# You'll see:
# - Downloading Python base image
# - Installing system dependencies
# - Installing Python packages (numpy, torch, ultralytics, etc.)
```

**Expected build output:**
```
[+] Building 1200.5s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [1/8] FROM docker.io/library/python:3.11-slim-bullseye
 => [2/8] RUN apt-get update && apt-get install...
 => [3/8] COPY requirements.txt .
 => [4/8] RUN pip install --upgrade pip && pip install...
 => [5/8] COPY backend/ ./backend/
 => [6/8] COPY frontend/ ./frontend/
 => [7/8] COPY models/ ./models/
 => [8/8] COPY data/ ./data/
 => exporting to image
```

### Step 8: Start Services

```bash
# Start both backend and frontend
docker compose up -d

# The -d flag runs containers in detached mode (background)
```

**Expected output:**
```
[+] Running 3/3
 ✔ Network dog-reid-network        Created
 ✔ Container dog-reid-backend      Started
 ✔ Container dog-reid-frontend     Started
```

### Step 9: Verify Services

```bash
# Check running containers
docker compose ps

# Should show:
# NAME                  STATUS              PORTS
# dog-reid-backend      Up 30 seconds       0.0.0.0:8000->8000/tcp
# dog-reid-frontend     Up 20 seconds       0.0.0.0:5000->5000/tcp

# Check logs
docker compose logs backend
docker compose logs frontend

# Or follow logs in real-time
docker compose logs -f
```

### Step 10: Test the Application

**From your Windows PC browser:**
```
http://<RASPBERRY_PI_IP>:5000
```

You should see the dog re-identification interface!

**Test live detection:**
```
http://<RASPBERRY_PI_IP>:5000/live
```

**Test backend API:**
```bash
curl http://<RASPBERRY_PI_IP>:8000/
```

---

## Managing Containers

### Basic Commands

**View running containers:**
```bash
docker compose ps
```

**View logs:**
```bash
# All services
docker compose logs

# Specific service
docker compose logs backend
docker compose logs frontend

# Follow logs (Ctrl+C to exit)
docker compose logs -f

# Last 50 lines
docker compose logs --tail=50
```

**Stop services:**
```bash
docker compose stop
```

**Start services:**
```bash
docker compose start
```

**Restart services:**
```bash
# Restart all
docker compose restart

# Restart specific service
docker compose restart backend
docker compose restart frontend
```

**Stop and remove containers:**
```bash
docker compose down
```

**Stop, remove, and delete volumes:**
```bash
docker compose down -v
```

### Updating the Application

**Pull latest code and rebuild:**
```bash
cd ~/dog-reidentification

# Stop containers
docker compose down

# Pull latest changes
git pull origin main

# Rebuild images
docker compose build

# Start services
docker compose up -d
```

### Accessing Container Shell

**Open shell in running container:**
```bash
# Backend container
docker compose exec backend /bin/bash

# Frontend container
docker compose exec frontend /bin/bash

# Inside container, you can:
ls -la              # List files
python              # Start Python interpreter
cat logs/app.log    # View logs
exit                # Exit container
```

### Resource Monitoring

**Check resource usage:**
```bash
docker stats

# Shows CPU, memory, network I/O for each container
# Press Ctrl+C to exit
```

**Check disk usage:**
```bash
docker system df

# Shows space used by images, containers, volumes
```

---

## Auto-Start on Boot

### Enable Docker to Start on Boot

```bash
# Enable Docker service
sudo systemctl enable docker

# Docker Compose containers will auto-restart with restart: unless-stopped
```

**Verify auto-start configuration:**
```bash
# Check Docker service
systemctl status docker

# Should show: "enabled"
```

**Test auto-start:**
```bash
# Reboot Raspberry Pi
sudo reboot

# Wait 2 minutes, then reconnect
ssh pi@<PI_IP>

# Check if containers are running
docker compose ps

# Should show both containers running
```

---

## Troubleshooting

### Common Issues

**1. Docker build fails with "out of memory":**
```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon

# Rebuild
docker compose build --no-cache
```

**2. Port already in use:**
```bash
# Find what's using the port
sudo lsof -i :5000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Or change ports in docker-compose.yml:
# ports:
#   - "5001:5000"  # Use port 5001 instead
```

**3. Containers keep restarting:**
```bash
# Check logs for errors
docker compose logs backend
docker compose logs frontend

# Common issues:
# - Missing model files
# - Incorrect permissions
# - Python errors

# Fix permissions
chmod -R 755 ~/dog-reidentification/data
chmod -R 755 ~/dog-reidentification/models
```

**4. Cannot access from browser:**
```bash
# Check containers are running
docker compose ps

# Check Raspberry Pi firewall
sudo ufw status

# If enabled, allow ports
sudo ufw allow 5000
sudo ufw allow 8000

# Test from Raspberry Pi itself
curl http://localhost:5000
curl http://localhost:8000
```

**5. "Permission denied" for Docker commands:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Logout and login
logout
# Then reconnect via SSH
```

**6. Slow build process:**
```bash
# This is normal on Raspberry Pi
# Build can take 15-30 minutes
# Be patient, especially during pip install

# Monitor build progress
docker compose build --progress=plain
```

**7. Container crashes during runtime:**
```bash
# Check memory usage
docker stats

# If using too much memory, optimize:
# - Reduce YOLO image size in inference_service.py
# - Increase swap space
# - Use Raspberry Pi 4 with 8GB RAM

# View container exit code
docker compose ps -a

# Restart with more memory info
docker compose up
```

### Health Checks

**Check container health:**
```bash
docker compose ps

# Look for "healthy" status
```

**Manual health check:**
```bash
# Backend
curl http://localhost:8000/

# Frontend
curl http://localhost:5000/
```

### Logs Location

**Container logs:**
```bash
# View logs
docker compose logs

# Logs are also in:
~/dog-reidentification/logs/
```

**Docker system logs:**
```bash
sudo journalctl -u docker -f
```

---

## Performance Optimization

### Raspberry Pi 4 Optimizations

**Edit `docker-compose.yml` to limit resources:**
```yaml
services:
  backend:
    # ... existing config ...
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

**Optimize inference settings:**
```bash
# Edit inference service
nano backend/inference_service.py

# Change these values:
# YOLO_IMAGE_SIZE = 416  # Reduced from 640
# MAX_CONCURRENT_REQUESTS = 1
```

### Cleanup Unused Resources

```bash
# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes

# WARNING: This removes all stopped containers and unused images
```

---

## Performance Expectations

### Raspberry Pi 4 (4GB) with Docker

| Operation | Time | Notes |
|-----------|------|-------|
| Build Images | 15-30 min | One-time setup |
| Container Start | 10-20 sec | Includes model loading |
| Single Image | 3-8 sec | Similar to native |
| Live Detection | 1-2 FPS | Recommended FPS |
| Memory Usage | 1.5-2.5 GB | Docker + Python |

**Note:** Docker adds ~100-200MB overhead compared to native installation.

---

## Advantages of Docker Deployment

✅ **Easy Setup:** No manual dependency installation  
✅ **Consistent Environment:** Works same on any system  
✅ **Isolated:** Doesn't affect system Python  
✅ **Portable:** Easy to move between Raspberry Pis  
✅ **Easy Updates:** Pull and rebuild  
✅ **Auto-Restart:** Containers restart on failure  
✅ **Easy Cleanup:** Remove everything with one command  

---

## Backing Up

### Backup Docker Volumes

```bash
# Create backup directory
mkdir -p ~/backups

# Backup database and FAISS index
docker compose exec backend tar -czf /tmp/data-backup.tar.gz /app/data
docker cp dog-reid-backend:/tmp/data-backup.tar.gz ~/backups/data-backup-$(date +%Y%m%d).tar.gz
```

### Restore Backup

```bash
# Stop services
docker compose down

# Extract backup
tar -xzf ~/backups/data-backup-20251116.tar.gz -C ~/dog-reidentification/

# Start services
docker compose up -d
```

---

## Comparison: Docker vs Native

| Aspect | Docker | Native |
|--------|--------|--------|
| Setup Time | 30-40 min | 40-60 min |
| Dependencies | Automatic | Manual |
| Performance | ~5% slower | Baseline |
| Memory | +200 MB | Baseline |
| Updates | Easy | Manual |
| Cleanup | Easy | Manual |
| Portability | Excellent | Platform-specific |
| Troubleshooting | Logs centralized | Scattered |

**Recommendation:** Use Docker for easier management and updates.

---

## Quick Reference

### Essential Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose stop

# View logs
docker compose logs -f

# Restart services
docker compose restart

# Update application
git pull && docker compose build && docker compose up -d

# Check status
docker compose ps

# Resource usage
docker stats

# Remove everything
docker compose down -v
```

### Access URLs

- **Upload Mode:** `http://<PI_IP>:5000`
- **Live Mode:** `http://<PI_IP>:5000/live`
- **Backend API:** `http://<PI_IP>:8000`

---

## Support

- **Documentation:** README.md, SESSION_SUMMARY.md
- **API Reference:** docs/API.md
- **GitHub Issues:** https://github.com/princejeshurun17/dog-reidentification/issues
- **Docker Docs:** https://docs.docker.com/

---

**Guide Version:** 1.0  
**Last Updated:** November 16, 2025  
**Tested On:** Raspberry Pi 4 (4GB), Raspberry Pi OS 64-bit (Bookworm)  
**Docker Version:** 24.x with Compose v2
