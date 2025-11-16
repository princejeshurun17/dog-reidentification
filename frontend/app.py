"""
Flask UI server for dog re-identification system.
Serves web interface and proxies requests to inference service.
"""
import os
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
# Use environment variable for Docker, fallback to localhost for local development
INFERENCE_SERVICE_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8000')
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
EMBEDDING_DIM = 2048  # Must match backend inference service (layer4)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/live')
def live():
    """Render live detection page."""
    return render_template('live.html')


@app.route('/api/process', methods=['POST'])
def process_image():
    """
    Handle image upload and forward to inference service.
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Save file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(file.filename)
        name, ext = os.path.splitext(filename)
        saved_filename = f"{timestamp}_{name}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        file.save(filepath)
        
        # Forward to inference service
        with open(filepath, 'rb') as f:
            files = {'file': (saved_filename, f, file.content_type)}
            response = requests.post(f"{INFERENCE_SERVICE_URL}/infer", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            result['uploaded_file'] = saved_filename
            result['upload_path'] = f"/uploads/{saved_filename}"
            return jsonify(result)
        else:
            return jsonify({
                'success': False,
                'error': f'Inference service error: {response.status_code}'
            }), 500
            
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': 'Cannot connect to inference service. Please ensure it is running.'
        }), 503
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/enroll', methods=['POST'])
def enroll_dog():
    """
    Enroll a new dog with provided metadata.
    Expects: name, contact_info, notes, embedding (from previous detection), image_path
    """
    try:
        data = request.get_json()
        
        print(f"[ENROLL] Received enrollment request: {data.get('name', 'Unknown')}")
        
        if not data or 'name' not in data:
            print("[ENROLL] Error: Missing dog name")
            return jsonify({'success': False, 'error': 'Dog name is required'}), 400
        
        if 'embedding' not in data:
            print("[ENROLL] Error: Missing embedding data")
            return jsonify({'success': False, 'error': 'Embedding data is required'}), 400
        
        # Import here to avoid circular dependency issues
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))
        from db import DogDatabase
        from faiss_store import FAISSStore
        import numpy as np
        
        print("[ENROLL] Initializing database and FAISS...")
        
        # Initialize database and FAISS with correct dimension
        db = DogDatabase()
        faiss_store = FAISSStore(embedding_dim=EMBEDDING_DIM)
        faiss_store.load_dog_ids()
        
        # Extract data
        name = data['name']
        contact_info = data.get('contact_info', '')
        notes = data.get('notes', '')
        embedding = np.array(data['embedding'], dtype=np.float32)
        image_path = data.get('image_path', '')
        
        embedding_norm = np.linalg.norm(embedding)
        print(f"[ENROLL] Embedding shape: {embedding.shape}, dtype: {embedding.dtype}, norm: {embedding_norm:.4f}")
        
        # Add to database
        print(f"[ENROLL] Adding dog '{name}' to database...")
        dog_id = db.add_dog(
            name=name,
            embedding=embedding,
            contact_info=contact_info,
            notes=notes,
            image_path=image_path
        )
        
        print(f"[ENROLL] Dog added with ID: {dog_id}")
        
        # Add to FAISS index
        print(f"[ENROLL] Adding embedding to FAISS index...")
        faiss_store.add_embedding(dog_id, embedding)
        
        print(f"[ENROLL] ✓ Successfully enrolled {name} (ID: {dog_id})")
        
        # Trigger inference service to reload FAISS index
        try:
            print(f"[ENROLL] Reloading inference service FAISS index...")
            # Increased timeout for Raspberry Pi
            reload_response = requests.post(f"{INFERENCE_SERVICE_URL}/reload", timeout=15)
            if reload_response.status_code == 200:
                print(f"[ENROLL] ✓ Inference service FAISS index reloaded")
            else:
                print(f"[ENROLL] ⚠ Warning: Could not reload inference service FAISS (status {reload_response.status_code})")
        except Exception as e:
            print(f"[ENROLL] ⚠ Warning: Could not reload inference service FAISS: {e}")
        
        return jsonify({
            'success': True,
            'message': f'Successfully enrolled {name}',
            'dog_id': dog_id
        })
        
    except Exception as e:
        print(f"[ENROLL] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/dogs', methods=['GET'])
def get_dogs():
    """
    Fetch list of all enrolled dogs from inference service.
    """
    try:
        # Increased timeout for Raspberry Pi
        response = requests.get(f"{INFERENCE_SERVICE_URL}/dogs", timeout=20)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """
    Fetch identification history from inference service.
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        # Increased timeout for Raspberry Pi
        response = requests.get(f"{INFERENCE_SERVICE_URL}/history?limit={limit}", timeout=20)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Fetch system statistics from inference service.
    """
    try:
        # Increased timeout for Raspberry Pi
        response = requests.get(f"{INFERENCE_SERVICE_URL}/stats", timeout=20)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/health')
def health():
    """Health check endpoint."""
    try:
        # Check inference service
        response = requests.get(f"{INFERENCE_SERVICE_URL}/", timeout=5)
        inference_healthy = response.status_code == 200
    except:
        inference_healthy = False
    
    return jsonify({
        'ui_service': 'running',
        'inference_service': 'running' if inference_healthy else 'unavailable'
    })


if __name__ == '__main__':
    # Use 0.0.0.0 to allow external connections (required for Docker)
    app.run(host='0.0.0.0', port=5000, debug=True)
