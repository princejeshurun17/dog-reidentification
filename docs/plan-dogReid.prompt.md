Plan: Flask UI + Inference Service

Overview
- Two-process Windows-friendly system: minimal Flask UI service plus standalone inference API.
- Uploaded images persist under data/uploads/ for auditing and later retraining.
- Inference service auto-selects CUDA when available, falls back to CPU otherwise.
- FAISS index and SQLite metadata stay in backend for identification + enrollment.

Components
1. Flask UI Server (frontend/)
   - Serves single-page browser UI (Jinja template + vanilla JS or lightweight framework).
   - Endpoints: GET / (upload form + gallery), POST /api/process (forwards file to inference service), GET /api/dogs (fetch known dogs from backend), POST /api/enroll (trigger enrollment flow when user confirms new dog).
   - Stores uploads on disk (data/uploads/<timestamp>.jpg) before invoking backend.
   - Handles CSRF/session basics, simple notification system for progress/errors.

2. Inference Service (backend/)
   - FastAPI or Flask API dedicated to ML workloads, separate process from UI.
   - Startup: load yolo.pt (dog detector) + dog.pt (re-ID embedder); detect CUDA availability via torch.cuda.is_available().
   - Endpoints: /infer (accepts image path or bytes, returns bounding boxes + embeddings + similarity results), /enroll (persist embedding + metadata), /dogs (list known metadata + stats).
   - Pipelines: YOLO runs to find dog faces (class 1). Crops resized to 224x224. Each crop processed by re-ID model to produce embedding vectors.

3. FAISS + SQLite Layer
   - SQLite DB (data/dogs.db) stores dog_id, name, contact info, notes, embedding vector serialized as blob, image path, timestamps.
   - FAISS index (IndexFlatIP or cosine-normalized) kept in memory, persisted via faiss.write_index to data/faiss.index.
   - On startup: load SQLite entries, rebuild FAISS if index missing/outdated.
   - Similarity threshold: default 0.70 (cosine). Embeddings normalized before similarity to align with FAISS IP search.
   - Enrollment flow: add row to SQLite, append embedding to FAISS, save index immediately for crash safety.

4. Data Management
   - Directory layout: data/uploads/ (images), data/faiss.index, data/dogs.db, logs/app.log.
   - Provide CLI utilities for rebuilding FAISS from SQLite, listing dogs, removing entries.
   - Background thread to checkpoint FAISS periodically or after batch updates.

5. Frontend UX Flow
   - User selects/upload image, preview rendered.
   - UI calls inference service; response lists detected faces with bounding boxes, similarity scores, and matched dog metadata (if score >= 0.70).
   - For unknown faces, UI prompts user to input dog details, then calls enrollment endpoint.
   - History table shows last N identifications with timestamps and top match scores.

6. Deployment & Setup
   - Python virtualenv for backend + UI (shared requirements file) including Flask, FastAPI, PyTorch, torchvision, faiss-cpu, SQLAlchemy, Pillow.
   - README instructions: install dependencies, place yolo.pt & dog.pt in repo root, run backend (`python backend/inference_service.py`) then UI (`python frontend/app.py`).
   - Optional CUDA instructions: install matching torch/torchvision/cuNNN wheels; inference service auto-switches via torch.cuda.is_available().

7. Testing & Validation
   - Unit tests for FAISS manager and SQLite persistence (pytest backend/tests/).
   - Integration script hitting inference endpoint with sample image, verifying JSON schema.
   - Manual QA steps documented: upload known dog image -> expect match; upload new dog -> enrollment path.

Next Steps
- Decide on FastAPI vs. Flask for inference service (leaning FastAPI for better async + docs).
- Define JSON schema contracts between UI and backend (docs/api.md).
- Plan for future multi-dog images (handle multiple detections in one shot).
