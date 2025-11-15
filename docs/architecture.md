# Dog Re-Identification System Architecture

## System Overview

The Dog Re-Identification System is a two-tier architecture consisting of:

1. **Frontend Layer**: Flask-based web UI for user interaction
2. **Backend Layer**: FastAPI-based inference service for ML operations

Both services run as separate processes on localhost, communicating via HTTP REST APIs.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         Browser                              │
│                   (http://localhost:5000)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask UI Server                           │
│                     (Port 5000)                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  - Serve HTML/CSS/JS                                 │   │
│  │  - Handle file uploads                               │   │
│  │  - Store images in data/uploads/                     │   │
│  │  - Proxy requests to inference service               │   │
│  │  - Manage enrollment flow                            │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP REST API
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                FastAPI Inference Service                     │
│                     (Port 8000)                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Model Loading & Inference                           │   │
│  │  ┌──────────┐  ┌──────────┐                          │   │
│  │  │ YOLO.pt  │  │  dog.pt  │                          │   │
│  │  │ Detector │  │  Re-ID   │                          │   │
│  │  └────┬─────┘  └────┬─────┘                          │   │
│  │       │             │                                 │   │
│  │       ▼             ▼                                 │   │
│  │  Face Detection → Crop → Embedding Generation        │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │         FAISS Store (faiss_store.py)                 │   │
│  │  - IndexFlatIP (Inner Product / Cosine Similarity)  │   │
│  │  - Normalized embeddings                             │   │
│  │  - In-memory index with disk persistence            │   │
│  │  - ID mapping: index_position → dog_id              │   │
│  └──────────┬───────────────────────────────────────────┘   │
│             │                                                │
│             ▼                                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          SQLite Database (db.py)                     │   │
│  │  Tables:                                             │   │
│  │  - dogs: metadata + serialized embeddings           │   │
│  │  - identifications: match history log               │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │     File System Storage      │
         │  - data/uploads/             │
         │  - data/dogs.db              │
         │  - data/faiss.index          │
         │  - data/faiss.index.ids.npy  │
         └──────────────────────────────┘
```

## Component Details

### 1. Flask UI Server (`frontend/app.py`)

**Responsibilities**:
- Serve static web interface (HTML/CSS/JS)
- Accept image uploads from browser
- Save uploaded files to `data/uploads/` with timestamp
- Forward processing requests to inference service
- Handle enrollment workflow with form data
- Provide API endpoints for frontend JavaScript

**Key Features**:
- File validation (type, size)
- CSRF protection (recommended for production)
- Error handling and user notifications
- Proxying to avoid CORS issues

**Endpoints**:
- `GET /`: Main HTML interface
- `POST /api/process`: Upload & process image
- `POST /api/enroll`: Register new dog
- `GET /api/dogs`: List all dogs
- `GET /api/history`: View identification history
- `GET /api/stats`: System statistics
- `GET /uploads/<file>`: Serve uploaded images

### 2. FastAPI Inference Service (`backend/inference_service.py`)

**Responsibilities**:
- Load and manage ML models (YOLO, Re-ID)
- Auto-detect CUDA availability, fallback to CPU
- Run dog face detection
- Crop and normalize faces to 224×224
- Generate embedding vectors
- Search FAISS index for matches
- Apply similarity threshold (default: 0.70)
- Log identifications to database

**Key Features**:
- Async endpoints with FastAPI
- Automatic OpenAPI documentation (`/docs`)
- Device detection and optimization
- Model caching (loaded once at startup)
- Batch processing support (future)

**Endpoints**:
- `GET /`: Health check
- `POST /infer`: Main inference pipeline
- `POST /enroll`: Enrollment (placeholder)
- `GET /dogs`: List enrolled dogs
- `GET /stats`: System statistics
- `GET /history`: Match history

### 3. FAISS Store (`backend/faiss_store.py`)

**Responsibilities**:
- Initialize FAISS IndexFlatIP for cosine similarity
- Normalize embeddings to unit vectors
- Add new embeddings with thread safety
- Search for k-nearest neighbors
- Persist index to disk
- Rebuild from database when needed
- Maintain dog_id mapping

**Implementation Details**:
- **Index Type**: `IndexFlatIP` (Inner Product)
- **Normalization**: L2 norm → cosine similarity via inner product
- **Thread Safety**: Lock for concurrent access
- **Persistence**: 
  - `data/faiss.index`: Binary FAISS index
  - `data/faiss.index.ids.npy`: NumPy array of dog IDs

**Key Methods**:
- `add_embedding(dog_id, embedding)`: Add single dog
- `search(embedding, k)`: Find k most similar
- `rebuild_from_database(dogs_data)`: Full rebuild
- `normalize_embedding(embedding)`: L2 normalization

### 4. Database Layer (`backend/db.py`)

**Responsibilities**:
- Manage SQLite connection and schema
- CRUD operations for dogs
- Serialize/deserialize embeddings (NumPy ↔ BLOB)
- Log identification events
- Query history and statistics

**Schema**:

**Table: dogs**
| Column | Type | Description |
|--------|------|-------------|
| dog_id | INTEGER PRIMARY KEY | Auto-increment ID |
| name | TEXT NOT NULL | Dog name |
| contact_info | TEXT | Owner contact |
| notes | TEXT | Additional info |
| embedding | BLOB | Serialized float32 array |
| image_path | TEXT | Path to reference image |
| created_at | TIMESTAMP | Registration time |
| updated_at | TIMESTAMP | Last update time |

**Table: identifications**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-increment ID |
| dog_id | INTEGER FK | Reference to dogs table |
| image_path | TEXT | Path to query image |
| similarity_score | REAL | Cosine similarity (0-1) |
| identified_at | TIMESTAMP | Match time |

**Key Methods**:
- `add_dog()`: Insert new dog
- `get_dog(dog_id)`: Retrieve by ID
- `get_all_dogs()`: List all
- `update_dog()`: Modify metadata
- `delete_dog()`: Remove dog and history
- `log_identification()`: Record match event

### 5. Web Frontend (`frontend/templates/index.html`)

**Responsibilities**:
- Drag-and-drop file upload
- Image preview
- Display detection results with bounding boxes
- Show similarity scores and match info
- Enrollment modal for unknown dogs
- Real-time notifications
- Statistics dashboard

**Technologies**:
- Vanilla JavaScript (no framework dependencies)
- CSS3 with gradients and animations
- Fetch API for HTTP requests
- FormData for file uploads

**UX Flow**:
1. User uploads image
2. Preview shown immediately
3. Spinner indicates processing
4. Results displayed:
   - **Match Found**: Green card with dog info & similarity
   - **Unknown Dog**: Orange card with "Register" button
5. Click "Register" → Modal opens
6. Submit form → Dog added to database
7. Stats updated automatically

## Data Flow

### Identification Flow

```
1. User uploads image
   ↓
2. Flask saves to data/uploads/TIMESTAMP_filename.jpg
   ↓
3. Flask forwards to FastAPI /infer
   ↓
4. FastAPI: YOLO detects faces (class 1)
   ↓
5. For each face:
   a. Crop to bbox
   b. Resize to 224×224
   c. Generate embedding via dog.pt
   d. Normalize embedding
   e. Search FAISS index
   f. If similarity ≥ 0.70: Match found
   g. Log to identifications table
   ↓
6. Return JSON with detections & matches
   ↓
7. Flask forwards to browser
   ↓
8. JavaScript renders results
```

### Enrollment Flow

```
1. User clicks "Register This Dog"
   ↓
2. Modal opens, pre-filled with embedding from detection
   ↓
3. User enters name, contact, notes
   ↓
4. JavaScript POSTs to /api/enroll with:
   - name, contact_info, notes
   - embedding (float array)
   - image_path
   ↓
5. Flask imports db & faiss_store modules
   ↓
6. Insert into dogs table (embedding as BLOB)
   ↓
7. Add normalized embedding to FAISS index
   ↓
8. Save FAISS index to disk
   ↓
9. Return success + dog_id
   ↓
10. JavaScript shows notification
    ↓
11. Stats refreshed
```

## Storage Layout

```
d:\FYP\frontend\
├── yolo.pt                    # YOLO detection model
├── dog.pt                     # Re-ID embedding model
├── requirements.txt           # Python dependencies
│
├── backend\
│   ├── inference_service.py  # FastAPI inference API
│   ├── db.py                  # SQLite database manager
│   └── faiss_store.py         # FAISS index manager
│
├── frontend\
│   ├── app.py                 # Flask UI server
│   └── templates\
│       └── index.html         # Web interface
│
├── data\                      # Created at runtime
│   ├── uploads\               # Uploaded images
│   ├── dogs.db                # SQLite database
│   ├── faiss.index            # FAISS binary index
│   └── faiss.index.ids.npy    # ID mappings
│
├── scripts\
│   ├── manage_db.py           # CLI database tools
│   └── test_integration.py    # Integration tests
│
├── docs\
│   ├── architecture.md        # This file
│   └── API.md                 # API documentation
│
└── README.md                  # Setup & usage guide
```

## Key Design Decisions

### Why Two Separate Services?

1. **Separation of Concerns**: UI logic vs. ML inference
2. **Independent Scaling**: Can run inference on GPU machine, UI on lightweight server
3. **Testing**: Easier to test inference service independently
4. **Development**: Frontend and ML teams can work separately
5. **Future-Proof**: Easy to migrate to microservices or Docker

### Why FastAPI for Inference?

1. **Performance**: Async support for concurrent requests
2. **Automatic Docs**: OpenAPI/Swagger UI at `/docs`
3. **Type Safety**: Pydantic models for validation
4. **Modern**: Native async/await, Python 3.8+ features

### Why Flask for UI?

1. **Simplicity**: Minimal boilerplate for static serving
2. **Mature**: Well-documented, stable
3. **Jinja2**: Server-side templating if needed
4. **Lightweight**: Low overhead for simple proxying

### Why FAISS?

1. **Speed**: Optimized similarity search (millions of vectors)
2. **Memory Efficient**: In-memory indexing with disk persistence
3. **Scalability**: Handles 10K+ dogs easily on CPU
4. **No External Service**: No need for vector DB server

### Why SQLite?

1. **Simplicity**: Single file, no server
2. **Reliability**: ACID compliant
3. **Performance**: Fast enough for < 100K records
4. **Portability**: Easy to backup/restore (copy file)
5. **Windows-Friendly**: No installation needed

### Why Cosine Similarity?

1. **Scale-Invariant**: Ignores embedding magnitude
2. **Interpretable**: 0 (opposite) to 1 (identical)
3. **Robust**: Works well with deep learning embeddings
4. **Standard**: Common in face/object recognition

### Why Store Embeddings in DB?

1. **Rebuild Capability**: Can reconstruct FAISS index
2. **Auditing**: Full history of enrolled embeddings
3. **Backup**: Single source of truth
4. **Debugging**: Inspect embeddings manually

## Performance Considerations

### Bottlenecks

1. **YOLO Inference**: 0.5-2s depending on device
2. **Re-ID Model**: 0.1-0.5s per face
3. **FAISS Search**: <0.1s for 10K entries
4. **Disk I/O**: Minimal (async file writes)

### Optimization Strategies

1. **GPU Acceleration**: 5-10× faster inference
2. **Model Quantization**: Reduce model size (future)
3. **Batch Processing**: Process multiple faces together
4. **Index Partitioning**: Use FAISS IVF for 100K+ entries
5. **Caching**: Cache recent embeddings (future)

### Scalability Limits

| Component | Limit | Workaround |
|-----------|-------|------------|
| SQLite | 100K dogs | Migrate to PostgreSQL |
| FAISS Flat | 1M embeddings | Use FAISS IVF/HNSW |
| Single Process | ~10 req/s | Load balancing |
| Local Storage | Disk size | Cloud storage (S3) |

## Security Considerations

### Current Implementation

- No authentication (localhost only)
- No encryption (HTTP)
- No input sanitization (basic file type check)
- No rate limiting

### Production Recommendations

1. **Authentication**: Add JWT or session-based auth
2. **HTTPS**: Use SSL certificates
3. **Input Validation**: Strict file type, size, content checks
4. **Rate Limiting**: Prevent abuse
5. **CORS**: Configure proper origins
6. **Secrets**: Environment variables for sensitive data
7. **Logging**: Audit all operations
8. **Backups**: Automated DB + FAISS backups

## Future Enhancements

### Short-Term

- [ ] Improve enrollment UX (auto-fill from upload)
- [ ] Add batch upload support
- [ ] Implement FAISS index hot-reload
- [ ] Add unit tests for all components
- [ ] Docker containerization

### Mid-Term

- [ ] User authentication & authorization
- [ ] Multi-user support with roles
- [ ] Advanced search filters (by date, similarity range)
- [ ] Export/import database functionality
- [ ] REST API versioning

### Long-Term

- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile app (React Native)
- [ ] Video stream processing
- [ ] Distributed FAISS with multiple shards
- [ ] Active learning & model retraining
- [ ] Integration with external databases

## Troubleshooting

### Common Issues

**Q: Services can't communicate**
- Check both are running (ports 5000 and 8000)
- Verify firewall settings
- Check `INFERENCE_SERVICE_URL` in Flask app

**Q: FAISS index out of sync with database**
- Run `python scripts\manage_db.py --rebuild`

**Q: CUDA out of memory**
- Use CPU mode or smaller batch size
- Reduce image resolution before upload

**Q: Slow inference on CPU**
- Expected behavior (2-5s per image)
- Consider GPU upgrade or cloud GPU

**Q: Database locked**
- Close other connections to `dogs.db`
- Ensure proper connection closing in code

## Testing Strategy

### Unit Tests

- Database CRUD operations
- FAISS add/search/rebuild
- Embedding normalization
- File upload validation

### Integration Tests

- Full pipeline: upload → detect → identify
- Enrollment flow end-to-end
- API contract testing
- Error handling

### Manual QA

- UI responsiveness
- Cross-browser compatibility
- Large file handling
- Concurrent user sessions

## Monitoring & Logging

### Recommended Metrics

- **Inference Latency**: P50, P95, P99
- **FAISS Search Time**: Average, max
- **Database Query Time**: Slow query log
- **Error Rates**: 4xx, 5xx by endpoint
- **Disk Usage**: uploads, database, index size
- **Memory Usage**: Peak, average

### Log Locations

- **Inference Service**: Console output (can redirect to file)
- **Flask UI**: Console output
- **Database**: SQLite journal files
- **Application Logs**: `data/logs/app.log` (to be implemented)

---

**Document Version**: 1.0  
**Last Updated**: November 15, 2025  
**Author**: AI Assistant
