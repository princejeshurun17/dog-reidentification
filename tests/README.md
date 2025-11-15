# Test Scripts

This directory contains all testing and evaluation scripts for the dog re-identification system.

## Running Tests

All tests should be run from the `tests/` directory:

```powershell
cd tests
python test_system.py
python quick_test.py
python compare_layers.py
python enhanced_inference.py
python evaluate_reid.py
```

## Test Scripts

### `test_system.py`
System integrity checks to verify all components are working correctly.

**Tests:**
- Embedding dimension consistency across modules
- Serialization/deserialization
- L2 normalization
- FAISS operations
- Database persistence

**Usage:**
```powershell
python test_system.py
```

### `quick_test.py`
Fast evaluation on a configurable number of dogs (default: 50).

**Features:**
- Gallery enrollment + query matching
- Temporary FAISS index (prevents corruption)
- Statistics output
- Latency metrics

**Usage:**
```powershell
python quick_test.py
```

**Output:** `../logs/quick_test_results.txt`

### `compare_layers.py`
Side-by-side comparison of Layer3 (1024-dim) vs Layer4 (2048-dim) architectures.

**Proved:** Layer4 achieves 84% accuracy vs Layer3's 52%

**Usage:**
```powershell
python compare_layers.py
```

### `enhanced_inference.py`
Test script with toggle flags for all enhancement features.

**Features:**
- Image augmentation (6 views)
- Layer ensemble (Layer3 + Layer4)
- Query expansion
- Adaptive thresholding
- Multi-metric search
- **Latency metrics tracking**

**Configuration:**
```python
USE_LAYER4 = True              # Use layer4 (2048-dim)
USE_AUGMENTATION = True        # Test multiple views
USE_ENSEMBLE = False           # Combine layer3 + layer4
USE_QUERY_EXPANSION = False    # Re-rank using top-K
USE_ADAPTIVE_THRESHOLD = True  # Confidence-based thresholding
```

**Usage:**
```powershell
python enhanced_inference.py
```

### `evaluate_reid.py`
Comprehensive evaluation with detailed report generation.

**Features:**
- Large-scale testing (100+ dogs)
- Rank-1/Rank-5 accuracy
- Distribution analysis
- Threshold recommendations

**Usage:**
```powershell
python evaluate_reid.py
```

**Output:** `evaluation_results.txt`

## Dataset Configuration

All test scripts use the PetFace dataset:

**Path:** `D:\FYP\data\PetFace\images\test\dog`

**Structure:**
```
test/dog/
├── 000000/
│   ├── 00.png
│   ├── 01.png
│   └── ...
├── 000001/
└── ...
```

To change the dataset path, modify `TEST_DATA_PATH` in each script.

## Latency Metrics

`quick_test.py` and `enhanced_inference.py` now track:

- **Face Detection**: YOLO inference time
- **Embedding Generation**: ResNet50 feature extraction time
- **Search**: FAISS similarity search time
- **Total Per Query**: End-to-end processing time (avg + p95)

## Notes

- All test scripts automatically handle relative paths to models and data
- Test results are saved to `../logs/` directory
- Temporary indices are used to prevent corruption of production FAISS index
- Tests run on CPU by default, CUDA if available
