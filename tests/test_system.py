"""
Quick system check for potential bugs.
"""
import numpy as np
import sys
import os
# Change to parent directory for proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

print("=" * 60)
print("SYSTEM INTEGRITY CHECK")
print("=" * 60)

# Test 1: Embedding dimension consistency
print("\n1. Checking embedding dimension consistency...")
from backend.inference_service import EMBEDDING_DIM as INFERENCE_DIM
from frontend.app import EMBEDDING_DIM as FLASK_DIM
from backend.faiss_store import FAISSStore

faiss_test = FAISSStore()
FAISS_DIM = faiss_test.embedding_dim

print(f"   Inference service EMBEDDING_DIM: {INFERENCE_DIM}")
print(f"   Flask app EMBEDDING_DIM: {FLASK_DIM}")
print(f"   FAISS default dimension: {FAISS_DIM}")

if INFERENCE_DIM == FLASK_DIM == FAISS_DIM:
    print("   ✓ All dimensions match: 1024")
else:
    print("   ✗ DIMENSION MISMATCH DETECTED!")
    sys.exit(1)

# Test 2: Embedding serialization/deserialization
print("\n2. Testing embedding serialization...")
test_embedding = np.random.randn(1024).astype(np.float32)
embedding_bytes = test_embedding.tobytes()
restored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

print(f"   Original shape: {test_embedding.shape}")
print(f"   Restored shape: {restored_embedding.shape}")
print(f"   Values match: {np.allclose(test_embedding, restored_embedding)}")

if not np.allclose(test_embedding, restored_embedding):
    print("   ✗ SERIALIZATION FAILED!")
    sys.exit(1)
else:
    print("   ✓ Serialization works correctly")

# Test 3: Normalization
print("\n3. Testing embedding normalization...")
raw_embedding = np.random.randn(1024).astype(np.float32)
normalized = faiss_test.normalize_embedding(raw_embedding)
norm = np.linalg.norm(normalized)

print(f"   Raw embedding norm: {np.linalg.norm(raw_embedding):.4f}")
print(f"   Normalized embedding norm: {norm:.4f}")

if abs(norm - 1.0) < 0.001:
    print("   ✓ Normalization produces unit vectors")
else:
    print(f"   ✗ NORMALIZATION ERROR: Expected norm ~1.0, got {norm:.4f}")
    sys.exit(1)

# Test 4: FAISS add/search
print("\n4. Testing FAISS add/search...")
test_store = FAISSStore(index_path="../data/test_faiss.index", embedding_dim=1024)

# Add embeddings
emb1 = np.random.randn(1024).astype(np.float32)
emb2 = np.random.randn(1024).astype(np.float32)
emb3 = emb1 + np.random.randn(1024).astype(np.float32) * 0.1  # Similar to emb1

test_store.add_embedding(1, emb1)
test_store.add_embedding(2, emb2)
test_store.add_embedding(3, emb3)

# Search with emb1 - should return dog_id=1 with highest similarity
results = test_store.search(emb1, k=3)
print(f"   Query with emb1, results: {results}")

if results[0][0] == 1 and results[0][1] > 0.95:
    print("   ✓ Exact match returns dog_id=1 with high similarity")
else:
    print(f"   ✗ SEARCH ERROR: Expected (1, >0.95), got {results[0]}")
    sys.exit(1)

# Test 5: Database operations
print("\n5. Testing database operations...")
from backend.db import DogDatabase

db = DogDatabase(db_path="../data/test_dogs.db")
test_emb = np.random.randn(1024).astype(np.float32)

dog_id = db.add_dog(
    name="Test Dog",
    embedding=test_emb,
    contact_info="test@test.com",
    notes="Test note"
)
print(f"   Added dog with ID: {dog_id}")

retrieved = db.get_dog(dog_id)
print(f"   Retrieved dog: {retrieved['name']}")
print(f"   Embedding shape: {retrieved['embedding'].shape}")
print(f"   Embedding match: {np.allclose(test_emb, retrieved['embedding'])}")

if retrieved['embedding'].shape[0] != 1024:
    print(f"   ✗ DATABASE ERROR: Expected 1024-dim, got {retrieved['embedding'].shape[0]}")
    sys.exit(1)

if not np.allclose(test_emb, retrieved['embedding']):
    print("   ✗ DATABASE ERROR: Embedding mismatch after save/load")
    sys.exit(1)

print("   ✓ Database operations working correctly")

# Test 6: Similarity threshold check
print("\n6. Checking similarity threshold...")
from backend.inference_service import SIMILARITY_THRESHOLD
print(f"   Current threshold: {SIMILARITY_THRESHOLD}")

if SIMILARITY_THRESHOLD == 0.995:
    print("   ⚠ WARNING: Threshold is very high (0.995)")
    print("   This may cause false negatives for same dog")
    print("   Consider lowering to 0.70-0.85 for testing")
elif SIMILARITY_THRESHOLD < 0.5:
    print("   ⚠ WARNING: Threshold is very low (<0.5)")
    print("   This may cause false positives")
else:
    print(f"   ✓ Threshold is reasonable ({SIMILARITY_THRESHOLD})")

# Cleanup
import os
if os.path.exists("../data/test_faiss.index"):
    os.remove("../data/test_faiss.index")
if os.path.exists("../data/test_faiss.index.ids.npy"):
    os.remove("../data/test_faiss.index.ids.npy")
if os.path.exists("../data/test_dogs.db"):
    os.remove("../data/test_dogs.db")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✓")
print("=" * 60)
print("\nNotes:")
print("- System is using layer3 features (1024-dim)")
print("- All dimensions are consistent across components")
print("- Serialization, normalization, and search are working")
print("- Database persistence is correct")
print(f"- Current similarity threshold: {SIMILARITY_THRESHOLD}")
print("\nThe system appears to be functioning correctly!")
