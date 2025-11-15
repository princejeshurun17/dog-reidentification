"""Quick test of the dog re-identification system on a small sample."""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime

from db import DogDatabase
from faiss_store import FAISSStore
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

print("="*60)
print("QUICK RE-ID SYSTEM TEST")
print("="*60)

# Configuration
TEST_DATA_PATH = r"D:\FYP\data\PetFace\images\test\dog"
NUM_DOGS = 100  # Test with just 5 dogs
IMAGES_PER_DOG = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load models
print("\nLoading models...")
yolo_model = YOLO("yolo.pt")
yolo_model.to(device)

checkpoint = torch.load("../models/dog.pt", map_location=device, weights_only=False)
reid_model = models.resnet50(weights=None)
reid_model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)
reid_model.layer4 = nn.Identity()
reid_model.fc = nn.Identity()
reid_model.to(device)
reid_model.eval()
print("✓ Models loaded")

# Helper functions
def detect_and_crop(image_path):
    """Detect face and crop."""
    image = Image.open(image_path).convert("RGB")
    results = yolo_model(image, verbose=False)
    
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 1:  # Face
                bbox = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox)
                cropped = image.crop((x1, y1, x2, y2))
                return cropped.resize((224, 224), Image.LANCZOS)
    return None

def generate_embedding(image):
    """Generate 1024-dim embedding."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img_array = np.array(image).astype(np.float32) / 255.0
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = reid_model(img_tensor)
    
    if len(features.shape) == 4:
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
    
    return features.squeeze().cpu().numpy().flatten()

# Load sample data
print(f"\nLoading {NUM_DOGS} dog identities...")
test_path = Path(TEST_DATA_PATH)
identity_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])[:NUM_DOGS]

gallery = []
queries = []

for folder in identity_folders:
    identity = folder.name
    images = sorted(list(folder.glob("*.png")))[:IMAGES_PER_DOG]
    
    if len(images) < 2:
        continue
    
    # First image goes to gallery
    cropped = detect_and_crop(str(images[0]))
    if cropped:
        emb = generate_embedding(cropped)
        gallery.append({'identity': identity, 'embedding': emb})
        print(f"  Gallery: {identity} - {images[0].name}")
    
    # Second image is query
    cropped = detect_and_crop(str(images[1]))
    if cropped:
        emb = generate_embedding(cropped)
        queries.append({'identity': identity, 'embedding': emb})
        print(f"  Query:   {identity} - {images[1].name}")

print(f"\n✓ Loaded {len(gallery)} gallery, {len(queries)} queries")

# Build FAISS index
print("\nBuilding FAISS index...")
# Create fresh index with unique path to avoid loading old data
import tempfile
temp_dir = tempfile.mkdtemp()
temp_index_path = os.path.join(temp_dir, "test_faiss.index")
faiss_store = FAISSStore(index_path=temp_index_path, embedding_dim=1024)
for idx, item in enumerate(gallery):
    faiss_store.add_embedding(idx, item['embedding'])
print(f"✓ Index built with {len(gallery)} embeddings")

# Test queries
print("\n" + "="*60)
print("TESTING QUERIES (threshold: 0.9165)")
print("="*60)

THRESHOLD = 0.9165
correct = 0
no_match_count = 0

for query in queries:
    results = faiss_store.search(query['embedding'], k=3)
    
    if results and results[0][1] >= THRESHOLD:
        top_idx, top_sim = results[0]
        predicted = gallery[top_idx]['identity']
        actual = query['identity']
        
        match = "✓" if predicted == actual else "✗"
        print(f"{match} Query {actual}: predicted {predicted} (sim: {top_sim:.4f})")
        
        if predicted == actual:
            correct += 1
    else:
        print(f"⚠ Query {query['identity']}: No match above threshold (best: {results[0][1]:.4f})" if results else f"⚠ Query {query['identity']}: No results")
        no_match_count += 1

accuracy = correct / len(queries) if queries else 0
print("\n" + "="*60)
print(f"RESULTS: {correct}/{len(queries)} correct = {accuracy:.1%} accuracy")
print(f"No matches (below threshold): {no_match_count}")
print("="*60)

# Analyze similarities
same_dog_sims = []
diff_dog_sims = []

for query in queries:
    results = faiss_store.search(query['embedding'], k=len(gallery))
    for idx, sim in results:
        if gallery[idx]['identity'] == query['identity']:
            same_dog_sims.append(sim)
        else:
            diff_dog_sims.append(sim)

if same_dog_sims and diff_dog_sims:
    print(f"\nSame dog similarities: mean={np.mean(same_dog_sims):.4f}, min={np.min(same_dog_sims):.4f}")
    print(f"Different dog similarities: mean={np.mean(diff_dog_sims):.4f}, max={np.max(diff_dog_sims):.4f}")
    print(f"Separation: {np.mean(same_dog_sims) - np.mean(diff_dog_sims):.4f}")
    
    # Threshold recommendation
    safe_threshold = (np.min(same_dog_sims) + np.max(diff_dog_sims)) / 2
    print(f"\nRecommended threshold: {safe_threshold:.4f}")

# Save results to text file
report = []
report.append("="*60)
report.append("QUICK RE-ID TEST RESULTS")
report.append("="*60)
report.append(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report.append(f"Device: {device}")
report.append(f"Dogs tested: {NUM_DOGS}")
report.append(f"Gallery size: {len(gallery)}")
report.append(f"Query size: {len(queries)}")
report.append(f"\nThreshold: {THRESHOLD}")
report.append(f"Accuracy: {correct}/{len(queries)} = {accuracy:.1%}")
report.append(f"No matches (below threshold): {no_match_count}")

if same_dog_sims and diff_dog_sims:
    report.append(f"\nSame dog similarities:")
    report.append(f"  Mean: {np.mean(same_dog_sims):.4f}")
    report.append(f"  Min: {np.min(same_dog_sims):.4f}")
    report.append(f"\nDifferent dog similarities:")
    report.append(f"  Mean: {np.mean(diff_dog_sims):.4f}")
    report.append(f"  Max: {np.max(diff_dog_sims):.4f}")
    report.append(f"\nSeparation: {np.mean(same_dog_sims) - np.mean(diff_dog_sims):.4f}")
    report.append(f"Recommended threshold: {safe_threshold:.4f}")

report.append("\n" + "="*60)

output_path = "quick_test_results.txt"
with open(output_path, 'w') as f:
    f.write("\n".join(report))

print(f"\n✓ Results saved to: {output_path}")

# Cleanup temp directory
import shutil
try:
    shutil.rmtree(temp_dir)
except:
    pass
