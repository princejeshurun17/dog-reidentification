"""Test layer4 (2048-dim) vs layer3 (1024-dim) performance."""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
from faiss_store import FAISSStore
import tempfile

TEST_DATA_PATH = r"D:\FYP\data\PetFace\images\test\dog"
NUM_DOGS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("yolo.pt").to(device)
checkpoint = torch.load("../models/dog.pt", map_location=device, weights_only=False)

def test_layer(use_layer4=True):
    """Test with layer3 or layer4."""
    reid_model = models.resnet50(weights=None)
    reid_model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)
    
    if use_layer4:
        # Use full backbone up to layer4 (2048-dim)
        reid_model.fc = nn.Identity()
        embedding_dim = 2048
        layer_name = "layer4"
    else:
        # Use only up to layer3 (1024-dim)
        reid_model.layer4 = nn.Identity()
        reid_model.fc = nn.Identity()
        embedding_dim = 1024
        layer_name = "layer3"
    
    reid_model.to(device).eval()
    print(f"\n{'='*60}")
    print(f"TESTING WITH {layer_name.upper()} ({embedding_dim}-dim)")
    print('='*60)
    
    def generate_embedding(image):
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
    
    def detect_and_crop(image_path):
        image = Image.open(image_path).convert("RGB")
        results = yolo_model(image, verbose=False)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 1:
                    bbox = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    cropped = image.crop((x1, y1, x2, y2))
                    return cropped.resize((224, 224), Image.LANCZOS)
        return None
    
    # Load data
    test_path = Path(TEST_DATA_PATH)
    identity_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])[:NUM_DOGS]
    
    gallery = []
    queries = []
    
    for folder in identity_folders:
        identity = folder.name
        images = sorted(list(folder.glob("*.png")))[:2]
        if len(images) < 2:
            continue
        
        cropped = detect_and_crop(str(images[0]))
        if cropped:
            gallery.append({'identity': identity, 'embedding': generate_embedding(cropped)})
        
        cropped = detect_and_crop(str(images[1]))
        if cropped:
            queries.append({'identity': identity, 'embedding': generate_embedding(cropped)})
    
    print(f"Gallery: {len(gallery)}, Queries: {len(queries)}")
    
    # Build FAISS
    temp_dir = tempfile.mkdtemp()
    temp_index = os.path.join(temp_dir, "test.index")
    faiss_store = FAISSStore(index_path=temp_index, embedding_dim=embedding_dim)
    
    for idx, item in enumerate(gallery):
        faiss_store.add_embedding(idx, item['embedding'])
    
    # Test queries
    correct = 0
    same_sims = []
    diff_sims = []
    
    for query in queries:
        results = faiss_store.search(query['embedding'], k=len(gallery))
        if results:
            top_idx, top_sim = results[0]
            if gallery[top_idx]['identity'] == query['identity']:
                correct += 1
            
            for idx, sim in results:
                if gallery[idx]['identity'] == query['identity']:
                    same_sims.append(sim)
                else:
                    diff_sims.append(sim)
    
    accuracy = correct / len(queries) if queries else 0
    separation = np.mean(same_sims) - np.mean(diff_sims) if same_sims and diff_sims else 0
    
    print(f"\nAccuracy: {correct}/{len(queries)} = {accuracy:.1%}")
    print(f"Same dog mean: {np.mean(same_sims):.4f}")
    print(f"Different dog mean: {np.mean(diff_sims):.4f}")
    print(f"Separation: {separation:.4f}")
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    return accuracy, separation

# Test both
layer3_acc, layer3_sep = test_layer(use_layer4=False)
layer4_acc, layer4_sep = test_layer(use_layer4=True)

print(f"\n{'='*60}")
print("COMPARISON")
print('='*60)
print(f"Layer3 (1024-dim): {layer3_acc:.1%} accuracy, {layer3_sep:.4f} separation")
print(f"Layer4 (2048-dim): {layer4_acc:.1%} accuracy, {layer4_sep:.4f} separation")

if layer4_acc > layer3_acc:
    print(f"\n✓ Layer4 is BETTER by {(layer4_acc - layer3_acc)*100:.1f}%")
else:
    print(f"\n✓ Layer3 is BETTER by {(layer3_acc - layer4_acc)*100:.1f}%")
