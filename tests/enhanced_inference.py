"""
Enhanced inference with multiple improvement strategies.
Each can be toggled on/off for testing.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO
import time

# =============================================================================
# IMPROVEMENT FLAGS - Toggle these on/off
# =============================================================================
USE_LAYER4 = True              # Use layer4 (2048-dim) instead of layer3 (1024-dim)
USE_AUGMENTATION = True        # Test multiple views of same image
USE_ENSEMBLE = False           # Combine layer3 + layer4 (requires both)
USE_QUERY_EXPANSION = False    # Re-rank using top-K average
USE_ADAPTIVE_THRESHOLD = True  # Use confidence-based thresholding

# =============================================================================
# 1. MODEL SETUP WITH LAYER SELECTION
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO("../models/yolo.pt").to(device)

checkpoint = torch.load("../models/dog.pt", map_location=device, weights_only=False)
reid_model = models.resnet50(weights=None)
reid_model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)

if USE_ENSEMBLE:
    # Keep all layers, extract from both layer3 and layer4
    reid_model.fc = nn.Identity()
    EMBEDDING_DIM = 3072  # 1024 + 2048
    print("Using ENSEMBLE mode (layer3 + layer4): 3072-dim")
elif USE_LAYER4:
    # Use full backbone to layer4
    reid_model.fc = nn.Identity()
    EMBEDDING_DIM = 2048
    print("Using LAYER4: 2048-dim")
else:
    # Use only to layer3
    reid_model.layer4 = nn.Identity()
    reid_model.fc = nn.Identity()
    EMBEDDING_DIM = 1024
    print("Using LAYER3: 1024-dim")

reid_model.to(device).eval()

# =============================================================================
# 2. IMAGE AUGMENTATION
# =============================================================================
def detect_and_crop(image_path):
    """Detect face and crop to 224x224."""
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

def augment_image(image):
    """Generate augmented versions of the image."""
    augmented = [image]  # Original
    
    if USE_AUGMENTATION:
        # Horizontal flip
        augmented.append(image.transpose(Image.FLIP_LEFT_RIGHT))
        
        # Small rotations
        augmented.append(image.rotate(5, fillcolor=(128, 128, 128)))
        augmented.append(image.rotate(-5, fillcolor=(128, 128, 128)))
        
        # Brightness adjustments
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(image)
        augmented.append(enhancer.enhance(1.1))  # Brighter
        augmented.append(enhancer.enhance(0.9))  # Darker
    
    return augmented

# =============================================================================
# 3. EMBEDDING GENERATION (with ensemble support)
# =============================================================================
def generate_single_embedding(image):
    """Generate embedding for one image."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    img_array = np.array(image).astype(np.float32) / 255.0
    for i in range(3):
        img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
    
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        if USE_ENSEMBLE:
            # Extract from both layers
            x = reid_model.conv1(img_tensor)
            x = reid_model.bn1(x)
            x = reid_model.relu(x)
            x = reid_model.maxpool(x)
            x = reid_model.layer1(x)
            x = reid_model.layer2(x)
            
            layer3_out = reid_model.layer3(x)
            layer4_out = reid_model.layer4(layer3_out)
            
            # Global average pooling
            layer3_feat = torch.nn.functional.adaptive_avg_pool2d(layer3_out, (1, 1))
            layer4_feat = torch.nn.functional.adaptive_avg_pool2d(layer4_out, (1, 1))
            
            # Concatenate
            features = torch.cat([layer3_feat.squeeze(), layer4_feat.squeeze()], dim=0)
        else:
            features = reid_model(img_tensor)
            if len(features.shape) == 4:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.squeeze()
    
    return features.cpu().numpy().flatten()

def generate_embedding(image):
    """Generate robust embedding with optional augmentation."""
    if USE_AUGMENTATION and image is not None:
        # Generate embeddings for all augmented versions
        augmented_images = augment_image(image)
        embeddings = [generate_single_embedding(aug_img) for aug_img in augmented_images]
        
        # Average the embeddings
        embedding = np.mean(embeddings, axis=0)
        print(f"  [AUG] Averaged {len(embeddings)} augmented views")
    else:
        embedding = generate_single_embedding(image)
    
    return embedding

# =============================================================================
# 4. SEARCH WITH MULTIPLE METRICS
# =============================================================================
def search_with_multiple_metrics(query_embedding, gallery_embeddings, k=5):
    """Search using both cosine similarity and euclidean distance."""
    # Normalize for cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    results = []
    for idx, gallery_emb in enumerate(gallery_embeddings):
        gallery_norm = gallery_emb / np.linalg.norm(gallery_emb)
        
        # Cosine similarity (higher = more similar)
        cosine_sim = np.dot(query_norm, gallery_norm)
        
        # Euclidean distance (lower = more similar)
        euclidean_dist = np.linalg.norm(query_embedding - gallery_emb)
        # Normalize to 0-1 range (approximate)
        max_dist = np.sqrt(EMBEDDING_DIM * 4)  # Rough max distance
        euclidean_sim = 1 - (euclidean_dist / max_dist)
        
        # Weighted combination
        combined_score = 0.7 * cosine_sim + 0.3 * euclidean_sim
        
        results.append((idx, combined_score, cosine_sim, euclidean_sim))
    
    # Sort by combined score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]

# =============================================================================
# 5. QUERY EXPANSION
# =============================================================================
def search_with_query_expansion(query_embedding, gallery_embeddings, gallery_data, k=5):
    """Re-rank using averaged features of top-K matches."""
    # Initial search
    initial_results = search_with_multiple_metrics(query_embedding, gallery_embeddings, k=5)
    
    if not USE_QUERY_EXPANSION:
        return initial_results
    
    # Average top-3 embeddings
    top_indices = [r[0] for r in initial_results[:3]]
    averaged_embedding = np.mean([gallery_embeddings[idx] for idx in top_indices], axis=0)
    
    # Re-search with averaged embedding
    final_results = search_with_multiple_metrics(averaged_embedding, gallery_embeddings, k=k)
    print(f"  [QE] Re-ranked using top-3 average")
    
    return final_results

# =============================================================================
# 6. ADAPTIVE THRESHOLDING
# =============================================================================
def apply_adaptive_threshold(results):
    """Apply confidence-based threshold decision."""
    if not results:
        return None, "No results"
    
    top_score = results[0][1]
    second_score = results[1][1] if len(results) > 1 else 0
    margin = top_score - second_score
    
    if USE_ADAPTIVE_THRESHOLD:
        # Very confident
        if top_score > 0.65 and margin > 0.10:
            return results[0], "High confidence"
        # Moderate confidence with good margin
        elif top_score > 0.50 and margin > 0.05:
            return results[0], "Moderate confidence"
        # Low confidence - reject
        else:
            return None, f"Low confidence (score={top_score:.3f}, margin={margin:.3f})"
    else:
        # Simple threshold
        THRESHOLD = 0.45  # Based on layer4 separation
        if top_score >= THRESHOLD:
            return results[0], "Above threshold"
        else:
            return None, f"Below threshold ({top_score:.3f} < {THRESHOLD})"

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    from pathlib import Path
    from faiss_store import FAISSStore
    import tempfile
    
    print("\n" + "="*70)
    print("ENHANCED RE-ID SYSTEM TEST")
    print("="*70)
    print(f"Augmentation: {'ON' if USE_AUGMENTATION else 'OFF'}")
    print(f"Ensemble: {'ON' if USE_ENSEMBLE else 'OFF'}")
    print(f"Query Expansion: {'ON' if USE_QUERY_EXPANSION else 'OFF'}")
    print(f"Adaptive Threshold: {'ON' if USE_ADAPTIVE_THRESHOLD else 'OFF'}")
    print("="*70)
    
    # Load test data
    TEST_DATA_PATH = r"D:\FYP\data\PetFace\images\test\dog"
    NUM_DOGS = 50
    
    test_path = Path(TEST_DATA_PATH)
    identity_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])[:NUM_DOGS]
    
    gallery = []
    queries = []
    
    # Timing metrics
    timings = {
        'detection': [],
        'embedding_generation': [],
        'search': [],
        'total_per_query': []
    }
    
    print("\nLoading dataset...")
    for folder in identity_folders:
        identity = folder.name
        images = sorted(list(folder.glob("*.png")))[:2]
        if len(images) < 2:
            continue
        
        t0 = time.time()
        cropped = detect_and_crop(str(images[0]))
        timings['detection'].append(time.time() - t0)
        if cropped:
            t0 = time.time()
            emb = generate_embedding(cropped)
            timings['embedding_generation'].append(time.time() - t0)
            gallery.append({'identity': identity, 'embedding': emb})
        
        t0 = time.time()
        cropped = detect_and_crop(str(images[1]))
        timings['detection'].append(time.time() - t0)
        if cropped:
            t0 = time.time()
            emb = generate_embedding(cropped)
            timings['embedding_generation'].append(time.time() - t0)
            queries.append({'identity': identity, 'embedding': emb})
    
    print(f"Gallery: {len(gallery)}, Queries: {len(queries)}")
    
    # Test
    gallery_embeddings = [g['embedding'] for g in gallery]
    
    correct = 0
    rejected = 0
    
    for query in queries:
        query_start = time.time()
        
        t0 = time.time()
        results = search_with_query_expansion(
            query['embedding'], 
            gallery_embeddings, 
            gallery,
            k=5
        )
        timings['search'].append(time.time() - t0)
        
        match, reason = apply_adaptive_threshold(results)
        
        timings['total_per_query'].append(time.time() - query_start)
        
        if match:
            predicted_idx = match[0]
            predicted = gallery[predicted_idx]['identity']
            actual = query['identity']
            
            if predicted == actual:
                correct += 1
        else:
            rejected += 1
    
    total = len(queries)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Correct: {correct}/{total} = {accuracy:.1%}")
    print(f"Rejected (no match): {rejected}")
    print(f"Effective accuracy: {correct}/{total-rejected} = {correct/(total-rejected)*100:.1%}" if total > rejected else "N/A")
    
    print(f"\n{'='*70}")
    print(f"LATENCY METRICS")
    print(f"{'='*70}")
    
    if timings['detection']:
        avg_detection = np.mean(timings['detection']) * 1000
        print(f"Face Detection: {avg_detection:.1f} ms (avg)")
    
    if timings['embedding_generation']:
        avg_embedding = np.mean(timings['embedding_generation']) * 1000
        print(f"Embedding Generation: {avg_embedding:.1f} ms (avg)")
    
    if timings['search']:
        avg_search = np.mean(timings['search']) * 1000
        print(f"Search: {avg_search:.1f} ms (avg)")
    
    if timings['total_per_query']:
        avg_total = np.mean(timings['total_per_query']) * 1000
        p95_total = np.percentile(timings['total_per_query'], 95) * 1000
        print(f"Total Per Query: {avg_total:.1f} ms (avg), {p95_total:.1f} ms (p95)")
    
    # Compute end-to-end for a new dog (detection + embedding + search)
    if timings['detection'] and timings['embedding_generation'] and timings['search']:
        end_to_end = (np.mean(timings['detection']) + 
                      np.mean(timings['embedding_generation']) + 
                      np.mean(timings['search'])) * 1000
        print(f"End-to-End (detect + embed + search): {end_to_end:.1f} ms")
