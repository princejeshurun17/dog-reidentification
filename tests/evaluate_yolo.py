"""
YOLO Model Evaluation Script
Tests dog face detection performance on a labeled dataset.
Evaluates precision, recall, false positives, and provides detailed statistics.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import requests
from pathlib import Path
from PIL import Image
import io
import numpy as np
from ultralytics import YOLO
import time

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ============================================================================
# CONFIGURATION
# ============================================================================
YOLO_MODEL_PATH = "../models/yolo.pt"
OUTPUT_DIR = "../logs/yolo_evaluation"
RESULTS_FILE = "../logs/yolo_evaluation_results_over6.txt"

# Detection thresholds to test
CONFIDENCE_THRESHOLDS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# Guardrails (matching inference service)
MIN_BBOX_SIZE = 50
MAX_BBOX_SIZE = 2000
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.0

# ============================================================================
# TEST DATASET
# ============================================================================
# Using a curated set of images with labels
# Format: {"url": "image_url", "has_dog": True/False, "description": "..."}

TEST_IMAGES = [
    # Positive samples (contains dogs) - 50 images
    {"url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb", "has_dog": True, "desc": "Golden retriever close-up"},
    {"url": "https://images.unsplash.com/photo-1552053831-71594a27632d", "has_dog": True, "desc": "Dog portrait outdoor"},
    {"url": "https://images.unsplash.com/photo-1561037404-61cd46aa615b", "has_dog": True, "desc": "Husky close-up"},
    {"url": "https://images.unsplash.com/photo-1588943211346-0908a1fb0b01", "has_dog": True, "desc": "Beagle puppy"},
    {"url": "https://images.unsplash.com/photo-1568572933382-74d440642117", "has_dog": True, "desc": "Border collie"},
    {"url": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e", "has_dog": True, "desc": "Pug face"},
    {"url": "https://images.unsplash.com/photo-1583511655826-05700d52f4d9", "has_dog": True, "desc": "German shepherd"},
    {"url": "https://images.unsplash.com/photo-1591160690555-5debfba289f0", "has_dog": True, "desc": "Corgi smiling"},
    {"url": "https://images.unsplash.com/photo-1596492784531-6e6eb5ea9993", "has_dog": True, "desc": "Dalmatian"},
    {"url": "https://images.unsplash.com/photo-1600077106724-946750eeaf3c", "has_dog": True, "desc": "Labrador retriever"},
    {"url": "https://images.unsplash.com/photo-1543466835-00a7907e9de1", "has_dog": True, "desc": "Shiba Inu"},
    {"url": "https://images.unsplash.com/photo-1558788353-f76d92427f16", "has_dog": True, "desc": "Poodle portrait"},
    {"url": "https://images.unsplash.com/photo-1534351450181-ea9f78427fe8", "has_dog": True, "desc": "Australian Shepherd"},
    {"url": "https://images.unsplash.com/photo-1537151608828-ea2b11777ee8", "has_dog": True, "desc": "Bulldog face"},
    {"url": "https://images.unsplash.com/photo-1597633425046-08f5110420b5", "has_dog": True, "desc": "Pitbull portrait"},
    {"url": "https://images.unsplash.com/photo-1601758228041-f3b2795255f1", "has_dog": True, "desc": "Boxer dog"},
    {"url": "https://images.unsplash.com/photo-1576201836106-db1758fd1c97", "has_dog": True, "desc": "Rottweiler"},
    {"url": "https://images.unsplash.com/photo-1555685812-4b943f1cb0eb", "has_dog": True, "desc": "Chihuahua"},
    {"url": "https://images.unsplash.com/photo-1598133894008-61f7fdb8cc3a", "has_dog": True, "desc": "Dachshund"},
    {"url": "https://images.unsplash.com/photo-1601979031925-424e53b6caaa", "has_dog": True, "desc": "Jack Russell"},
    {"url": "https://images.unsplash.com/photo-1600804931749-2da4ce26c869", "has_dog": True, "desc": "Samoyed"},
    {"url": "https://images.unsplash.com/photo-1605468845049-22e577e5c4c8", "has_dog": True, "desc": "Cocker Spaniel"},
    {"url": "https://images.unsplash.com/photo-1629903112904-4e7d2695bbbb", "has_dog": True, "desc": "Bernese Mountain Dog"},
    {"url": "https://images.unsplash.com/photo-1598133893773-de3574464ef0", "has_dog": True, "desc": "Great Dane"},
    {"url": "https://images.unsplash.com/photo-1580489944761-15a19d654956", "has_dog": True, "desc": "Yorkshire Terrier"},
    {"url": "https://images.unsplash.com/photo-1615751072497-5f5169febe17", "has_dog": True, "desc": "Maltese dog"},
    {"url": "https://images.unsplash.com/photo-1560807707-8cc77767d783", "has_dog": True, "desc": "Akita"},
    {"url": "https://images.unsplash.com/photo-1592194996308-7b43878e84a6", "has_dog": True, "desc": "Doberman"},
    {"url": "https://images.unsplash.com/photo-1530281700549-e82e7bf110d6", "has_dog": True, "desc": "Chow Chow"},
    {"url": "https://images.unsplash.com/photo-1477884213360-7e9d7dcc1e48", "has_dog": True, "desc": "Mixed breed dog"},
    {"url": "https://images.unsplash.com/photo-1585559604959-6388fe69c92a", "has_dog": True, "desc": "Basset Hound"},
    {"url": "https://images.unsplash.com/photo-1560743173-567a3b5658b1", "has_dog": True, "desc": "Saint Bernard"},
    {"url": "https://images.unsplash.com/photo-1599003662596-760a8e2c9f05", "has_dog": True, "desc": "Schnauzer"},
    {"url": "https://images.unsplash.com/photo-1594149929607-8a96c0450456", "has_dog": True, "desc": "Bichon Frise"},
    {"url": "https://images.unsplash.com/photo-1553882809-a4f57e59501d", "has_dog": True, "desc": "West Highland Terrier"},
    {"url": "https://images.unsplash.com/photo-1551717743-49959800b1f6", "has_dog": True, "desc": "Springer Spaniel"},
    {"url": "https://images.unsplash.com/photo-1605026068816-1097d4ccaeb0", "has_dog": True, "desc": "Pomeranian"},
    {"url": "https://images.unsplash.com/photo-1625316708582-7c38734be31d", "has_dog": True, "desc": "Bullmastiff"},
    {"url": "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e", "has_dog": True, "desc": "Weimaraner"},
    {"url": "https://images.unsplash.com/photo-1516734212186-a967f81ad0d7", "has_dog": True, "desc": "Newfoundland"},
    {"url": "https://images.unsplash.com/photo-1568393691622-c7ba131d63b4", "has_dog": True, "desc": "Boston Terrier"},
    {"url": "https://images.unsplash.com/photo-1541364983171-a8ba01e95cfc", "has_dog": True, "desc": "Bull Terrier"},
    {"url": "https://images.unsplash.com/photo-1598133893008-61f7fdb8cc3a", "has_dog": True, "desc": "Miniature Dachshund"},
    {"url": "https://images.unsplash.com/photo-1560807707-8cc77767d783", "has_dog": True, "desc": "Alaskan Malamute"},
    {"url": "https://images.unsplash.com/photo-1548199973-03cce0bbc87b", "has_dog": True, "desc": "Rhodesian Ridgeback"},
    {"url": "https://images.unsplash.com/photo-1557053910-d9eadeed1c58", "has_dog": True, "desc": "Vizsla"},
    {"url": "https://images.unsplash.com/photo-1601758125946-6ec2ef64daf8", "has_dog": True, "desc": "Bloodhound"},
    {"url": "https://images.unsplash.com/photo-1612536981114-d7b96fc84c0b", "has_dog": True, "desc": "Whippet"},
    {"url": "https://images.unsplash.com/photo-1588847858111-35a0b221b3e1", "has_dog": True, "desc": "Afghan Hound"},
    {"url": "https://images.unsplash.com/photo-1611003228941-98852ba62227", "has_dog": True, "desc": "Greyhound"},
    
    # Negative samples (no dogs - should NOT detect) - 50 images
    {"url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba", "has_dog": False, "desc": "Cat portrait"},
    {"url": "https://images.unsplash.com/photo-1574158622682-e40e69881006", "has_dog": False, "desc": "Cat close-up"},
    {"url": "https://images.unsplash.com/photo-1583337130417-3346a1be7dee", "has_dog": False, "desc": "White cat"},
    {"url": "https://images.unsplash.com/photo-1503023345310-bd7c1de61c7d", "has_dog": False, "desc": "Human portrait male"},
    {"url": "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6", "has_dog": False, "desc": "Person outdoors"},
    {"url": "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d", "has_dog": False, "desc": "Man portrait beard"},
    {"url": "https://images.unsplash.com/photo-1494790108377-be9c29b29330", "has_dog": False, "desc": "Woman portrait"},
    {"url": "https://images.unsplash.com/photo-1511367461989-f85a21fda167", "has_dog": False, "desc": "Bird close-up"},
    {"url": "https://images.unsplash.com/photo-1591608971362-f08b2a75731a", "has_dog": False, "desc": "Forest landscape"},
    {"url": "https://images.unsplash.com/photo-1506905925346-21bda4d32df4", "has_dog": False, "desc": "Mountain scenery"},
    {"url": "https://images.unsplash.com/photo-1529778873920-4da4926a72c2", "has_dog": False, "desc": "Orange tabby cat"},
    {"url": "https://images.unsplash.com/photo-1533738363-b7f9aef128ce", "has_dog": False, "desc": "Siamese cat"},
    {"url": "https://images.unsplash.com/photo-1492370284958-c20b15c692d2", "has_dog": False, "desc": "Black cat"},
    {"url": "https://images.unsplash.com/photo-1517849845537-4d257902454a", "has_dog": False, "desc": "Maine Coon cat"},
    {"url": "https://images.unsplash.com/photo-1491485880348-85d48a9e5312", "has_dog": False, "desc": "Persian cat"},
    {"url": "https://images.unsplash.com/photo-1438761681033-6461ffad8d80", "has_dog": False, "desc": "Woman smiling"},
    {"url": "https://images.unsplash.com/photo-1500648767791-00dcc994a43e", "has_dog": False, "desc": "Man professional"},
    {"url": "https://images.unsplash.com/photo-1534528741775-53994a69daeb", "has_dog": False, "desc": "Girl portrait"},
    {"url": "https://images.unsplash.com/photo-1542190891-2093d38760f2", "has_dog": False, "desc": "Elderly person"},
    {"url": "https://images.unsplash.com/photo-1552374196-c4e7ffc6e126", "has_dog": False, "desc": "Baby portrait"},
    {"url": "https://images.unsplash.com/photo-1552053831-71594a27632d", "has_dog": False, "desc": "Horse portrait"},
    {"url": "https://images.unsplash.com/photo-1563281577-a7be47e20db9", "has_dog": False, "desc": "Rabbit"},
    {"url": "https://images.unsplash.com/photo-1425082661705-1834bfd09dca", "has_dog": False, "desc": "Parrot bird"},
    {"url": "https://images.unsplash.com/photo-1549214309-fbd57ff15ff9", "has_dog": False, "desc": "Hamster"},
    {"url": "https://images.unsplash.com/photo-1535268647677-300dbf3d78d1", "has_dog": False, "desc": "Eagle bird"},
    {"url": "https://images.unsplash.com/photo-1470093851219-69951fcbb533", "has_dog": False, "desc": "Sunset landscape"},
    {"url": "https://images.unsplash.com/photo-1441974231531-c6227db76b6e", "has_dog": False, "desc": "Forest path"},
    {"url": "https://images.unsplash.com/photo-1505142468610-359e7d316be0", "has_dog": False, "desc": "Beach ocean"},
    {"url": "https://images.unsplash.com/photo-1472214103451-9374bd1c798e", "has_dog": False, "desc": "Mountain peak"},
    {"url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b", "has_dog": False, "desc": "Snow mountains"},
    {"url": "https://images.unsplash.com/photo-1469474968028-56623f02e42e", "has_dog": False, "desc": "Nature waterfall"},
    {"url": "https://images.unsplash.com/photo-1475924156734-496f6cac6ec1", "has_dog": False, "desc": "Sunset clouds"},
    {"url": "https://images.unsplash.com/photo-1502082553048-f009c37129b9", "has_dog": False, "desc": "City skyline"},
    {"url": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b", "has_dog": False, "desc": "Building architecture"},
    {"url": "https://images.unsplash.com/photo-1478760329108-5c3ed9d495a0", "has_dog": False, "desc": "Food burger"},
    {"url": "https://images.unsplash.com/photo-1504674900247-0877df9cc836", "has_dog": False, "desc": "Food salad"},
    {"url": "https://images.unsplash.com/photo-1490645935967-10de6ba17061", "has_dog": False, "desc": "Food pizza"},
    {"url": "https://images.unsplash.com/photo-1516594798947-e65505dbb29d", "has_dog": False, "desc": "Flower roses"},
    {"url": "https://images.unsplash.com/photo-1490750967868-88aa4486c946", "has_dog": False, "desc": "Flower tulips"},
    {"url": "https://images.unsplash.com/photo-1502224562085-639556652f33", "has_dog": False, "desc": "Plant leaves"},
    {"url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64", "has_dog": False, "desc": "Car sports"},
    {"url": "https://images.unsplash.com/photo-1542362567-b07e54358753", "has_dog": False, "desc": "Car vintage"},
    {"url": "https://images.unsplash.com/photo-1511919884226-fd3cad34687c", "has_dog": False, "desc": "Book stack"},
    {"url": "https://images.unsplash.com/photo-1497633762265-9d179a990aa6", "has_dog": False, "desc": "Workspace desk"},
    {"url": "https://images.unsplash.com/photo-1484480974693-6ca0a78fb36b", "has_dog": False, "desc": "Coffee cup"},
    {"url": "https://images.unsplash.com/photo-1495521821757-a1efb6729352", "has_dog": False, "desc": "Laptop computer"},
    {"url": "https://images.unsplash.com/photo-1511385348-e43d7a4005b5", "has_dog": False, "desc": "Technology phone"},
    {"url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9", "has_dog": False, "desc": "Headphones"},
    {"url": "https://images.unsplash.com/photo-1519681393784-d120267933ba", "has_dog": False, "desc": "Night sky stars"},
    {"url": "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d", "has_dog": False, "desc": "Person casual"},
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def download_image(url, timeout=10):
    """Download image from URL."""
    try:
        # Add params to get smaller image from Unsplash
        url_with_params = f"{url}?w=800&q=80"
        response = requests.get(url_with_params, timeout=timeout)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None


def validate_detection(bbox, confidence, threshold):
    """
    Apply same guardrails as inference service.
    Returns (is_valid, reason)
    """
    # Confidence check
    if confidence < threshold:
        return False, f"Low confidence: {confidence:.3f}"
    
    # Calculate dimensions
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    # Size checks
    if width < MIN_BBOX_SIZE or height < MIN_BBOX_SIZE:
        return False, f"Too small: {width:.0f}x{height:.0f}"
    
    if width > MAX_BBOX_SIZE or height > MAX_BBOX_SIZE:
        return False, f"Too large: {width:.0f}x{height:.0f}"
    
    # Aspect ratio check
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False, f"Bad aspect ratio: {aspect_ratio:.2f}"
    
    return True, "Valid"


def detect_dogs(image, model, confidence_threshold):
    """
    Detect dog faces with validation.
    Returns: (valid_detections, all_detections, rejected_reasons)
    """
    results = model(image, verbose=False)
    
    all_detections = []
    valid_detections = []
    rejected_reasons = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            
            # Only check face detections (class 1)
            if class_id == 1:
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0])
                
                all_detections.append({
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id
                })
                
                # Validate with guardrails
                is_valid, reason = validate_detection(bbox, confidence, confidence_threshold)
                
                if is_valid:
                    valid_detections.append({
                        "bbox": bbox,
                        "confidence": confidence
                    })
                else:
                    rejected_reasons.append(reason)
    
    return valid_detections, all_detections, rejected_reasons


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def evaluate_yolo():
    """Main evaluation function."""
    
    print("=" * 80)
    print("YOLO DOG FACE DETECTION - EVALUATION")
    print("=" * 80)
    print()
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    model = YOLO(YOLO_MODEL_PATH)
    print("✓ Model loaded\n")
    
    # Download all test images
    print(f"Downloading {len(TEST_IMAGES)} test images...")
    images_data = []
    for idx, item in enumerate(TEST_IMAGES):
        print(f"  [{idx+1}/{len(TEST_IMAGES)}] {item['desc']}... ", end="")
        image = download_image(item['url'])
        if image:
            images_data.append({
                "image": image,
                "has_dog": item["has_dog"],
                "desc": item["desc"]
            })
            print("✓")
        else:
            print("✗ Failed")
    
    print(f"\n✓ Downloaded {len(images_data)}/{len(TEST_IMAGES)} images\n")
    
    # Count ground truth
    total_images = len(images_data)
    positive_images = sum(1 for d in images_data if d["has_dog"])
    negative_images = total_images - positive_images
    
    print(f"Dataset: {total_images} images ({positive_images} with dogs, {negative_images} without)\n")
    print("=" * 80)
    
    # Test each confidence threshold
    results_summary = []
    
    for conf_thresh in CONFIDENCE_THRESHOLDS:
        print(f"\nTesting with confidence threshold: {conf_thresh}")
        print("-" * 80)
        
        true_positives = 0   # Correctly detected dogs
        false_positives = 0  # Detected dogs where none exist
        true_negatives = 0   # Correctly identified no dogs
        false_negatives = 0  # Missed dogs that exist
        
        total_detections = 0
        total_rejected = 0
        latencies = []
        
        for idx, data in enumerate(images_data):
            image = data["image"]
            has_dog = data["has_dog"]
            desc = data["desc"]
            
            # Run detection with timing
            start_time = time.time()
            valid_dets, all_dets, rejected = detect_dogs(image, model, conf_thresh)
            latency = (time.time() - start_time) * 1000  # ms
            latencies.append(latency)
            
            detected_dog = len(valid_dets) > 0
            total_detections += len(all_dets)
            total_rejected += len(rejected)
            
            # Calculate metrics
            if has_dog and detected_dog:
                true_positives += 1
                result = "✓ TP"
            elif has_dog and not detected_dog:
                false_negatives += 1
                result = "✗ FN"
            elif not has_dog and not detected_dog:
                true_negatives += 1
                result = "✓ TN"
            else:  # not has_dog and detected_dog
                false_positives += 1
                result = "✗ FP"
            
            print(f"  {result} | {desc:40s} | Dets: {len(valid_dets)} | Latency: {latency:.1f}ms")
        
        # Calculate statistics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (true_positives + true_negatives) / total_images if total_images > 0 else 0
        
        avg_latency = np.mean(latencies) if latencies else 0
        
        print()
        print(f"Results at threshold {conf_thresh}:")
        print(f"  True Positives:  {true_positives}/{positive_images}")
        print(f"  False Negatives: {false_negatives}/{positive_images}")
        print(f"  True Negatives:  {true_negatives}/{negative_images}")
        print(f"  False Positives: {false_positives}/{negative_images}")
        print()
        print(f"  Precision: {precision:.2%} (of detected dogs, how many were real?)")
        print(f"  Recall:    {recall:.2%} (of real dogs, how many did we find?)")
        print(f"  F1 Score:  {f1_score:.2%}")
        print(f"  Accuracy:  {accuracy:.2%}")
        print()
        print(f"  Total raw detections: {total_detections}")
        print(f"  Rejected by guardrails: {total_rejected}")
        print(f"  Average latency: {avg_latency:.1f}ms")
        
        results_summary.append({
            "threshold": conf_thresh,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "accuracy": accuracy,
            "tp": true_positives,
            "fp": false_positives,
            "tn": true_negatives,
            "fn": false_negatives,
            "avg_latency": avg_latency
        })
    
    # Save detailed results
    print("\n" + "=" * 80)
    print("SUMMARY - Best Threshold Selection")
    print("=" * 80)
    
    with open(RESULTS_FILE, 'w') as f:
        f.write("YOLO Dog Face Detection - Evaluation Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {total_images} images ({positive_images} with dogs, {negative_images} without)\n\n")
        
        f.write("Threshold | Precision | Recall | F1 Score | Accuracy | FP | FN | Latency\n")
        f.write("-" * 80 + "\n")
        
        best_f1 = max(results_summary, key=lambda x: x['f1'])
        best_precision = max(results_summary, key=lambda x: x['precision'])
        best_recall = max(results_summary, key=lambda x: x['recall'])
        
        for r in results_summary:
            marker = ""
            if r == best_f1:
                marker = " <- Best F1"
            elif r == best_precision:
                marker = " <- Best Precision"
            elif r == best_recall:
                marker = " <- Best Recall"
            
            line = f"{r['threshold']:.2f}      | {r['precision']:.1%}     | {r['recall']:.1%}  | {r['f1']:.1%}    | {r['accuracy']:.1%}    | {r['fp']}  | {r['fn']}  | {r['avg_latency']:.0f}ms{marker}\n"
            f.write(line)
            print(line.rstrip())
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("Recommendations:\n")
        f.write(f"- Best F1 Score (balanced): {best_f1['threshold']}\n")
        f.write(f"- Best Precision (fewer false positives): {best_precision['threshold']}\n")
        f.write(f"- Best Recall (find more dogs): {best_recall['threshold']}\n")
        f.write(f"\nCurrent production threshold: 0.45\n")
        f.write(f"Guardrails: Min size={MIN_BBOX_SIZE}px, Max size={MAX_BBOX_SIZE}px, ")
        f.write(f"Aspect ratio={MIN_ASPECT_RATIO}-{MAX_ASPECT_RATIO}\n")
    
    print(f"\n✓ Detailed results saved to: {RESULTS_FILE}")
    print("\nRecommended threshold:")
    print(f"  - Balanced (F1):     {best_f1['threshold']} (Precision: {best_f1['precision']:.1%}, Recall: {best_f1['recall']:.1%})")
    print(f"  - High Precision:    {best_precision['threshold']} (Fewer false positives)")
    print(f"  - High Recall:       {best_recall['threshold']} (Find more dogs)")


if __name__ == "__main__":
    evaluate_yolo()
