"""
Dog Re-Identification System Evaluation Script

Tests the system on the PetFace dataset with the following structure:
images/test/dog/
    ├── 000000/
    │   ├── 00.png
    │   ├── 01.png
    │   └── ...
    ├── 000001/
    │   └── ...

Each folder represents a unique dog identity.
"""
import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime
from collections import defaultdict
import json

# Add backend to path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, backend_path)

# Set environment variable before importing
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from db import DogDatabase
from faiss_store import FAISSStore
import torch
import torch.nn as nn
import torchvision.models as models
from ultralytics import YOLO

# Configuration
TEST_DATA_PATH = r"D:\FYP\data\PetFace\images\test\dog"
YOLO_MODEL_PATH = "yolo.pt"
REID_MODEL_PATH = "dog.pt"
CROP_SIZE = (224, 224)
EMBEDDING_DIM = 1024
SIMILARITY_THRESHOLD = 0.70

# Evaluation settings
MAX_IDENTITIES = 100  # Limit number of dog identities to test (set to None for all)
IMAGES_PER_IDENTITY = 2  # How many images per dog to use
QUERY_SPLIT = 0.5  # Use first 50% for gallery, rest for queries

# Output
RESULTS_FILE = "evaluation_results.txt"


class DogReIDEvaluator:
    """Evaluates dog re-identification system."""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        self.yolo_model.to(self.device)
        print(f"✓ YOLO model loaded")
        
        # Load Re-ID model
        checkpoint = torch.load(REID_MODEL_PATH, map_location=self.device, weights_only=False)
        self.reid_model = models.resnet50(pretrained=False)
        
        if isinstance(checkpoint, dict) and 'state_dict_backbone' in checkpoint:
            self.reid_model.load_state_dict(checkpoint['state_dict_backbone'], strict=False)
        else:
            raise ValueError("Unexpected model format")
        
        # Use layer3 for 1024-dim features
        self.reid_model.layer4 = nn.Identity()
        self.reid_model.fc = nn.Identity()
        self.reid_model.to(self.device)
        self.reid_model.eval()
        print(f"✓ Re-ID model loaded (layer3, 1024-dim)")
        
        # Initialize FAISS
        self.faiss_store = FAISSStore(embedding_dim=EMBEDDING_DIM)
        print(f"✓ FAISS initialized")
        
        # Statistics
        self.stats = {
            'total_identities': 0,
            'processed_identities': 0,
            'gallery_size': 0,
            'query_size': 0,
            'face_detection_failures': 0,
            'successful_extractions': 0,
            'rank1_correct': 0,
            'rank5_correct': 0,
            'similarities': [],
            'same_dog_similarities': [],
            'different_dog_similarities': [],
        }
    
    def detect_and_crop_face(self, image_path):
        """Detect dog face and return cropped image."""
        try:
            image = Image.open(image_path).convert("RGB")
            results = self.yolo_model(image, verbose=False)
            
            # Find face detection (class 1)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    if class_id == 1:  # Face
                        bbox = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, bbox)
                        cropped = image.crop((x1, y1, x2, y2))
                        resized = cropped.resize(CROP_SIZE, Image.LANCZOS)
                        return resized
            
            return None
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def generate_embedding(self, image):
        """Generate embedding from cropped face image."""
        # ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        img_array = np.array(image).astype(np.float32) / 255.0
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - mean[i]) / std[i]
        
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.reid_model(img_tensor)
        
        # Apply global average pooling for layer3
        if len(features.shape) == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        
        embedding = features.squeeze().cpu().numpy().flatten()
        return embedding
    
    def load_dataset(self):
        """Load test dataset and split into gallery and query sets."""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        test_path = Path(TEST_DATA_PATH)
        if not test_path.exists():
            raise FileNotFoundError(f"Test data path not found: {TEST_DATA_PATH}")
        
        # Get all identity folders
        identity_folders = sorted([d for d in test_path.iterdir() if d.is_dir()])
        self.stats['total_identities'] = len(identity_folders)
        print(f"Found {len(identity_folders)} dog identities")
        
        # Limit identities if specified
        if MAX_IDENTITIES:
            identity_folders = identity_folders[:MAX_IDENTITIES]
            print(f"Using first {MAX_IDENTITIES} identities for evaluation")
        
        gallery_data = []
        query_data = []
        
        print("\nProcessing images...")
        for identity_folder in identity_folders:
            identity_id = identity_folder.name
            image_files = sorted(list(identity_folder.glob("*.png")) + list(identity_folder.glob("*.jpg")))
            
            if len(image_files) < IMAGES_PER_IDENTITY:
                continue
            
            # Use first images
            images_to_process = image_files[:IMAGES_PER_IDENTITY]
            split_point = max(1, int(len(images_to_process) * QUERY_SPLIT))
            
            gallery_images = images_to_process[:split_point]
            query_images = images_to_process[split_point:]
            
            # Process gallery images
            for img_path in gallery_images:
                cropped = self.detect_and_crop_face(str(img_path))
                if cropped:
                    embedding = self.generate_embedding(cropped)
                    gallery_data.append({
                        'identity': identity_id,
                        'path': str(img_path),
                        'embedding': embedding
                    })
                    self.stats['successful_extractions'] += 1
                else:
                    self.stats['face_detection_failures'] += 1
            
            # Process query images
            for img_path in query_images:
                cropped = self.detect_and_crop_face(str(img_path))
                if cropped:
                    embedding = self.generate_embedding(cropped)
                    query_data.append({
                        'identity': identity_id,
                        'path': str(img_path),
                        'embedding': embedding
                    })
                    self.stats['successful_extractions'] += 1
                else:
                    self.stats['face_detection_failures'] += 1
            
            self.stats['processed_identities'] += 1
            if self.stats['processed_identities'] % 10 == 0:
                print(f"  Processed {self.stats['processed_identities']}/{len(identity_folders)} identities...")
        
        self.stats['gallery_size'] = len(gallery_data)
        self.stats['query_size'] = len(query_data)
        
        print(f"\n✓ Dataset loaded:")
        print(f"  Gallery: {len(gallery_data)} images")
        print(f"  Query: {len(query_data)} images")
        print(f"  Face detection failures: {self.stats['face_detection_failures']}")
        
        return gallery_data, query_data
    
    def build_gallery(self, gallery_data):
        """Build FAISS index from gallery embeddings."""
        print("\n" + "="*60)
        print("BUILDING GALLERY INDEX")
        print("="*60)
        
        for idx, item in enumerate(gallery_data):
            self.faiss_store.add_embedding(idx, item['embedding'])
        
        print(f"✓ Gallery index built with {len(gallery_data)} embeddings")
    
    def evaluate_queries(self, gallery_data, query_data):
        """Evaluate queries against gallery."""
        print("\n" + "="*60)
        print("EVALUATING QUERIES")
        print("="*60)
        
        rank1_correct = 0
        rank5_correct = 0
        total_queries = len(query_data)
        
        for idx, query in enumerate(query_data):
            # Search in gallery
            results = self.faiss_store.search(query['embedding'], k=5)
            
            if not results:
                continue
            
            # Get predicted identities
            top1_idx = results[0][0]
            top1_identity = gallery_data[top1_idx]['identity']
            top1_similarity = results[0][1]
            
            top5_identities = [gallery_data[r[0]]['identity'] for r in results]
            
            # Check if correct
            true_identity = query['identity']
            
            if top1_identity == true_identity:
                rank1_correct += 1
                self.stats['same_dog_similarities'].append(top1_similarity)
            else:
                self.stats['different_dog_similarities'].append(top1_similarity)
            
            if true_identity in top5_identities:
                rank5_correct += 1
            
            # Store similarity
            self.stats['similarities'].append(top1_similarity)
            
            if (idx + 1) % 50 == 0:
                print(f"  Evaluated {idx + 1}/{total_queries} queries...")
        
        self.stats['rank1_correct'] = rank1_correct
        self.stats['rank5_correct'] = rank5_correct
        
        print(f"\n✓ Evaluation complete")
    
    def compute_metrics(self):
        """Compute evaluation metrics."""
        metrics = {}
        
        # Rank-1 and Rank-5 accuracy
        if self.stats['query_size'] > 0:
            metrics['rank1_accuracy'] = self.stats['rank1_correct'] / self.stats['query_size']
            metrics['rank5_accuracy'] = self.stats['rank5_correct'] / self.stats['query_size']
        else:
            metrics['rank1_accuracy'] = 0.0
            metrics['rank5_accuracy'] = 0.0
        
        # Similarity statistics
        if self.stats['similarities']:
            metrics['mean_similarity'] = np.mean(self.stats['similarities'])
            metrics['median_similarity'] = np.median(self.stats['similarities'])
            metrics['std_similarity'] = np.std(self.stats['similarities'])
            metrics['min_similarity'] = np.min(self.stats['similarities'])
            metrics['max_similarity'] = np.max(self.stats['similarities'])
        
        # Same vs different dog similarities
        if self.stats['same_dog_similarities']:
            metrics['same_dog_mean'] = np.mean(self.stats['same_dog_similarities'])
            metrics['same_dog_std'] = np.std(self.stats['same_dog_similarities'])
            metrics['same_dog_min'] = np.min(self.stats['same_dog_similarities'])
        
        if self.stats['different_dog_similarities']:
            metrics['different_dog_mean'] = np.mean(self.stats['different_dog_similarities'])
            metrics['different_dog_std'] = np.std(self.stats['different_dog_similarities'])
            metrics['different_dog_max'] = np.max(self.stats['different_dog_similarities'])
        
        return metrics
    
    def generate_report(self, metrics):
        """Generate detailed evaluation report."""
        report = []
        report.append("="*80)
        report.append("DOG RE-IDENTIFICATION SYSTEM EVALUATION REPORT")
        report.append("="*80)
        report.append(f"\nEvaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset: {TEST_DATA_PATH}")
        report.append(f"Device: {self.device}")
        
        report.append("\n" + "-"*80)
        report.append("DATASET STATISTICS")
        report.append("-"*80)
        report.append(f"Total identities in dataset: {self.stats['total_identities']}")
        report.append(f"Processed identities: {self.stats['processed_identities']}")
        report.append(f"Gallery size: {self.stats['gallery_size']} images")
        report.append(f"Query size: {self.stats['query_size']} images")
        report.append(f"Successful extractions: {self.stats['successful_extractions']}")
        report.append(f"Face detection failures: {self.stats['face_detection_failures']}")
        
        report.append("\n" + "-"*80)
        report.append("MODEL CONFIGURATION")
        report.append("-"*80)
        report.append(f"YOLO model: {YOLO_MODEL_PATH}")
        report.append(f"Re-ID model: {REID_MODEL_PATH}")
        report.append(f"Embedding dimension: {EMBEDDING_DIM}")
        report.append(f"Feature extraction: ResNet50 layer3")
        report.append(f"Similarity metric: Cosine similarity (L2 normalized)")
        report.append(f"Similarity threshold: {SIMILARITY_THRESHOLD}")
        
        report.append("\n" + "-"*80)
        report.append("RETRIEVAL PERFORMANCE")
        report.append("-"*80)
        report.append(f"Rank-1 Accuracy: {metrics['rank1_accuracy']:.2%}")
        report.append(f"  (Correct matches: {self.stats['rank1_correct']}/{self.stats['query_size']})")
        report.append(f"Rank-5 Accuracy: {metrics['rank5_accuracy']:.2%}")
        report.append(f"  (Correct matches: {self.stats['rank5_correct']}/{self.stats['query_size']})")
        
        report.append("\n" + "-"*80)
        report.append("SIMILARITY SCORE ANALYSIS")
        report.append("-"*80)
        report.append(f"Overall Statistics:")
        report.append(f"  Mean similarity: {metrics.get('mean_similarity', 0):.4f}")
        report.append(f"  Median similarity: {metrics.get('median_similarity', 0):.4f}")
        report.append(f"  Std deviation: {metrics.get('std_similarity', 0):.4f}")
        report.append(f"  Min similarity: {metrics.get('min_similarity', 0):.4f}")
        report.append(f"  Max similarity: {metrics.get('max_similarity', 0):.4f}")
        
        if 'same_dog_mean' in metrics:
            report.append(f"\nSame Dog Matches (True Positives):")
            report.append(f"  Count: {len(self.stats['same_dog_similarities'])}")
            report.append(f"  Mean: {metrics['same_dog_mean']:.4f}")
            report.append(f"  Std: {metrics['same_dog_std']:.4f}")
            report.append(f"  Min: {metrics['same_dog_min']:.4f}")
        
        if 'different_dog_mean' in metrics:
            report.append(f"\nDifferent Dog Matches (False Positives):")
            report.append(f"  Count: {len(self.stats['different_dog_similarities'])}")
            report.append(f"  Mean: {metrics['different_dog_mean']:.4f}")
            report.append(f"  Std: {metrics['different_dog_std']:.4f}")
            report.append(f"  Max: {metrics['different_dog_max']:.4f}")
        
        # Discrimination analysis
        if 'same_dog_mean' in metrics and 'different_dog_mean' in metrics:
            separation = metrics['same_dog_mean'] - metrics['different_dog_mean']
            report.append(f"\nDiscrimination Analysis:")
            report.append(f"  Separation (same_mean - diff_mean): {separation:.4f}")
            if separation > 0.1:
                report.append(f"  Status: GOOD - Clear separation between same/different dogs")
            elif separation > 0.05:
                report.append(f"  Status: MODERATE - Some overlap expected")
            else:
                report.append(f"  Status: POOR - Significant overlap, consider tuning")
        
        report.append("\n" + "-"*80)
        report.append("THRESHOLD RECOMMENDATIONS")
        report.append("-"*80)
        
        if 'same_dog_min' in metrics and 'different_dog_max' in metrics:
            safe_threshold = (metrics['same_dog_min'] + metrics['different_dog_max']) / 2
            report.append(f"Safe threshold (midpoint): {safe_threshold:.4f}")
            report.append(f"Conservative threshold: {metrics['same_dog_min']:.4f}")
            report.append(f"Aggressive threshold: {metrics['different_dog_max']:.4f}")
            report.append(f"Current threshold: {SIMILARITY_THRESHOLD}")
        
        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    def run_evaluation(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*80)
        print("STARTING DOG RE-IDENTIFICATION EVALUATION")
        print("="*80)
        
        # Load dataset
        gallery_data, query_data = self.load_dataset()
        
        if not gallery_data or not query_data:
            print("\n✗ Insufficient data for evaluation")
            return
        
        # Build gallery
        self.build_gallery(gallery_data)
        
        # Evaluate queries
        self.evaluate_queries(gallery_data, query_data)
        
        # Compute metrics
        metrics = self.compute_metrics()
        
        # Generate report
        report = self.generate_report(metrics)
        
        # Print to console
        print("\n" + report)
        
        # Save to file (in the same directory as script)
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), RESULTS_FILE)
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Report saved to: {output_path}")
        
        return metrics


if __name__ == "__main__":
    evaluator = DogReIDEvaluator()
    metrics = evaluator.run_evaluation()
