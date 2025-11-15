"""
Integration test for the dog re-identification system.
Tests the full pipeline: upload → detect → identify/enroll
"""
import requests
import os
import sys


INFERENCE_URL = "http://127.0.0.1:8000"
UI_URL = "http://127.0.0.1:5000"


def test_health_checks():
    """Test that both services are running."""
    print("\n=== Testing Health Checks ===")
    
    # Test inference service
    try:
        response = requests.get(f"{INFERENCE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✓ Inference service is running")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Inference service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to inference service: {e}")
        return False
    
    # Test UI service
    try:
        response = requests.get(f"{UI_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ UI service is running")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ UI service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to UI service: {e}")
        return False
    
    return True


def test_stats_endpoint():
    """Test statistics endpoint."""
    print("\n=== Testing Statistics Endpoint ===")
    
    try:
        response = requests.get(f"{INFERENCE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✓ Stats endpoint working")
            print(f"  Total dogs: {stats['database']['total_dogs']}")
            print(f"  FAISS embeddings: {stats['faiss']['total_embeddings']}")
            print(f"  Device: {stats['device']}")
            print(f"  Threshold: {stats['similarity_threshold']}")
            return True
        else:
            print(f"✗ Stats endpoint failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error testing stats: {e}")
        return False


def test_list_dogs():
    """Test listing dogs endpoint."""
    print("\n=== Testing List Dogs Endpoint ===")
    
    try:
        response = requests.get(f"{INFERENCE_URL}/dogs", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✓ List dogs endpoint working")
            print(f"  Total dogs: {data['total']}")
            if data['dogs']:
                print("  Sample dogs:")
                for dog in data['dogs'][:3]:
                    print(f"    - {dog['name']} (ID: {dog['dog_id']})")
            return True
        else:
            print(f"✗ List dogs failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error listing dogs: {e}")
        return False


def test_inference_with_sample(image_path):
    """Test inference endpoint with a sample image."""
    print(f"\n=== Testing Inference with {image_path} ===")
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        print("  Skipping inference test.")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{INFERENCE_URL}/infer", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Inference completed successfully")
            print(f"  Total faces detected: {result.get('total_faces', 0)}")
            
            if result.get('detections'):
                for i, detection in enumerate(result['detections']):
                    print(f"\n  Detection {i+1}:")
                    print(f"    Confidence: {detection['confidence']:.2f}")
                    print(f"    Embedding dim: {detection['embedding_dim']}")
                    
                    if detection.get('matches'):
                        match = detection['matches'][0]
                        print(f"    Match found: {match['name']}")
                        print(f"    Similarity: {match['similarity']:.2%}")
                    else:
                        print(f"    No match (unknown dog)")
            
            return result
        else:
            print(f"✗ Inference failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error during inference: {e}")
        return None


def test_ui_process_endpoint(image_path):
    """Test UI process endpoint."""
    print(f"\n=== Testing UI Process Endpoint ===")
    
    if not os.path.exists(image_path):
        print(f"✗ Image file not found: {image_path}")
        print("  Skipping UI test.")
        return None
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{UI_URL}/api/process", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ UI process endpoint working")
            print(f"  Uploaded file: {result.get('uploaded_file')}")
            print(f"  Detections: {result.get('total_faces', 0)}")
            return result
        else:
            print(f"✗ UI process failed with status {response.status_code}")
            print(f"  Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Error during UI processing: {e}")
        return None


def run_all_tests(sample_image=None):
    """Run all tests."""
    print("=" * 70)
    print("Dog Re-ID System Integration Tests")
    print("=" * 70)
    
    # Test 1: Health checks
    if not test_health_checks():
        print("\n✗ Services are not running. Please start them first:")
        print("  Terminal 1: python backend\\inference_service.py")
        print("  Terminal 2: python frontend\\app.py")
        return
    
    # Test 2: Stats
    test_stats_endpoint()
    
    # Test 3: List dogs
    test_list_dogs()
    
    # Test 4: Inference (if image provided)
    if sample_image:
        test_inference_with_sample(sample_image)
        test_ui_process_endpoint(sample_image)
    else:
        print("\n⚠ No sample image provided. Skipping inference tests.")
        print("  Usage: python test_integration.py <path_to_image>")
    
    print("\n" + "=" * 70)
    print("Tests completed!")
    print("=" * 70)


if __name__ == '__main__':
    sample_image = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_tests(sample_image)
