"""
FAISS index manager for efficient similarity search of dog embeddings.
"""
import os
import numpy as np
import faiss
from typing import List, Tuple, Optional
import threading


class FAISSStore:
    """Manages FAISS index for dog face embeddings with cosine similarity."""
    
    def __init__(self, index_path: str = "data/faiss.index", embedding_dim: int = 2048):
        """
        Initialize FAISS index manager.
        
        Args:
            index_path: Path to save/load FAISS index
            embedding_dim: Dimension of embedding vectors (default 2048 for layer4)
        """
        self.index_path = index_path
        self.embedding_dim = embedding_dim
        self.index = None
        self.dog_ids = []  # Maps index position to dog_id
        self.lock = threading.Lock()
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        self._init_index()
    
    def _init_index(self):
        """Initialize or load FAISS index."""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index for cosine similarity (Inner Product after normalization)."""
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        print(f"Created new FAISS index with dimension {self.embedding_dim}")
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding to unit length for cosine similarity.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding
        """
        embedding = embedding.astype(np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def add_embedding(self, dog_id: int, embedding: np.ndarray, already_normalized: bool = False):
        """
        Add a dog embedding to the index.
        
        Args:
            dog_id: Database ID of the dog
            embedding: Embedding vector (raw, not normalized)
            already_normalized: If True, skip normalization
        """
        with self.lock:
            # Normalize for cosine similarity (unless already normalized)
            if already_normalized:
                normalized = embedding.reshape(1, -1)
            else:
                normalized = self.normalize_embedding(embedding).reshape(1, -1)
            
            # Add to index
            self.index.add(normalized)
            self.dog_ids.append(dog_id)
            
            # Save index
            self.save_index()
    
    def search(self, embedding: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """
        Search for similar embeddings.
        
        Args:
            embedding: Query embedding vector
            k: Number of nearest neighbors to return
            
        Returns:
            List of (dog_id, similarity_score) tuples
        """
        if self.index.ntotal == 0:
            print("[FAISS SEARCH] Index is empty, no results")
            return []
        
        with self.lock:
            # Normalize query embedding
            normalized = self.normalize_embedding(embedding).reshape(1, -1)
            
            # Search (returns distances and indices)
            k = min(k, self.index.ntotal)
            distances, indices = self.index.search(normalized, k)
            
            print(f"[FAISS SEARCH] Query embedding shape: {embedding.shape}")
            print(f"[FAISS SEARCH] Index has {self.index.ntotal} embeddings, {len(self.dog_ids)} dog IDs")
            print(f"[FAISS SEARCH] Raw distances: {distances[0]}")
            print(f"[FAISS SEARCH] Raw indices: {indices[0]}")
            
            # Convert to list of (dog_id, similarity_score)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(self.dog_ids):
                    dog_id = self.dog_ids[idx]
                    # Distance is inner product (cosine similarity for normalized vectors)
                    similarity = float(dist)
                    results.append((dog_id, similarity))
                    print(f"[FAISS SEARCH] Mapped index {idx} to dog_id {dog_id}, similarity {similarity:.4f}")
                else:
                    print(f"[FAISS SEARCH] Invalid index {idx} (dog_ids length: {len(self.dog_ids)})")
            
            return results
    
    def rebuild_from_database(self, dogs_data: List[Tuple[int, np.ndarray]]):
        """
        Rebuild the entire FAISS index from database.
        
        Args:
            dogs_data: List of (dog_id, embedding) tuples
        """
        with self.lock:
            # Create new index
            self._create_new_index()
            self.dog_ids = []
            
            if not dogs_data:
                self.save_index()
                return
            
            # Normalize all embeddings (embeddings from DB are raw/unnormalized)
            embeddings = []
            for dog_id, embedding in dogs_data:
                # Check if embedding is already normalized (norm close to 1.0)
                norm = np.linalg.norm(embedding)
                if abs(norm - 1.0) < 0.01:
                    # Already normalized, use as-is
                    normalized = embedding
                    print(f"  Dog {dog_id}: embedding already normalized (norm={norm:.4f})")
                else:
                    # Not normalized, normalize it
                    normalized = self.normalize_embedding(embedding)
                    print(f"  Dog {dog_id}: normalizing embedding (norm={norm:.4f} -> 1.0)")
                embeddings.append(normalized)
                self.dog_ids.append(dog_id)
            
            # Add all at once
            embeddings_array = np.vstack(embeddings)
            self.index.add(embeddings_array)
            
            # Save
            self.save_index()
            print(f"Rebuilt FAISS index with {len(dogs_data)} embeddings")
    
    def remove_embedding(self, dog_id: int):
        """
        Remove a dog's embedding from the index.
        Note: FAISS doesn't support efficient removal, so we rebuild the index.
        
        Args:
            dog_id: Database ID of the dog to remove
        """
        with self.lock:
            if dog_id in self.dog_ids:
                # Remove from mapping
                idx = self.dog_ids.index(dog_id)
                self.dog_ids.pop(idx)
                
                # FAISS doesn't support removal, so we need to rebuild
                # For production, consider using IndexIDMap or rebuilding periodically
                print(f"Removed dog_id {dog_id}. Index rebuild needed for full cleanup.")
    
    def save_index(self):
        """Persist FAISS index to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            # Also save dog_ids mapping
            ids_path = self.index_path + ".ids.npy"
            np.save(ids_path, np.array(self.dog_ids, dtype=np.int32))
        except Exception as e:
            print(f"Error saving FAISS index: {e}")
    
    def load_dog_ids(self):
        """Load dog_ids mapping from disk."""
        ids_path = self.index_path + ".ids.npy"
        if os.path.exists(ids_path):
            try:
                self.dog_ids = np.load(ids_path).tolist()
                print(f"Loaded {len(self.dog_ids)} dog ID mappings")
            except Exception as e:
                print(f"Error loading dog IDs: {e}")
                self.dog_ids = []
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "total_embeddings": self.index.ntotal if self.index else 0,
            "dimension": self.embedding_dim,
            "dog_ids_count": len(self.dog_ids)
        }
