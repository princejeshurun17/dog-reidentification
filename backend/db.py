"""
Database models and utilities for dog re-identification system.
"""
import os
import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import json
import numpy as np


class DogDatabase:
    """Manages SQLite database for dog metadata and embeddings."""
    
    def __init__(self, db_path: str = "data/dogs.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dogs (
                dog_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                contact_info TEXT,
                notes TEXT,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS identifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dog_id INTEGER,
                image_path TEXT,
                similarity_score REAL,
                identified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (dog_id) REFERENCES dogs (dog_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_dog(self, name: str, embedding: np.ndarray, 
                contact_info: str = "", notes: str = "", 
                image_path: str = "") -> int:
        """Add a new dog to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize embedding as bytes
        embedding_bytes = embedding.astype(np.float32).tobytes()
        
        cursor.execute("""
            INSERT INTO dogs (name, contact_info, notes, embedding, image_path)
            VALUES (?, ?, ?, ?, ?)
        """, (name, contact_info, notes, embedding_bytes, image_path))
        
        dog_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return dog_id
    
    def get_dog(self, dog_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a dog by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM dogs WHERE dog_id = ?", (dog_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            dog = dict(row)
            # Deserialize embedding
            embedding_bytes = dog['embedding']
            dog['embedding'] = np.frombuffer(embedding_bytes, dtype=np.float32)
            return dog
        return None
    
    def get_all_dogs(self) -> List[Dict[str, Any]]:
        """Retrieve all dogs from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM dogs ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        
        dogs = []
        for row in rows:
            dog = dict(row)
            # Deserialize embedding
            embedding_bytes = dog['embedding']
            dog['embedding'] = np.frombuffer(embedding_bytes, dtype=np.float32)
            dogs.append(dog)
        
        return dogs
    
    def update_dog(self, dog_id: int, **kwargs):
        """Update dog information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build update query dynamically
        allowed_fields = ['name', 'contact_info', 'notes', 'image_path']
        update_fields = []
        values = []
        
        for field in allowed_fields:
            if field in kwargs:
                update_fields.append(f"{field} = ?")
                values.append(kwargs[field])
        
        if not update_fields:
            conn.close()
            return
        
        # Always update timestamp
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        values.append(dog_id)
        
        query = f"UPDATE dogs SET {', '.join(update_fields)} WHERE dog_id = ?"
        cursor.execute(query, values)
        
        conn.commit()
        conn.close()
    
    def delete_dog(self, dog_id: int):
        """Delete a dog from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM dogs WHERE dog_id = ?", (dog_id,))
        cursor.execute("DELETE FROM identifications WHERE dog_id = ?", (dog_id,))
        
        conn.commit()
        conn.close()
    
    def log_identification(self, dog_id: int, image_path: str, similarity_score: float):
        """Log an identification event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO identifications (dog_id, image_path, similarity_score)
            VALUES (?, ?, ?)
        """, (dog_id, image_path, similarity_score))
        
        conn.commit()
        conn.close()
    
    def get_recent_identifications(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent identification history."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.*, d.name 
            FROM identifications i
            LEFT JOIN dogs d ON i.dog_id = d.dog_id
            ORDER BY i.identified_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def get_dog_count(self) -> int:
        """Get total number of dogs in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM dogs")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
