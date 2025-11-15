"""
CLI utility for managing the dog re-identification database.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from db import DogDatabase
from faiss_store import FAISSStore
import argparse


def list_dogs():
    """List all dogs in the database."""
    db = DogDatabase()
    dogs = db.get_all_dogs()
    
    if not dogs:
        print("No dogs in database.")
        return
    
    print(f"\n{'ID':<5} {'Name':<20} {'Contact':<25} {'Created At':<20}")
    print("-" * 70)
    
    for dog in dogs:
        print(f"{dog['dog_id']:<5} {dog['name']:<20} {dog.get('contact_info', ''):<25} {dog.get('created_at', ''):<20}")
    
    print(f"\nTotal: {len(dogs)} dogs")


def rebuild_faiss():
    """Rebuild FAISS index from database."""
    db = DogDatabase()
    faiss_store = FAISSStore()
    
    print("Fetching all dogs from database...")
    dogs = db.get_all_dogs()
    
    if not dogs:
        print("No dogs in database. Nothing to rebuild.")
        return
    
    print(f"Rebuilding FAISS index with {len(dogs)} embeddings...")
    dogs_data = [(dog['dog_id'], dog['embedding']) for dog in dogs]
    faiss_store.rebuild_from_database(dogs_data)
    
    print("✓ FAISS index rebuilt successfully!")
    print(f"  Total embeddings: {faiss_store.index.ntotal}")


def show_stats():
    """Show database and FAISS statistics."""
    db = DogDatabase()
    faiss_store = FAISSStore()
    faiss_store.load_dog_ids()
    
    dog_count = db.get_dog_count()
    faiss_stats = faiss_store.get_stats()
    
    print("\n=== Database Statistics ===")
    print(f"Total dogs: {dog_count}")
    
    print("\n=== FAISS Index Statistics ===")
    print(f"Total embeddings: {faiss_stats['total_embeddings']}")
    print(f"Embedding dimension: {faiss_stats['dimension']}")
    print(f"Dog ID mappings: {faiss_stats['dog_ids_count']}")
    
    if dog_count != faiss_stats['total_embeddings']:
        print("\n⚠ WARNING: Database and FAISS index are out of sync!")
        print("  Run 'python manage_db.py --rebuild' to fix.")


def delete_dog(dog_id):
    """Delete a dog from database and FAISS index."""
    db = DogDatabase()
    faiss_store = FAISSStore()
    
    # Check if dog exists
    dog = db.get_dog(dog_id)
    if not dog:
        print(f"Error: Dog with ID {dog_id} not found.")
        return
    
    print(f"Deleting dog: {dog['name']} (ID: {dog_id})")
    
    # Delete from database
    db.delete_dog(dog_id)
    
    # Remove from FAISS (note: requires rebuild for full cleanup)
    faiss_store.remove_embedding(dog_id)
    
    print("✓ Dog deleted from database.")
    print("⚠ Note: Run 'python manage_db.py --rebuild' for full FAISS cleanup.")


def show_history(limit=10):
    """Show recent identification history."""
    db = DogDatabase()
    history = db.get_recent_identifications(limit)
    
    if not history:
        print("No identification history.")
        return
    
    print(f"\n{'ID':<5} {'Dog Name':<20} {'Similarity':<12} {'Identified At':<20}")
    print("-" * 70)
    
    for record in history:
        name = record.get('name', 'Unknown')
        similarity = f"{record.get('similarity_score', 0):.2%}"
        timestamp = record.get('identified_at', '')
        
        print(f"{record['id']:<5} {name:<20} {similarity:<12} {timestamp:<20}")


def main():
    parser = argparse.ArgumentParser(description='Dog Re-ID Database Management CLI')
    parser.add_argument('--list', action='store_true', help='List all dogs')
    parser.add_argument('--stats', action='store_true', help='Show database statistics')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild FAISS index from database')
    parser.add_argument('--delete', type=int, metavar='DOG_ID', help='Delete dog by ID')
    parser.add_argument('--history', type=int, nargs='?', const=10, metavar='LIMIT', help='Show identification history (default: 10)')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute commands
    if args.list:
        list_dogs()
    
    if args.stats:
        show_stats()
    
    if args.rebuild:
        rebuild_faiss()
    
    if args.delete:
        delete_dog(args.delete)
    
    if args.history is not None:
        show_history(args.history)


if __name__ == '__main__':
    main()
