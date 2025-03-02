import os
import glob
import sys
import argparse

def get_database_path(provided_path=None):
    """Determine database path from command line or auto-detect"""
    if provided_path:
        if not os.path.exists(provided_path):
            print(f"\n❌ Error: Database file '{provided_path}' not found!")
            sys.exit(1)
        return provided_path, False

    # Look for .db files in current directory
    db_files = glob.glob("*.db")
    
    if not db_files:
        print("\n❌ Error: No .db files found in current directory!")
        print("Please provide the database path as a command line argument:")
        print("python main.py path/to/your/database.db")
        sys.exit(1)
    
    if len(db_files) > 1:
        print("\n❌ Multiple database files found in current directory:")
        for db_file in db_files:
            print(f"  - {db_file}")
        print("\nPlease specify which database to use:")
        print("python main.py path/to/your/database.db")
        sys.exit(1)
    
    print(f"\n📌 Auto-detected database: {db_files[0]}")
    return db_files[0], True

def prompt_rebuild_vectorstore(cache_path):
    """Ask user whether to rebuild vector store"""
    while True:
        response = input("\n⚠️ Database has changed. Rebuild vector store? (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Skype Chat History Analyzer')
    parser.add_argument('database', nargs='?', help='Path to the database file')
    return parser.parse_args() 