"""
Database validation module for Skype Analyzer.

This module provides functions to validate the structure of a Skype database.
"""

import sqlite3
import pandas as pd
from tabulate import tabulate

def validate_database(db_path):
    """
    Validate the structure of a Skype database.
    
    Args:
        db_path (str): Path to the Skype database file.
        
    Returns:
        bool: True if the database is valid, False otherwise.
    """
    print("\n🔍 Validating database structure...")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        
        # Check if required tables exist
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table';", 
            conn
        )
        
        required_tables = ['Messages', 'Conversations']
        missing_tables = [table for table in required_tables if table not in tables['name'].values]
        
        if missing_tables:
            print(f"❌ Missing required tables: {', '.join(missing_tables)}")
            conn.close()
            return False
            
        # Check Messages table structure
        messages_columns = pd.read_sql_query(
            "PRAGMA table_info(Messages);", 
            conn
        )
        
        required_columns = ['id', 'convo_id', 'author', 'timestamp', 'body_xml']
        missing_columns = [col for col in required_columns if col not in messages_columns['name'].values]
        
        if missing_columns:
            print(f"❌ Messages table is missing required columns: {', '.join(missing_columns)}")
            conn.close()
            return False
            
        # Check Conversations table structure
        conv_columns = pd.read_sql_query(
            "PRAGMA table_info(Conversations);", 
            conn
        )
        
        required_columns = ['id', 'displayname', 'type']
        missing_columns = [col for col in required_columns if col not in conv_columns['name'].values]
        
        if missing_columns:
            print(f"❌ Conversations table is missing required columns: {', '.join(missing_columns)}")
            conn.close()
            return False
            
        # Check if there are messages in the database
        message_count = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM Messages WHERE body_xml IS NOT NULL;", 
            conn
        ).iloc[0]['count']
        
        if message_count == 0:
            print("❌ No messages found in the database.")
            conn.close()
            return False
            
        # Print database statistics
        print("\n✅ Database structure is valid!")
        print(f"📊 Found {message_count:,} messages")
        
        # Detailed table structure information is now hidden from end users
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error validating database: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python db_validation.py <path_to_database>")
        sys.exit(1)
        
    db_path = sys.argv[1]
    is_valid = validate_database(db_path)
    
    sys.exit(0 if is_valid else 1) 