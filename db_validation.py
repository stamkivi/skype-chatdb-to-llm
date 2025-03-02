import sqlite3
import pandas as pd
import argparse
import glob
import sys
import os
import random
from tabulate import tabulate

def get_database_path(provided_path=None):
    """
    Determine the database path from command line arg or by scanning current directory.
    Returns tuple of (path, was_autodetected).
    """
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
        print("python db_validation.py path/to/your/database.db")
        sys.exit(1)
    
    if len(db_files) > 1:
        print("\n❌ Multiple database files found in current directory:")
        for db_file in db_files:
            print(f"  - {db_file}")
        print("\nPlease specify which database to use:")
        print("python db_validation.py path/to/your/database.db")
        sys.exit(1)
    
    print(f"\n📌 Auto-detected database: {db_files[0]}")
    return db_files[0], True

def analyze_database(db_path):
    print("\n🔍 Analyzing chat database...")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')

    # 1. Total message count
    message_count = pd.read_sql_query("""
        SELECT COUNT(*) as count 
        FROM Messages 
        WHERE body_xml IS NOT NULL
    """, conn).iloc[0]['count']
    
    print(f"\n📊 Total messages: {message_count:,}")

    # 2. Date range
    date_range = pd.read_sql_query("""
        SELECT 
            datetime(MIN(timestamp), 'unixepoch') as earliest,
            datetime(MAX(timestamp), 'unixepoch') as latest
        FROM Messages
        WHERE body_xml IS NOT NULL
    """, conn).iloc[0]
    
    print(f"📅 Date range: {date_range['earliest']} to {date_range['latest']}")

    # 3. Top 10 participants
    print("\n👥 Top 10 participants by message count:")
    top_participants = pd.read_sql_query("""
        SELECT 
            author,
            COUNT(*) as message_count,
            COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Messages WHERE body_xml IS NOT NULL) as percentage
        FROM Messages
        WHERE body_xml IS NOT NULL
        GROUP BY author
        ORDER BY message_count DESC
        LIMIT 10
    """, conn)
    
    print(tabulate(top_participants, headers='keys', tablefmt='pretty', floatfmt=".1f"))

    # 4. Random 5 messages with conversation names
    print("\n📝 Random 5 messages from the chat history:")
    random_messages = pd.read_sql_query("""
        SELECT 
            datetime(m.timestamp, 'unixepoch') as timestamp,
            m.author,
            m.body_xml as message,
            c.displayname as conversation_name,
            CASE 
                WHEN c.displayname IS NULL THEN 'Direct Message'
                ELSE 'Group Chat'
            END as chat_type
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5
    """, conn)

    for idx, msg in random_messages.iterrows():
        print("\n" + "─" * 80)
        print(f"📅 Timestamp: {msg['timestamp']}")
        print(f"👥 Chat: {msg['conversation_name'] if msg['conversation_name'] else 'Direct Message'}")
        print(f"👤 Author: {msg['author']}")
        print(f"💬 Message:\n{msg['message']}")

    conn.close()

if __name__ == "__main__":
    print("\n🔍 Chat Database Validation Tool")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chat Database Validation Tool')
    parser.add_argument('database', nargs='?', help='Path to the database file')
    args = parser.parse_args()
    
    # Get database path
    db_path, auto_detected = get_database_path(args.database)
    
    # Analyze the database
    analyze_database(db_path) 