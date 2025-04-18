#!/usr/bin/env python3
"""
Skype Conversation Browser and Exporter

This script allows browsing Skype conversations and exporting them to markdown files.
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime
from tabulate import tabulate

def get_conversations(db_path):
    """Get list of conversations with statistics"""
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    
    # Get conversation statistics
    conversations = pd.read_sql_query("""
        SELECT 
            c.displayname as title,
            COUNT(DISTINCT m.author) as member_count,
            COUNT(*) as message_count,
            date(MIN(datetime(m.timestamp, 'unixepoch'))) as first_message,
            date(MAX(datetime(m.timestamp, 'unixepoch'))) as last_message
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        GROUP BY c.displayname
        HAVING COUNT(*) > 1
        ORDER BY message_count DESC, member_count DESC;
    """, conn)
    
    conn.close()
    return conversations

def export_conversation(db_path, conversation_title, output_dir):
    """Export a conversation to markdown"""
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    
    # Get all messages from the conversation
    messages = pd.read_sql_query("""
        SELECT 
            datetime(m.timestamp, 'unixepoch') as date,
            m.author,
            m.body_xml as message
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE c.displayname = ? AND m.body_xml IS NOT NULL
        ORDER BY m.timestamp;
    """, conn, params=(conversation_title,))
    
    conn.close()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from conversation title
    filename = f"{conversation_title.replace('/', '_').replace(':', '_')}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Write to markdown file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {conversation_title}\n\n")
        for _, msg in messages.iterrows():
            f.write(f"**{msg['author']}** ({msg['date']}):\n")
            f.write(f"{msg['message']}\n\n")
    
    return filepath

def display_conversations(conversations, page=0, page_size=20):
    """Display a page of conversations"""
    start_idx = page * page_size
    end_idx = start_idx + page_size
    page_conversations = conversations.iloc[start_idx:end_idx]
    
    print(f"\n📊 Available Conversations (Page {page + 1} of {(len(conversations) + page_size - 1) // page_size}):")
    print(tabulate(
        page_conversations,
        headers=['#', 'Title', 'Members', 'Messages', 'First Message', 'Last Message'],
        showindex=range(start_idx, min(end_idx, len(conversations))),
        tablefmt='pretty'
    ))

def main():
    print("\n🔍 Skype Conversation Browser")
    print("=" * 50)
    
    # Look for .db files in current directory
    db_files = [f for f in os.listdir('.') if f.endswith('.db')]
    
    if len(db_files) == 1:
        db_path = db_files[0]
        print(f"\nFound database file: {db_path}")
    else:
        # Get database path from user
        db_path = input("\nEnter path to Skype database file: ").strip()
    
    if not os.path.exists(db_path):
        print("❌ Database file not found!")
        return
    
    # Get conversations
    conversations = get_conversations(db_path)
    page = 0
    page_size = 20
    
    while True:
        display_conversations(conversations, page, page_size)
        
        print("\nCommands:")
        print("  [number] - Export conversation")
        print("  n - Next page")
        print("  p - Previous page")
        print("  q - Quit")
        
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'n':
            if (page + 1) * page_size < len(conversations):
                page += 1
            else:
                print("❌ Already on the last page!")
        elif choice == 'p':
            if page > 0:
                page -= 1
            else:
                print("❌ Already on the first page!")
        else:
            try:
                idx = int(choice)
                if idx < 0 or idx >= len(conversations):
                    print("❌ Invalid selection!")
                    continue
                    
                # Export selected conversation
                conversation = conversations.iloc[idx]
                output_dir = "chat_export"
                filepath = export_conversation(db_path, conversation['title'], output_dir)
                print(f"\n✅ Exported conversation to: {filepath}")
                
            except ValueError:
                print("❌ Please enter a valid number or command!")
                continue

if __name__ == "__main__":
    main() 