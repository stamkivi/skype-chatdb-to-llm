import sqlite3
import pandas as pd
from tabulate import tabulate
from datetime import datetime

def load_messages(db_path):
    """Load and analyze messages from Skype database"""
    print("\n📚 Connecting to database...")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    print("✓ Database connection established")
    
    print("\n📥 Loading messages and analyzing data...")
    
    # Get first and last messages
    timeline = pd.read_sql_query("""
        SELECT 
            datetime(m.timestamp, 'unixepoch') AS date,
            m.author,
            m.body_xml AS message,
            c.displayname as conversation_name
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL 
        AND m.timestamp IN (
            SELECT MIN(timestamp) FROM Messages WHERE body_xml IS NOT NULL
            UNION
            SELECT MAX(timestamp) FROM Messages WHERE body_xml IS NOT NULL
        )
        ORDER BY m.timestamp ASC;
    """, conn)
    
    # Get a random message
    random_msg = pd.read_sql_query("""
        SELECT 
            datetime(m.timestamp, 'unixepoch') AS date,
            m.author,
            m.body_xml AS message,
            c.displayname as conversation_name
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 1;
    """, conn)
    
    # Get conversation stats
    conv_stats = pd.read_sql_query("""
        SELECT 
            CASE 
                WHEN c.displayname IS NULL THEN 'Direct Messages'
                ELSE 'Group Chats'
            END as chat_type,
            COUNT(DISTINCT m.convo_id) as conversation_count,
            COUNT(*) as message_count
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        GROUP BY chat_type
        ORDER BY chat_type;
    """, conn)
    
    # Get top 10 authors
    top_authors = pd.read_sql_query("""
        SELECT 
            author,
            COUNT(*) as message_count,
            ROUND(COUNT(*) * 100.0 / (
                SELECT COUNT(*) 
                FROM Messages 
                WHERE body_xml IS NOT NULL
            ), 1) as percentage
        FROM Messages
        WHERE body_xml IS NOT NULL
        GROUP BY author
        ORDER BY message_count DESC
        LIMIT 10;
    """, conn)
    
    # Load all messages for processing
    df = pd.read_sql_query("""
        SELECT 
            datetime(m.timestamp, 'unixepoch') AS date,
            m.author,
            m.body_xml AS message,
            c.displayname as conversation_name,
            CASE 
                WHEN c.displayname IS NULL THEN 'Direct Message'
                ELSE 'Group Chat'
            END as chat_type
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL 
        ORDER BY m.timestamp ASC;
    """, conn)
    
    conn.close()
    
    # Print detailed statistics
    print("\n📊 Chat Database Summary:")
    print("=" * 50)
    print(f"Total Messages: {len(df):,}")
    
    print("\n📅 Timeline:")
    print(f"First Message: {timeline.iloc[0]['date']} by {timeline.iloc[0]['author']}")
    print(f"Last Message:  {timeline.iloc[1]['date']} by {timeline.iloc[1]['author']}")
    
    print("\n🎲 Random Message Sample:")
    print(f"Date: {random_msg.iloc[0]['date']}")
    print(f"Author: {random_msg.iloc[0]['author']}")
    print(f"Chat: {random_msg.iloc[0]['conversation_name'] if pd.notna(random_msg.iloc[0]['conversation_name']) else 'Direct Message'}")
    msg_preview = random_msg.iloc[0]['message']
    print(f"Message: {msg_preview[:100]}..." if len(msg_preview) > 100 else msg_preview)
    
    print("\n👥 Conversation Statistics:")
    for _, row in conv_stats.iterrows():
        print(f"{row['chat_type']}: {row['conversation_count']:,} conversations ({row['message_count']:,} messages)")
    
    print("\n👤 Top 10 Most Active Participants:")
    print(tabulate(top_authors, headers='keys', tablefmt='pretty', floatfmt=".1f"))
    
    print("\n✓ Database loading complete")
    return df

def get_context_messages(db_path, conversation, timestamp_str, window=3600):
    """Get messages before and after a specific timestamp in a conversation"""
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    
    try:
        # Convert date string to timestamp for comparison
        date_obj = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        timestamp = int(date_obj.timestamp())
        
        # Query for messages before and after in the same conversation
        context_query = """
        SELECT 
            datetime(m.timestamp, 'unixepoch') AS date,
            m.author,
            m.body_xml AS message,
            c.displayname as conversation_name,
            CASE 
                WHEN c.displayname IS NULL THEN 'Direct Message'
                ELSE 'Group Chat'
            END as chat_type
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE 
            (c.displayname = ? OR (c.displayname IS NULL AND ? IS NULL))
            AND m.timestamp BETWEEN ? - ? AND ? + ?
        ORDER BY m.timestamp
        LIMIT 5;
        """
        
        # Use conversation name as filter
        params = (conversation, conversation, timestamp, window, timestamp, window)
        context_df = pd.read_sql_query(context_query, conn, params=params)
    except Exception as e:
        # Fallback if date parsing fails
        context_df = pd.DataFrame()
        print(f"Error getting context: {str(e)}")
    finally:
        conn.close()
    
    return context_df 