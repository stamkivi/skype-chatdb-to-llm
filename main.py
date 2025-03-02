import sqlite3
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from tqdm import tqdm
import time
import sys
import os
import hashlib
import json
import argparse
import glob
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

def loading_spinner():
    spinner = ['|', '/', '-', '\\']
    return spinner

def load_messages(db_path):
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
    
    # Investigate conversation types in more detail
    conv_analysis = pd.read_sql_query("""
        SELECT 
            c.displayname,
            c.type as conv_type,
            COUNT(DISTINCT m.convo_id) as conversation_count,
            COUNT(*) as message_count,
            GROUP_CONCAT(DISTINCT m.author) as participants
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        GROUP BY c.displayname, c.type
        ORDER BY message_count DESC
        LIMIT 10;
    """, conn)
    
    # Get conversation type distribution
    conv_type_stats = pd.read_sql_query("""
        SELECT 
            c.type as conv_type,
            COUNT(DISTINCT m.convo_id) as conversation_count,
            COUNT(*) as message_count,
            COUNT(DISTINCT m.author) as unique_authors
        FROM Messages m
        LEFT JOIN Conversations c ON m.convo_id = c.id
        WHERE m.body_xml IS NOT NULL
        GROUP BY c.type;
    """, conn)
    
    # Check table structure
    table_info = pd.read_sql_query("""
        SELECT sql FROM sqlite_master 
        WHERE type='table' AND name IN ('Conversations', 'Messages');
    """, conn)
    
    print("\n🔍 Database Structure Analysis:")
    print("=" * 50)
    print("\nTable Definitions:")
    for _, row in table_info.iterrows():
        print(f"\n{row['sql']}")
    
    print("\n📊 Conversation Types Distribution:")
    print(tabulate(conv_type_stats, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\n🔎 Top 10 Conversations by Message Count:")
    print(tabulate(conv_analysis, headers='keys', tablefmt='pretty', showindex=False))
    
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
    
    print("\n✓ Database loading complete")
    return df

def create_documents(df):
    print("\n🔄 Converting messages to documents...")
    documents = []
    
    # First, create individual message documents
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing messages"):
        # Create a rich text representation that includes metadata in the content
        content = f"""
[Message Info]
Author: {row["author"]}
Date: {row["date"]}
Chat: {row["conversation_name"] if pd.notna(row["conversation_name"]) else "Direct Message"}
Type: {row["chat_type"]}

[Message Content]
{row["message"]}
"""
        # Keep the metadata for reference, but include everything in content too
        metadata = {
            "author": row["author"],
            "date": row["date"],
            "chat_type": row["chat_type"],
            "conversation": row["conversation_name"] if pd.notna(row["conversation_name"]) else "Direct Message"
        }
        documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    # Then, create conversation summary documents
    print("\nCreating conversation summaries...")
    for conv_name, conv_df in df.groupby('conversation_name'):
        chat_type = "Group Chat" if pd.notna(conv_name) else "Direct Message"
        conv_name = conv_name if pd.notna(conv_name) else "Direct Message"
        
        summary = f"""
[Conversation Summary]
Name: {conv_name}
Type: {chat_type}
Participants: {', '.join(conv_df['author'].unique())}
Message Count: {len(conv_df)}
Date Range: {conv_df['date'].min()} to {conv_df['date'].max()}

This is a {chat_type.lower()} between {', '.join(conv_df['author'].unique())}.
"""
        documents.append(Document(
            page_content=summary,
            metadata={
                "type": "conversation_summary",
                "conversation": conv_name,
                "chat_type": chat_type
            }
        ))
    
    return documents

def calculate_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def save_db_hash(db_path, cache_path):
    """Save database hash to a metadata file"""
    hash_value = calculate_file_hash(db_path)
    metadata = {"db_hash": hash_value}
    metadata_path = os.path.join(cache_path, "metadata.json")
    
    os.makedirs(cache_path, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    return hash_value

def check_db_changed(db_path, cache_path):
    """Check if database has changed since last vectorstore creation"""
    metadata_path = os.path.join(cache_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        return True
    
    current_hash = calculate_file_hash(db_path)
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    return current_hash != metadata.get("db_hash")

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
    """Ask user whether to rebuild the vector store"""
    while True:
        response = input("\n⚠️  Database has changed since last vector store creation.\n"
                        "Would you like to rebuild the vector store? This may take several minutes. (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("\n⚠️  Warning: Using existing vector store with outdated data!")
            return False
        print("Please answer 'y' or 'n'")

def build_vector_store(documents, db_path, cache_path="chat_vectorstore"):
    print("\n🔍 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✓ Embedding model loaded")
    
    print("\n📊 Creating vector store...")
    print("This may take several minutes depending on the number of messages...")
    start_time = time.time()
    
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Save the vectorstore and database hash
    print("\n💾 Saving vector store to disk...")
    vectorstore.save_local(cache_path)
    save_db_hash(db_path, cache_path)
    print("✓ Vector store cached")
    
    duration = time.time() - start_time
    print(f"✓ Vector store created in {duration:.1f} seconds")
    return vectorstore

class ChatAnalyzer:
    def __init__(self, vectorstore, chat_stats, db_path):
        print("\n🤖 Initializing chat model...")
        
        # Enhanced system prompt
        system_prompt = """You are analyzing a Skype chat history database. When answering questions:

1. Always include specific message quotes when relevant
2. For each quote, include:
   - The author's name
   - The approximate date (if known)
   - The conversation context
3. Use the following format for quotes:
   > [Author] at [timestamp] in [Conversation]: "[Exact message text]"
4. When summarizing, include key quotes that support your summary
5. If you're referring to multiple messages, list them clearly
6. Always maintain the original wording of messages
7. If you're unsure about a message's context, say so

The chat history spans from {first_date} to {last_date} and includes messages from {num_participants} participants across {num_conversations} conversations."""

        # Enhanced query template
        prompt_template = """
{system_prompt}

Question: {query}

Please analyze the relevant messages and provide a detailed answer, including specific message quotes where appropriate.
"""

        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k": 20,
                "lambda_mult": 0.7
            }
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OllamaLLM(
                model="mistral",
                temperature=0.7,
                system=system_prompt.format(**chat_stats)
            ),
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True
        )
        print("✓ Chat model ready")

        self.conversation_history = []
        self.last_references = []
        self.vectorstore = vectorstore
        self.db_path = db_path  # Store database path for context queries

    def query(self, question):
        print("\n🔄 Processing your question...")
        spinner = loading_spinner()
        i = 0
        response = ""
        
        # Build context from conversation history
        context = "\n".join(
            f"Q: {q}\nA: {a}" 
            for q, a in self.conversation_history[-3:]  # Keep last 3 Q/A pairs
        )
        
        # Enhanced prompt with conversation history
        full_prompt = f"""
{context}

Current Question: {question}

Please consider the conversation history above when answering.
"""
        
        def spinning_cursor():
            sys.stdout.write('\r' + f"Thinking {spinner[i%4]}")
            sys.stdout.flush()
        
        while not response:
            spinning_cursor()
            i += 1
            result = self.qa_chain.invoke({"query": full_prompt})
            response = result['result'] if isinstance(result, dict) else result
            time.sleep(0.1)
        
        # Update conversation history
        self.conversation_history.append((question, response))
        
        # Add source message references - FIXED VERSION
        self.last_references = []  # Clear previous references
        
        # Check if source_documents exists in the result
        if isinstance(result, dict) and 'source_documents' in result:
            source_docs = result['source_documents']
            response += "\n\nReferences:\n"
            
            for idx, doc in enumerate(source_docs[:3]):  # Show top 3 references
                # Extract metadata safely
                author = doc.metadata.get('author', 'Unknown')
                date = doc.metadata.get('date', 'Unknown date')
                conversation = doc.metadata.get('conversation', 'Unknown chat')
                
                # Add to response
                response += f"- {author} in {conversation} ({date})\n"
                
                # Store for later retrieval
                self.last_references.append({
                    'author': author,
                    'date': date,
                    'conversation': conversation,
                    'content': doc.page_content
                })
                
            # Debug info
            print(f"DEBUG: Stored {len(self.last_references)} references")
        else:
            print("DEBUG: No source_documents found in result")
            # Print result structure to debug
            if isinstance(result, dict):
                print(f"DEBUG: Result keys: {list(result.keys())}")
        
        sys.stdout.write('\r' + ' ' * 20 + '\r')  # Clear the spinner
        return response

    def get_full_message(self, reference_index=0):
        """Get full content of a referenced message with context"""
        if not self.last_references:
            return "No recent references available"
        
        if reference_index >= len(self.last_references):
            return "Invalid reference index"
            
        ref = self.last_references[reference_index]
        
        # Connect to the database to get context messages
        conn = sqlite3.connect(self.db_path)
        conn.text_factory = lambda b: b.decode(errors='ignore')
        
        # Parse the date to get a timestamp
        try:
            # Convert date string to timestamp for comparison
            date_obj = datetime.strptime(ref['date'], '%Y-%m-%d %H:%M:%S')
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
                c.displayname = ? 
                AND m.timestamp BETWEEN ? - 3600 AND ? + 3600  -- 1 hour before and after
            ORDER BY m.timestamp
            LIMIT 5;  -- Get a few messages for context
            """
            
            # Use conversation name as filter
            params = (ref['conversation'], timestamp, timestamp)
            context_df = pd.read_sql_query(context_query, conn, params=params)
        except Exception as e:
            # Fallback if date parsing fails
            context_df = pd.DataFrame()
            print(f"Error getting context: {str(e)}")
        
        conn.close()
        
        if context_df.empty:
            return f"""
Full message:

> {ref['author']} at {ref['date']} in {ref['conversation']}:
{ref['content']}

(No context messages found)
"""
        
        # Build the response
        response = "\nMessage context:\n"
        target_content = ref['content']
        
        for _, msg in context_df.iterrows():
            is_target = target_content in msg['message']
            
            response += f"""
> {msg['author']} at {msg['date']} in {msg['conversation_name'] or 'Direct Message'}:
{msg['message']}
"""
            if is_target:
                response += "  <-- This is the message you asked about\n"
        
        return response

if __name__ == "__main__":
    print("\n🚀 Starting Chat Analysis System")
    print("=" * 50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Chat Analysis System')
    parser.add_argument('database', nargs='?', help='Path to the database file')
    args = parser.parse_args()
    
    # Get database path
    db_path, auto_detected = get_database_path(args.database)
    cache_path = "chat_vectorstore"  # Path where the vector store will be cached
    
    # Check if we can use existing cache
    cache_exists = os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "metadata.json"))
    db_changed = check_db_changed(db_path, cache_path) if cache_exists else True
    
    if cache_exists and not db_changed:
        print("\n📂 Loading cached vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = FAISS.load_local(
            cache_path, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("✓ Vector store loaded from cache")
        
        # We need to load basic stats even when using cache
        df = load_messages(db_path)
    else:
        if cache_exists:
            print("\n🔄 Database has changed, rebuilding vector store...")
        else:
            print("\n🆕 No cache found, creating new vector store...")
        
        df = load_messages(db_path)
        documents = create_documents(df)
        vectorstore = build_vector_store(documents, db_path, cache_path)
    
    # Calculate stats for the system prompt
    chat_stats = {
        "first_date": df['date'].min(),
        "last_date": df['date'].max(),
        "num_participants": df['author'].nunique(),
        "num_conversations": df['conversation_name'].nunique()
    }
    
    analyzer = ChatAnalyzer(vectorstore, chat_stats, db_path)
    
    print("\n🤖 Chat system ready! You can now ask questions about your chat history.")
    print("Commands:")
    print("- Type 'full message' to see the complete text of the last referenced message")
    print("- Type 'full message 2' to see the second last referenced message")
    print("- Type 'exit' or 'quit' to end the session")
    print("=" * 50)
    
    while True:
        question = input("\n❓ Ask a question about your chats: ").strip().lower()
        
        if question in ["exit", "quit"]:
            print("\n👋 Goodbye!")
            break
            
        # Handle full message requests
        if question.startswith("full message"):
            try:
                # Extract message number if specified
                if len(question.split()) > 2:
                    idx = int(question.split()[2]) - 1
                else:
                    idx = 0
                print(analyzer.get_full_message(idx))
                continue
            except Exception as e:
                print(f"Error showing message: {str(e)}")
                continue
                
        response = analyzer.query(question)
        print("\n💡 Response:")
        print(response)
        print("\n" + "-" * 50)
