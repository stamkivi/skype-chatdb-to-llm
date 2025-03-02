import os
import time
import hashlib
import json
from tqdm import tqdm
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def create_documents(df):
    """Convert dataframe to document objects"""
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
    
    print(f"✓ Created {len(documents)} documents")
    return documents

def calculate_file_hash(file_path):
    """Calculate hash of a file"""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_db_hash(db_path, cache_path):
    """Save database hash to cache"""
    db_hash = calculate_file_hash(db_path)
    hash_path = os.path.join(cache_path, "db_hash.json")
    with open(hash_path, 'w') as f:
        json.dump({"hash": db_hash, "path": db_path}, f)

def check_db_changed(db_path, cache_path):
    """Check if database has changed since last run"""
    hash_path = os.path.join(cache_path, "db_hash.json")
    if not os.path.exists(hash_path):
        return True
    
    with open(hash_path, 'r') as f:
        data = json.load(f)
    
    current_hash = calculate_file_hash(db_path)
    return current_hash != data["hash"]

def build_vector_store(documents, db_path, cache_path="chat_vectorstore"):
    """Build or load vector store"""
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

def load_vector_store(cache_path, embeddings=None):
    """Load vector store from cache"""
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    return FAISS.load_local(
        cache_path, 
        embeddings,
        allow_dangerous_deserialization=True
    ) 