import os
import pandas as pd
from .database import load_messages
from .vectorstore import create_documents, build_vector_store, load_vector_store
from .chat_analyzer import ChatAnalyzer
from .utils import get_database_path, prompt_rebuild_vectorstore, parse_arguments
from .db_validation import validate_database
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    """Main entry point for the application"""
    print("\n🚀 Starting Chat Analysis System")
    print("=" * 50)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Get database path
    db_path, auto_detected = get_database_path(args.database)
    
    # Validate database structure
    if not validate_database(db_path):
        print("\n❌ Database validation failed. Please check your database file.")
        return
    
    cache_path = "chat_vectorstore"  # Path where the vector store will be cached
    
    # Check if we can use existing cache
    cache_exists = os.path.exists(cache_path) and os.path.exists(os.path.join(cache_path, "index.faiss"))
    
    # We need to load basic stats even when using cache
    df = load_messages(db_path)
    
    # Calculate stats for the system prompt
    chat_stats = {
        "first_date": df['date'].min(),
        "last_date": df['date'].max(),
        "num_participants": df['author'].nunique(),
        "num_conversations": df['conversation_name'].nunique()
    }
    
    # Check if database has changed
    from .vectorstore import check_db_changed
    db_changed = check_db_changed(db_path, cache_path) if cache_exists else True
    
    if cache_exists and not db_changed:
        print("\n📂 Loading cached vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vectorstore = load_vector_store(cache_path, embeddings)
        print("✓ Vector store loaded from cache")
    else:
        if cache_exists:
            if prompt_rebuild_vectorstore(cache_path):
                print("\n🔄 Database has changed, rebuilding vector store...")
                documents = create_documents(df)
                vectorstore = build_vector_store(documents, db_path, cache_path)
            else:
                print("\n📂 Using existing vector store...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                vectorstore = load_vector_store(cache_path, embeddings)
                print("✓ Vector store loaded from cache")
        else:
            print("\n🆕 No cache found, creating new vector store...")
            documents = create_documents(df)
            vectorstore = build_vector_store(documents, db_path, cache_path)
    
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