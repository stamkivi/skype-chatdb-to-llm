import time
import sys
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import pandas as pd
from .database import get_context_messages

def loading_spinner():
    """Return spinner characters for loading animation"""
    return ['|', '/', '-', '\\']

class ChatAnalyzer:
    """Main class for analyzing chat history"""
    def __init__(self, vectorstore, chat_stats, db_path):
        print("\n🤖 Initializing chat model...")
        
        # Create a system prompt that explains the data structure and expected behavior
        system_prompt = """You are analyzing a Skype chat history database. The documents you have access to are of two types:

1. Individual Messages:
   - Each message includes metadata (Author, Date, Chat name, Chat type) and the actual message content
   - Messages can be from either direct messages or group chats
   - The metadata is structured in [Message Info] sections
   - The actual message content follows in [Message Content] sections

2. Conversation Summaries:
   - These provide overview information about each chat
   - Include participant lists, message counts, and date ranges
   - Marked with [Conversation Summary] headers

When answering questions:
- For questions about specific people, look for their names in both Author fields and message content
- For questions about specific conversations, use the Chat/conversation_name fields and Conversation Summaries
- For timeline questions, pay attention to the Date fields
- When summarizing conversations, combine information from both the Conversation Summary and relevant messages
- If asked about a specific time period, filter by the dates in the metadata
- Always mention whether information comes from direct messages or group chats when relevant
- If you're not sure about something, say so rather than making assumptions
- Always include specific message quotes when relevant
- For each quote, include the author's name, date, and conversation context

The chat history spans from {first_date} to {last_date} and includes messages from {num_participants} participants across {num_conversations} conversations."""

        # Create a template that combines the system prompt with the user's question
        prompt_template = """
{system_prompt}

Question: {query}

Please analyze the relevant messages and provide a detailed answer, including specific message quotes where appropriate.
"""
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Use maximum marginal relevance
            search_kwargs={
                "k": 10,  # Fetch more documents
                "fetch_k": 20,  # Consider more candidates
                "lambda_mult": 0.7  # Balance relevance with diversity
            }
        )

        # Initialize the QA chain with the prompt template
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
        
        self.db_path = db_path
        self.conversation_history = []
        self.last_references = []
        self.vectorstore = vectorstore
        
        print("✓ Chat model ready")
        
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
        
        # Add source message references
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
        
        sys.stdout.write('\r' + ' ' * 20 + '\r')  # Clear the spinner
        return response
        
    def get_full_message(self, reference_index=0):
        """Get full content of a referenced message with context"""
        if not self.last_references:
            return "No recent references available"
        
        if reference_index >= len(self.last_references):
            return "Invalid reference index"
            
        ref = self.last_references[reference_index]
        
        # Get context messages from database
        context_df = get_context_messages(
            self.db_path, 
            ref['conversation'], 
            ref['date'],
            window=3600  # 1 hour window
        )
        
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