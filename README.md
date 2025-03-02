# Skype Chat History Analyzer

This tool allows you to analyze your Skype chat history using AI. It loads your Skype chat database, processes the messages, and lets you ask questions about your chat history using natural language.

## Features

- 📚 Loads and processes Skype chat database (`.db` files)
- 🔍 Creates searchable vector embeddings of your messages
- 💾 Caches processed data for faster subsequent runs
- 🤖 Uses Mistral AI model through Ollama for analysis
- 📊 Provides detailed chat statistics and insights
- 💬 Supports both group chats and direct messages
- 🕒 Maintains conversation context and timeline

## Prerequisites

1. Python 3.8 or higher
2. [Ollama](https://ollama.ai/) installed and running
3. The Mistral model pulled in Ollama (`ollama pull mistral`)
4. Your Skype chat database file (`.db`)

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd skype-chatdb-to-llm
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running:
```bash
ollama serve
```

## Usage

1. Start the analyzer by running:
```bash
python main.py [path-to-your-skype-database.db]
```
If you don't specify a database path, the script will look for `.db` files in the current directory.

2. The tool will:
   - Load and analyze your chat database
   - Show statistics about your chat history
   - Create or load a vector store of your messages
   - Start an interactive prompt

3. You can then ask questions about your chat history, for example:
   - "What were the main topics discussed in [group name]?"
   - "Summarize my conversations with [person] about [topic]"
   - "What was discussed in May 2023?"
   - "Find all messages mentioning [specific topic]"

4. Type 'exit' or 'quit' to end the session.

## Example Questions

## ⚠️ Database Compatibility Notice

This tool works with Skype's `main.db` SQLite database file from older Skype versions. Specifically:

- ✅ **Known to work**: Skype 5.x (circa 2012)
- ❌ **Not compatible**: Skype 8.0 and newer

Starting with Skype 8.0, Microsoft [stopped storing chat history in local SQLite databases](https://answers.microsoft.com/en-us/skype/forum/all/does-anyone-here-know-when-skype-stopped-saving/b3ab41e3-3c05-4e5d-b49a-34aadf9ec23c). Newer versions store messages in the cloud instead.

### Finding Your Skype Database (For Old Skype Versions)

If you have an old Skype installation or backup, the `main.db` file location depends on your operating system:

- **Windows**:
  ```
  %AppData%\Skype\<your_skype_name>\main.db
  ```

- **macOS**:
  ```
  ~/Library/Application Support/Skype/<your_skype_name>/main.db
  ```

- **Linux**:
  ```
  ~/.Skype/<your_skype_name>/main.db
  ```

Make a copy of this file before using it with this tool.

## Technical Details

The tool:
1. Loads messages from your Skype SQLite database
2. Creates embeddings using HuggingFace's sentence-transformers
3. Stores these in a FAISS vector database for efficient searching
4. Uses Mistral through Ollama to understand and answer questions
5. Caches processed data to speed up subsequent runs

## Data Privacy

⚠️ **IMPORTANT: Chat Privacy Considerations**

Your Skype chat history likely contains personal, private, and potentially sensitive conversations. To protect your privacy:

- By default, this tool processes everything locally on your machine
- Only the local Ollama/Mistral setup is officially supported and recommended
- The vector store is saved locally in the `chat_vectorstore` directory
- No data is sent to external services (except to Ollama running locally)

### Privacy Best Practices
1. Always work with a copy of your database file
2. Keep your vector store in a secure location
3. Be cautious about who has access to your machine while processing
4. Consider the privacy implications before enabling any cloud features

## Alternative LLM Options

By default, this tool uses Mistral through Ollama for local analysis. This is the **recommended approach** for privacy and data security.

### Local Options (Recommended) 🔒

These options keep your chat data on your machine:

Change the model name in `ChatAnalyzer.__init__`:
```python
llm=OllamaLLM(model="mistral")  # Default
```
to any other model you have pulled in Ollama:
```python
llm=OllamaLLM(model="llama2")  # Need: ollama pull llama2
# or
llm=OllamaLLM(model="codellama")  # Need: ollama pull codellama
# or
llm=OllamaLLM(model="neural-chat")  # Need: ollama pull neural-chat
```

### Cloud Options (⚠️ USE AT YOUR OWN RISK)

**WARNING**: Using cloud-based LLMs means sending your chat history to external servers. This is **NOT RECOMMENDED** for private conversations. If you choose to proceed:
- Your chat data will be sent to third-party servers
- Data retention policies of these services may apply
- Privacy and confidentiality cannot be guaranteed
- Additional costs will apply
- You accept all associated privacy and security risks

If you understand and accept these risks, you can modify the LLM initialization:

#### OpenAI (GPT-3.5/4)
```bash
pip install openai
```
```python
from langchain_openai import ChatOpenAI

# In ChatAnalyzer.__init__:
llm=ChatOpenAI(
    model_name="gpt-4",  # or "gpt-3.5-turbo"
    openai_api_key="your-api-key"
) 
```

## Troubleshooting

1. If you see "No .db files found":
   - Make sure you're in the correct directory
   - Provide the full path to your Skype database file

2. If Ollama connection fails:
   - Ensure Ollama is running (`ollama serve`)
   - Verify the Mistral model is installed (`ollama pull mistral`)

3. For "Database has changed" messages:
   - Choose 'y' to rebuild the vector store if you want to include recent messages
   - Choose 'n' to use the cached version (faster but might miss recent messages)

## Cache Management

The tool creates a `chat_vectorstore` directory to cache processed data. To force a fresh analysis:
1. Delete the `chat_vectorstore` directory
2. Run the script again

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements!

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.