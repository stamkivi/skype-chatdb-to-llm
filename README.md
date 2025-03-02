# Skype Chat History Analyzer

This tool allows you to analyze your Skype chat history using AI. It loads your Skype chat database, processes the messages, and lets you ask questions about your chat history using natural language.

## Features

- Load and analyze Skype chat databases (main.db)
- Generate statistics about your chat history
- Create a vector store for semantic search of your messages
- Ask questions about your chat history using natural language
- View full message context with surrounding messages

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/skype-chatdb-to-llm.git
cd skype-chatdb-to-llm
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have Ollama installed and running with the Mistral model:
```bash
ollama pull mistral
```

## Usage

1. Find your Skype chat database file:
   - Windows: `%AppData%\Skype\<your-username>\main.db`
   - macOS: `~/Library/Application Support/Skype/<your-username>/main.db`
   - Linux: `~/.Skype/<your-username>/main.db`

2. Copy the database file to a location of your choice.

3. Run the analyzer:
```bash
python main.py path/to/your/main.db
```

If you don't specify a database path, the tool will look for `.db` files in the current directory.

4. Ask questions about your chat history:
```
❓ Ask a question about your chats: Who did I chat with the most?
```

5. Special commands:
   - `full message`: Show the complete text of the last referenced message with context
   - `full message 2`: Show the second last referenced message with context
   - `exit` or `quit`: End the session

## Data Privacy

This tool processes your chat data locally on your machine. No data is sent to external servers except when using the Ollama API, which also runs locally. Your chat history remains private.

## Compatibility

This tool is designed to work with Skype's SQLite database format. It has been tested with main.db for 5.x versions of Skype (circa 2012 backups), but database schema compatibility was definitely broken around Skype 8.0.

## Contributing

Feel free to open issues or submit pull requests if you have suggestions for improvements!

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.