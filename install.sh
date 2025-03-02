#!/bin/bash
# Installation script for Skype Analyzer

echo "🚀 Installing Skype Analyzer..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "⚠️ Ollama is not installed. You will need to install it to use this tool."
    echo "Visit https://ollama.ai/ for installation instructions."
fi

# Create a virtual environment
echo "📦 Creating a virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install the package
echo "📥 Installing dependencies..."
pip3 install -r requirements.txt

echo "🔧 Installing Skype Analyzer..."
pip3 install -e .

# Pull the Mistral model if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "🤖 Pulling the Mistral model for Ollama..."
    ollama pull mistral
fi

echo "✅ Installation complete!"
echo ""
echo "To use Skype Analyzer:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the analyzer: skype-analyzer"
echo "   or: python3 main.py"
echo ""
echo "Enjoy analyzing your Skype chats! 🎉" 