#!/usr/bin/env python3
"""
Test script to verify the skype_analyzer package imports correctly.
"""

import sys

def test_imports():
    """Test importing all modules from the package."""
    try:
        from skype_analyzer import database, vectorstore, chat_analyzer, utils, cli
        print("✅ All modules imported successfully!")
        
        # Print version
        from skype_analyzer import __version__
        print(f"📦 Package version: {__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Testing skype_analyzer package imports...")
    success = test_imports()
    sys.exit(0 if success else 1) 