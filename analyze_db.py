#!/usr/bin/env python3
"""
Skype Database Analysis Tool

This script analyzes a Skype database file and prints statistics.
"""

from skype_analyzer.database import load_messages
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_db.py <path_to_database>")
        sys.exit(1)
        
    db_path = sys.argv[1]
    load_messages(db_path)  # This function prints statistics 