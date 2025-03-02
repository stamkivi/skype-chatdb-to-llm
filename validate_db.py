#!/usr/bin/env python3
"""
Skype Database Validation Tool

This script validates the structure of a Skype database file.
"""

from skype_analyzer.db_validation import validate_database
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_db.py <path_to_database>")
        sys.exit(1)
        
    db_path = sys.argv[1]
    is_valid = validate_database(db_path)
    
    sys.exit(0 if is_valid else 1) 