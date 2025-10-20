"""
Pytest configuration file.

This file helps pytest find modules in the project.
"""
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))