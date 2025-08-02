#!/usr/bin/env python3
"""
Simple launcher for the Cricket Analytics UI
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    print("🏏 Starting Cricket Analytics UI...")
    
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "cricket_ui.py",
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]) 