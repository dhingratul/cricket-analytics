#!/usr/bin/env python3
"""
Launcher for Cricket Analytics Streamlit UI
"""
import subprocess
import sys
import os
from pathlib import Path

def run_ui():
    """Launch the cricket analytics UI"""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    ui_file = script_dir / "cricket_ui.py"
    
    # Make sure we're in the project root (parent of src)
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Launch Streamlit with the correct path
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        str(ui_file),
        "--server.port", "8501",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ])

if __name__ == "__main__":
    run_ui() 