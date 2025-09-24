#!/usr/bin/env python3
"""
Streamlit Launcher Script
Runs the Council Meeting Chatbot UI (4-chatbot.py) without needing to type the long command.
"""

import subprocess
import sys
import os

def main():
    python_exe = sys.executable  # Use current venv’s Python
    script_path = os.path.join(os.path.dirname(__file__), "4-chatbot.py")

    print("🚀 Launching Council Meeting Chatbot...")
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit app: {e}")

if __name__ == "__main__":
    main()
