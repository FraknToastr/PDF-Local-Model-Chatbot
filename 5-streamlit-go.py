#!/usr/bin/env python3
"""
Wrapper script to launch Streamlit app for the chatbot.
Usage:
    python 5-streamlit-go.py
"""

import subprocess
import sys
import os

def main():
    python_exe = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), "4-chatbot.py")

    try:
        print("🚀 Starting Streamlit app...\n")
        subprocess.run([python_exe, "-m", "streamlit", "run", script_path])
    except KeyboardInterrupt:
        print("\n🛑 Streamlit app stopped by user.")
    except Exception as e:
        print(f"\n❌ Error while running Streamlit app: {e}")

if __name__ == "__main__":
    main()
