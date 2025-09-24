#!/usr/bin/env python3
"""
Council Meeting Chatbot Launcher with Reset Option
--------------------------------------------------
This script can reset the data pipeline and then launch the Streamlit chatbot UI.
- Step 1: Ask user if they want to reset.
- Step 2: If yes → run Script 2 (chunking) and Script 3 (build LanceDB).
- Step 3: Launch Script 4 (Streamlit chatbot).
"""

import subprocess
import sys
import os
import shutil

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
LANCEDB_DIR = os.path.join(DATA_DIR, "lancedb_data")

SCRIPT2 = os.path.join(BASE_DIR, "2-hybrid-chunking-multiple-PDFs.py")
SCRIPT3 = os.path.join(BASE_DIR, "3-build-lancedb.py")
SCRIPT4 = os.path.join(BASE_DIR, "4-chatbot.py")

def run_script(script_path):
    """Run a Python script with current interpreter."""
    python_exe = sys.executable
    print(f"\n🚀 Running {os.path.basename(script_path)}...")
    try:
        subprocess.run([python_exe, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ {os.path.basename(script_path)} failed with code {e.returncode}")
        sys.exit(1)

def reset_data():
    """Delete chunks.pkl and LanceDB directory."""
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
        print(f"🗑️ Deleted {CHUNKS_FILE}")
    if os.path.exists(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR, ignore_errors=True)
        print(f"🗑️ Deleted {LANCEDB_DIR}")

def main():
    choice = input("🔄 Do you want to reset data and rebuild? (y/n): ").strip().lower()
    if choice == "y":
        reset_data()
        run_script(SCRIPT2)
        run_script(SCRIPT3)

    # Always launch chatbot
    python_exe = sys.executable
    print("\n🚀 Launching Council Meeting Chatbot...")
    try:
        subprocess.run([python_exe, "-m", "streamlit", "run", SCRIPT4], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to launch Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
