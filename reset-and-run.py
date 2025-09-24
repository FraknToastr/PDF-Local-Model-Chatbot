# reset-and-run.py
"""
Reset pipeline, then run scripts 2, 3, and 5 with user confirmation between steps.
"""

import os
import shutil
import subprocess
import sys

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")

SCRIPTS = [
    ("2-hybrid-chunking-multiple-PDFs.py", "Build chunks from PDFs"),
    ("3-build-lancedb.py", "Build LanceDB index"),
    ("5-streamlit-go.py", "Launch Streamlit chatbot"),
]

def ask_continue(message: str) -> bool:
    """Ask user if they want to continue."""
    reply = input(f"\n➡️ {message} (y/n): ").strip().lower()
    return reply == "y"

def reset_pipeline():
    print("🔄 Resetting pipeline...")

    # Delete chunks file
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)
        print(f"   ✅ Removed {CHUNKS_FILE}")
    else:
        print("   ℹ️ No chunks.pkl found to delete")

    # Delete LanceDB data folder
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print(f"   ✅ Removed {DB_DIR}")
    else:
        print("   ℹ️ No LanceDB directory found to delete")

def run_script(script_name: str):
    """Run a Python script in the current venv."""
    python_exe = sys.executable
    print(f"\n🚀 Running {script_name}...\n")
    result = subprocess.run([python_exe, script_name])
    if result.returncode != 0:
        prin
