import subprocess
import sys
import os

# Always use the Python executable of the current venv
python_exe = sys.executable
script_path = os.path.join(os.path.dirname(__file__), "4-chatbot.py")

subprocess.run([python_exe, "-m", "streamlit", "run", script_path])
