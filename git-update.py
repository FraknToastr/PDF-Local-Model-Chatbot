#!/usr/bin/env python3
"""
Helper script to update the local repository from origin.
It runs `git fetch origin` followed by `git pull origin <branch>`.
"""

import subprocess
import sys

def run_command(cmd):
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {' '.join(cmd)}")
        sys.exit(e.returncode)

def main():
    # Default branch, override via argument
    branch = sys.argv[1] if len(sys.argv) > 1 else "main"

    print(f"🔄 Fetching from origin...")
    run_command(["git", "fetch", "origin"])

    print(f"⬇️ Pulling latest changes from origin/{branch}...")
    run_command(["git", "pull", "origin", branch])

    print(f"✅ Repository is up to date with origin/{branch}.")

if __name__ == "__main__":
    main()
