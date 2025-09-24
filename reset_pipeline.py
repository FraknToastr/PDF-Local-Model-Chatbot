import os
import shutil
import logging
import subprocess

# --- Config ---
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
LANCEDB_DIR = os.path.join(DATA_DIR, "lancedb_data")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def reset_pipeline():
    logger.info("🔄 Resetting pipeline (without deleting scraped PDFs)...")

    # Delete all_chunks.pkl
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        logger.info(f"🗑️ Deleted {OUTPUT_FILE}")
    else:
        logger.info("✅ No all_chunks.pkl found, skipping.")

    # Delete LanceDB data directory
    if os.path.exists(LANCEDB_DIR):
        shutil.rmtree(LANCEDB_DIR)
        logger.info(f"🗑️ Deleted {LANCEDB_DIR}")
    else:
        logger.info("✅ No lancedb_data folder found, skipping.")

    # Delete Streamlit session history if stored in data/
    session_file = os.path.join(DATA_DIR, "session_history.pkl")
    if os.path.exists(session_file):
        os.remove(session_file)
        logger.info(f"🗑️ Deleted {session_file}")
    else:
        logger.info("✅ No session_history.pkl found, skipping.")

    # Clear Streamlit cache
    try:
        subprocess.run(["streamlit", "cache", "clear"], check=True)
        logger.info("🗑️ Streamlit cache cleared.")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear Streamlit cache automatically: {e}")

    logger.info("✨ Reset complete! Your PDFs are still in the data folder.")

if __name__ == "__main__":
    reset_pipeline()
