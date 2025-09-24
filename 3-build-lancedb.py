import os
import pickle
import logging
import lancedb
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Embedding model
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def build_lancedb():
    logger.info("🚀 Building LanceDB...")

    # Load embedding model with truncation
    embedder = SentenceTransformer(MODEL_ID)
    embedder.max_seq_length = 512  # ✅ enforce safe truncation
    logger.info(f"Loaded embedding model: {MODEL_ID}")

    # Load chunks
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Embed all chunks
    records = []
    for entry in chunks:
        text = entry["chunk"]["text"].strip()
        if not text:
            continue

        # Safe embedding with truncation
        emb = embedder.encode(text, normalize_embeddings=True).tolist()

        record = {
            "vector": emb,
            "text": text,
            "source_file": entry.get("metadata", {}).get("source_file"),
            "page_number": entry.get("metadata", {}).get("page_number"),
            "date": entry.get("metadata", {}).get("date"),
        }
        records.append(record)

    # Open/create LanceDB
    db = lancedb.connect(DB_DIR)

    # Drop table if it exists (fresh rebuild)
    if TABLE_NAME in db.table_names():
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=records)
    logger.info(f"✅ Created table '{TABLE_NAME}' with {len(records)} records.")


if __name__ == "__main__":
    build_lancedb()
