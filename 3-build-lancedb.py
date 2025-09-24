import os
import pickle
import logging

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

import torch

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    logger.info("🚀 Using GPU for embeddings")
else:
    logger.warning("⚠️ Using CPU for embeddings")


def main(embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"):
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        logger.error("❌ No chunks found. Nothing to index.")
        return

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # --- Load embedder ---
    embedder = SentenceTransformer(embedding_model_name, device=device)

    # --- Generate embeddings ---
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # --- Attach vectors back into chunks ---
    for c, emb in zip(chunks, embeddings):
        c["vector"] = emb.tolist()

    # --- Define schema for LanceDB ---
    schema = pa.schema([
        ("vector", pa.list_(pa.float32(), len(embeddings[0]))),
        ("text", pa.string()),
        ("source_file", pa.string()),
        ("page_number", pa.int32()),
        ("date", pa.string()),
    ])

    # --- Connect to LanceDB ---
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping existing table '{TABLE_NAME}'")
        db.drop_table(TABLE_NAME)

    logger.info(f"Creating new table with schema: {TABLE_NAME}")
    table = db.create_table(TABLE_NAME, schema=schema, data=[])

    # --- Insert chunks ---
    table.add(chunks)

    logger.info(f"✅ Inserted {len(chunks)} rows into table '{TABLE_NAME}'")


if __name__ == "__main__":
    main()
