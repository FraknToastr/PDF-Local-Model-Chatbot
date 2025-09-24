import os
import pickle
import logging
import json
import torch
import lancedb
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

# Choose embedding model here
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # or "mixedbread-ai/mxbai-embed-large-v1"

# Infer vector dimension
VECTOR_DIM = 384 if "MiniLM" in MODEL_ID else 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # --- Init embedding model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"🚀 Using {device.upper()} for embeddings")
    embedder = SentenceTransformer(MODEL_ID, device=device)
    embedder.max_seq_length = 512  # truncate long texts safely

    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"📦 Loaded {len(chunks)} chunks")

    # --- Prepare DB ---
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping old table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    logger.info(f"🆕 Creating table: {TABLE_NAME}")
    table = db.create_table(
        TABLE_NAME,
        data=[
            {
                "vector": [0.0] * VECTOR_DIM,
                "text": "",
                "source_file": "",
                "page_number": 0,
                "date": "",
            }
        ],
        mode="overwrite"
    )

    # --- Embed & Insert ---
    records = []
    for ch in chunks:
        text = ch["chunk"]["text"]
        metadata = ch["metadata"]

        vec = embedder.encode(text, convert_to_numpy=True).tolist()

        records.append({
            "vector": vec,
            "text": text,
            "source_file": metadata.get("source_file", ""),
            "page_number": metadata.get("page_number", 0),
            "date": metadata.get("date", ""),
        })

    table.add(records)
    logger.info(f"✅ Inserted {len(records)} rows into LanceDB")

    # --- Save embedding metadata ---
    meta_path = os.path.join(DB_DIR, "embedding_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({"model_id": MODEL_ID, "vector_dim": VECTOR_DIM}, f)
    logger.info(f"💾 Saved embedding metadata to {meta_path}")

if __name__ == "__main__":
    main()
