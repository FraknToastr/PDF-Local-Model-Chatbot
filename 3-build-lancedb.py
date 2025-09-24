import os
import pickle
import logging
import argparse
import json
import torch
import lancedb
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
META_FILE = os.path.join(DB_DIR, "embedding_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(model_id: str):
    logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Loading embedding model: {model_id}")

    embed_model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    vector_dim = embed_model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {vector_dim}")

    # Load chunks
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    if not chunks:
        logger.error("❌ No chunks found. Aborting.")
        return

    # Convert to LanceDB-compatible format
    records = []
    for c in chunks:
        text = c["chunk"]["text"]
        metadata = c["metadata"]
        embedding = embed_model.encode(text).tolist()
        records.append({
            "vector": embedding,
            "text": text,
            "source_file": metadata.get("source_file"),
            "page_number": metadata.get("page_number"),
            "date": metadata.get("date"),
        })

    # Save into LanceDB
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        logger.info(f"Replacing existing table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=records)
    logger.info(f"✅ Created LanceDB table '{TABLE_NAME}' with {table.count_rows()} rows.")

    # Save embedding metadata
    with open(META_FILE, "w") as f:
        json.dump({"model_id": model_id, "vector_dim": vector_dim}, f)
    logger.info(f"💾 Saved embedding metadata: {META_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Embedding model to use")
    args = parser.parse_args()
    main(args.model_id)
