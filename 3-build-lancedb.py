import os
import pickle
import logging
import lancedb
import pyarrow as pa
import torch
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # --- Check GPU ---
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.warning("⚠️ No GPU detected, using CPU")

    # --- Load embedding model ---
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ Missing chunks file: {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # --- Generate embeddings ---
    records = []
    for c in chunks:
        text = c.get("text", "")
        if not text.strip():
            continue

        emb = embedder.encode(text, convert_to_numpy=True).tolist()
        records.append({
            "vector": emb,
            "text": text,
            "source_file": c.get("source_file"),
            "page_number": c.get("page_number"),
            "date": c.get("date")
        })

    logger.info(f"Prepared {len(records)} records with embeddings.")

    if not records:
        logger.error("❌ No valid chunks to insert into LanceDB.")
        return

    # --- Define schema ---
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), len(records[0]["vector"]))),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("date", pa.string())
    ])

    # --- Connect to LanceDB ---
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping existing table '{TABLE_NAME}'")
        db.drop_table(TABLE_NAME)

    logger.info(f"Creating new table with schema: {TABLE_NAME}")
    table = db.create_table(TABLE_NAME, schema=schema)  # ✅ FIXED (no empty data=[])

    # --- Insert data ---
    table.add(records)

    logger.info(f"✅ Inserted {len(records)} rows into table '{TABLE_NAME}'")
    logger.info(f"📋 Final schema: {table.schema}")

if __name__ == "__main__":
    main()
