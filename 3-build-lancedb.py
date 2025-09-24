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
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}")
        return []
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


def build_lancedb(chunks, model_id):
    logger.info(f"🔍 Loading embedding model: {model_id}")
    model = SentenceTransformer(model_id, device="cuda" if torch.cuda.is_available() else "cpu")

    # Connect to DB
    db = lancedb.connect(DB_DIR)

    # Define schema
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32())),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("date", pa.string()),
    ])

    # Create table (old behavior)
    table = db.create_table(TABLE_NAME, schema=schema, data=[])
    logger.info(f"🆕 Creating new table: {TABLE_NAME}")

    # Insert data
    records = []
    for c in chunks:
        text = c.get("chunk", {}).get("text") if "chunk" in c else c.get("text", "")
        metadata = c.get("metadata", {})
        embedding = model.encode(text).tolist()
        records.append({
            "vector": embedding,
            "text": text,
            "source_file": metadata.get("source_file"),
            "page_number": metadata.get("page_number"),
            "date": metadata.get("date"),
        })

    if records:
        table.add(records)
        logger.info(f"✅ Inserted {len(records)} rows into table: {TABLE_NAME}")
    else:
        logger.warning("⚠️ No records to insert into LanceDB.")


def main():
    chunks = load_chunks()
    if not chunks:
        logger.error("❌ No chunks to process. Run script 2 first.")
        return

    build_lancedb(chunks, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
