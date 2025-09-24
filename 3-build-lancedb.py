import os
import pickle
import logging
import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_lancedb(chunks, model_name):
    """Build LanceDB table from chunks using SentenceTransformer embeddings"""
    # Load embedding model
    logger.info(f"🔍 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info(f"✅ Embedding model loaded (device={model.device})")

    # Connect to LanceDB
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)
    logger.info(f"📦 Connected to LanceDB at {DB_DIR}")

    # Define schema
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), model.get_sentence_embedding_dimension())),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("date", pa.string()),
    ])

    # Drop and recreate the table fresh
    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping old table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    logger.info(f"🆕 Creating new table: {TABLE_NAME}")
    table = db.create_table(TABLE_NAME, schema=schema, data=[])

    # Build records
    records = []
    for i, c in enumerate(chunks):
        text = c.get("text", "").strip()
        metadata = c.get("metadata", {})
        if not text:
            continue

        records.append({
            "text": text,
            "vector": model.encode(text).tolist(),
            "source_file": metadata.get("source_file", ""),
            "page_number": metadata.get("page_number", -1),
            "date": metadata.get("date", ""),
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  🔨 Encoded {i+1}/{len(chunks)} chunks...")

    if not records:
        logger.warning("⚠️ No valid chunks to insert! Table will remain empty.")
    else:
        logger.info(f"📥 Inserting {len(records)} rows into LanceDB...")
        table.add(records)
        logger.info("✅ Data inserted successfully.")

    # Schema check
    logger.info("📊 Final table schema:")
    logger.info(table.schema)


def main():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}. Run script 2 first.")
        return

    # Load chunks
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # Build LanceDB
    build_lancedb(chunks, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
