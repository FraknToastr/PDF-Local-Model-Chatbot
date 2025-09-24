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

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_lancedb(chunks, model_name):
    logger.info(f"🔍 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    # Ensure GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"✅ Embedding model loaded (device={device})")

    # Prepare LanceDB connection
    logger.info(f"📦 Connecting to LanceDB at {DB_DIR}...")
    db = lancedb.connect(DB_DIR)

    # Drop and recreate the table fresh
    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping old table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), 384)),  # 384 for MiniLM
            pa.field("text", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("page_number", pa.int32()),
            pa.field("date", pa.string()),
        ]
    )

    logger.info(f"🆕 Creating new table: {TABLE_NAME}")
    # ✅ FIXED: don’t pass data=[]
    table = db.create_table(TABLE_NAME, schema=schema)

    # Build rows for insertion
    records = []
    for c in chunks:
        text = c["text"]
        metadata = c.get("metadata", {})

        # Compute embedding
        vector = model.encode(text).tolist()

        records.append(
            {
                "vector": vector,
                "text": text,
                "source_file": metadata.get("source_file"),
                "page_number": metadata.get("page_number"),
                "date": metadata.get("date"),
            }
        )

    if records:
        logger.info(f"➕ Inserting {len(records)} records...")
        table.add(records)
    else:
        logger.warning("⚠️ No rows to insert! Check your chunking pipeline.")

    logger.info("✅ LanceDB build complete.")


def main():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ Chunks file not found: {CHUNKS_FILE}")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    build_lancedb(chunks, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
