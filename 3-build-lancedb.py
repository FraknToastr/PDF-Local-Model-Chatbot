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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def build_lancedb(chunks, model_name=DEFAULT_MODEL):
    logger.info(f"🔍 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    device = "cuda" if model.device.type == "cuda" else "cpu"
    logger.info(f"✅ Embedding model loaded (device={device})")

    logger.info(f"📦 Connecting to LanceDB at {DB_DIR}...")
    db = lancedb.connect(DB_DIR)

    # --- Define schema ---
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), list_size=model.get_sentence_embedding_dimension())),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("date", pa.string())
    ])

    # --- Drop existing table if present ---
    if TABLE_NAME in db.table_names():
        logger.info(f"🔁 Dropping existing table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    logger.info(f"🆕 Creating new table: {TABLE_NAME}")
    table = db.create_table(TABLE_NAME, schema=schema)

    # --- Build rows ---
    rows = []
    for c in chunks:
        text = c.get("chunk", {}).get("text")
        if not text:
            continue
        embedding = model.encode(text).tolist()
        rows.append({
            "vector": embedding,
            "text": text,
            "source_file": c.get("metadata", {}).get("source_file"),
            "page_number": c.get("metadata", {}).get("page_number"),
            "date": c.get("metadata", {}).get("date"),
        })

    if rows:
        logger.info(f"📥 Inserting {len(rows)} rows into LanceDB...")
        table.add(rows)
        logger.info("✅ Insert complete.")
    else:
        logger.warning("⚠️ No rows to insert! Check your chunking pipeline.")

    logger.info(f"📊 Table schema:\n{table.schema}")


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
