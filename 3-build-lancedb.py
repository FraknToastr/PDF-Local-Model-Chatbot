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

    # Define schema explicitly
    schema = pa.schema([
        pa.field("vector", pa.list_(pa.float32(), model.get_sentence_embedding_dimension())),
        pa.field("text", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("page_number", pa.int32()),
        pa.field("date", pa.string()),
    ])

    # Create or open table
    if TABLE_NAME in db.table_names():
        table = db.open_table(T_
