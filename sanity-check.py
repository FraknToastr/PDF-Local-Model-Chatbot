# sanity-check.py
"""
Quick script to verify chunks and LanceDB embeddings.
Run this after 2-hybrid-chunking-multiple-PDFs.py and 3-build-lancedb.py
"""

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
CHECK_KEYWORD = "Lomax-Smith"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sanity-check")

# --- Load chunks ---
if not os.path.exists(CHUNKS_FILE):
    logger.error(f"❌ No chunks file found at {CHUNKS_FILE}")
    exit(1)

with open(CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

logger.info(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

# Show first 3 chunks for inspection
for i, c in enumerate(chunks[:3]):
    text = c.get("chunk", {}).get("text") if isinstance(c.get("chunk"), dict) else c.get("text")
    meta = c.get("metadata", {})
    logger.info(f"Sample chunk {i+1}: {str(text)[:100]}...")
    logger.info(f"   Metadata: {meta}")

# --- Keyword search ---
hits = [
    c for c in chunks
    if CHECK_KEYWORD.lower() in (
        (c.get("chunk", {}).get("text", "").lower() if isinstance(c.get("chunk"), dict) else c.get("text", "").lower())
    )
]

logger.info(f"🔍 Found {len(hits)} chunks containing '{CHECK_KEYWORD}'")
if hits:
    for h in hits[:3]:
        text = h.get("chunk", {}).get("text") if isinstance(h.get("chunk"), dict) else h.get("text")
        logger.info(f"   → {text[:120]}")

# --- LanceDB check ---
if not os.path.exists(DB_DIR):
    logger.error(f"❌ LanceDB directory not found at {DB_DIR}. Run script 3 first.")
    exit(1)

db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    logger.error(f"❌ Table '{TABLE_NAME}' not found in LanceDB")
    exit(1)

table = db.open_table(TABLE_NAME)
logger.info(f"✅ LanceDB table '{TABLE_NAME}' contains {table.count_rows()} rows")

# --- Test embedding + query ---
logger.info("Embedding test query using SentenceTransformer...")
model = SentenceTransformer(EMBED_MODEL)
query_vec = model.encode(["Who is the Lord Mayor of Adelaide?"], convert_to_numpy=True).tolist()[0]

results = table.search(query_vec, vector_column_name="vector").limit(3).to_list()
logger.info("🔎 Nearest neighbors from LanceDB:")
for r in results:
    text = r.get("text", "")
    logger.info(f"   → {text[:120]}...")
    logger.info(f"     Source: {r.get('source_file')} page {r.get('page_number')} date {r.get('date')}")
