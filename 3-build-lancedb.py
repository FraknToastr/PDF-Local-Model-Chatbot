# 3-build-lancedb.py
"""
Build LanceDB from chunked PDF text.
- Loads chunks created by script 2 (all_chunks.pkl)
- Embeds with the chosen model
- Stores in LanceDB with schema-safe structure
"""

import os
import pickle
import logging
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import pyarrow as pa

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

# Default embedding model (overridable in chatbot UI)
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunk file found at {CHUNKS_FILE}. Run script 2 first.")
        return []
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"✅ Loaded {len(chunks)} chunks from {CHUNKS_FILE}")
    return chunks


def get_embedding_model(model_id: str):
    """
    Return a function that generates embeddings given a list of texts.
    Supports both HuggingFace and SentenceTransformers.
    """
    logger.info(f"🔍 Loading embedding model: {model_id}")

    if model_id.startswith("sentence-transformers/"):
        model = SentenceTransformer(model_id)
        return model, lambda texts: model.encode(texts, convert_to_numpy=True).tolist()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        def embed(texts):
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embeddings.tolist()

        return model, embed


def build_lancedb(chunks, model_id=DEFAULT_MODEL):
    if not chunks:
        logger.error("❌ No chunks to index. Exiting.")
        return

    # Initialize DB
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    # Load model + embedding function
    model, embed_fn = get_embedding_model(model_id)

    # Prepare schema
    schema = pa.schema([
        ("vector", pa.list_(pa.float32())),
        ("text", pa.string()),
        ("source_file", pa.string()),
        ("page_number", pa.int32()),
        ("date", pa.string()),
    ])

    # Drop table if exists
    if TABLE_NAME in db.table_names():
        logger.info(f"🗑️ Dropping old table: {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, schema=schema, data=[])

    # Insert data
    batch = []
    for i, entry in enumerate(chunks, start=1):
        text = entry.get("chunk", {}).get("text") or entry.get("text")
        metadata = entry.get("metadata", {})

        if not text:
            continue

        vec = embed_fn([text])[0]

        batch.append({
            "vector": vec,
            "text": text,
            "source_file": metadata.get("source_file"),
            "page_number": metadata.get("page_number"),
            "date": metadata.get("date"),
        })

        if i % 50 == 0:
            table.add(batch)
            logger.info(f"Inserted {i}/{len(chunks)} chunks...")
            batch = []

    if batch:
        table.add(batch)

    logger.info(f"✅ Finished inserting {len(chunks)} chunks into LanceDB ({TABLE_NAME})")


def main():
    chunks = load_chunks()
    build_lancedb(chunks, DEFAULT_MODEL)


if __name__ == "__main__":
    main()
