import os
import pickle
import logging
import argparse
import torch
import lancedb
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_PATH = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Embedding model config ---
DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

class ChunkEmbedding(BaseModel):
    vector: Vector(1024)  # vector length matches embedding model
    text: str
    source_file: str
    page_number: int | None = None
    date: str | None = None

def embed_texts(model, tokenizer, texts, device="cpu", batch_size=8):
    """Generate embeddings for a list of texts"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Take mean pooling
            emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings.extend(emb)
    return np.array(embeddings)

def main(model_id):
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        all_chunks = pickle.load(f)

    if not all_chunks:
        logger.error("❌ No chunks found in all_chunks.pkl. Did script 2 produce 0 chunks?")
        return

    logger.info(f"Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}")

    # --- Load embedding model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ Using CPU (GPU not available)")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    logger.info(f"Model loaded successfully: {model_id}")

    # --- Prepare texts ---
    texts = [c["chunk"]["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    # --- Compute embeddings ---
    logger.info("Computing embeddings...")
    vectors = embed_texts(model, tokenizer, texts, device=device, batch_size=8)

    logger.info(f"✅ Generated embeddings: {vectors.shape}")

    # --- Open LanceDB ---
    db = lancedb.connect(DB_PATH)

    # Drop existing table if it exists
    if TABLE_NAME in db.table_names():
        logger.warning(f"⚠️ Dropping existing table {TABLE_NAME}")
        db.drop_table(TABLE_NAME)

    # Create new table
    logger.info(f"Creating new table: {TABLE_NAME}")
    table = db.create_table(TABLE_NAME, schema=ChunkEmbedding)

    # Insert records
    records = []
    for vec, text, meta in zip(vectors, texts, metadatas):
        records.append(
            {
                VECTOR_COL: vec.tolist(),
                "text": text,
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "date": meta.get("date"),
            }
        )

    table.add(records)
    logger.info(f"✅ Inserted {len(records)} records into LanceDB table '{TABLE_NAME}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Embedding model ID")
    args = parser.parse_args()
    main(args.model_id)
