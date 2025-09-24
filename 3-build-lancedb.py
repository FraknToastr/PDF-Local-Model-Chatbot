import os
import pickle
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import argparse

# --- Config defaults ---
CHUNKS_FILE = "data/all_chunks.pkl"
DB_URI = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
DEFAULT_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def embed_func(texts, tokenizer, model):
    """Embed a list of texts into vectors."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()

def main(model_id: str):
    # --- Load embedding model ---
    logger.info(f"Loading embedding model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    logger.info("Model loaded successfully.")

    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, "rb") as f:
        all_chunks = pickle.load(f)

    logger.info(f"Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}")

    # --- Connect to LanceDB ---
    db = lancedb.connect(DB_URI)

    # Drop existing table (optional if re-running)
    if TABLE_NAME in db.table_names():
        logger.info(f"Table {TABLE_NAME} already exists. Dropping it for rebuild.")
        db.drop_table(TABLE_NAME)

    table = db.create_table(TABLE_NAME, data=[])

    # Save metadata about the embedding model
    meta_table = db.create_table(
        "embedding_metadata",
        data=[{"model_id": model_id}],
        mode="overwrite"
    )
    logger.info(f"Saved embedding model metadata: {model_id}")

    # --- Insert embeddings ---
    records = []
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i+batch_size]

        texts = []
        metas = []
        for c in batch:
            if isinstance(c.get("chunk"), dict):
                texts.append(c["chunk"]["text"])  # new format
            else:
                texts.append(c["chunk"].text)  # old format
            metas.append(c.get("metadata", {}))

        embeddings = embed_func(texts, tokenizer, model)
        for j, text in enumerate(texts):
            records.append({
                "vector": embeddings[j],
                "text": text,
                "metadata": metas[j]
            })

        logger.info(f"Processed {i+len(batch)} / {len(all_chunks)} chunks")

    # Write to table
    logger.info(f"Inserting {len(records)} records into LanceDB...")
    table.add(records)
    logger.info("✅ Ingestion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build LanceDB index from PDF chunks.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID for embeddings (default: mixedbread-ai/mxbai-embed-large-v1)"
    )
    args = parser.parse_args()
    main(args.model_id)
