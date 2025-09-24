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

def embed_func(texts, tokenizer, model, device):
    """Embed a list of texts into vectors."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()

def main(model_id: str):
    # --- Detect GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 Using GPU: {gpu_name}")
        dtype = torch.float16
    else:
        device = torch.device("cpu")
        logger.warning("⚠️ No GPU detected, using CPU")
        dtype = torch.float32

    # --- Load embedding model ---
    logger.info(f"Loading embedding model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(device)
    logger.info("Model loaded successfully.")

    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, "rb") as f:
        all_chunks = pickle.load(f)

    logger.info(f"Loaded {len(all_chunks)} chunks from {CHUNKS_FILE}")

    # --- Connect to LanceDB ---
    db = lancedb.connect(DB_URI)

    # Check if table exists already
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        existing_count = table.count_rows()
        logger.info(f"Resuming: Found existing table with {existing_count} records.")
    else:
        logger.info(f"Creating new table: {TABLE_NAME}")
        table = db.create_table(TABLE_NAME, data=[])
        existing_count = 0

    # Save metadata about the embedding model
    db.create_table(
        "embedding_metadata",
        data=[{"model_id": model_id}],
        mode="overwrite"
    )
    logger.info(f"Saved embedding model metadata: {model_id}")

    # --- Resume from where we left off ---
    total_chunks = len(all_chunks)
    if existing_count >= total_chunks:
        logger.info("✅ All chunks already embedded. Nothing to do.")
        return

    logger.info(f"Starting from chunk {existing_count} / {total_chunks}")

    # --- Insert embeddings batch by batch ---
    batch_size = 32
    for i in range(existing_count, total_chunks, batch_size):
        batch = all_chunks[i:i+batch_size]

        texts = []
        metas = []
        for c in batch:
            if isinstance(c.get("chunk"), dict):
                texts.append(c["chunk"]["text"])  # new format
            else:
                texts.append(c["chunk"].text)  # old format
            metas.append(c.get("metadata", {}))

        embeddings = embed_func(texts, tokenizer, model, device)

        records = []
        for j, text in enumerate(texts):
            records.append({
                "vector": embeddings[j],
                "text": text,
                "metadata": metas[j]
            })

        # Write batch to LanceDB immediately (saves progress!)
        table.add(records)
        logger.info(f"✅ Saved batch {i // batch_size + 1} → processed {i+len(batch)} / {total_chunks} chunks")

    logger.info("🎉 Ingestion complete!")

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
