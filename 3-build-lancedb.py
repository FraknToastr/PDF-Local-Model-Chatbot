import os
import pickle
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import argparse
import pyarrow as pa
from tqdm import tqdm

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

def infer_schema_from_sample(sample):
    """Infer LanceDB schema from a sample record, flattening metadata fields."""
    fields = [
        pa.field("vector", pa.list_(pa.float32())),
        pa.field("text", pa.string()),
    ]

    for k, v in sample.items():
        if k in ["vector", "text"]:
            continue
        if isinstance(v, str):
            fields.append(pa.field(k, pa.string()))
        elif isinstance(v, int):
            fields.append(pa.field(k, pa.int64()))
        elif isinstance(v, float):
            fields.append(pa.field(k, pa.float64()))
        else:
            fields.append(pa.field(k, pa.string()))  # fallback

    return pa.schema(fields)

def flatten_record(vector, text, metadata):
    """Flatten metadata into top-level fields for LanceDB."""
    record = {"vector": vector, "text": text}
    if isinstance(metadata, dict):
        for k, v in metadata.items():
            record[k] = v
    return record

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
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(device)
    logger.info("Model loaded successfully.")

    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")

    with open(CHUNKS_FILE, "rb") as f:
        all_chunks = pickle.load(f)

    total_chunks = len(all_chunks)
    logger.info(f"Loaded {total_chunks} chunks from {CHUNKS_FILE}")

    # --- Connect to LanceDB ---
    db = lancedb.connect(DB_URI)

    # Check if table exists already
    if TABLE_NAME in db.table_names():
        table = db.open_table(TABLE_NAME)
        existing_count = table.count_rows()
        logger.info(f"Resuming: Found existing table with {existing_count} records.")
    else:
        # Infer schema from first chunk
        first_chunk = all_chunks[0]
        if isinstance(first_chunk.get("chunk"), dict):
            text_val = first_chunk["chunk"]["text"]
        else:
            text_val = first_chunk["chunk"].text

        sample_record = {
            "vector": [0.0],
            "text": text_val,
            **first_chunk.get("metadata", {})
        }

        schema = infer_schema_from_sample(sample_record)
        logger.info(f"Creating new table with inferred + flattened schema: {TABLE_NAME}")
        table = db.create_table(TABLE_NAME, schema=schema)
        existing_count = 0

        # 🔎 Dump schema
        logger.info("📋 Table schema:")
        for field in schema:
            logger.info(f"  - {field.name}: {field.type}")

    # Save metadata about the embedding model
    db.create_table(
        "embedding_metadata",
        data=[{"model_id": model_id}],
        mode="overwrite"
    )
    logger.info(f"Saved embedding model metadata: {model_id}")

    # --- Resume check ---
    if existing_count >= total_chunks:
        logger.info("✅ All chunks already embedded. Nothing to do.")
        return

    logger.info(f"Starting from chunk {existing_count} / {total_chunks}")

    # --- Insert embeddings batch by batch with tqdm ---
    batch_size = 32
    pbar = tqdm(total=total_chunks, initial=existing_count, desc="Embedding chunks", unit="chunk")

    sample_printed = False  # to only show first sample once

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
            records.append(flatten_record(embeddings[j], text, metas[j]))

        # Save progress immediately (batch-safe)
        table.add(records)

        # Print first sample to verify schema + metadata
        if not sample_printed:
            logger.info("🔎 Sample inserted record:")
            for k, v in records[0].items():
                if k == "vector":
                    logger.info(f"  {k}: [embedding length = {len(v)}]")
                else:
                    logger.info(f"  {k}: {v}")
            sample_printed = True

        # Update progress bar
        pbar.update(len(batch))

    pbar.close()
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
