import os
import pickle
import logging
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
import pyarrow as pa

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"
MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_model(model_id: str):
    logger.info(f"Loading embedding model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

    if torch.cuda.is_available():
        model = model.to("cuda")
        logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("⚠️ No GPU detected, using CPU")

    return tokenizer, model

def embed_texts(tokenizer, model, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

def main():
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ Chunks file not found: {CHUNKS_FILE}. Run script 2 first.")
        return

    # Load chunks
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    logger.info(f"Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    if not chunks:
        logger.error("❌ No chunks to embed. Exiting.")
        return

    # Load model
    tokenizer, model = load_model(MODEL_ID)

    # Prepare texts & metadata
    texts = [c["chunk"]["text"] for c in chunks]
    metadata = [c.get("metadata", {}) for c in chunks]

    # Generate embeddings
    embeddings = embed_texts(tokenizer, model, texts)

    # Connect to LanceDB
    db = lancedb.connect(DB_DIR)

    # Define schema
    schema = pa.schema([
        (VECTOR_COL, pa.list_(pa.float32())),
        ("text", pa.string()),
        ("source_file", pa.string()),
        ("page_number", pa.int64()),
        ("date", pa.string()),
    ])

    # Check if table already exists
    if TABLE_NAME in db.table_names():
        logger.info(f"📂 Table '{TABLE_NAME}' already exists → checking for duplicates")
        table = db.open_table(TABLE_NAME)

        # Fetch existing keys (source_file + page_number + text)
        existing = set()
        for row in table.to_list():
            key = (row.get("source_file"), row.get("page_number"), row.get("text"))
            existing.add(key)

        # Build new records, skip duplicates
        new_records = []
        for emb, txt, meta in zip(embeddings, texts, metadata):
            key = (meta.get("source_file"), meta.get("page_number"), txt)
            if key in existing:
                continue
            rec = {
                VECTOR_COL: emb.tolist(),
                "text": txt,
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "date": meta.get("date"),
            }
            new_records.append(rec)

        if new_records:
            table.add(new_records)
            logger.info(f"✅ Appended {len(new_records)} new unique records to '{TABLE_NAME}'")
        else:
            logger.info("ℹ️ No new unique records to add")

    else:
        logger.info(f"📂 Table '{TABLE_NAME}' does not exist → creating new table")
        records = []
        for emb, txt, meta in zip(embeddings, texts, metadata):
            rec = {
                VECTOR_COL: emb.tolist(),
                "text": txt,
                "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"),
                "date": meta.get("date"),
            }
            records.append(rec)

        table = db.create_table(TABLE_NAME, schema=schema, data=records, mode="create")
        logger.info(f"✅ Created new table '{TABLE_NAME}' with {len(records)} records")

    # Sanity check: confirm DB exists
    if os.path.exists(DB_DIR):
        logger.info(f"📦 LanceDB directory: {DB_DIR}")
    else:
        logger.error(f"❌ Expected LanceDB directory missing: {DB_DIR}")

    # Print schema dump
    logger.info("📋 Table schema:")
    for field in table.schema:
        logger.info(f"  - {field.name}: {field.type}")

    # Print table stats
    logger.info(f"🔢 Total records in '{TABLE_NAME}': {table.count_rows()}")

if __name__ == "__main__":
    main()
