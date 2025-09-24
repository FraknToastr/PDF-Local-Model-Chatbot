import os
import pickle
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
import logging

# --- Config ---
CHUNKS_FILE = "data/all_chunks.pkl"
DB_URI = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Load embedding model ---
logger.info(f"Loading embedding model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
logger.info("Model loaded successfully.")


def embed_func(texts):
    """Embed a list of texts into vectors."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        model.device
    )
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()


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

# --- Insert embeddings ---
records = []
batch_size = 32
for i in range(0, len(all_chunks), batch_size):
    batch = all_chunks[i : i + batch_size]

    # Handle both new + old chunk formats
    texts = []
    metas = []
    for c in batch:
        if isinstance(c.get("chunk"), dict):
            # New format: {"chunk": {"text": ...}}
            texts.append(c["chunk"]["text"])
        else:
            # Old format: {"chunk": <DoclingChunkObject>}
            texts.append(c["chunk"].text)
        metas.append(c.get("metadata", {}))

    embeddings = embed_func(texts)
    for j, text in enumerate(texts):
        records.append({"vector": embeddings[j], "text": text, "metadata": metas[j]})

    logger.info(f"Processed {i + len(batch)} / {len(all_chunks)} chunks")

# Write to table
logger.info(f"Inserting {len(records)} records into LanceDB...")
table.add(records)
logger.info("✅ Ingestion complete!")
