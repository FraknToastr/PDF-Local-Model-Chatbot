import os
import pickle
import logging
import argparse
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# --- Config ---
DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Detect GPU ---
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    logger.warning("⚠️ No GPU detected, using CPU")

def embed_texts(model, tokenizer, texts, batch_size=16):
    """Embed texts into vectors using Hugging Face model."""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            outputs = model(**tokens)
            # Take mean pooling of last hidden state
            hidden_states = outputs.last_hidden_state
            batch_embeds = hidden_states.mean(dim=1).cpu().numpy()
            embeddings.append(batch_embeds)
    return np.vstack(embeddings)

def main(model_id: str):
    # --- Load chunks ---
    if not os.path.exists(CHUNKS_FILE):
        logger.error(f"❌ No chunks file found at {CHUNKS_FILE}. Run script 2 first.")
        return

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        logger.error("❌ Chunks file is empty (0 chunks). Run script 2 successfully first.")
        return

    texts = [c["chunk"]["text"] for c in chunks if c.get("chunk", {}).get("text")]
    metadata = [c.get("metadata", {}) for c in chunks if c.get("chunk", {}).get("text")]

    if not texts:
        logger.error("❌ No valid text chunks found in chunks file.")
        return

    logger.info(f"Loaded {len(texts)} text chunks from {CHUNKS_FILE}")

    # --- Load embedding model ---
    logger.info(f"Loading embedding model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    logger.info("Model loaded successfully.")

    # --- Generate embeddings ---
    logger.info("Embedding chunks...")
    embeddings = embed_texts(model, tokenizer, texts, batch_size=16)
    logger.info(f"✅ Generated embeddings shape: {embeddings.shape}")

    # --- Connect to LanceDB ---
    os.makedirs(DB_DIR, exist_ok=True)
    db = lancedb.connect(DB_DIR)

    if TABLE_NAME in db.table_names():
        logger.info(f"Table '{TABLE_NAME}' already exists → dropping and recreating.")
        db.drop_table(TABLE_NAME)

    # --- Insert into table ---
    logger.info(f"Creating new table: {TABLE_NAME}")
    data = []
    for emb, text, meta in zip(embeddings, texts, metadata):
        row = {"vector": emb.tolist(), "text": text}
        row.update(meta)  # merge metadata like page_number, source_file, date
        data.append(row)

    table = db.create_table(TABLE_NAME, data=data)
    logger.info(f"✅ Inserted {len(data)} rows into table '{TABLE_NAME}' at {DB_DIR}")

    # --- Quick sample ---
    sample = table.head(1).to_pandas()
    logger.info(f"🔎 Sample row from table:\n{sample}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str,
                        default="mixedbread-ai/mxbai-embed-large-v1",
                        help="Embedding model to use")
    args = parser.parse_args()
    main(args.model_id)
