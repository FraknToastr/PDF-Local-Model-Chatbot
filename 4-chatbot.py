import os
import json
import logging
import torch
import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
DATA_DIR = "data"
DB_DIR = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"
META_FILE = os.path.join(DB_DIR, "embedding_metadata.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Load metadata ---
if not os.path.exists(META_FILE):
    st.error("❌ No embedding metadata found. Please run script 3 first.")
    st.stop()

with open(META_FILE, "r") as f:
    db_meta = json.load(f)

db_model_id = db_meta["model_id"]
db_vector_dim = db_meta["vector_dim"]

# --- Sidebar UI ---
st.sidebar.title("⚙️ Settings")

embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0 if "MiniLM" in db_model_id else 1,
)

# Enforce DB metadata
if embedding_model_name != db_model_id:
    st.sidebar.warning(f"⚠️ DB was built with {db_model_id}. Using that instead for compatibility.")
    embedding_model_name = db_model_id

top_k = st.sidebar.slider("Top-K results", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.2, 0.1)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256, step=64)

suppress_context = st.sidebar.checkbox("Suppress PDF context", value=False)
if st.sidebar.button("🔄 Reset Conversation"):
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.sidebar.success("Conversation reset!")

# --- Load embedding model ---
embed_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")

# --- Load LLM ---
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- Load DB ---
db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    st.error(f"❌ Table '{TABLE_NAME}' not found. Run script 3 first.")
    st.stop()

table = db.open_table(TABLE_NAME)

# --- Chat UI ---
st.title("📚 PDF Local Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_query = st.chat_input("Ask something about the PDFs...")

if user_query:
    # Embed query
    query_embedding = embed_model.encode(user_query).tolist()

    if len(query_embedding) != db_vector_dim:
        st.error(f"❌ Dimension mismatch: DB expects {db_vector_dim}, got {len(query_embedding)}. Rebuild DB with script 3.")
        st.stop()

    # Search DB
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    context = "\n".join([r["text"] for r in results]) if not suppress_context else ""

    # Build prompt
    prompt = f"Answer the question based ONLY on the provided PDF context.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=(temperature > 0),   # greedy if temp=0
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state["messages"].append(("user", user_query))
    st.session_state["messages"].append(("assistant", response))

# --- Display chat ---
for sender, msg in st.session_state["messages"]:
    if sender == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
