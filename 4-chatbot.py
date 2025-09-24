import os
import json
import logging
import streamlit as st
import torch
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
DB_DIR = os.path.join("data", "lancedb_data")
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load DB metadata ---
meta_path = os.path.join(DB_DIR, "embedding_metadata.json")
if not os.path.exists(meta_path):
    st.error("❌ No embedding metadata found. Run script 3 first.")
    st.stop()

with open(meta_path, "r") as f:
    db_meta = json.load(f)
db_model_id = db_meta["model_id"]
db_vector_dim = db_meta["vector_dim"]

# --- Streamlit Sidebar Controls ---
st.sidebar.header("⚙️ Chatbot Settings")

user_model_choice = st.sidebar.selectbox(
    "Embedding model:",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0 if "MiniLM" in db_model_id else 1
)

if user_model_choice != db_model_id:
    st.sidebar.warning(
        f"⚠️ DB was built with {db_model_id}. Overriding to match DB."
    )
model_id = db_model_id

top_k = st.sidebar.slider("Top-K (retrieved docs)", 1, 20, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
max_new_tokens = st.sidebar.slider("Max new tokens", 50, 2000, 512)

reset_conv = st.sidebar.button("🔄 Reset Conversation")

# --- State ---
if reset_conv or "history" not in st.session_state:
    st.session_state.history = []

# --- Load Embedder ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer(model_id, device=device)
embedder.max_seq_length = 512

# --- Load DB ---
db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    st.error(f"❌ Table {TABLE_NAME} not found. Run script 3 first.")
    st.stop()

table = db.open_table(TABLE_NAME)

# --- Load LLM (local HuggingFace) ---
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # swap if needed
tok = AutoTokenizer.from_pretrained(LLM_MODEL)
llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, device_map="auto")

# --- Streamlit UI ---
st.title("📄 Local PDF Chatbot")

query = st.text_input("Ask a question about the PDFs:")

if query:
    query_vec = embedder.encode(query, convert_to_numpy=True).tolist()

    results = table.search(query_vec, vector_column_name="vector").limit(top_k).to_list()

    context = "\n".join([r["text"] for r in results])
    prompt = f"Answer based only on the context below.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tok(prompt, return_tensors="pt").to(device)
    outputs = llm.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k
    )

    answer = tok.decode(outputs[0], skip_special_tokens=True)

    st.write("### 💬 Answer")
    st.write(answer)

    # Save to history
    st.session_state.history.append({"q": query, "a": answer})

# --- Show conversation history ---
if st.session_state.history:
    st.write("### 📜 Conversation History")
    for h in st.session_state.history:
        st.markdown(f"**Q:** {h['q']}\n\n**A:** {h['a']}")
