import os
import logging
import streamlit as st
import lancedb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Streamlit setup ---
st.set_page_config(page_title="Council Meeting Chatbot", page_icon="🏛️", layout="wide")
st.markdown("<h1 style='text-align:center; color:white;'>Council Meeting Chatbot</h1>", unsafe_allow_html=True)

# --- Load LanceDB ---
logger.info(f"📦 Connecting to LanceDB at {DB_DIR}...")
db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

# --- Load embedding model ---
logger.info(f"🔍 Loading embedding model: {DEFAULT_MODEL}")
embed_model = SentenceTransformer(DEFAULT_MODEL)
embed_dim = embed_model.get_sentence_embedding_dimension()
logger.info(f"✅ Embedding model loaded (dim={embed_dim})")

# --- Dimension check ---
table_dim = None
for field in table.schema:
    if field.name == "vector":
        table_dim = field.type.list_size
        break

if table_dim and table_dim != embed_dim:
    st.error(f"❌ Embedding dimension mismatch! Table={table_dim}, Model={embed_dim}")
    st.stop()

# --- Load LLM ---
logger.info(f"🤖 Loading LLM: {LLM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
logger.info("✅ LLM ready")

# --- Chat UI ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    top_k = st.slider("Top-K (context docs)", 1, 10, 5)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    max_tokens = st.slider("Max new tokens", 50, 1000, 300)

for msg in st.session_state["messages"]:
    role, text = msg
    bubble_color = "#1e90ff" if role == "user" else "#2e8b57"
    st.markdown(
        f"<div style='background-color:{bubble_color}; padding:10px; border-radius:10px; color:white; margin:5px 0;'>{role}: {text}</div>",
        unsafe_allow_html=True
    )

query = st.chat_input("Ask about Adelaide council meetings...")

if query:
    st.session_state["messages"].append(("user", query))

    # --- Embed query ---
    query_embedding = embed_model.encode(query).tolist()

    # --- Search DB ---
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No relevant context found."

    # --- Build prompt ---
    prompt = f"""You are a factual assistant. Answer ONLY using the provided context. 
If the context is empty or irrelevant, say "I could not find information in the council meeting records."

Context:
{context}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state["messages"].append(("assistant", answer))

    # --- Rerun UI ---
    st.rerun()
