import os
import logging
import streamlit as st
import torch
import lancedb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- DB Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# --- Detect device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Using device: {DEVICE}")

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Local Model Chatbot", layout="wide")
st.title("📄 PDF Local Model Chatbot")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")

    embed_model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "mixedbread-ai/mxbai-embed-large-v1"
        ],
        index=0
    )

    llm_model_name = st.selectbox(
        "LLM model",
        [
            "microsoft/phi-3-mini-4k-instruct",  # ✅ lightweight + free
            "mistralai/Mistral-7B-Instruct-v0.2" # 🔒 requires HF login
        ],
        index=0
    )

    top_k = st.slider("Top-K results", 1, 10, 3)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, step=0.1)
    max_new_tokens = st.slider("Max new tokens", 50, 1000, 300)

    suppress_context = st.checkbox("Suppress context (ignore PDF chunks)", value=False)
    if st.button("🔄 Reset Conversation"):
        if "history" in st.session_state:
            del st.session_state["history"]
        st.success("Conversation reset!")

# --- Cache models ---
@st.cache_resource
def load_embedding_model(name):
    logger.info(f"Loading embedding model: {name}")
    return SentenceTransformer(name, device=DEVICE)

@st.cache_resource
def load_llm_and_tokenizer(name):
    logger.info(f"Loading LLM model: {name}")
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto"
    )
    return tok, model

# Load models
embed_model = load_embedding_model(embed_model_name)
tokenizer, llm_model = load_llm_and_tokenizer(llm_model_name)

# --- LanceDB ---
db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    st.error("No LanceDB table found. Run script 3 (build-lancedb) first.")
    st.stop()
table = db.open_table(TABLE_NAME)

# --- Chat Input ---
if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.chat_input("Ask me something about the PDFs...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state["history"].append(("user", user_query))

    # --- Embed query ---
    query_embedding = embed_model.encode(user_query).tolist()

    # --- Retrieve context ---
    context_text = ""
    if not suppress_context:
        results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
        if results:
            context_text = "\n\n".join([r["text"] for r in results if "text" in r])
    else:
        results = []

    # --- Build prompt ---
    if suppress_context or not context_text.strip():
        prompt = f"Answer the following question:\n{user_query}"
    else:
        prompt = f"Use the context below to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {user_query}"

    # --- Generate response ---
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Display ---
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state["history"].append(("assistant", answer))

# --- Show history ---
for role, msg in st.session_state["history"]:
    with st.chat_message(role):
        st.write(msg)
