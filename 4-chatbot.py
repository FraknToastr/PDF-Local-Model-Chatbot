import os
import re
import streamlit as st
import lancedb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="https://www.cityofadelaide.com.au/common/base/img/favicon.ico",  # ✅ official favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme styling + chat bubbles
st.markdown("""
    <style>
        body { background-color: #111; color: #eee; }
        .user-bubble {
            background-color: #2c2f38;
            color: #fff;
            padding: 8px 12px;
            border-radius: 12px;
            margin: 4px 0;
            text-align: right;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #0057b7;
            color: #fff;
            padding: 12px;
            border-radius: 12px;
            margin: 4px 0;
            float: left;
            clear: both;
            max-width: 85%;
        }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Logo
# --------------------------
st.image("https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg", width=200)
st.title("Council Meeting Chatbot")

# --------------------------
# Load DB + Embedding Models
# --------------------------
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"

db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

embedding_models = {
    "sentence-transformers/all-MiniLM-L6-v2": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "mixedbread-ai/mxbai-embed-large-v1": SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
}
default_embed_model = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------
# Load LLM (local Hugging Face)
# --------------------------
@st.cache_resource
def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------
# Deduplication helper
# --------------------------
def deduplicate_chunks(chunks):
    seen = set()
    unique = []
    for ch in chunks:
        text = ch.get("text", "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text)
        if key not in seen:
            seen.add(key)
            unique.append(ch)
    return unique

# --------------------------
# Boost name/title matches
# --------------------------
def boost_name_chunks(results):
    boosted = []
    for r in results:
        txt = r.get("text", "").lower()
        if any(keyword in txt for keyword in ["lord mayor", "councillor", "dr", "ms", "mr"]):
            r["_score"] = r.get("_score", 1.0) * 1.5
        boosted.append(r)
    return sorted(boosted, key=lambda x: -x.get("_score", 1.0))

# --------------------------
# Chat State
# --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------
# Sidebar Settings
# --------------------------
st.sidebar.header("⚙️ Settings")
embed_model_choice = st.sidebar.selectbox("Embedding model", list(embedding_models.keys()), index=0)
top_k = st.sidebar.slider("Top-K Chunks", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_new_tokens = st.sidebar.slider("Max New Tokens", 50, 1000, 300)
use_context = st.sidebar.checkbox("Restrict to PDF context only", value=True)

if st.sidebar.button("Reset Conversation"):
    st.session_state.messages = []

# --------------------------
# Input Box
# --------------------------
user_input = st.chat_input("Ask a question about the council meeting PDFs...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

# --------------------------
# Conversation Loop
# --------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]

    # Embed query
    embedder = embedding_models[embed_model_choice]
    query_embedding = embedder.encode(query).tolist()

    # Search LanceDB
    results = table.search(query_embedding, vector_column_name=VECTOR_COL).limit(top_k).to_list()
    results = deduplicate_chunks(results)
    results = boost_name_chunks(results)

    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No context found."

    # Build prompt
    if use_context and context != "No context found.":
        prompt = f"Answer the question strictly using the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    else:
        prompt = f"Question: {query}\nAnswer:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0.0),
        temperature=temperature if temperature > 0.0 else 1.0,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Trim echo
    if "Answer:" in response:
        response = response.split("Answer:", 1)[-1].strip()

    st.session_state.messages.append({"role": "bot", "content": response})
    st.markdown(f"<div class='bot-bubble'>{response}</div>", unsafe_allow_html=True)
