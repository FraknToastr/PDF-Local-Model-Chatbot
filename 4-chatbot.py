import os
import logging
import lancedb
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Config ---
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "lancedb_data")
TABLE_NAME = "adelaide_agendas"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Sidebar ---
st.sidebar.image(LOGO_URL, width=200)
st.sidebar.title("⚙️ Settings")

embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0,
)

top_k = st.sidebar.slider("Top-K results", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 50, 500, 200)
suppress_context = st.sidebar.checkbox("Suppress context display", value=True)

if st.sidebar.button("Reset Conversation"):
    st.session_state["history"] = []
    st.rerun()

# --- Title ---
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
        <img src="{LOGO_URL}" alt="City of Adelaide" width="60">
        <h1 style="color: white; margin: 0;">Council Meeting Chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Connect DB ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Load models ---
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer(embedding_model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        torch_dtype="auto",
        device_map="auto"
    )
    return embed_model, tokenizer, model

import torch
embed_model, tokenizer, model = load_models()

# --- Chat input ---
user_query = st.chat_input("Ask about Adelaide Council meetings...")

if "history" not in st.session_state:
    st.session_state["history"] = []

if user_query:
    # Embed query
    query_embedding = embed_model.encode(user_query).tolist()
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    # Build context string
    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No context found."

    # Prompt
    prompt = f"""Answer factually based only on the following context:
{context}

Question: {user_query}
Answer:"""

    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.1),
    )
    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Extract only the answer part ---
    response = None
    if "Answer:" in raw_response:
        response = raw_response.split("Answer:")[-1].strip()
        response = "Answer: " + response  # keep prefix
    else:
        # fallback: take the last line
        sentences = raw_response.strip().split("\n")
        if sentences:
            response = sentences[-1].strip()
        else:
            response = raw_response.strip()

    # Save clean Q&A only
    st.session_state["history"].append(("user", user_query))
    st.session_state["history"].append(("bot", response))

# --- Display history ---
for role, text in st.session_state["history"]:
    if role == "user":
        st.markdown(
            f"<div style='background-color: #1a1c23; color: white; padding: 10px; border-radius: 10px; float: right; clear: both; margin: 5px 0; max-width: 70%; text-align: right;'>{text}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background-color: #0b5ed7; color: white; padding: 10px; border-radius: 10px; float: left; clear: both; margin: 5px 0; max-width: 70%;'>{text}</div>",
            unsafe_allow_html=True,
        )
