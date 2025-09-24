import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from typing import List

# --- Config ---
DB_URI = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# --- Sidebar controls ---
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=10, value=3, step=1)

embed_model_choice = st.sidebar.selectbox(
    "Embedding model (for queries)",
    options=[
        "mixedbread-ai/mxbai-embed-large-v1",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ],
    index=0
)

chat_model_choice = st.sidebar.selectbox(
    "Chat model",
    options=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "tiiuae/falcon-7b-instruct"
    ],
    index=0
)

# --- Load embedding model dynamically ---
@st.cache_resource
def load_embedder(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

embed_tokenizer, embed_model = load_embedder(embed_model_choice)

def embed_func(texts: List[str]):
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(embed_model.device)
    with torch.no_grad():
        embeddings = embed_model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()

# --- Load chat model dynamically ---
@st.cache_resource
def load_chat_model(model_id: str):
    chat_pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return chat_pipe

chat_pipe = load_chat_model(chat_model_choice)

# --- Connect to LanceDB ---
db = lancedb.connect(DB_URI)
if TABLE_NAME not in_
