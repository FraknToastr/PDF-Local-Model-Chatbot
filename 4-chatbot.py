import os
import streamlit as st
import lancedb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"

# --- UI Setup ---
st.set_page_config(page_title="Council Meeting Chatbot", page_icon=LOGO_URL, layout="centered")

# Custom CSS for dark theme + chat bubbles
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #ffffff;
    }
    .user-bubble {
        background-color: #2e7d32;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: right;
    }
    .bot-bubble {
        background-color: #424242;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Controls ---
st.sidebar.image(LOGO_URL, use_column_width=True)
st.sidebar.title("⚙️ Settings")

embedding_model_name = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0,
)

generation_model_name = st.sidebar.selectbox(
    "Generation Model",
    ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
    index=0,
)

top_k = st.sidebar.slider("Top-K (context chunks)", 1, 20, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7, 0.1)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 512, 64)
suppress_boilerplate = st.sidebar.checkbox("Suppress ceremonial text", value=True)

# --- Load DB ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Load Models ---
@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_generation_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype=torch.float16)
    return tokenizer, model

embedder = load_embedding_model(embedding_model_name)
tokenizer, model = load_generation_model(generation_model_name)

# --- Deduplication Helper ---
def clean_text(text):
    ceremonial_phrases = [
        "Council acknowledges that we are meeting on traditional Country",
        "We recognise and respect their cultural heritage",
        "Minute's silence in memory",
        "Council Pledge",
    ]
    for phrase in ceremonial_phrases:
        if suppress_boilerplate and phrase.lower() in text.lower():
            return ""
    return text.strip()

# --- Chat UI ---
st.image(LOGO_URL, width=120)
st.title("Council Meeting Chatbot")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_query = st.chat_input("Ask a question about the council meeting agendas...")

if user_query:
    # Embed query
    query_embedding = embedder.encode(user_query).tolist()

    # Search LanceDB (but don’t dump results to screen)
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
    context = "\n\n".join([clean_text(r.get("text", "")) for r in results if r.get("text")]) or "No relevant context found."

    # Build prompt (internal use only)
    prompt = f"Answer factually based only on the following context:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.1),
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save only Q&A to history
    st.session_state["history"].append(("user", user_query))
    st.session_state["history"].append(("bot", response))

# --- Display Chat ---
for role, text in st.session_state["history"]:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{text}</div>", unsafe_allow_html=True)
