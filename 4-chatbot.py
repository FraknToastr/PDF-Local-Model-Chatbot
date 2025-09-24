import os
import logging
import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Must be the first Streamlit command
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="🏛️",
    layout="wide"
)

# --- Dark theme background ---
st.markdown(
    """
    <style>
        .stApp {
            background-color: #111111;
            color: white;
        }
        .user-bubble {
            background-color: #1f77b4;
            color: white;
            padding: 12px;
            border-radius: 12px;
            margin: 6px 0;
            text-align: right;
        }
        .bot-bubble {
            background-color: #444444;
            color: white;
            padding: 12px;
            border-radius: 12px;
            margin: 6px 0;
            text-align: left;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logo + Title ---
st.image("https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg", width=150)
st.title("Council Meeting Chatbot")

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- LanceDB setup ---
DB_PATH = os.path.join("data", "lancedb_data")
TABLE_NAME = "adelaide_agendas"

db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Embedding model ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

# --- LLM model ---
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- Sidebar controls ---
st.sidebar.header("Model Controls")
top_k = st.sidebar.slider("Top-K results", 1, 10, 3)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256)

# ✅ Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state["history"] = []

# --- Deduplication phrases ---
DEDUPLICATE_PATTERNS = [
    "Acknowledgement of Country",
    "The Lord Mayor will state",
    "Minutes silence",
    "Council acknowledges"
]

def deduplicate_chunks(chunks):
    seen = set()
    filtered = []
    for ch in chunks:
        text = ch.get("text", "").strip()
        if any(p in text for p in DEDUPLICATE_PATTERNS):
            key = next((p for p in DEDUPLICATE_PATTERNS if p in text), None)
            if key and key in seen:
                continue
            seen.add(key)
        filtered.append(ch)
    return filtered

# --- Chat state ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- User input ---
query = st.chat_input("Ask about Adelaide Council meetings...")

if query:
    # --- Embed query ---
    query_embedding = embedder.encode([query])[0].tolist()

    # --- Search LanceDB ---
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
    results = deduplicate_chunks(results)

    # --- Build context (used internally, not displayed) ---
    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No relevant context found."

    # --- Generate answer ---
    prompt = f"Answer the question using only the council PDF context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Append to history ---
    st.session_state["history"].append({"user": query, "bot": answer})

# --- Display conversation ---
for h in st.session_state["history"]:
    st.markdown(f"<div class='user-bubble'>{h['user']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-bubble'>{h['bot']}</div>", unsafe_allow_html=True)
