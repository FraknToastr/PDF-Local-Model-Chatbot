import os
import logging
import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# Logo (try SVG first, fallback to PNG if it fails)
LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"
LOCAL_LOGO = "data/logo.png"

# --- Streamlit UI ---
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="https://www.cityofadelaide.com.au/favicon.ico",  # City of Adelaide favicon
    layout="centered",
)

# Dark theme styling
st.markdown(
    """
    <style>
    body { background-color: #1E1E1E; color: white; }
    .user-bubble {
        background-color: #2E86C1;
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-bubble {
        background-color: #333333;
        color: #F1F1F1;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Try to display the SVG logo, fallback to local PNG if needed
try:
    st.image(LOGO_URL, use_column_width=False, width=150, output_format="SVG")
except Exception:
    if os.path.exists(LOCAL_LOGO):
        st.image(LOCAL_LOGO, use_column_width=False, width=150)
    else:
        st.write("⚠️ Logo missing")

st.title("Council Meeting Chatbot")

# --- Load DB ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Load Embedding Model ---
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Load LLM ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
st.sidebar.header("Chat Controls")
top_k = st.sidebar.slider("Top-K Results", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256)
suppress_context = st.sidebar.checkbox("Suppress PDF context", value=False)

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.messages = []
    st.rerun()

# --- Chat Input ---
user_input = st.chat_input("Ask a question about Council Meetings...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Encode query
    query_embedding = embedder.encode(user_input).tolist()

    # Search DB unless context is suppressed
    context = ""
    if not suppress_context:
        results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
        if results:
            context = "\n\n".join([r.get("text", "") for r in results if "text" in r])
        else:
            context = "No context found."

    # Construct prompt
    prompt = f"Answer the question based only on the council documents.\n\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=temperature,
    )
    bot_reply = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Append bot reply
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

# --- Display Messages ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
