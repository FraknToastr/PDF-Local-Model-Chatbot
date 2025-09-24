import streamlit as st
import torch
import lancedb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
LOGO_PATH = "data/logo.png"  # <-- put your white transparent logo here

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Dark Theme CSS + Chat Bubbles ---
st.markdown(
    """
    <style>
    body {
        background-color: #121212;
        color: #e0e0e0;
    }
    .user-bubble {
        background-color: #1E88E5;
        color: white;
        padding: 10px;
        border-radius: 12px;
        margin: 5px;
        text-align: right;
        float: right;
        clear: both;
        max-width: 70%;
    }
    .bot-bubble {
        background-color: #333333;
        color: #f0f0f0;
        padding: 10px;
        border-radius: 12px;
        margin: 5px;
        text-align: left;
        float: left;
        clear: both;
        max-width: 70%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Logo and Title ---
if LOGO_PATH:
    st.image(LOGO_PATH, use_column_width=False, width=150)
st.title("Council Meeting Chatbot")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")

embedding_model_name = st.sidebar.selectbox(
    "Embedding Model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0
)

llm_model_name = st.sidebar.selectbox(
    "LLM Model",
    ["mistralai/Mistral-7B-Instruct-v0.2", "HuggingFaceH4/zephyr-7b-beta"],
    index=0
)

top_k = st.sidebar.slider("Top-K (context chunks)", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 50, 800, 300)

filter_boilerplate = st.sidebar.checkbox("Filter boilerplate context", value=True)

reset_chat = st.sidebar.button("🗑️ Reset conversation")

# --- Helper: Build Context ---
def build_context(results, suppress_boilerplate=True):
    seen = set()
    context_parts = []

    boilerplate_phrases = [
        "We pray for wisdom, courage, empathy",  # Lord Mayor’s prayer
        "whilst seeking and respecting the opinions of others"
    ]

    for r in results:
        text = r.get("text", "").strip()
        if not text or text in seen:
            continue
        if suppress_boilerplate and any(bp in text for bp in boilerplate_phrases):
            continue
        seen.add(text)
        context_parts.append(text)

    return "\n\n".join(context_parts) if context_parts else "No context found."

# --- Load Models ---
@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

@st.cache_resource
def load_llm(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

embedder = load_embedding_model(embedding_model_name)
tokenizer, model = load_llm(llm_model_name)

# --- Load DB ---
db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

# --- Session State ---
if "history" not in st.session_state or reset_chat:
    st.session_state.history = []

# --- Chat UI ---
user_input = st.chat_input("Ask me about Council Meeting documents...")

if user_input:
    # Embed query
    query_embedding = embedder.encode(user_input).tolist()

    # Vector search
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    # Build context
    context = build_context(results, suppress_boilerplate=filter_boilerplate)

    # Prompt
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip prompt from response
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()

    # Save to history
    st.session_state.history.append({"role": "user", "text": user_input})
    st.session_state.history.append({"role": "bot", "text": response})

# --- Render Chat ---
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['text']}</div>", unsafe_allow_html=True)
