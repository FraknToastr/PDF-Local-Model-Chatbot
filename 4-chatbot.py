import os
import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
APP_TITLE = "Council Meeting Chatbot"
LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"

# --- Streamlit Setup ---
st.set_page_config(page_title=APP_TITLE, page_icon="🏛️", layout="wide")

# Sidebar settings
st.sidebar.image(LOGO_URL, use_column_width=True)
st.sidebar.title("Settings")

embedding_model_choice = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0
)

top_k = st.sidebar.slider("Top-K Results", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256)

# Suppress context option
suppress_context = st.sidebar.checkbox("Suppress context in responses", value=True)

# Reset button
if st.sidebar.button("Reset Conversation"):
    if "messages" in st.session_state:
        del st.session_state["messages"]
    st.rerun()

# --- Title and Logo ---
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:15px;">
        <img src="{LOGO_URL}" alt="City of Adelaide" width="140">
        <h1 style="margin:0;">{APP_TITLE}</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Load Models ---
@st.cache_resource
def load_embedder(model_name):
    return SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_chat_model():
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return tokenizer, model

embedder = load_embedder(embedding_model_choice)
tokenizer, model = load_chat_model()

# --- Database ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Deduplication / Filtering ---
def filter_context(chunks):
    cleaned = []
    for r in chunks:
        text = r.get("text", "")
        # Remove boilerplate ceremonial lines
        if any(phrase in text for phrase in [
            "The Lord Mayor will state",
            "Acknowledgement of Country",
            "prayer",
            "silence",
            "closure",
        ]):
            continue
        cleaned.append(r)
    return cleaned

def boost_member_chunks(chunks):
    """Move member list chunks to the front if available"""
    member_chunks = [c for c in chunks if c.get("metadata", {}).get("is_member_list")]
    other_chunks = [c for c in chunks if not c.get("metadata", {}).get("is_member_list")]
    return member_chunks + other_chunks if member_chunks else chunks

# --- Chat Logic ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_query = st.chat_input("Ask about the Council Meeting PDFs...")
if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])

if st.session_state["messages"] and st.session_state["messages"][-1]["role"] == "user":
    query = st.session_state["messages"][-1]["content"]

    # Embed and search
    query_embedding = embedder.encode([query])[0].tolist()
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    # Filter + boost
    results = filter_context(results)
    results = boost_member_chunks(results)

    context = "\n\n".join([r["text"] for r in results]) if results else "No context found."

    # Strict prompt
    prompt = f"""
    You are a factual assistant. Answer strictly based only on the council PDF context below.

    Context:
    {context}

    Question: {query}
    Answer: (reply using only names/info from the context. Do not define or explain general roles.)
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only model answer
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()

    # Save + show in separate bubble
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)
