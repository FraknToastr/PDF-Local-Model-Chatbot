import os
import json
import streamlit as st
import torch
import lancedb
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Paths ---
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "lancedb_data")
META_FILE = os.path.join(DB_PATH, "embedding_metadata.json")

# --- Load LanceDB ---
db = lancedb.connect(DB_PATH)
TABLE_NAME = "adelaide_agendas"
table = db.open_table(TABLE_NAME)

# --- Load embedding metadata ---
embedding_model = "unknown"
embedding_dim = "unknown"
if os.path.exists(META_FILE):
    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedding_model = meta.get("model_name", "unknown")
    embedding_dim = meta.get("embedding_dim", "unknown")

# --- Sidebar UI ---
st.sidebar.header("⚙️ Settings")
st.sidebar.markdown(f"**Embedding model:** `{embedding_model}`")
st.sidebar.markdown(f"**Vector dimension:** `{embedding_dim}`")

# Debug info
st.sidebar.markdown("### Debug info")
try:
    row_count = table.count_rows()
    st.sidebar.markdown(f"**Rows in DB:** {row_count}")
    st.sidebar.markdown(f"**Schema:** {table.schema}")
except Exception as e:
    st.sidebar.error(f"DB error: {e}")

# --- Chat model selection ---
st.sidebar.markdown("### Chat model")
chat_model = st.sidebar.selectbox(
    "Choose a chat model",
    ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
    index=0
)

# --- Load LLM ---
@st.cache_resource
def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_model(chat_model)

# --- Conversation State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

def reset_conversation():
    st.session_state["messages"] = []

st.sidebar.button("🧹 Reset conversation", on_click=reset_conversation)

# --- Main Chat UI ---
st.title("📑 PDF Local Model Chatbot")

user_input = st.chat_input("Ask me a question about the PDFs...")
if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # TODO: Retrieve context from LanceDB (currently disabled)
    context = ""

    # Build prompt
    prompt = f"Context:\n{context}\n\nUser: {user_input}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False  # Greedy decoding for factual consistency
        )

    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.session_state["messages"].append({"role": "assistant", "content": reply})

# --- Display Messages ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
