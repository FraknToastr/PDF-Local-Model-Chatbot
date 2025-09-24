import os
import logging
import streamlit as st
import lancedb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_DIR = os.path.join("data", "lancedb_data")
TABLE_NAME = "adelaide_agendas"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load LanceDB ---
db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    st.error(f"❌ Table '{TABLE_NAME}' not found in LanceDB. Run Script 3 first.")
    st.stop()

table = db.open_table(TABLE_NAME)
schema = table.schema
logger.info(f"📋 Opened table '{TABLE_NAME}' with schema: {schema}")

# --- Streamlit UI ---
st.title("📄 PDF Local Model Chatbot")

with st.sidebar:
    st.header("⚙️ Settings")
    embedding_model_name = st.selectbox(
        "Embedding model",
        ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
        index=0 if DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2" else 1,
    )
    top_k = st.slider("Top-K Results", 1, 20, 5)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
    max_new_tokens = st.slider("Max New Tokens", 64, 1024, 256)
    suppress_context = st.checkbox("Suppress context (ignore PDFs)", value=False)
    reset_chat = st.button("Reset Conversation")

# --- Session State ---
if "history" not in st.session_state or reset_chat:
    st.session_state.history = []

# --- Device selection ---
if torch.cuda.is_available():
    device = "cuda"
    logger.info(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    logger.warning("⚠️ No GPU detected, using CPU")

# --- Load embeddings ---
embedder = SentenceTransformer(embedding_model_name, device=device)

# --- Load LLM ---
logger.info(f"Loading LLM: {LLM_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

# --- Chatbot loop ---
user_input = st.text_input("💬 Ask a question:")
if user_input:
    # Embed query
    query_embedding = embedder.encode(user_input, convert_to_numpy=True).tolist()

    # Context retrieval
    if not suppress_context:
        try:
            results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
            context = "\n\n".join(
                [r.get("text", "") for r in results if isinstance(r, dict) and "text" in r]
            ) if results else "No context found."
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            context = "No context (vector search failed)."
    else:
        context = ""

    # Build prompt
    prompt = f"Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=(temperature > 0.0),
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save history
    st.session_state.history.append((user_input, answer))

# --- Show conversation ---
st.subheader("📝 Conversation")
for i, (q, a) in enumerate(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")
