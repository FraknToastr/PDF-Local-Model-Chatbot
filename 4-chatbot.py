import os
import logging
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import lancedb
from sentence_transformers import SentenceTransformer  # ✅ embedding models

# --- Config ---
DB_DIR = os.path.join("data", "lancedb_data")
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load LanceDB ---
db = lancedb.connect(DB_DIR)
if TABLE_NAME not in db.table_names():
    raise RuntimeError(f"❌ Table {TABLE_NAME} not found in LanceDB. Run script 3 first.")

table = db.open_table(TABLE_NAME)

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Local Model Chatbot", layout="wide")
st.title("📄 PDF Local Model Chatbot")

with st.sidebar:
    st.header("⚙️ Settings")

    # Embedding model selection
    embedding_model_id = st.selectbox(
        "Choose embedding model",
        [
            "mixedbread-ai/mxbai-embed-large-v1",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
    )

    # LLM selection
    model_id = st.selectbox(
        "Choose LLM",
        [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "NousResearch/Llama-2-13b-chat-hf",
        ],
        index=0,
    )

    top_k = st.slider("Top-K results", 1, 20, 5)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_new_tokens = st.slider("Max new tokens", 50, 2000, 512, 50)

    show_sources = st.checkbox("Show Sources", value=True)
    debug_mode = st.checkbox("Debug Mode: Show raw chunks", value=False)  # ✅ new

st.write("Ask a question based on the ingested PDFs:")

user_query = st.text_input("Your question:")

# --- Model Loading (lazy) ---
@st.cache_resource
def load_model(model_id: str):
    logger.info(f"Loading model {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model

if user_query:
    tokenizer, model = load_model(model_id)

    # --- Embed query ---
    logger.info(f"Using embedding model: {embedding_model_id}")
    embedder = SentenceTransformer(embedding_model_id)
    query_embedding = embedder.encode(user_query).tolist()

    # --- Search LanceDB ---
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    if not results:
        st.error("No relevant chunks found in database. Try re-running script 2 and 3.")
    else:
        # --- Build context ---
        context_chunks = [r["text"] for r in results if "text" in r]
        context = "\n\n".join(context_chunks)

        # --- Grounded system prompt ---
        prompt = f"""
You are a PDF chatbot. Only answer using the provided context from PDFs.
If the answer is not in the context, reply: 'I could not find relevant information in the PDFs.'

### Context:
{context}

### Question:
{user_query}

### Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # --- Display Answer ---
        st.subheader("🤖 Answer")
        st.write(answer)

        if show_sources:
            st.subheader("📚 Sources")
            for r in results:
                meta = r.get("metadata", {})
                st.markdown(
                    f"- **{meta.get('source_file', 'Unknown')}**, "
                    f"Page {meta.get('page_number', '?')} "
                    f"(Date: {meta.get('date', 'N/A')})"
                )

        # --- Debug: show raw chunks ---
        if debug_mode:
            st.subheader("🛠️ Debug: Raw Retrieved Chunks")
            for i, r in enumerate(results, start=1):
                st.markdown(f"**Chunk {i}:**")
                st.code(r.get("text", "")[:1000])  # limit preview to 1000 chars
