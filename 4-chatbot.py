import os
import logging
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import lancedb
from sentence_transformers import SentenceTransformer

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

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("⚙️ Settings")

    embedding_model_id = st.selectbox(
        "Choose embedding model",
        [
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/multi-qa-mpnet-base-dot-v1",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
        index=0,
    )

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
    temperature = st.slider("Temperature", 0.0, 1.5, 0.1, 0.1)
    max_new_tokens = st.slider("Max new tokens", 50, 2000, 512, 50)

    show_sources = st.checkbox("Show Sources", value=True)
    suppress_context = st.checkbox("🚫 Suppress Context (ignore LanceDB chunks)", value=False)

    # ✅ Reset button
    if st.button("🔄 Reset Conversation"):
        st.session_state.history = []
        st.success("Conversation history cleared.")

st.write("Ask a question based on the ingested PDFs:")

user_query = st.text_input("Your question:")

# --- Model Loader ---
@st.cache_resource
def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model

if user_query:
    tokenizer, model = load_model(model_id)

    if suppress_context:
        context = ""
        results = []
    else:
        embedder = SentenceTransformer(embedding_model_id)
        query_embedding = embedder.encode(user_query).tolist()
        results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()
        context = "\n\n".join([r["text"] for r in results if "text" in r])

    # ✅ Grounded by default
    prompt = f"""
You are a strict PDF chatbot.
Only answer if the provided context contains relevant information from the PDFs. 
If the answer is not in the context, reply exactly: "I could not find relevant information in the PDFs."

### Context:
{context}

### Question:
{user_query}

### Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": (temperature > 0),
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Save to history ---
    st.session_state.history.append({"question": user_query, "answer": answer})

    # --- Display ---
    st.subheader("🤖 Answer")
    st.write(answer)

    if show_sources and not suppress_context:
        st.subheader("📚 Sources")
        for r in results:
            meta = r.get("metadata", {})
            st.markdown(
                f"- **{meta.get('source_file', 'Unknown')}**, "
                f"Page {meta.get('page_number', '?')} "
                f"(Date: {meta.get('date', 'N/A')})"
            )

# --- Show conversation history ---
if st.session_state.history:
    st.subheader("📝 Conversation History")
    for i, entry in enumerate(st.session_state.history, 1):
        st.markdown(f"**Q{i}:** {entry['question']}")
        st.markdown(f"**A{i}:** {entry['answer']}")
