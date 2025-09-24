import os
import logging
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import lancedb
import re

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load DB ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Embedding models ---
EMBED_MODELS = {
    "MiniLM (384d)": "sentence-transformers/all-MiniLM-L6-v2",
    "E5 Large (1024d)": "intfloat/e5-large-v2",
}

# --- Load LLM ---
LLM_ID = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(LLM_ID)
model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Adelaide Agenda Chatbot")

embed_choice = st.selectbox("Choose embedding model", list(EMBED_MODELS.keys()))
embed_model = SentenceTransformer(EMBED_MODELS[embed_choice])

top_k = st.slider("Top-K results", 1, 10, 5)
temperature = st.slider("Temperature", 0.1, 1.5, 0.7, step=0.1)
max_tokens = st.slider("Max new tokens", 64, 1024, 256)

reset_chat = st.button("🔄 Reset conversation")
if reset_chat:
    st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Ask a question:")

def search_db(query_text, boost_entities=True):
    query_embedding = embed_model.encode(query_text).tolist()
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    # --- NER fallback ---
    if (not results or len(results) == 0) and boost_entities:
        logger.info("⚠️ No vector hits, falling back to keyword/entity search")
        entities = re.findall(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", query_text)
        for ent in entities:
            keyword_hits = table.search(ent).limit(top_k).to_list()
            if keyword_hits:
                results = keyword_hits
                break

    return results

if query:
    st.session_state.history.append(("You", query))
    results = search_db(query)

    context = "\n\n".join([r["chunk"]["text"] for r in results]) if results else "No context found."

    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
        top_k=50
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[-1].strip()
    st.session_state.history.append(("Assistant", answer))

# --- Chat history display ---
for speaker, msg in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**🧑 You:** {msg}")
    else:
        st.markdown(f"**🤖 Assistant:** {msg}")
