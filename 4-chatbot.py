import os
import streamlit as st
import lancedb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# --- UI ---
st.title("📄 PDF RAG Chatbot")
st.write("Ask questions based on the loaded PDF content.")

# --- Model selectors ---
embedding_model_name = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0,
)
llm_model_name = st.sidebar.selectbox(
    "LLM model",
    ["mistralai/Mistral-7B-Instruct-v0.2", "meta-llama/Llama-2-7b-chat-hf"],
    index=0,
)

top_k = st.sidebar.slider("Top-K results", 1, 10, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 50, 1000, 300)

# --- Load embedding model ---
embedder = SentenceTransformer(embedding_model_name)

# --- Connect to LanceDB ---
db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

# --- Load LLM ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
).to(device)

# --- Chat input ---
query = st.text_input("💬 Enter your question")

if query:
    # Embed query
    query_embedding = embedder.encode(query).tolist()

    # Search LanceDB
    results = table.search(query_embedding, vector_column_name="vector").limit(top_k).to_list()

    # ✅ Schema-safe context assembly
    context = "\n\n".join(
        [f"[{r.get('source_file')}, page {r.get('page_number')}] {r.get('text','')}" for r in results]
    ) if results else "No context found."

    # Display retrieved context for debugging
    with st.expander("Show Retrieved Context"):
        st.write(context)

    # Build prompt
    prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown("### 🤖 Answer")
    st.write(answer)
