import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from typing import List

# --- Config ---
DB_URI = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# --- Sidebar controls ---
st.sidebar.header("⚙️ Settings")
top_k = st.sidebar.slider("Number of sources to retrieve", min_value=1, max_value=10, value=3, step=1)

embed_model_choice = st.sidebar.selectbox(
    "Embedding model",
    options=[
        "mixedbread-ai/mxbai-embed-large-v1",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ],
    index=0
)

chat_model_choice = st.sidebar.selectbox(
    "Chat model",
    options=[
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "tiiuae/falcon-7b-instruct"
    ],
    index=0
)

# --- Load embedding model dynamically ---
@st.cache_resource
def load_embedder(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    return tokenizer, model

embed_tokenizer, embed_model = load_embedder(embed_model_choice)

def embed_func(texts: List[str]):
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(embed_model.device)
    with torch.no_grad():
        embeddings = embed_model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()

# --- Load chat model dynamically ---
@st.cache_resource
def load_chat_model(model_id: str):
    chat_pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return chat_pipe

chat_pipe = load_chat_model(chat_model_choice)

# --- Connect to LanceDB ---
db = lancedb.connect(DB_URI)
if TABLE_NAME not in db.table_names():
    raise RuntimeError(f"Table '{TABLE_NAME}' not found. Run 2-build-lancedb.py first.")
table = db.open_table(TABLE_NAME)

# --- Search helper ---
def search_pdfs(query: str, top_k: int = 5):
    query_vector = embed_func([query])[0]
    return table.search(query=query_vector).limit(top_k).to_list()

# --- Build prompt ---
def build_prompt(query: str, results):
    context_texts = []
    for r in results:
        src = r.get("metadata", {}).get("source_file", "Unknown source")
        context_texts.append(f"[Source: {src}]\n{r['text']}")
    context = "\n\n".join(context_texts)

    prompt = f"""
You are a helpful assistant. Answer the user's question strictly based on the PDF context below.
If the answer is not in the context, say "I could not find that in the PDFs."

Question: {query}

Context:
{context}

Answer with citations (e.g., [Source: filename.pdf]).
"""
    return prompt

# --- Generate answer locally ---
def generate_answer(prompt: str):
    out = chat_pipe(
        prompt,
        max_new_tokens=500,
        temperature=0.2,
        do_sample=False
    )
    return out[0]["generated_text"][len(prompt):].strip()

# --- Streamlit UI ---
st.set_page_config(page_title="PDF-grounded Chatbot", layout="wide")
st.title("📄 Adelaide PDF Chatbot")
st.write("Ask a question, and I will answer using only the council agenda PDFs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Type your question about the PDFs...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve chunks
    results = search_pdfs(query, top_k=top_k)

    # Build grounded prompt
    prompt = build_prompt(query, results)

    # Generate local answer
    answer = generate_answer(prompt)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})

# --- Display chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📂 Show Sources"):
                for r in msg["sources"]:
                    src = r.get("metadata", {}).get("source_file", "Unknown source")
                    preview = r["text"][:300].replace("\n", " ") + ("..." if len(r["text"]) > 300 else "")
                    st.markdown(f"**Source:** {src}")
                    st.write(preview)
                    st.markdown("---")
