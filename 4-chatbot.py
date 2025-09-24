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
    "Embedding model (for queries)",
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

# --- Detect GPU ---
if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🚀 Using GPU: {gpu_name}")
else:
    device = "cpu"
    st.sidebar.warning("⚠️ Using CPU")

# --- Load embedding model dynamically ---
@st.cache_resource
def load_embedder(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    ).to(device)
    return tokenizer, model

embed_tokenizer, embed_model = load_embedder(embed_model_choice)

def embed_func(texts: List[str]):
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        embeddings = embed_model(**inputs).last_hidden_state[:, 0, :]
    return embeddings.cpu().to(torch.float32).numpy()

# --- Load chat model dynamically ---
@st.cache_resource
def load_chat_model(model_id: str):
    chat_pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return chat_pipe

chat_pipe = load_chat_model(chat_model_choice)

# --- Connect to LanceDB ---
db = lancedb.connect(DB_URI)
if TABLE_NAME not in db.table_names():
    st.error("❌ No table found in LanceDB. Run Script 3 to build the database.")
    st.stop()

table = db.open_table(TABLE_NAME)

# --- Chatbot UI ---
st.title("📄 PDF Local Model Chatbot")

user_query = st.text_input("Ask a question about the documents:")
if user_query:
    # Embed query
    query_embedding = embed_func([user_query])[0]

    # Retrieve relevant chunks
    results = table.search(query_embedding).limit(top_k).to_list()

    # Build context string
    context = "\n\n".join([r["text"] for r in results])

    # Run chat model
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"
    response = chat_pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.3)[0]["generated_text"]

    # Display response
    st.subheader("💬 Answer")
    st.write(response)

    # Show sources
    with st.expander("📚 Sources"):
        for r in results:
            st.markdown(f"- {r['metadata'].get('source_file', 'Unknown source')}")
