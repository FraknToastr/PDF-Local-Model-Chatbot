import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from typing import List
import threading

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

# --- Load embedding model ---
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

# --- Load chat model ---
@st.cache_resource
def load_chat_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    return tokenizer, model

chat_tokenizer, chat_model = load_chat_model(chat_model_choice)

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
    # Reset stop flag at new query
    if "stop_requested" not in st.session_state:
        st.session_state.stop_requested = False
    else:
        st.session_state.stop_requested = False

    # Progress bar
    progress = st.progress(0, text="🔎 Embedding query...")

    # Step 1: Embed query
    query_embedding = embed_func([user_query])[0]
    progress.progress(25, text="📂 Searching LanceDB...")

    # Step 2: Retrieve relevant chunks
    results = table.search(query_embedding).limit(top_k).to_list()
    progress.progress(50, text="📝 Preparing context...")

    # Build context string
    context = "\n\n".join([r["text"] for r in results])

    # Step 3: Prepare generation
    prompt = f"Answer the question based only on the context below:\n\n{context}\n\nQuestion: {user_query}\nAnswer:"

    streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs=chat_tokenizer(prompt, return_tensors="pt").to(device),
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
    )

    thread = threading.Thread(target=chat_model.generate, kwargs=generation_kwargs)
    thread.start()

    progress.progress(75, text="🤖 Generating answer...")

    # --- Streaming output ---
    st.subheader("💬 Answer")
    answer_placeholder = st.empty()
    streamed_text = ""

    # Stop button
    stop_button = st.button("⏹ Stop Generation")

    for new_text in streamer:
        if st.session_state.stop_requested:
            break
        streamed_text += new_text
        answer_placeholder.markdown(streamed_text)

        if stop_button:
            st.session_state.stop_requested = True

    progress.progress(100, text="✅ Done!")

    # Show sources
    with st.expander("📚 Sources"):
        for r in results:
            st.markdown(f"- {r['metadata'].get('source_file', 'Unknown source')}")
