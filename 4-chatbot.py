import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModel, TextIteratorStreamer
from typing import List
import threading
import gc

# --- Config ---
DB_URI = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"  # name of the embedding column in LanceDB

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

# --- Schema check ---
schema_fields = [f.name for f in table.schema]
if VECTOR_COL not in schema_fields:
    st.error(f"❌ No '{VECTOR_COL}' column found in LanceDB table. Schema fields: {schema_fields}")
    st.stop()

# --- Helper to stop generation ---
def stop_generation(thread: threading.Thread):
    st.session_state.stop_requested = True
    if thread.is_alive():
        del thread
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    st.warning("⏹ Generation stopped by user")

# --- Helper to clear chat ---
def clear_chat():
    st.session_state.user_query = ""
    st.session_state.stop_requested = False
    st.session_state.answer_text = ""
    st.session_state.sources = []
    st.session_state.progress = 0
    st.session_state.chat_history = []
    st.experimental_rerun()

# --- Chatbot UI ---
st.title("📄 PDF Local Model Chatbot")

# Initialize session state
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""
if "sources" not in st.session_state:
    st.session_state.sources = []
if "progress" not in st.session_state:
    st.session_state.progress = 0
if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Chat"):
        clear_chat()
with col2:
    if st.button("⏹ Stop Generation"):
        st.session_state.stop_requested = True

# Input
user_query = st.text_input("Ask a question about the documents:", value=st.session_state.user_query)

if user_query:
    st.session_state.user_query = user_query
    st.session_state.stop_requested = False

    # Progress bar
    progress = st.progress(0, text="🔎 Embedding query...")

    # Step 1: Embed query
    query_embedding = embed_func([user_query])[0]
    progress.progress(25, text="📂 Searching LanceDB...")

    # Step 2: Retrieve relevant chunks (explicit vector col)
    results = table.search(query_embedding, vector_column_name=VECTOR_COL).limit(top_k).to_list()
    st.session_state.sources = results
    progress.progress(50, text="📝 Preparing context...")

    # Build context string with citations
    context_parts = []
    for idx, r in enumerate(results, 1):
        context_parts.append(f"[S{idx}] {r['text']}")
    context = "\n\n".join(context_parts)

    # Step 3: Prepare generation
    prompt = (
        f"Answer the question based only on the context below. "
        f"When you use information from a source, cite it with [S#].\n\n"
        f"{context}\n\nQuestion: {user_query}\nAnswer:"
    )

    streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        inputs=chat_tokenizer(prompt, return_tensors="pt").to(device),
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
    )

    gen_thread = threading.Thread(target=chat_model.generate, kwargs=generation_kwargs)
    gen_thread.start()

    progress.progress(75, text="🤖 Generating answer...")

    # --- Streaming output ---
    st.subheader("💬 Answer")
    answer_placeholder = st.empty()
    streamed_text = ""

    for new_text in streamer:
        if st.session_state.stop_requested:
            break
        streamed_text += new_text
        st.session_state.answer_text = streamed_text
        answer_placeholder.markdown(streamed_text)

    progress.progress(100, text="✅ Done!")

    # Save into chat history
    if not st.session_state.stop_requested:
        st.session_state.chat_history.append({
            "question": user_query,
            "answer": st.session_state.answer_text,
            "sources": st.session_state.sources
        })

# --- Chat history panel ---
if st.session_state.chat_history:
    st.subheader("📜 Chat History")
    for i, turn in enumerate(st.session_state.chat_history, 1):
        with st.expander(f"Q{i}: {turn['question'][:60]}..."):
            st.markdown(f"**Q{i}:** {turn['question']}")
            st.markdown(f"**A{i}:** {turn['answer']}")
            if turn["sources"]:
                st.markdown("**Sources:**")
                for idx, r in enumerate(turn["sources"], 1):
                    meta_lines = [f"🔖 S{idx} → {r.get('source_file','Unknown file')}"]
                    for k, v in r.items():
                        if k not in [VECTOR_COL, "text", "source_file"] and v is not None:
                            meta_lines.append(f"- **{k}**: {v}")
                    st.markdown("\n".join(meta_lines))
