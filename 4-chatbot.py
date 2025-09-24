import os
import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="🗂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject dark theme CSS and chat bubbles
st.markdown(
    """
    <style>
    body, .stApp { background-color: #0e1117; color: #f0f2f6; }
    .user-bubble {
        background-color: #1e2130;
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 6px 0px;
        max-width: 80%;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #004AAD;
        color: white;
        padding: 12px;
        border-radius: 12px;
        margin: 6px 0px;
        max-width: 80%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header with Logo ---
col1, col2 = st.columns([0.15, 0.85])
with col1:
    st.image(LOGO_URL, use_column_width=True)
with col2:
    st.title("Council Meeting Chatbot")

st.write("Ask questions about the City of Adelaide council meeting documents.")

# --- Load DB ---
db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

# --- Load Embedding Model ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedder = load_embedder()

# --- Load LLM ---
@st.cache_resource
def load_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_llm()

# --- Load NER model for fallback ---
@st.cache_resource
def load_ner():
    return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

ner_pipeline = load_ner()

# --- Chat State ---
if "history" not in st.session_state:
    st.session_state["history"] = []

# --- Chat UI ---
with st.container():
    for entry in st.session_state["history"]:
        role, text = entry
        if role == "user":
            st.markdown(f"<div class='chat-container'><div class='user-bubble'>{text}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-container'><div class='bot-bubble'>{text}</div></div>", unsafe_allow_html=True)

# --- Input ---
query = st.chat_input("Type your question here...")
if query:
    st.session_state["history"].append(("user", query))

    # --- Embed Query ---
    query_embedding = embedder.encode([query])[0]

    # --- Search Vector DB ---
    results = table.search(query_embedding, vector_column_name="vector").limit(5).to_list()

    # --- Fallback: if no results or weak matches ---
    if not results:
        ents = ner_pipeline(query)
        keywords = [ent["word"] for ent in ents if ent["entity_group"] in ["PER", "ORG", "LOC"]]
        if keywords:
            keyword = keywords[0]
            results = table.search(query_embedding, vector_column_name="vector").filter(f"text CONTAINS '{keyword}'").limit(5).to_list()

    # --- Build Context ---
    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No context found."

    # --- Generate Answer ---
    inputs = tokenizer(
        f"Answer the following question strictly using the provided context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:",
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
        top_k=50
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- Append Bot Response ---
    st.session_state["history"].append(("bot", answer))

    # --- Rerun to update UI ---
    st.rerun()
