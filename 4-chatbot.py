import os
import logging
import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- Config ---
DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Logo + branding
LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"
APP_TITLE = "Council Meeting Chatbot"

# Boosted councillor names
BOOST_TERMS = [
    "Lord Mayor", "Jane Lomax-Smith", "Councillor",
    "Abrahimzadeh", "Couros", "Davis", "Giles",
    "Martin", "Siebentritt", "Snape", "Cabada"
]

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Load Embedding Model ---
@st.cache_resource
def load_embedder():
    logger.info(f"Load pretrained SentenceTransformer: {DEFAULT_MODEL}")
    model = SentenceTransformer(DEFAULT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    return model

embedder = load_embedder()

# --- Load LanceDB ---
@st.cache_resource
def load_table():
    db = lancedb.connect(DB_PATH)
    return db.open_table(TABLE_NAME)

table = load_table()

# --- Load LLM ---
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

tokenizer, model = load_llm()

# --- UI ---
st.set_page_config(page_title=APP_TITLE, page_icon="🏛️", layout="wide")
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #0e1117;
            color: #fafafa;
        }}
        .user-bubble {{
            background-color: #1e90ff;
            color: white;
            padding: 10px;
            border-radius: 12px;
            margin: 5px 0;
            text-align: right;
            max-width: 80%;
            margin-left: auto;
        }}
        .bot-bubble {{
            background-color: #2c2c2c;
            color: #fafafa;
            padding: 10px;
            border-radius: 12px;
            margin: 5px 0;
            text-align: left;
            max-width: 80%;
            margin-right: auto;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(LOGO_URL, width=180)
st.title(APP_TITLE)

# Reset button
if st.button("Reset Conversation"):
    st.session_state["messages"] = []

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Boosted Search ---
def boosted_search(query, top_k=5):
    q_embed = embedder.encode([query])[0]

    # Normal vector search
    results = table.search(q_embed, vector_column_name="vector").limit(top_k * 2).to_list()

    # Boost scoring
    boosted_results = []
    for r in results:
        score = r["_distance"] * -1  # LanceDB returns distance, lower is better
        text = r.get("text", "")
        meta = r.get("metadata", {})

        # Boost councillor/member list
        if meta.get("is_member_list"):
            score += 2.0

        # Boost if query contains councillor names
        for term in BOOST_TERMS:
            if term.lower() in query.lower() or term.lower() in text.lower():
                score += 1.0

        boosted_results.append((score, text))

    # Sort by boosted score
    boosted_results.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in boosted_results[:top_k]]

# --- Chat logic ---
def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Chat UI ---
user_query = st.chat_input("Ask about Council Meetings...")

if user_query:
    # Save user bubble
    st.session_state["messages"].append(("user", user_query))

    # Context from boosted search
    hits = boosted_search(user_query, top_k=5)
    context = "\n\n".join(hits) if hits else "No relevant context found in the agendas."

    # Strict grounding
    prompt = f"Answer ONLY using the following council agenda context. If unclear, say 'I don’t know based on the available documents.'\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"

    answer = generate_answer(prompt)

    # Save bot bubble
    st.session_state["messages"].append(("bot", answer))

# Render conversation
for role, msg in st.session_state["messages"]:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)
