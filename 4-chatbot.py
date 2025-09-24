import os
import re
import streamlit as st
import lancedb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# -------------------------------
# Config
# -------------------------------
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOGO_URL = "https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg"

# -------------------------------
# Suppression rules
# -------------------------------
CEREMONIAL_PATTERNS = [
    r"The Lord Mayor will state:.*?(First Nations who are present today\.)",  # Acknowledgement of Country
    r"© 20\d{2} City of Adelaide\. All Rights Reserved\.",                    # Copyright footer
    r"Our Adelaide Bold\. Aspirational\. Innovative\.",                       # Motto
    r"The Lord Mayor’s Prayer:.*?(Amen)",                                     # Prayer
    r"Pledge of Loyalty:.*?(loyalty to Australia and to the people of South Australia\.)",
    r"Council will observe a Memorial Silence.*?(?:\n|\Z)",                   # Memorial Silence
]

def strip_ceremonial(text: str) -> str:
    """Remove repeated ceremonial/boilerplate sections from a chunk."""
    for pat in CEREMONIAL_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()

# -------------------------------
# Load DB + models
# -------------------------------
db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

embedder = SentenceTransformer(EMBED_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# -------------------------------
# UI
# -------------------------------
st.set_page_config(
    page_title="Council Meeting Chatbot",
    page_icon="https://www.cityofadelaide.com.au/favicon.ico",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: white; }
    .user-bubble {
        background-color: #2E86C1; padding: 10px; border-radius: 10px; margin: 5px; color: white;
    }
    .bot-bubble {
        background-color: #1B2631; padding: 10px; border-radius: 10px; margin: 5px; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image(LOGO_URL, width=180)
st.markdown("<h1 style='text-align: center;'>Council Meeting Chatbot</h1>", unsafe_allow_html=True)

# -------------------------------
# Chat state
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask about Adelaide council meetings...")

if user_input:
    # Embed query
    query_embedding = embedder.encode(user_input).tolist()
    results = table.search(query_embedding, vector_column_name="vector").limit(5).to_list()

    # Build context (with suppression)
    context = "\n\n".join(
        [strip_ceremonial(r["text"]) for r in results if "text" in r]
    ) if results else "No context found."

    # Generate response
    prompt = f"Context:\n{context}\n\nQuestion: {user_input}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Update history
    st.session_state.history.append((user_input, answer))

# -------------------------------
# Display conversation
# -------------------------------
for q, a in st.session_state.history:
    st.markdown(f"<div class='user-bubble'>You: {q}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='bot-bubble'>Bot: {a}</div>", unsafe_allow_html=True)
