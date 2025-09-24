# 4-chatbot.py
import streamlit as st
import lancedb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

DB_PATH = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"

st.set_page_config(page_title="Council Meeting Chatbot", page_icon="https://www.cityofadelaide.com.au/favicon.ico")

st.markdown(
    """
    <style>
        body { background-color: #121212; color: #fff; }
        .user-bubble {
            background-color: #2c2c34;
            color: #fff;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            float: right;
            clear: both;
        }
        .bot-bubble {
            background-color: #0d47a1;
            color: #fff;
            padding: 10px 15px;
            border-radius: 15px;
            margin: 5px;
            max-width: 70%;
            float: left;
            clear: both;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("https://www.cityofadelaide.com.au/common/base/img/coa-logo-white.svg", width=180)
st.title("Council Meeting Chatbot")

db = lancedb.connect(DB_PATH)
table = db.open_table(TABLE_NAME)

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
gen_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(gen_model_id)
model = AutoModelForCausalLM.from_pretrained(
    gen_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def boosted_search(query, top_k=5):
    query_embedding = embed_model.encode([query])[0]
    results = table.search(query_embedding, vector_column_name=VECTOR_COL).limit(top_k).to_list()

    # Boost with keyword match
    if any(kw in query.lower() for kw in ["who", "name", "lord mayor"]):
        keyword_hits = table.search(query_embedding, vector_column_name=VECTOR_COL).limit(50).to_list()
        keyword_hits = [r for r in keyword_hits if re.search(r"Lord Mayor", r.get("text", ""), re.I)]
        results.extend(keyword_hits)

    # Deduplicate boilerplate
    seen = set()
    filtered = []
    for r in results:
        txt = r.get("text", "")
        if not txt.strip():
            continue
        if any(pat.lower() in txt.lower() for pat in [
            "we pray for wisdom, courage, empathy, understanding and guidance"
        ]):
            continue
        if txt not in seen:
            seen.add(txt)
            filtered.append(r)

    return filtered[:top_k]

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    results = boosted_search(user_input, top_k=5)
    context = "\n\n".join([r.get("text", "") for r in results]) if results else "No context found."

    prompt = f"Answer the following question strictly using the provided context.\n\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, do_sample=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", answer))

if st.button("Reset Chat"):
    st.session_state.chat_history = []

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='user-bubble'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg}</div>", unsafe_allow_html=True)
