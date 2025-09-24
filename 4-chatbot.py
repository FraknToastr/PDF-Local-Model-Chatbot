import streamlit as st
import lancedb
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch

# --- Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"

# --- Load DB ---
db = lancedb.connect(DB_DIR)
table = db.open_table(TABLE_NAME)

# --- Load Models ---
@st.cache_resource
def load_models(embed_model_id, lm_model_id):
    embedder = SentenceTransformer(embed_model_id)
    tokenizer = AutoTokenizer.from_pretrained(lm_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        lm_model_id,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    return embedder, tokenizer, model

# --- UI ---
st.title("📄 Adelaide Agenda Chatbot")
st.sidebar.header("⚙️ Settings")

embed_model_id = st.sidebar.selectbox(
    "Embedding model",
    ["sentence-transformers/all-MiniLM-L6-v2", "mixedbread-ai/mxbai-embed-large-v1"],
    index=0,
)
lm_model_id = st.sidebar.text_input("Language Model ID", "mistralai/Mistral-7B-Instruct-v0.2")

top_k = st.sidebar.slider("Top-K Results", 1, 20, 5)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7)
max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 256)

suppress_context = st.sidebar.checkbox("Suppress context (ignore PDFs)?", value=False)

# Load models
embedder, tokenizer, model = load_models(embed_model_id, lm_model_id)

# Conversation state
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.messages = []

# --- Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about Adelaide Agendas...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            context = "No context found."
            if not suppress_context:
                # Embed query
                query_embedding = embedder.encode(user_input).tolist()

                # LanceDB search
                results = table.search(query_embedding, vector_column_name="vector").limit(50).to_list()

                # --- NEW: Keyword boosting fallback ---
                keywords = ["lord mayor", "councillor", "ceo", "mayor"]
                boosted = []
                for r in results:
                    text = r.get("text", "").lower()
                    score = 1.0  # baseline
                    for kw in keywords:
                        if kw in user_input.lower() and kw in text:
                            score += 5.0  # boost heavily
                    boosted.append((score, r))

                boosted.sort(key=lambda x: x[0], reverse=True)
                top_results = [r for _, r in boosted[:top_k]]

                context = "\n\n".join([r.get("text", "") for r in top_results]) if top_results else "No context found."

            # Build prompt
            prompt = f"Answer the question based only on the context.\n\nContext:\n{context}\n\nQuestion: {user_input}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0.0),
            )
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
