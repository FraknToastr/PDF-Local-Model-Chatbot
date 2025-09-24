import streamlit as st
import lancedb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config ---
DB_DIR = "data/lancedb_data"
TABLE_NAME = "adelaide_agendas"
VECTOR_COL = "vector"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"  # local instruct model
EMBED_MODEL_ID = "mixedbread-ai/mxbai-embed-large-v1"

# --- Load embedding model ---
@st.cache_resource
def load_embed_model():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    model = AutoModel.from_pretrained(
        EMBED_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

def embed_texts(tokenizer, model, texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return emb

# --- Load LLM model ---
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    return tokenizer, model

def generate_response(llm_tokenizer, llm_model, prompt, max_new_tokens=512):
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    return llm_tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="PDF Chatbot", layout="wide")
    st.title("📄 PDF Local Model Chatbot")

    query = st.text_input("Ask a question about the PDFs:")

    if query:
        # Load embedding model
        embed_tokenizer, embed_model = load_embed_model()
        query_embedding = embed_texts(embed_tokenizer, embed_model, [query])[0]

        # Connect to LanceDB
        db = lancedb.connect(DB_DIR)
        if TABLE_NAME not in db.table_names():
            st.error(f"Table '{TABLE_NAME}' not found in {DB_DIR}. Run Script 3 first.")
            return
        table = db.open_table(TABLE_NAME)

        # Search LanceDB ✅ explicit vector_column_name
        top_k = 5
        results = table.search(query_embedding, vector_column_name=VECTOR_COL).limit(top_k).to_list()

        if not results:
            st.warning("No relevant chunks found in the database.")
            return

        # Build context
        context = "\n\n".join([r["text"] for r in results])
        prompt = f"Answer the question based only on the following context:\n\n{context}\n\nQuestion: {query}\nAnswer:"

        # Generate response
        llm_tokenizer, llm_model = load_llm()
        answer = generate_response(llm_tokenizer, llm_model, prompt)

        # Display
        st.subheader("💡 Answer")
        st.write(answer)

        # Show sources
        with st.expander("📂 Show Sources"):
            for r in results:
                st.markdown(f"- **{r.get('source_file')}** (p{r.get('page_number')}, {r.get('date')})")

if __name__ == "__main__":
    main()
