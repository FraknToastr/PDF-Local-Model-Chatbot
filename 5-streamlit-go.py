import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import lancedb

# --------------------------
# Streamlit config (must be first!)
# --------------------------
st.set_page_config(
    page_title="PDF Local Model Chatbot",
    page_icon="🏛️",
    layout="wide"
)

# --------------------------
# Load model + tokenizer
# --------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # or your chosen local model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --------------------------
# LanceDB connection
# --------------------------
db = lancedb.connect("./lancedb")
table = db.open_table("pdf_chunks")

# --------------------------
# Streamlit UI
# --------------------------
st.title("🏛️ PDF Local Model Chatbot")
st.markdown("Ask questions based only on your PDF documents.")

user_question = st.text_input("Enter your question:")

if user_question:
    # --------------------------
    # Search LanceDB for relevant chunks
    # --------------------------
    results = table.search(user_question).limit(5).to_list()
    
    if not results:
        st.warning("⚠️ No relevant information found in the PDFs.")
        st.stop()

    # Build context from chunks
    context = "\n\n".join([r["text"] for r in results if "text" in r])

    if not context.strip():
        st.warning("⚠️ I couldn’t find anything relevant in the PDFs.")
        st.stop()

    # --------------------------
    # Force model to only use retrieved context
    # --------------------------
    prompt = f"""
You are a helpful assistant. 
Answer the question only using the following PDF extracts:

{context}

Question: {user_question}

Answer (if not in extracts, say "I couldn’t find this in the PDFs."):
"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.3,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --------------------------
    # Display Answer
    # --------------------------
    st.subheader("💬 Answer")
    st.write(answer.replace(prompt, "").strip())

    # --------------------------
    # Show Sources
    # --------------------------
    with st.expander("📄 Show Sources"):
        for r in results:
            st.markdown(f"**File:** {r.get('source', 'Unknown')} — Page {r.get('page_number', '?')}")
            st.write(r.get("text", ""))
