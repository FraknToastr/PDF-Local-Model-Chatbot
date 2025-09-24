🔄 Pipeline Order

Scraper → 1-Agenda-Frontsheet-Extraction.py

Scrapes Adelaide Council meeting pages.

Downloads Agenda Frontsheet PDFs.

Optionally extracts them with Docling.

Outputs PDFs into data/ folder.

Chunker (with overlap) → 3-hybrid-chunking-multiple-PDFs.py

Walks through data/ folder.

Converts each PDF → chunks with Docling HybridChunker.

Expands into overlapping chunks (sliding window).

Saves results into data/all_chunks.pkl.

Embed + Index → 2-build-lancedb.py

Loads chunks from all_chunks.pkl.

Embeds them with Hugging Face embeddings model.

Creates/refreshes LanceDB table (adelaide_agendas).

Chatbot (Streamlit UI) → 5-chatbot.py

Lets you ask questions.

Retrieves relevant chunks from LanceDB.

Uses a local Hugging Face instruct model (Mistral, LLaMA, Falcon, etc.) to generate an answer.

Shows answer + expandable “Show Sources” with retrieved text and PDF filenames.

Sidebar controls:

Adjust number of sources (top_k).

Switch chat model dynamically.

This guarantees your chatbot is:

Grounded in PDFs only ✅

Transparent about which PDF every answer comes from ✅

Fully local (no OpenAI API needed) ✅

Flexible (swap models and retrieval depth live in the UI) ✅
