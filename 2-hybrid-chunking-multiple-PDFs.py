import os
import pickle
import logging
from datetime import datetime
from docling.document_converter import DocumentConverter, InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, PdfPipelineOptions
from docling.chunking import HybridChunker
import torch

# --- Config ---
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Detect device ---
if torch.cuda.is_available():
    device = "cuda"
    logger.info("🚀 Using GPU (CUDA) for Docling")
else:
    device = "cpu"
    logger.warning("⚠️ No GPU detected, using CPU")

def extract_date_from_path(path: str):
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        year, month_part = parts[-3], parts[-2]
        if year.isdigit() and month_part[:2].isdigit():
            return f"{year}-{month_part[:2]}-01"
    return None

def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")

    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter({InputFormat.PDF: pipeline})

    # ✅ Smaller chunks for agendas, with overlap
    chunker = HybridChunker(chunk_size=350, overlap=60, use_titles=True)

    logger.info("Initialization successful.")

    # Load existing chunks
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} existing chunks from {OUTPUT_FILE}.")
    else:
        all_chunks = []

    processed_files = {c["metadata"].get("source_file") for c in all_chunks if "metadata" in c}

    pdf_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(DATA_DIR)
        for f in files if f.lower().endswith(".pdf")
    ]
    logger.info(f"Found {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, DATA_DIR)
        if rel_path in processed_files:
            logger.info(f"Skipping already processed file: {rel_path}")
            continue

        logger.info(f"Processing new PDF: {rel_path}")
        try:
            result = converter.convert(pdf_path)
            doc = result.document  # ✅ correct attribute

            chunks = chunker.chunk(doc)

            for ch in chunks:
                metadata = {
                    "source_file": rel_path,
                    "page_number": getattr(ch, "page_number", None),
                    "date": extract_date_from_path(rel_path)
                }
                all_chunks.append({"chunk": {"text": ch.text}, "metadata": metadata})

        except Exception as e:
            logger.error(f"Failed to process '{rel_path}': {e}")
            continue

    # Save
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"✅ Saved {len(all_chunks)} chunks → {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pdfs_for_chunking()
