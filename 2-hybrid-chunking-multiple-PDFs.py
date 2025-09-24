import os
import pickle
import logging
import re
from datetime import datetime
from docling.document_converter import DocumentConverter
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
    """
    Try to parse a date from a filename like:
    data/2025/04_April/Agenda_Frontsheet_1259.pdf
    → returns '2025-04-01' as a fallback
    """
    parts = path.replace("\\", "/").split("/")
    date_val = None

    if len(parts) >= 3:
        year = parts[-3]
        month_part = parts[-2]
        month_match = re.match(r"(\d{2})_", month_part)
        if year.isdigit() and month_match:
            yyyy = int(year)
            mm = int(month_match.group(1))
            try:
                date_val = datetime(yyyy, mm, 1).strftime("%Y-%m-%d")
            except Exception:
                date_val = None

    return date_val

def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")

    # ✅ Proper pipeline initialization with PdfPipelineOptions
    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter(pipeline)

    # ✅ Hybrid chunker
    chunker = HybridChunker(chunk_size=500, overlap=50)

    logger.info("Initialization successful.")

    # Load existing chunks if any
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} existing chunks from {OUTPUT_FILE}.")
    else:
        all_chunks = []

    processed_files = {c.get("metadata", {}).get("source_file") for c in all_chunks if "metadata" in c}

    # Scan for PDFs
    logger.info(f"Scanning for PDF files in '{DATA_DIR}'...")
    pdf_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    logger.info(f"Found {len(pdf_files)} PDF file(s).")

    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, DATA_DIR)

        if rel_path in processed_files:
            logger.info(f"Skipping already processed file: {rel_path}")
            continue

        logger.info(f"Processing new PDF: {rel_path}")

        try:
            doc = converter.convert(pdf_path)
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

    # Save updated chunks
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Chunks saved successfully! Total: {len(all_chunks)}")
    logger.info(f"✅ Wrote {len(all_chunks)} chunks → {OUTPUT_FILE}")

    # Extra check: confirm file exists
    if os.path.exists(OUTPUT_FILE):
        size_kb = os.path.getsize(OUTPUT_FILE) / 1024
        logger.info(f"📦 File created at {OUTPUT_FILE} ({size_kb:.1f} KB)")
    else:
        logger.error(f"❌ Expected output file missing: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pdfs_for_chunking()
