# 2-hybrid-chunking-multiple-PDFs.py
import os
import pickle
import logging
import re
from datetime import datetime
from docling.document_converter import DocumentConverter, InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, PdfPipelineOptions
from docling.chunking import HybridChunker
import torch

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
    logger.info("🚀 Using GPU (CUDA) for Docling")
else:
    device = "cpu"
    logger.warning("⚠️ No GPU detected, using CPU")

# Patterns of boilerplate to skip/deduplicate
CEREMONIAL_PATTERNS = [
    "We pray for wisdom, courage, empathy, understanding and guidance",
]

def extract_date_from_path(path: str):
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

def is_boilerplate(text: str) -> bool:
    for pat in CEREMONIAL_PATTERNS:
        if pat.lower() in text.lower():
            return True
    return False

def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")
    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter({InputFormat.PDF: pipeline})

    # Hybrid chunker with overlap + combine short lines
    chunker = HybridChunker(chunk_size=500, overlap=50)

    logger.info("Initialization successful.")

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} existing chunks.")
    else:
        all_chunks = []

    processed_files = {c.get("metadata", {}).get("source_file") for c in all_chunks}

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
            result = converter.convert(pdf_path)
            doc = result.document

            chunks = chunker.chunk(doc)

            merged_chunks = []
            buffer = ""
            for ch in chunks:
                txt = ch.text.strip()
                if not txt:
                    continue
                if is_boilerplate(txt):
                    continue
                if len(txt) < 80:  # merge very short lines
                    buffer += " " + txt
                else:
                    if buffer:
                        txt = buffer.strip() + " " + txt
                        buffer = ""
                    merged_chunks.append(txt)
            if buffer:
                merged_chunks.append(buffer.strip())

            for txt in merged_chunks:
                metadata = {
                    "source_file": rel_path,
                    "date": extract_date_from_path(rel_path),
                }
                all_chunks.append({"text": txt, "metadata": metadata})

        except Exception as e:
            logger.error(f"Failed to process '{rel_path}': {e}")
            continue

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Chunks saved successfully! Total: {len(all_chunks)}")

if __name__ == "__main__":
    process_pdfs_for_chunking()
