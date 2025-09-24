#!/usr/bin/env python3
"""
2-hybrid-chunking-multiple-PDFs-clean.py
- Processes PDFs into cleaned chunks
- Removes ceremonial filler (prayers, pledges, silences)
- Deduplicates repeated lines
- Adds metadata flag for member lists
"""

import os
import pickle
import logging
import re
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
    """Extract year/month from filename path like data/2025/04_April/file.pdf"""
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        year = parts[-3]
        month_part = parts[-2]
        match = re.match(r"(\d{2})_", month_part)
        if year.isdigit() and match:
            yyyy, mm = int(year), int(match.group(1))
            try:
                return datetime(yyyy, mm, 1).strftime("%Y-%m-%d")
            except Exception:
                return None
    return None

def clean_text(text: str) -> str:
    """Remove ceremonial filler and deduplicate repeated lines."""
    if not text:
        return ""

    # Strip common ceremonial filler
    patterns = [
        r"The Lord Mayor will state:.*?(respecting the opinions of others\.')",
        r"May we in this meeting.*?(those we serve\.')",
        r"Council acknowledges that we are meeting on traditional Country.*?(today\.')",
        r"A minute of silence.*?(observed\.)",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)

    # Deduplicate repeated lines inside one chunk
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    seen = set()
    deduped = []
    for ln in lines:
        if ln not in seen:
            deduped.append(ln)
            seen.add(ln)

    return "\n".join(deduped)

def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")

    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter({InputFormat.PDF: pipeline})
    chunker = HybridChunker(chunk_size=500, overlap=50)

    logger.info("Initialization successful.")

    all_chunks = []
    processed_files = set()

    # Scan PDFs
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
            result = converter.convert(pdf_path)
            doc = result.document

            chunks = chunker.chunk(doc)

            for ch in chunks:
                cleaned = clean_text(ch.text)
                if not cleaned.strip():
                    continue

                metadata = {
                    "source_file": rel_path,
                    "page_number": getattr(ch, "page_number", None),
                    "date": extract_date_from_path(rel_path),
                    "is_member_list": (
                        "lord mayor" in cleaned.lower() or "councillor" in cleaned.lower()
                    ),
                }
                all_chunks.append({"text": cleaned, "metadata": metadata})

        except Exception as e:
            logger.error(f"❌ Failed to process '{rel_path}': {e}")
            continue

    # Save updated chunks
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"✅ Chunks saved successfully! Total: {len(all_chunks)}")
    logger.info(f"📦 File written to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_pdfs_for_chunking()
