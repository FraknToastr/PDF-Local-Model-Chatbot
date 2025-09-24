import os
import pickle
import logging
import re
import json
from datetime import datetime

import torch
from docling.document_converter import DocumentConverter, InputFormat
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline, PdfPipelineOptions
from docling.chunking import HybridChunker
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# --- Config ---
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "all_chunks.pkl")
COUNCILLOR_FILE = os.path.join(DATA_DIR, "councillors.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Device detect ---
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    logger.info("🚀 Using GPU (CUDA)")
else:
    logger.warning("⚠️ Using CPU only")

# --- Load NER model for councillor extraction ---
logger.info("🔎 Loading NER model for councillor detection...")
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
ner_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
ner_pipeline = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, device=device)


def extract_date_from_path(path: str):
    """Parse a date from a filename path like: 2025/04_April/...pdf"""
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        year = parts[-3]
        month_part = parts[-2]
        match = re.match(r"(\d{2})_", month_part)
        if year.isdigit() and match:
            try:
                return datetime(int(year), int(match.group(1)), 1).strftime("%Y-%m-%d")
            except Exception:
                return None
    return None


def extract_councillor_names(text: str):
    """Extract potential councillor names from text using regex + NER."""
    names = set()

    # Quick regex for capitalised words after "Councillor" or "Lord Mayor"
    regex_matches = re.findall(r"(Councillor\s+[A-Z][a-zA-Z\-']+(?:\s+[A-Z][a-zA-Z\-']+)*)", text)
    for match in regex_matches:
        names.add(match.strip())

    # Run NER
    ner_results = ner_pipeline(text)
    current_name = []
    for ent in ner_results:
        if ent["entity"].endswith("PER"):
            word = ent["word"].replace("##", "")
            current_name.append(word)
        else:
            if current_name:
                names.add(" ".join(current_name))
                current_name = []
    if current_name:
        names.add(" ".join(current_name))

    return list(names)


def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")

    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter({InputFormat.PDF: pipeline})

    chunker = HybridChunker(chunk_size=500, overlap=50)

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} existing chunks")
    else:
        all_chunks = []

    councillor_names = set()

    processed_files = {c["metadata"]["source_file"] for c in all_chunks if "metadata" in c}

    # Scan for PDFs
    pdf_files = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, f))

    logger.info(f"Found {len(pdf_files)} PDF(s).")

    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, DATA_DIR)
        if rel_path in processed_files:
            logger.info(f"Skipping already processed file: {rel_path}")
            continue

        logger.info(f"Processing new PDF: {rel_path}")
        try:
            result = converter.convert(pdf_path)
            doc = result.document
            chunks = list(chunker.chunk(doc))

            for ch in chunks:
                text = ch.text
                metadata = {
                    "source_file": rel_path,
                    "page_number": getattr(ch, "page_number", None),
                    "date": extract_date_from_path(rel_path),
                    "is_member_list": "Councillor" in text or "Lord Mayor" in text
                }
                all_chunks.append({"text": text, "metadata": metadata})

                # Councillor extraction
                if metadata["is_member_list"]:
                    found_names = extract_councillor_names(text)
                    councillor_names.update(found_names)

        except Exception as e:
            logger.error(f"❌ Failed to process {rel_path}: {e}")
            continue

    # Save updated chunks
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    logger.info(f"✅ Saved {len(all_chunks)} chunks → {OUTPUT_FILE}")

    # Save councillor list
    if councillor_names:
        with open(COUNCILLOR_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(list(councillor_names)), f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Extracted {len(councillor_names)} councillor names → {COUNCILLOR_FILE}")
    else:
        logger.warning("⚠️ No councillor names extracted!")


if __name__ == "__main__":
    process_pdfs_for_chunking()
