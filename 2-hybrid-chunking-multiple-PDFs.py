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

# Quiet some HF warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

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
        year = parts[-3]
        month_part = parts[-2]
        m = re.match(r"(\d{2})_", month_part)
        if year.isdigit() and m:
            try:
                return datetime(int(year), int(m.group(1)), 1).strftime("%Y-%m-%d")
            except Exception:
                return None
    return None

def ensure_doc_list(conv_result):
    """Normalize ConversionResult → list[DoclingDocument]."""
    if hasattr(conv_result, "documents") and conv_result.documents:
        docs = conv_result.documents
    elif hasattr(conv_result, "document") and conv_result.document is not None:
        docs = conv_result.document
    else:
        raise ValueError("ConversionResult has neither .document nor .documents")

    # If it’s a single DoclingDocument
    try:
        from docling_core.types.doc.document import DoclingDocument  # type: ignore
        if isinstance(docs, DoclingDocument):
            return [docs]
    except Exception:
        pass

    # If it’s iterable, force into list
    if hasattr(docs, "__iter__") and not isinstance(docs, list):
        docs = list(docs)
    if not isinstance(docs, list):
        docs = [docs]
    return docs

def process_pdfs_for_chunking():
    logger.info("Initializing document converter and chunker...")

    pipeline_options = PdfPipelineOptions()
    pipeline = StandardPdfPipeline(pipeline_options=pipeline_options)
    converter = DocumentConverter({InputFormat.PDF: pipeline})

    chunker = HybridChunker(chunk_size=500, overlap=50)

    logger.info("Initialization successful.")

    # Load existing chunks if any
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "rb") as f:
            all_chunks = pickle.load(f)
        logger.info(f"Loaded {len(all_chunks)} existing chunks from {OUTPUT_FILE}.")
    else:
        all_chunks = []

    processed_files = {c.get("metadata", {}).get("source_file")
                       for c in all_chunks if "metadata" in c}

    # Find PDFs
    logger.info(f"Scanning for PDF files in '{DATA_DIR}'...")
    pdf_files = []
    for root, _, files in os.walk(DATA_DIR):
        for name in files:
            if name.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, name))
    logger.info(f"Found {len(pdf_files)} PDF file(s).")

    total_added = 0
    success_count = 0
    fail_count = 0

    for pdf_path in pdf_files:
        rel_path = os.path.relpath(pdf_path, DATA_DIR)
        if rel_path in processed_files:
            logger.info(f"Skipping already processed file: {rel_path}")
            continue

        logger.info(f"Processing new PDF: {rel_path}")
        try:
            result = converter.convert(pdf_path)
            docs = ensure_doc_list(result)

            added = 0
            for doc in docs:
                # ✅ Pass DoclingDocument directly
                chunks = chunker.chunk(doc)
                chunks = list(chunks) if hasattr(chunks, "__iter__") else chunks
                logger.info(f"Produced {len(chunks)} chunks from {rel_path}")

                if chunks:
                    first_text = getattr(chunks[0], "text", None) or chunks[0].get("text", "")
                    logger.info(f"🔎 First chunk preview ({rel_path}): {first_text[:200]!r}")

                for ch in chunks:
                    text = getattr(ch, "text", None) or ch.get("text")
                    page_number = getattr(ch, "page_number", None) or ch.get("page_number")
                    if not text:
                        continue
                    all_chunks.append({
                        "chunk": {"text": text},
                        "metadata": {
                            "source_file": rel_path,
                            "page_number": page_number,
                            "date": extract_date_from_path(rel_path),
                        },
                    })
                    added += 1

            if added > 0:
                success_count += 1
            else:
                fail_count += 1

            total_added += added
            logger.info(f"✅ Added {added} chunks from {rel_path}")

        except Exception as e:
            fail_count += 1
            logger.error(f"Failed to process '{rel_path}': {e}")
            continue

    # Save updated chunks
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    logger.info(f"Chunks saved successfully! Total: {len(all_chunks)}")
    logger.info(f"✅ Wrote {len(all_chunks)} chunks → {OUTPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        size_kb = os.path.getsize(OUTPUT_FILE) / 1024
        logger.info(f"📦 File created at {OUTPUT_FILE} ({size_kb:.1f} KB)")

    # --- Final Summary ---
    logger.info("=== Processing Summary ===")
    logger.info(f"PDFs succeeded (≥1 chunk): {success_count}")
    logger.info(f"PDFs failed/no chunks:     {fail_count}")
    logger.info(f"Total chunks added:        {total_added}")
    logger.info("==========================")

if __name__ == "__main__":
    process_pdfs_for_chunking()
