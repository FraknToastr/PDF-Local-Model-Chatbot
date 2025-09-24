import pickle
import os
import logging
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from typing import List

# --- Configuration ---
MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP = 50  # tokens to overlap between chunks
DATA_FOLDER = "data"
OUTPUT_FILE = "data/all_chunks.pkl"
PROCESSED_LOG_FILE = "data/processed_files.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def load_processed_log() -> set:
    """Loads the set of file paths that have already been processed."""
    if os.path.exists(PROCESSED_LOG_FILE):
        with open(PROCESSED_LOG_FILE, "r") as f:
            return set(line.strip() for line in f)
    return set()


def save_processed_log(filepath: str):
    """Saves the path of a successfully processed file."""
    with open(PROCESSED_LOG_FILE, "a") as f:
        f.write(filepath + "\n")


def find_pdf_files(data_dir: str) -> List[Path]:
    """Recursively finds all PDF files in a given directory."""
    logger.info(f"Scanning for PDF files in '{data_dir}'...")
    pdf_files = list(Path(data_dir).rglob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF file(s).")
    return pdf_files


def sliding_window_chunks(text: str, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Create overlapping text windows from a chunk's text."""
    words = text.split()
    new_chunks = []
    start = 0
    while start < len(words):
        end = min(start + MAX_CHUNK_TOKENS, len(words))
        window = " ".join(words[start:end])
        new_chunks.append(window)
        if end == len(words):
            break
        start = end - overlap  # move back by overlap
    return new_chunks


def process_pdfs_for_chunking():
    """Main function to process all PDFs in the data folder for hybrid chunking with overlap."""
    if not os.path.exists(DATA_FOLDER):
        logger.error(
            f"Data folder '{DATA_FOLDER}' not found. Please run the web scraper first."
        )
        return

    try:
        # Initialize docling components
        logger.info("Initializing document converter and chunker...")
        converter = DocumentConverter()
        chunker = HybridChunker(max_tokens=MAX_CHUNK_TOKENS)
        logger.info("Initialization successful.")
    except ImportError as e:
        logger.error(f"Failed to initialize docling components: {e}")
        return

    processed_files = load_processed_log()
    all_chunks = []

    # Check for existing chunks file and load them
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "rb") as f:
                all_chunks = pickle.load(f)
            logger.info(f"Loaded {len(all_chunks)} existing chunks from {OUTPUT_FILE}.")
        except Exception as e:
            logger.warning(
                f"Could not load existing chunks: {e}. Starting with an empty list."
            )
            all_chunks = []

    pdf_files = find_pdf_files(DATA_FOLDER)
    newly_processed_count = 0

    for pdf_path in pdf_files:
        str_path = str(pdf_path)
        if str_path in processed_files:
            logger.info(f"Skipping already processed file: {str_path}")
            continue

        logger.info(f"Processing new PDF: {str_path}")

        try:
            # Convert PDF to a docling document object
            conversion_result = converter.convert(str_path)

            # Handle different return types from different docling versions.
            if hasattr(conversion_result, "document"):
                document = conversion_result.document
            else:
                document = conversion_result

            if not document:
                logger.warning(
                    f"Error: Could not convert document '{str_path}'. Skipping."
                )
                continue

            # Chunk the document using the HybridChunker
            base_chunks = list(chunker.chunk(document))

            # Apply sliding window overlap to each chunk
            for chunk in base_chunks:
                overlapped_texts = sliding_window_chunks(chunk.text, CHUNK_OVERLAP)
                for win_text in overlapped_texts:
                    chunk_data = {
                        "chunk": {"text": win_text},  # store plain dict for portability
                        "metadata": {"source_file": str_path},
                    }
                    all_chunks.append(chunk_data)

            logger.info(
                f"Created {len(base_chunks)} base chunks and expanded to {len(all_chunks)} total with overlap."
            )

            # Log the processed file to avoid re-processing
            save_processed_log(str_path)
            newly_processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process '{str_path}': {e}")

    if newly_processed_count > 0:
        # Save all chunks to a single file
        logger.info(f"Saving a total of {len(all_chunks)} chunks to {OUTPUT_FILE}...")
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
        logger.info("Chunks saved successfully!")
    else:
        logger.info("No new PDF files to process. Chunk file remains unchanged.")


if __name__ == "__main__":
    process_pdfs_for_chunking()
