import requests
from bs4 import BeautifulSoup
import re
import os
import json
import time
from urllib.parse import urljoin, urlparse
import logging
import pickle
from typing import List, Optional
from pathlib import Path

# --------------------------------------------------------------
# Import docling components with correct paths
# --------------------------------------------------------------
print("🔍 Importing docling components...")

DocumentConverter = None
DoclingDocumentType = None

try:
    # Import DocumentConverter from the correct location
    from docling.document_converter import DocumentConverter

    print("✅ DocumentConverter imported successfully from docling.document_converter")

    # Try to find the correct document type
    print("🔍 Looking for document types in docling.datamodel...")

    try:
        import docling.datamodel as datamodel

        datamodel_contents = [
            item for item in dir(datamodel) if not item.startswith("_")
        ]
        print(f"📋 Available in docling.datamodel: {datamodel_contents}")

        # Try common document type names
        document_candidates = [
            "Document",
            "DoclingDocument",
            "BaseDocument",
            "ConvertedDocument",
        ]

        for candidate in document_candidates:
            if hasattr(datamodel, candidate):
                DoclingDocumentType = getattr(datamodel, candidate)
                print(f"✅ Found document type: {candidate}")
                break

        if DoclingDocumentType is None:
            # If no specific type found, look for anything with 'Document' in the name
            doc_like_items = [
                item for item in datamodel_contents if "document" in item.lower()
            ]
            if doc_like_items:
                print(f"🔍 Found document-like items: {doc_like_items}")
                # Use the first one we find
                DoclingDocumentType = getattr(datamodel, doc_like_items[0])
                print(f"✅ Using document type: {doc_like_items[0]}")
            else:
                print("⚠️ No document type found in datamodel, using dynamic typing")
                DoclingDocumentType = object

    except ImportError as e:
        print(f"❌ Could not import docling.datamodel: {e}")
        print("⚠️ Using dynamic document typing")
        DoclingDocumentType = object

except ImportError as e:
    print(f"❌ Failed to import DocumentConverter: {e}")
    print("PDF extraction will be disabled")

# Final status
if DocumentConverter is not None:
    print("✅ Docling setup complete - extraction will be enabled")
    if DoclingDocumentType != object:
        print(f"✅ Will use document type: {DoclingDocumentType.__name__}")
    else:
        print(
            "✅ Will use dynamic document typing (compatible with all docling versions)"
        )
else:
    print("❌ Docling setup failed - extraction will be disabled")


class AdelaideMeetingScraper:
    def __init__(self, enable_extraction=True):
        self.base_url = "https://meetings.cityofadelaide.com.au/"
        self.main_page_url = (
            "https://meetings.cityofadelaide.com.au/ieListMeetings.aspx?CId=167&Year=0"
        )
        self.data_folder = "data"
        self.processed_file = os.path.join(self.data_folder, "processed_links.json")
        self.documents_file = os.path.join(self.data_folder, "documents.pkl")
        self.session = requests.Session()
        self.processed_links = self.load_processed_links()
        self.enable_extraction = enable_extraction and DocumentConverter is not None

        # Initialize document converter if available
        if self.enable_extraction and DocumentConverter is not None:
            try:
                print("🔧 Initializing DocumentConverter...")
                self.converter = DocumentConverter()
                self.extracted_documents = []
                print("✅ PDF extraction enabled and ready")

            except Exception as e:
                print(f"❌ Could not initialize DocumentConverter: {e}")
                print(f"❌ Error type: {type(e).__name__}")
                print("🔧 Trying alternative initialization...")

                # Try alternative initialization approaches
                try:
                    # Some versions might need specific parameters
                    self.converter = DocumentConverter(format_options={})
                    self.extracted_documents = []
                    print("✅ PDF extraction enabled with alternative initialization")
                except Exception as e2:
                    print(f"❌ Alternative initialization also failed: {e2}")
                    print("PDF extraction will be disabled")
                    self.enable_extraction = False
                    self.converter = None
                    self.extracted_documents = []
        else:
            self.converter = None
            self.extracted_documents = []
            if DocumentConverter is None:
                print("❌ PDF extraction disabled - docling not available")
            else:
                print("🚫 PDF extraction disabled by user")

        # Setup logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Create data folder if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)

        # Headers to appear more like a regular browser
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def load_processed_links(self):
        """Load previously processed links from JSON file"""
        if os.path.exists(self.processed_file):
            try:
                with open(self.processed_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                self.logger.warning(
                    "Could not load processed links file, starting fresh"
                )
                return {"main_page": "", "meeting_links": {}}
        return {"main_page": "", "meeting_links": {}}

    def save_processed_links(self):
        """Save processed links to JSON file"""
        try:
            with open(self.processed_file, "w") as f:
                json.dump(self.processed_links, f, indent=2)
        except IOError as e:
            self.logger.error(f"Could not save processed links: {e}")

    def save_extracted_documents(self):
        """Save extracted documents to pickle file"""
        if self.extracted_documents:
            try:
                with open(self.documents_file, "wb") as f:
                    pickle.dump(self.extracted_documents, f)
                self.logger.info(
                    f"Saved {len(self.extracted_documents)} extracted documents to {self.documents_file}"
                )
            except Exception as e:
                self.logger.error(f"Could not save extracted documents: {e}")

    def load_existing_documents(self):
        """Load existing extracted documents if available"""
        if os.path.exists(self.documents_file):
            try:
                with open(self.documents_file, "rb") as f:
                    self.extracted_documents = pickle.load(f)
                self.logger.info(
                    f"Loaded {len(self.extracted_documents)} existing extracted documents"
                )
            except Exception as e:
                self.logger.warning(f"Could not load existing documents: {e}")
                self.extracted_documents = []

    def extract_pdf_content(self, pdf_path: str, meeting_id: str):
        """Extract content from a PDF file using docling"""
        if not self.enable_extraction:
            return None

        try:
            self.logger.info(f"Extracting content from {pdf_path}")

            # Convert PDF to docling document
            result = self.converter.convert(pdf_path)

            # Handle different return types from different docling versions
            if hasattr(result, "document"):
                # Some versions return a result object with a document attribute
                doc = result.document
            else:
                # Others return the document directly
                doc = result

            # Add metadata about the source
            metadata_dict = {
                "source_file": pdf_path,
                "meeting_id": meeting_id,
                "extraction_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Try different ways to set metadata depending on docling version
            if hasattr(doc, "metadata"):
                if doc.metadata is None:
                    doc.metadata = metadata_dict
                elif isinstance(doc.metadata, dict):
                    doc.metadata.update(metadata_dict)
                else:
                    # If metadata exists but isn't a dict, try to convert or replace
                    doc.metadata = metadata_dict
            elif hasattr(doc, "meta"):
                if doc.meta is None:
                    doc.meta = metadata_dict
                elif isinstance(doc.meta, dict):
                    doc.meta.update(metadata_dict)
                else:
                    doc.meta = metadata_dict
            else:
                # If no metadata attribute exists, try to add one
                try:
                    doc.metadata = metadata_dict
                except:
                    # If we can't set metadata, that's okay, continue without it
                    pass

            self.logger.info(
                f"✓ Successfully extracted content from meeting {meeting_id}"
            )
            return doc

        except Exception as e:
            self.logger.error(f"❌ Error extracting PDF {pdf_path}: {e}")
            return None

    def get_page(self, url, timeout=10):
        """Fetch a web page with error handling"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None

    def find_meeting_links(self, soup):
        """Find all links with &Mid= parameter"""
        meeting_links = []

        # Look for links containing &Mid=
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if "&Mid=" in href or "&MId=" in href:  # Handle both cases
                full_url = urljoin(self.base_url, href)
                meeting_links.append(full_url)

        return list(set(meeting_links))  # Remove duplicates

    def find_agenda_frontsheet_pdf(self, soup, base_url):
        """Find the Agenda Frontsheet PDF link on a meeting page"""
        pdf_links = []

        # Look for PDF links that contain "agenda" and "frontsheet" in URL or link text
        for link in soup.find_all("a", href=True):
            href = link["href"]
            link_text = link.get_text().strip().lower()
            href_lower = href.lower()

            # Check if it's a PDF link
            if href_lower.endswith(".pdf") or ".pdf" in href_lower:
                # Check for "agenda" and "frontsheet" in either URL or link text
                has_agenda = "agenda" in href_lower or "agenda" in link_text
                has_frontsheet = "frontsheet" in href_lower or "frontsheet" in link_text

                if has_agenda and has_frontsheet:
                    full_url = urljoin(base_url, href)
                    pdf_links.append((full_url, link.get_text().strip()))
                    self.logger.debug(f"Found agenda frontsheet PDF: {full_url}")

        return pdf_links

    def extract_date_from_url(self, pdf_url):
        """Extract date from PDF URL to determine folder structure"""
        import re
        from datetime import datetime

        # Look for date patterns in the URL
        # Pattern 1: 22nd-Apr-2025 format
        date_match = re.search(r"(\d{1,2})[a-z]{2}-([A-Za-z]{3})-(\d{4})", pdf_url)
        if date_match:
            day, month_abbr, year = date_match.groups()
            try:
                # Convert month abbreviation to number
                date_obj = datetime.strptime(f"{day}-{month_abbr}-{year}", "%d-%b-%Y")
                return date_obj.year, date_obj.month, date_obj.strftime("%B")
            except ValueError:
                pass

        # Pattern 2: Other common date formats
        date_patterns = [
            r"(\d{4})-(\d{2})-(\d{2})",  # 2025-04-22
            r"(\d{2})-(\d{2})-(\d{4})",  # 22-04-2025
            r"(\d{4})(\d{2})(\d{2})",  # 20250422
        ]
        for pattern in date_patterns:
            match = re.search(pattern, pdf_url)
            if match:
                try:
                    if (
                        pattern == r"(\d{4})-(\d{2})-(\d{2})"
                        or pattern == r"(\d{4})(\d{2})(\d{2})"
                    ):
                        year, month, day = match.groups()
                    else:  # dd-mm-yyyy
                        day, month, year = match.groups()

                    year, month = int(year), int(month)
                    if 1 <= month <= 12 and 2020 <= year <= 2030:  # Reasonable bounds
                        date_obj = datetime(year, month, 1)
                        return year, month, date_obj.strftime("%B")
                except (ValueError, TypeError):
                    continue

        # If no date found, return current year and a default folder
        current_year = datetime.now().year
        return current_year, 0, "Unknown_Date"

    def download_pdf(self, pdf_url, filename, meeting_id=None):
        """Download a PDF file into appropriate year/month folder and optionally extract content"""
        try:
            response = self.get_page(pdf_url)
            if response:
                # Extract date to determine folder structure
                year, month_num, month_name = self.extract_date_from_url(pdf_url)

                # Create year/month folder structure
                if month_num == 0:
                    # If we couldn't determine the month, just use year
                    folder_path = os.path.join(self.data_folder, str(year), month_name)
                else:
                    folder_path = os.path.join(
                        self.data_folder, str(year), f"{month_num:02d}_{month_name}"
                    )
                os.makedirs(folder_path, exist_ok=True)

                # Save file in the appropriate folder
                filepath = os.path.join(folder_path, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                self.logger.info(f"Downloaded: {filename} -> {folder_path}")

                # Extract PDF content if extraction is enabled
                if self.enable_extraction and meeting_id:
                    extracted_doc = self.extract_pdf_content(filepath, meeting_id)
                    if extracted_doc:
                        self.extracted_documents.append(extracted_doc)

                return filepath  # Return the full path for record keeping
        except IOError as e:
            self.logger.error(f"Error saving {filename}: {e}")
            return False

    def sanitize_filename(self, filename):
        """Remove invalid characters from filename"""
        # Remove invalid characters for filenames
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename

    def extract_meeting_id(self, url):
        """Extract meeting ID from URL"""
        match = re.search(r"[Mm][Ii]d=(\d+)", url)
        return match.group(1) if match else "unknown"

    def scrape_meetings(self):
        """Main scraping function"""
        self.logger.info("Starting Adelaide Council meeting scraper...")

        # Load existing extracted documents if available
        if self.enable_extraction:
            self.load_existing_documents()

        # Get the main page
        response = self.get_page(self.main_page_url)
        if not response:
            self.logger.error("Could not fetch main page")
            return

        soup = BeautifulSoup(response.content, "html.parser")

        # Find all meeting links
        meeting_links = self.find_meeting_links(soup)
        self.logger.info(f"Found {len(meeting_links)} meeting links")

        # Process meeting links
        processed_count = 0
        skipped_count = 0
        downloaded_count = 0

        for link in meeting_links:
            meeting_id = self.extract_meeting_id(link)

            # Check if this meeting has already been processed
            if meeting_id in self.processed_links.get("meeting_links", {}):
                self.logger.info(f"Skipping meeting {meeting_id} (already processed)")
                skipped_count += 1
                continue

            self.logger.info(f"Processing meeting link: {link}")
            meeting_response = self.get_page(link)
            if not meeting_response:
                continue

            meeting_soup = BeautifulSoup(meeting_response.content, "html.parser")
            pdf_links = self.find_agenda_frontsheet_pdf(
                meeting_soup, meeting_response.url
            )

            if pdf_links:
                for pdf_url, link_text in pdf_links:
                    filename = self.sanitize_filename(
                        f"Agenda_Frontsheet_{meeting_id}.pdf"
                    )
                    downloaded_path = self.download_pdf(pdf_url, filename, meeting_id)
                    if downloaded_path:
                        downloaded_count += 1
                        # Mark this meeting as processed after successful download
                        if "meeting_links" not in self.processed_links:
                            self.processed_links["meeting_links"] = {}
                        self.processed_links["meeting_links"][meeting_id] = (
                            downloaded_path
                        )
            else:
                self.logger.info(f"No Agenda Frontsheet PDF found for {link}")

            processed_count += 1
            # Save progress every 10 downloads to avoid losing data if interrupted
            if processed_count % 10 == 0:
                self.save_processed_links()

        self.logger.info("--- Scraping Complete ---")
        self.logger.info(f"Total meetings processed: {processed_count}")
        self.logger.info(f"Total PDFs downloaded: {downloaded_count}")
        self.logger.info(f"Total links skipped (already processed): {skipped_count}")

        # Final save
        self.save_processed_links()
        self.save_extracted_documents()

    def process_local_pdfs(self):
        """Process all PDF files found in the data folder"""
        self.logger.info("Starting local PDF processing...")

        # Load existing extracted documents if available
        if self.enable_extraction:
            self.load_existing_documents()

        for root, dirs, files in os.walk(self.data_folder):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    filepath = os.path.join(root, filename)
                    meeting_id = "local_file_" + os.path.basename(
                        root
                    )  # Use folder name for ID
                    self.extract_pdf_content(filepath, meeting_id)

        self.logger.info("--- Local PDF Processing Complete ---")
        self.logger.info(
            f"Extracted content from {len(self.extracted_documents)} local PDFs"
        )
        self.save_extracted_documents()

    def get_stats(self):
        """Display folder structure and statistics"""
        print("\n=== Folder Structure ===")
        folder_counts = {}
        for root, dirs, files in os.walk(self.data_folder):
            pdf_count = sum(1 for f in files if f.lower().endswith(".pdf"))

            if pdf_count > 0:
                # Get relative path from data folder
                rel_path = os.path.relpath(root, self.data_folder)
                folder_counts[rel_path] = pdf_count

        if folder_counts:
            for folder, count in sorted(folder_counts.items()):
                print(f"  {folder}/: {count} PDF{'s' if count != 1 else ''}")
        else:
            print("  No organized folders found yet")


def has_pdfs_in_data_folder(data_folder: str) -> bool:
    """Check if any PDF files exist in the specified folder or its subdirectories."""
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(".pdf"):
                return True
    return False


def main():
    # You can disable extraction by passing enable_extraction=False
    scraper = AdelaideMeetingScraper(enable_extraction=True)

    try:
        # Check if there are any PDFs in the data folder
        if has_pdfs_in_data_folder(scraper.data_folder):
            print("Local PDFs found. Processing existing files...")
            scraper.process_local_pdfs()
        else:
            print("No local PDFs found. Starting web scraping...")
            scraper.scrape_meetings()

        # Always run get_stats() to see the results
        scraper.get_stats()

    except KeyboardInterrupt:
        print("\nScraping interrupted by user")
        scraper.save_processed_links()
        if scraper.enable_extraction:
            scraper.save_extracted_documents()
    except Exception as e:
        scraper.logger.error(f"Unexpected error: {e}")
        scraper.save_processed_links()
        if scraper.enable_extraction:
            scraper.save_extracted_documents()


if __name__ == "__main__":
    main()
