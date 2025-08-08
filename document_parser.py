# document_parser.py
import fitz # PyMuPDF
from docx import Document as DocxDocument # python-docx
import pytesseract # pytesseract for OCR
from PIL import Image
import pandas as pd # pandas for CSV/DB
import io
import re
import sqlite3 # sqlite3 for DB
import os

# --- Helper Functions for Text Processing ---

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
    - Replaces common whitespace characters with a single space.
    - Removes leading/trailing whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\n\r\t\s]+', ' ', text) # Replace newlines, tabs, multiple spaces with single space
    text = text.strip()
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """
    Splits a large string of text into smaller, overlapping chunks.
    Args:
        text (str): The full text content to chunk.
        chunk_size (int): The maximum length of each chunk.
        overlap (int): The number of characters to overlap between consecutive chunks.
    Returns:
        list[str]: A list of cleaned text chunks.
    """
    if not text:
        return []

    chunks = []
    current_position = 0
    while current_position < len(text):
        end_position = min(current_position + chunk_size, len(text))
        chunk = text[current_position:end_position]
        chunks.append(clean_text(chunk)) # Clean each chunk

        # Move forward, accounting for overlap
        current_position += (chunk_size - overlap)
        if current_position >= len(text) - overlap and end_position == len(text):
            # If the next start position would be very close to the end,
            # or we've reached the end, break to avoid tiny final chunks.
            break
    return chunks

# --- Document Specific Parsers ---

def parse_pdf_text(file_content: bytes) -> str:
    """Parses text from a PDF file using PyMuPDF."""
    text_content = ""
    try:
        with fitz.open(stream=file_content, filetype="pdf") as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_content += page.get_text()
    except Exception as e:
        print(f"Error parsing PDF with PyMuPDF: {e}")
        text_content = "" # Ensure empty if parsing fails
    return text_content

def parse_docx_text(file_content: bytes) -> str:
    """Parses text from a DOCX file using python-docx."""
    text_content = []
    try:
        doc = DocxDocument(io.BytesIO(file_content))
        for paragraph in doc.paragraphs:
            text_content.append(paragraph.text)
        # Join paragraphs with a space to maintain separation but not too much newlines
    except Exception as e:
        print(f"Error parsing DOCX with python-docx: {e}")
        return ""
    return " ".join(text_content)

def parse_txt_text(file_content: bytes) -> str:
    """Reads text directly from a TXT file."""
    try:
        # Attempt UTF-8, then fall back to Latin-1 if decoding fails
        return file_content.decode('utf-8')
    except UnicodeDecodeError:
        return file_content.decode('latin-1', errors='ignore')
    except Exception as e:
        print(f"Error parsing TXT: {e}")
        return ""

def parse_image_ocr(file_content: bytes) -> str:
    """Performs OCR on an image file (JPG, PNG) using pytesseract."""
    try:
        image = Image.open(io.BytesIO(file_content))
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not in your PATH. Please install Tesseract OCR engine.")
        return ""
    except Exception as e:
        print(f"Error during OCR for image: {e}")
        return ""

def parse_scanned_pdf_ocr(file_content: bytes) -> str:
    """Performs OCR on a scanned PDF using PyMuPDF to render pages to images, then pytesseract."""
    full_text = []
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            # Render page to a high-resolution pixmap (image)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Render at 2x resolution for better OCR
            img_bytes = pix.tobytes(format="png") # Convert pixmap to PNG bytes
            image = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(image)
            full_text.append(text)
        pdf_document.close()
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not in your PATH. Please install Tesseract OCR engine.")
        return ""
    except Exception as e:
        print(f"Error during scanned PDF OCR: {e}")
        return ""
    return "\n".join(full_text)

def parse_csv_text(file_content: bytes) -> str:
    """Parses content from a CSV file using pandas, converting DataFrame to string."""
    try:
        # Decode bytes to string, then use io.StringIO for pandas to read as text
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        # Convert DataFrame to a string representation for ingestion, keeping CSV format for consistency
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Error parsing CSV with pandas: {e}")
        return ""

def parse_sqlite_db_text(file_content: bytes) -> str:
    """
    Parses content from an SQLite .db file.
    It attempts to read all table schemas and data from the database.
    """
    db_content = io.BytesIO(file_content)
    conn = None
    try:
        # Create a temporary file to load the DB content, as sqlite3.connect doesn't directly
        # support a BytesIO object for existing databases.
        temp_db_path = "temp_db_for_parsing.db"
        with open(temp_db_path, "wb") as f:
            f.write(db_content.read())
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        all_db_info = []

        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table_name_tuple in tables:
            table_name = table_name_tuple[0]
            all_db_info.append(f"\n--- Table: {table_name} ---")

            # Get column names
            cursor.execute(f"PRAGMA table_info('{table_name}');")
            columns = [col[1] for col in cursor.fetchall()]
            all_db_info.append(f"Columns: {', '.join(columns)}")

            # Get table data (limiting rows to avoid excessively large output for large tables)
            cursor.execute(f"SELECT * FROM '{table_name}' LIMIT 100;") # Limit to first 100 rows
            rows = cursor.fetchall()
            for row in rows:
                all_db_info.append(str(row)) # Simple string representation of rows

        return "\n".join(all_db_info)
    except sqlite3.Error as e:
        print(f"Error parsing SQLite DB: {e}")
        return ""
    except Exception as e:
        print(f"Unexpected error with SQLite DB: {e}")
        return ""
    finally:
        if conn:
            conn.close()
        # Clean up the temporary file
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)

# --- Main Document Parsing Function ---

def parse_document(file_content: bytes, filename: str) -> list[str]:
    """
    Parses the document based on its file extension and returns a list of text chunks.
    Handles different document types and attempts OCR for scanned PDFs/images.
    """
    file_extension = os.path.splitext(filename)[1].lower()
    full_text = ""

    if file_extension == '.pdf':
        # First, try to extract text directly from the PDF
        full_text = parse_pdf_text(file_content)
        # If text is minimal (heuristic for scanned PDF), try OCR
        if len(clean_text(full_text)) < 50: # If less than 50 characters, assume scanned
            # Corrected print statement:
            print(f"Text content very low. PDF '{filename}' might be scanned. Attempting OCR...")
            full_text = parse_scanned_pdf_ocr(file_content)
    elif file_extension == '.docx':
        full_text = parse_docx_text(file_content)
    elif file_extension == '.txt':
        full_text = parse_txt_text(file_content)
    elif file_extension in ['.jpg', '.png', '.jpeg']:
        full_text = parse_image_ocr(file_content)
    elif file_extension == '.csv':
        full_text = parse_csv_text(file_content)
    elif file_extension == '.db': # Assuming SQLite .db files
        full_text = parse_sqlite_db_text(file_content)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Supported types: .pdf, .docx, .txt, .jpg, .png, .csv, .db.")

    if not full_text.strip():
        raise ValueError(f"Could not extract meaningful text from {filename}. It might be empty, corrupted, or an unsupported format despite extension.")

    # Convert all content into clean, meaningful chunks of text (with overlap)
    return chunk_text(full_text)