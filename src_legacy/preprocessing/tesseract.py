"""OCR utility using Tesseract for PDFs that cannot be parsed directly."""

import os

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

# Set up source folder containing PDFs requiring OCR
SOURCE_DIR = os.path.join("data", "reports_corrupted")

# Optional: configure path to tesseract binary if it is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Linux
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Windows


def ocr_pdf_to_text(pdf_path: str) -> str:
    """Convert a PDF to text using Tesseract OCR.

    The PDF is first rendered to images and then each page is processed
    individually.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        str: Extracted text.
    """

    text = ""
    try:
        # Convert PDF to list of images (one per page)
        images = convert_from_path(pdf_path, dpi=300)
        for img in tqdm(images, desc=f"Extracting text from {pdf_path}"):
            page_text = pytesseract.image_to_string(img)
            text += page_text + "\n"
    except Exception as e:  # pragma: no cover - diagnostic output
        print(f"Failed to OCR {pdf_path}: {e}")
    return text


def process_all_pdfs(source_dir: str) -> None:
    """Run OCR on all PDFs within ``source_dir``.

    Args:
        source_dir: Directory containing PDF files to process.
    """

    for root, _, files in os.walk(source_dir):
        for file in tqdm(files, desc=f"Processing PDFs in {root}"):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                txt_path = os.path.join(root, file.replace(".pdf", ".txt"))

                print(f"OCR processing: {pdf_path}")
                text = ocr_pdf_to_text(pdf_path)
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"Saved OCR output: {txt_path}")


if __name__ == "__main__":
    process_all_pdfs(SOURCE_DIR)
