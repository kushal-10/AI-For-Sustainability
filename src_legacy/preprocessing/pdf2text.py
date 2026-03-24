"""Utilities for extracting text from PDF reports without OCR."""

import logging
import os

import fitz
from tqdm import tqdm

# Expected PDF layout: data/reports/FIRM/YEAR.pdf

# Setup a base logger
logging.basicConfig(
    filename=os.path.join("src", "data_processing", "pdf2text.log"),
    level=logging.INFO,
    filemode="w",
)


def get_txt_content(file: str = "temp") -> str:
    """Extract text content from a PDF using PyMuPDF.

    Args:
        file: Path to the PDF file.

    Returns:
        str: Extracted text content.
    """

    txt_content = ""
    with fitz.open(file) as doc:
        for page in doc:
            txt_content += "\n" + page.get_text()

    # Sanity check for valid PDF content
    logging.info(f"Doc content for {file}::\n {txt_content[:250]}")
    logging.info("*" * 50)
    return txt_content


def write_txt(content: str = "temp", file: str = "temp.txt") -> None:
    """Save text content to disk.

    The file is saved as ``results.txt`` in the directory
    ``data/texts/COMPANY/YEAR``.

    Args:
        content: Text content to be saved.
        file: Destination path for the ``.txt`` file.

    Raises:
        AssertionError: If ``file`` does not end with ``.txt``.
    """

    # Ensure the output file path is sensible
    assert file.endswith(".txt"), "File path is not valid. Path should end with .txt"
    with open(file, "w") as f:
        f.write(content)
    logging.info(f"Saved file : {file}")
    logging.info("*" * 50)


if __name__ == '__main__':

    txt_dir = os.path.join("data", "texts")
    dir = os.path.join("data", "reports")

    # Walk through the reports and extract text
    companies = os.listdir(os.path.join(dir))
    for company in tqdm(companies):
        if not os.path.isdir(os.path.join(dir, company)):
            continue
        years = os.listdir(os.path.join(dir, company))
        for year in years:
            if year.endswith('.pdf'):
                year_temp = year.replace('.pdf', '')
                results_dir = os.path.join(txt_dir, company, year_temp)

                # Only process those PDFs for which .../COMPANY/YEAR folder does not exist
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                    logging.info(f"Getting docs for - {results_dir}")
                    doc_content = get_txt_content(os.path.join(dir, company, year))
                    write_txt(doc_content, os.path.join(results_dir, 'results.txt'))
                    logging.info(f"DONE for file - {results_dir}")
                    logging.info("%" * 100)

                # Also check for PDFs whose .../COMPANY/YEAR folder exists but
                # /results.txt has not been produced yet, or got interrupted while processing.
                elif os.path.isdir(results_dir) and not os.listdir(results_dir):
                    logging.info(f"Getting docs for - {results_dir}")
                    doc_content = get_txt_content(os.path.join(dir, company, year))
                    write_txt(doc_content, os.path.join(results_dir, 'results.txt'))
                    logging.info("%" * 100)

