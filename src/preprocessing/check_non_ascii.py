# Check file for tracking pdf to text fail cases

import os
import logging
from tqdm import tqdm

from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # For consistent results across runs

logging.basicConfig(
    filename=os.path.join("src", "data_processing", "check_non_ascii.log"),
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    filemode='a'
)

def is_non_ascii(text, threshold=0.5):
    """
    Return True if > %threshold of content in PDF is non-ascii
    """
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / max(len(text), 1) > threshold

def scan_txt_files(base_dir):
    """
    Scan each .txt file in base_dir for non-ASCII characters and detect language.
    Log file path, language, and first 100 characters.
    """
    non_ascii_files = []
    total_files = 0
    for root, _, files in tqdm(os.walk(base_dir)):
        for file in files:
            if file.endswith(".txt") and not file.endswith(".ascii.txt"):
                total_files += 1
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    if is_non_ascii(content):
                        lang = detect(content)
                        preview = content[:100].replace("\n", " ").replace("\r", "")
                        logging.info(f"{full_path} — Detected language: {lang} — Preview: {preview}")
                        non_ascii_files.append(full_path)
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")

    logging.info(f"Found {len(non_ascii_files)} non_ascii files out of {total_files}")
    print(non_ascii_files)
    print(len(non_ascii_files))
    return non_ascii_files

if __name__ == '__main__':
    txt_dir = os.path.join("data", "texts")
    bad_txts = scan_txt_files(txt_dir)
