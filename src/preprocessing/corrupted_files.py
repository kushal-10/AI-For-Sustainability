# After initial pass, some PDFs were not properly processed, this file identifies those cases

import os
import shutil
from tqdm import tqdm
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0  # for reproducibility

# The directories containing processed text files, some might be empty, or in German Language
results_dir = "data/texts"

# Find files that are empty
corrupted_files = []
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file == "results.txt":
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if not content:  # empty or only whitespace
                    corrupted_files.append(file_path)

# Find files that contain German Language Text
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith("results.txt"):
            file_path = os.path.join(root, file)
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                # Read a sample of the file (e.g., first 1000 chars)
                text = f.read(10000)
                try:
                    lang = detect(text)
                    if lang == "de":
                        corrupted_files.append(file_path)
                except Exception as e:
                    print(f"Could not detect language for {file_path}: {e}")


for file in tqdm(corrupted_files):
    source_file = file
    dest_file = file.replace("texts", "texts_corrupted")
    shutil.copy(source_file, dest_file)