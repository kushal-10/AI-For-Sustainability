import os
import json
from typing import List, Dict, Any
import logging

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.classification.prompts import get_classifications, create_batch_object
from src.utils.file_utils import load_json
from src.filtering.fuzzy_search import is_ai_related

logging.basicConfig(
    filename=os.path.join("src", "classification", "submit_requests.log"),  # log file path
    filemode="w",                     # "w" overwrite, "a" append
    level=logging.INFO,            # only WARNING+ messages
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

client = OpenAI()
MODEL = "gpt-4.1-mini"
BASE_DIR = os.path.join("data", "texts")

T = 0.5
BATCH_DIR = os.path.join("data", "batches_41_mini_pse")
# PATCH_DIR = os.path.join(BATCH_DIR, "patched_max_tokens_50")
PATCH_DIR = BATCH_DIR # hotfix Porsche SE

if not os.path.exists(BATCH_DIR):
    os.makedirs(BATCH_DIR)

def save_batch(batches: List, batch_num: int):
    jsonl_path = os.path.join(BATCH_DIR, f"batch_{batch_num}.jsonl")

    with open(jsonl_path, 'w') as file:
        for obj in batches:
            file.write(json.dumps(obj) + '\n')


def _create_batches(split_paths: List):

    batches = []
    batch_num = 0

    embedding_sentences = 0
    fuzzy_sentences = 0

    for sp in tqdm(split_paths):
        csv_path = sp.replace("splits.json", "similarity_scores.csv").replace("texts", "scores_csv")
        assert os.path.exists(csv_path), f"{csv_path} does not exist"

        json_data = load_json(sp)
        csv_df = pd.read_csv(csv_path)

        columns = list(csv_df.columns)

        for i in range(len(csv_df)):
            sentence_id = str(int(csv_df.iloc[i][columns[0]]))
            sentence = json_data[sentence_id]
            for j in range(1, len(columns)):
                score = csv_df.iloc[i][columns[j]]
                if score >= T:
                    batch_obj = create_batch_object(sentence, sentence_id, csv_path, model=MODEL)
                    batches.append(batch_obj)
                    embedding_sentences += 1

                    if len(batches) >= 20000:
                        save_batch(batches, batch_num)
                        batch_num += 1
                        batches = []
                    break

                if j == len(columns) - 1: # Last value, sentence did not clear threshold
                    if is_ai_related(sentence): # Handle lost context cases
                        batch_obj = create_batch_object(sentence, sentence_id, csv_path, model=MODEL)
                        batches.append(batch_obj)
                        fuzzy_sentences += 1

                        if len(batches) >= 20000:
                            save_batch(batches, batch_num)
                            batch_num += 1
                            batches = []


    save_batch(batches, batch_num)

    print(f"Created {batch_num} batches")
    print(f"Embedding sentences: {embedding_sentences}")
    print(f"Fuzzy sentences: {fuzzy_sentences}")


def create_batches():

    splits = []
    for dirname, _, filenames in os.walk(BASE_DIR):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            if filename.endswith("splits.json"):
                if "Porsche SE" in dirname:
                    splits.append(filepath)

    _create_batches(splits)


def submit_requests():
    BATCH_DIR = PATCH_DIR
    batch_files = [
        os.path.join(BATCH_DIR, f)
        for f in os.listdir(BATCH_DIR)
        if f.endswith(".jsonl")
    ]

    for path in tqdm(batch_files):
        batch_file = client.files.create(
            file=open(path, "rb"),
            purpose="batch"
        )
        logger.info("/" * 50)
        logger.info(f"BATCH - {batch_file}")
        logger.info(f"For path - {path}")
        logger.info("/" * 50)

        logger.info(f"batch_file: {batch_file}")
        logger.info(f"batch_file.id: {batch_file.id}")

        batch_job = client.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        logger.info(f"batch_job.id: {batch_job.id}")
        logger.info(f"batch_job.output_file_id: {batch_job.output_file_id}")


if __name__ == "__main__":

    # create_batches()
    submit_requests()
