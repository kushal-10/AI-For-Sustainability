#!/usr/bin/env python3
import os, re, csv
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# --- Your utils ---
from src.embedding_filter.utils import detect_german, ai_embeddings, ai_embeddings_de, sdg_embeddings_de, sdg_embeddings
from src.utils.file_utils import load_json

BASE_DIR = Path("data/texts")
OUT_ROOT = Path("data/scores_csv")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 1024
ROUND_DECIMALS = 2

# ---------------- Device & model ----------------
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
if device == "cuda":
    torch.set_float32_matmul_precision("high")

model = SentenceTransformer(MODEL_NAME, device=device)

# ---------------- Helpers ----------------
def _refs_for_lang(is_german: bool):
    if is_german:
        ai_sel = {**ai_embeddings_de, **ai_embeddings}
        sdg_sel = sdg_embeddings_de
    else:
        ai_sel = ai_embeddings
        sdg_sel = sdg_embeddings

    # SDG keys sorted numerically if possible
    sdg_keys = list(sdg_sel.keys())
    try:
        sdg_keys.sort(key=lambda k: int(re.sub(r"\D", "", str(k)) or "0"))
    except Exception:
        sdg_keys.sort()
    ai_keys = sorted(ai_sel.keys())

    def to_norm_mat(keys, dct):
        vecs = [torch.as_tensor(dct[k], dtype=torch.float32, device=device) for k in keys]
        M = torch.stack(vecs, dim=0)
        return torch.nn.functional.normalize(M, dim=1)

    sdg_mat = to_norm_mat(sdg_keys, sdg_sel)
    ai_mat  = to_norm_mat(ai_keys,  ai_sel)
    return (sdg_keys, sdg_mat), (ai_keys, ai_mat)

def encode_texts(texts: List[str]) -> torch.Tensor:
    return model.encode(
        texts,
        convert_to_tensor=True,
        device=device,
        show_progress_bar=False,
        normalize_embeddings=True
    )

# ---------------- Main ----------------
def process_partition(results_txt: Path):
    company = results_txt.parts[-3]
    year    = int(results_txt.parts[-2])

    out_dir = OUT_ROOT / company / str(year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "similarity_scores.csv"

    # Resume logic: skip if CSV exists and is non-empty
    if out_csv.exists() and out_csv.stat().st_size > 0:
        # print(f"⏭️  Skip {company}/{year}: CSV already exists.")
        return

    splits_path = results_txt.with_name("splits.json")
    if not splits_path.exists():
        return

    text_blob  = results_txt.read_text(encoding="utf-8")
    is_german  = detect_german(text_blob)
    splits: Dict[str, str] = load_json(str(splits_path))
    if not splits:
        return

    (sdg_keys, sdg_mat), (ai_keys, ai_mat) = _refs_for_lang(is_german)

    header = ["sentence_id"] + [f"sdg_{k}" for k in sdg_keys] + ai_keys

    # deterministic order by sentence_id (as stored in splits.json)
    items = sorted(((k, v) for k, v in splits.items()), key=lambda x: int(x[0]))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        for i in range(0, len(items), BATCH_SIZE):
            chunk = items[i:i+BATCH_SIZE]
            sent_ids = [c[0] for c in chunk]
            texts    = [c[1] for c in chunk]

            sent_emb   = encode_texts(texts)
            sdg_scores = (sent_emb @ sdg_mat.T).detach().cpu().numpy()
            ai_scores  = (sent_emb @ ai_mat.T ).detach().cpu().numpy()

            for row_idx, sid in enumerate(sent_ids):
                row = [sid]
                row += [round(float(sdg_scores[row_idx, j]), ROUND_DECIMALS) for j in range(len(sdg_keys))]
                row += [round(float(ai_scores[row_idx, j]), ROUND_DECIMALS) for j in range(len(ai_keys))]
                w.writerow(row)

            del sent_emb, sdg_scores, ai_scores
            if device == "cuda":
                torch.cuda.empty_cache()


def main():
    results = []
    for dirname, _, filenames in os.walk(BASE_DIR):
        for filename in filenames:
            if filename.endswith("results.txt"):
                results.append(Path(dirname) / filename)
    results.sort()

    for p in tqdm(results, desc="Scoring (resumable CSV)"):
        process_partition(p)

if __name__ == "__main__":
    main()
