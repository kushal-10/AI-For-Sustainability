#!/usr/bin/env python3
import json
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, util

# ---------------- Config ----------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INPUT_JSON = Path("kw_data/nist_airc.json")
OUTPUT_JSON = Path("kw_data/ai_keywords.json")
QUERY = "artificial intelligence"
THRESHOLD = 0.5   # tune this

# ---------------- Device & model ----------------
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = SentenceTransformer(MODEL_NAME, device=device)

# ---------------- Load data ----------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    kw_data = json.load(f)

# English words only
english_keywords = list(kw_data.keys())

# ---------------- Embedding ----------------
query_emb = model.encode(QUERY, convert_to_tensor=True, device=device, normalize_embeddings=True)
kw_embs = model.encode(english_keywords, convert_to_tensor=True, device=device, normalize_embeddings=True)

# ---------------- Similarity ----------------
sims = util.cos_sim(query_emb, kw_embs).cpu().numpy().flatten()

# Filter by threshold
filtered = {
    word: kw_data[word]
    for word, score in zip(english_keywords, sims)
    if score >= THRESHOLD
}

# # ---------------- Save ----------------
# OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
# with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#     json.dump(filtered, f, ensure_ascii=False, indent=2)
#
# print(f"Filtered {len(filtered)} AI-related keywords saved to {OUTPUT_JSON}")

print(filtered.keys(), len(filtered))