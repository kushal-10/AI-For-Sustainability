import os
import re
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker

# NOTE: langchain_community.HuggingFaceEmbeddings is deprecated; prefer langchain_huggingface
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback

import nltk

from src.utils.file_utils import load_text, save_json

# ---------- Hard constants ----------
MIN_SEMANTIC_CHUNK_TOKENS = 10
MAX_CHUNK_TOKENS = 512
PARAPHRASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_BATCH_SIZE = 4096  # tune per VRAM; try 32 if you hit OOM

# Enable sentence tokenizer (uncomment if punkt is missing on the machine)
# nltk.download("punkt", quiet=True)

# ---------- MPS/GPU helpers ----------
def _torch_device():
    # Prefer Apple MPS, then CUDA, else CPU
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def _advise_mps_env():
    # Helpful for unsupported ops to avoid crashes; user can opt-in
    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # safe default on macOS

# ---------- strict token tools (exact cap at 512) ----------
def _get_tokenizer():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        raise RuntimeError("Install tiktoken: `pip install tiktoken` for exact 512-token capping.") from e

def count_tokens_strict(text: str) -> int:
    enc = _get_tokenizer()
    return len(enc.encode(text))

def hard_cap_by_tokens(text: str, max_tokens: int) -> List[str]:
    enc = _get_tokenizer()
    ids = enc.encode(text)
    return [enc.decode(ids[i:i+max_tokens]) for i in range(0, len(ids), max_tokens)]

def pack_sentences_greedy_strict(sentences: List[str], max_tokens: int) -> List[str]:
    enc = _get_tokenizer()
    chunks, cur_ids = [], []
    for s in sentences:
        s_ids = enc.encode(s)
        if len(s_ids) > max_tokens:
            if cur_ids:
                chunks.append(enc.decode(cur_ids)); cur_ids = []
            chunks.extend(hard_cap_by_tokens(s, max_tokens))
            continue
        if len(cur_ids) + len(s_ids) <= max_tokens:
            cur_ids.extend(s_ids)
        else:
            if cur_ids:
                chunks.append(enc.decode(cur_ids))
            cur_ids = list(s_ids)
    if cur_ids:
        chunks.append(enc.decode(cur_ids))
    return chunks

# ---------- Splitter base ----------
class Splitter(ABC):
    def __init__(self, files: List[str]) -> None:
        self.files = files

    @abstractmethod
    def split(self, txt_file) -> Dict:
        """Split a given file and return a dict -> {id: chunk_text}"""

    @staticmethod
    def get_sentence_dict(sentence_list: List[str]) -> Dict:
        return {str(i): s for i, s in enumerate(sentence_list)}

# ---------- Naive (sentence) ----------
class NaiveSplitter(Splitter):
    @staticmethod
    def clean_pdf_text(raw_text: str) -> str:
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"Page\s+\d+", "", text)
        text = re.sub(r"-\n", "", text)
        text = re.sub(r"\n([a-z])", r" \1", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\u00ad", "").replace("\u2009", "").strip()
        return text

    def split(self, txt_file) -> Dict:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        text = self.clean_pdf_text(text)
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return self.get_sentence_dict(sentences)

# ---------- Semantic (local, GPU-aware, batched, 512-cap) ----------
class SemanticSplitterHF(Splitter):
    """
    Local semantic splitter with exact hard-cap at 512 tokens.
    Uses Apple MPS on Mac M-series automatically when available.
    """
    _embeddings = None  # per-process singleton

    @classmethod
    def _get_embeddings(cls):
        if cls._embeddings is None:
            _advise_mps_env()
            device = _torch_device()
            cls._embeddings = HuggingFaceEmbeddings(
                model_name=PARAPHRASE_MODEL,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": HF_BATCH_SIZE,  # leverage GPU batching
                    # "show_progress_bar": False,
                },
            )
        return cls._embeddings

    def __init__(self, files: List[str]) -> None:
        super().__init__(files)
        embeddings = self._get_embeddings()
        # NB: SemanticChunker has *no* max cap; we enforce 512 after this.
        self.text_splitter = SemanticChunker(
            embeddings,
            min_chunk_size=MIN_SEMANTIC_CHUNK_TOKENS,
            # You can tune thresholds if needed:
            # breakpoint_threshold_type="percentile",
            # breakpoint_threshold_amount=95,
        )

    def _enforce_max_tokens(self, text: str) -> List[str]:
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        if sentences:
            chunks = pack_sentences_greedy_strict(sentences, MAX_CHUNK_TOKENS)
        else:
            chunks = hard_cap_by_tokens(text, MAX_CHUNK_TOKENS)
        # safety pass
        final = []
        for c in chunks:
            if count_tokens_strict(c) > MAX_CHUNK_TOKENS:
                final.extend(hard_cap_by_tokens(c, MAX_CHUNK_TOKENS))
            else:
                final.append(c)
        return final

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        docs = self.text_splitter.create_documents([text_content])
        final_chunks: List[str] = []
        for d in docs:
            final_chunks.extend(self._enforce_max_tokens(d.page_content))
        return {str(i): c for i, c in enumerate(final_chunks)}

# ---------- Stubs ----------
class SpacySplitter(Splitter):
    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        return {}

class NLTKSplitter(Splitter):
    """
    Sentence-based splitter using NLTK's Punkt tokenizer.
    - Cleans common PDF artifacts (line breaks, hyphenation, page numbers).
    - Splits into sentences with nltk.sent_tokenize.
    - Packs sentences greedily to respect the 512-token hard cap
      using the shared `pack_sentences_greedy_strict` + tiktoken.
    """

    @staticmethod
    def _ensure_punkt():
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    @staticmethod
    def clean_pdf_text(raw_text: str) -> str:
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"Page\s+\d+", "", text)       # drop "Page 12"
        text = re.sub(r"-\n", "", text)              # de-hyphenate
        text = re.sub(r"\n([a-z])", r" \1", text)    # join broken lines mid-sentence
        text = re.sub(r"\n\s*\n+", "\n\n", text)     # collapse extra blank lines
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces
        text = text.replace("\u00ad", "").replace("\u2009", "").strip()
        return text

    def split(self, txt_file) -> Dict:
        self._ensure_punkt()
        text = load_text(txt_file)
        text = self.clean_pdf_text(text)

        # Sentence tokenize
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]

        # Pack into <=512-token chunks (uses tiktoken)
        chunks = pack_sentences_greedy_strict(sentences, MAX_CHUNK_TOKENS)

        # Return as {id: chunk}
        return {str(i): c for i, c in enumerate(chunks)}

# ---------- Sister-folder output helpers ----------
def compute_out_root(root_dir: str) -> str:
    """
    Given a root_dir like .../data/sample_texts,
    return a sister folder like .../data/sample_jsons.
    If the last component ends with '_texts', swap to '_jsons'; else append '_jsons'.
    """
    root_dir = os.path.abspath(root_dir)
    parent = os.path.dirname(root_dir)
    last = os.path.basename(os.path.normpath(root_dir))
    if last.endswith("_texts"):
        last_out = last[:-6] + "jsons"
    else:
        last_out = last + "_jsons"
    return os.path.join(parent, last_out)

def get_output_path(results_path: str, splitter_name: str, root_dir: str) -> str:
    """
    Map .../root_dir/<subdirs>/results.txt
      -> .../sister_root/<same subdirs>/splits_<splitter>.json
    """
    out_root = compute_out_root(root_dir)
    rel_dir = os.path.relpath(os.path.dirname(results_path), start=root_dir)
    out_dir = os.path.join(out_root, rel_dir)
    return os.path.join(out_dir, f"splits_{splitter_name}.json")

# ---------- File discovery ----------
def find_results_files(root_dir: str) -> List[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn == "results.txt":
                matches.append(os.path.join(dirpath, fn))
    return matches

def select_splitter(splitter_name: str, files: List[str]) -> Splitter:
    name = splitter_name.lower()
    if name == "semantic":
        return SemanticSplitterHF(files)
    elif name == "naive":
        return NaiveSplitter(files)
    elif name == "nltk":
        return NLTKSplitter(files)
    elif name == "spacy":
        return SpacySplitter(files)
    else:
        raise ValueError("Unknown splitter. Choose from: semantic, naive, nltk, spacy")

# ---------- Worker (runs in each process) ----------
def _process_one(args: Tuple[str, str, str]) -> Tuple[str, str, str]:
    fpath, splitter_name, root_dir = args
    try:
        splitter = select_splitter(splitter_name, [fpath])
        splits = splitter.split(fpath)
        out_path = get_output_path(fpath, splitter_name, root_dir)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_json(out_path, splits)
        return ("ok", fpath, out_path)
    except Exception as e:
        return ("err", fpath, str(e))

# ---------- Main ----------
def main(root_dir: str, splitter_name: str, workers: int):
    files = find_results_files(root_dir)
    if not files:
        print(f"No results.txt files found under {root_dir}")
        return

    # macOS: prefer 'spawn' to avoid forking a process that already has Torch state
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # Single-process path (debug-friendly)
    if workers <= 1:
        splitter = select_splitter(splitter_name, files)
        pbar = tqdm(files, desc=f"Splitting with '{splitter_name}'")
        for fpath in pbar:
            status, fp, info = _process_one((fpath, splitter_name, root_dir))
            if status == "err":
                pbar.write(f"[ERROR] {fp}: {info}")
        return

    # Multi-process path
    tasks = [(fp, splitter_name, root_dir) for fp in files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Splitting ({workers} workers)"):
            status, fpath, info = fut.result()
            if status == "err":
                print(f"[ERROR] {fpath}: {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch split results.txt within a folder tree (GPU-aware, parallel).")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root folder containing subfolders with results.txt files.")
    parser.add_argument("--splitter", type=str, required=True,
                        choices=["semantic", "naive", "nltk", "spacy"],
                        help="Which splitter to use.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Process files in parallel with N workers (recommend <= CPU cores).")
    args = parser.parse_args()
    main(args.root_dir, args.splitter, args.workers)



"""
python3 src/preprocessing/splitter.py --root_dir data/sample_texts --splitter nltk --workers 1
"""