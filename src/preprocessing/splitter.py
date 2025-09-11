import os
import re
import argparse
from abc import ABC, abstractmethod
from typing import List, Dict

from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import nltk

from src.utils.file_utils import load_text, save_json

nltk.download("punkt")

PARAPHRASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def count_tokens(text: str) -> int:
    """
    Approx token count.
    Uses tiktoken if available; otherwise falls back to a simple heuristic (whitespace split).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Heuristic: ~1 token per word; punctuation counted as part of words is okay for an upper bound.
        return len(re.findall(r"\S+", text))

def pack_sentences_greedy(sentences: List[str], max_tokens: int) -> List[str]:
    """
    Greedy bin-packing of sentences into chunks <= max_tokens.
    """
    chunks, cur, cur_tokens = [], [], 0
    for s in sentences:
        t = count_tokens(s)
        # if a single sentence is longer than max_tokens, hard-split by words
        if t > max_tokens:
            words = s.split()
            buf, btoks = [], 0
            for w in words:
                wt = 1  # heuristic
                if btoks + wt > max_tokens and buf:
                    chunks.append(" ".join(buf).strip())
                    buf, btoks = [], 0
                buf.append(w)
                btoks += wt
            if buf:
                chunks.append(" ".join(buf).strip())
            continue

        if cur_tokens + t <= max_tokens:
            cur.append(s)
            cur_tokens += t
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            cur, cur_tokens = [s], t
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]

class Splitter(ABC):
    def __init__(self, files: List[str]) -> None:
        self.files = files

    @abstractmethod
    def split(self, txt_file) -> Dict:
        """
        Split a given file and return a dict in format -> key: text_chunk
        """

    @staticmethod
    def get_sentence_dict(sentence_list: List[str]) -> Dict:
        splits = {}
        for i in range(len(sentence_list)):
            splits[str(i)] = sentence_list[i]
        return splits


class NaiveSplitter(Splitter):
    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    @staticmethod
    def clean_pdf_text(raw_text: str) -> str:
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"Page\s+\d+", "", text)          # Remove page numbers
        text = re.sub(r"-\n", "", text)                 # Fix hyphenation
        text = re.sub(r"\n([a-z])", r" \1", text)       # Merge broken lines in paragraph
        text = re.sub(r"\n\s*\n+", "\n\n", text)        # Normalize paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\u00ad", "")               # Soft hyphen
        text = text.replace("\u2009", "")               # Thin space
        text = text.strip()
        return text

    def split(self, txt_file) -> Dict:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        text = self.clean_pdf_text(text)
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return self.get_sentence_dict(sentences)


class SemanticSplitterHF(Splitter):
    """
    Semantic splitter using local HuggingFace embeddings (free/offline).
    Always uses 'paraphrase-multilingual-MiniLM-L12-v2' with normalized embeddings.
    Enforces a max token size per chunk via recursive, sentence-aware re-splitting.
    """
    def __init__(self, files: List[str], min_chunk_size: int, max_chunk_size: int) -> None:
        super().__init__(files)
        self.min_chunk_size = int(min_chunk_size)
        self.max_chunk_size = int(max_chunk_size)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=PARAPHRASE_MODEL,
            model_kwargs={"device": "cuda" if _has_cuda() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.text_splitter = SemanticChunker(
            self.embeddings,
            min_chunk_size=self.min_chunk_size
            # You can tune breakpoint thresholds if needed:
            # breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95
        )

    def _enforce_max_tokens(self, text: str) -> List[str]:
        if count_tokens(text) <= self.max_chunk_size:
            return [text]
        # Split by sentences and greedily pack under max_chunk_size
        sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        if not sentences:
            # Fallback: hard split by words
            return pack_sentences_greedy([text], self.max_chunk_size)
        return pack_sentences_greedy(sentences, self.max_chunk_size)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        docs = self.text_splitter.create_documents([text_content])

        # Enforce max tokens per chunk
        final_chunks: List[str] = []
        for d in docs:
            enforced = self._enforce_max_tokens(d.page_content)
            final_chunks.extend(enforced)

        return {str(i): c for i, c in enumerate(final_chunks)}


class SpacySplitter(Splitter):
    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        # TODO: implement spaCy sentence splitter if needed
        return {}


class NLTKSplitter(Splitter):
    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        # TODO: implement NLTK-specific behavior if different from NaiveSplitter
        return {}


def find_results_files(root_dir: str) -> List[str]:
    matches = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn == "results.txt":
                matches.append(os.path.join(dirpath, fn))
    return matches


def get_output_path(results_path: str, splitter_name: str) -> str:
    folder = os.path.dirname(results_path)
    return os.path.join(folder, f"splits_{splitter_name}.json")


def select_splitter(splitter_name: str, files: List[str], min_chunk_size: int, max_chunk_size: int) -> Splitter:
    name = splitter_name.lower()
    if name == "semantic":
        return SemanticSplitterHF(files, min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size)
    elif name == "naive":
        return NaiveSplitter(files)
    elif name == "nltk":
        return NLTKSplitter(files)
    elif name == "spacy":
        return SpacySplitter(files)
    else:
        raise ValueError("Unknown splitter. Choose from: semantic, naive, nltk, spacy")


def main(root_dir: str, splitter_name: str, min_chunk_size: int, max_chunk_size: int):
    files = find_results_files(root_dir)
    if not files:
        print(f"No results.txt files found under {root_dir}")
        return

    splitter = select_splitter(splitter_name, files, min_chunk_size, max_chunk_size)

    for fpath in tqdm(files, desc=f"Splitting with '{splitter_name}'"):
        try:
            splits = splitter.split(fpath)
            out_path = get_output_path(fpath, splitter_name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            save_json(out_path, splits)
        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch split results.txt files within a folder tree.")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root folder containing subfolders with results.txt files.")
    parser.add_argument("--splitter", type=str, required=True,
                        choices=["semantic", "naive", "nltk", "spacy"],
                        help="Which splitter to use.")
    parser.add_argument("--min_chunk_size", type=int, required=True,
                        help="Minimum semantic chunk size (SemanticChunker parameter).")
    parser.add_argument("--max_chunk_size", type=int, required=True,
                        help="Maximum tokens per chunk (approx). Oversized chunks are re-split recursively.")

    args = parser.parse_args()
    main(args.root_dir, args.splitter, args.min_chunk_size, args.max_chunk_size)
