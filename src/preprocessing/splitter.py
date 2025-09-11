import os
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import re

from tqdm import tqdm
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
import nltk

from src.utils.file_utils import load_text, save_json

nltk.download("punkt")

class Splitter(ABC):

    def __init__(self, files: List[str]) -> None:
        self.files = files

    @abstractmethod
    def split(self, txt_file) -> Dict:
        """
        Implement different splitter methods

        Split a given file and return a dict in format -> key: text_chunk
        :param txt_file: Path to the text file
        :return: splits - a dict in format -> key: text_chunk
        """

    @staticmethod
    def get_sentence_dict(sentence_list: List[str]) -> Dict:
        """
        Convert a list of raw sentences into a dict with sentence ids
        :param sentence_list: A list of raw sentences
        :return: a dict in format -> key: text_chunk
        """

        splits = {}
        for i in range(len(sentence_list)):
            splits[str(i)] = sentence_list[i]

        return splits

class NaiveSplitter(Splitter):

    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    @staticmethod
    def clean_pdf_text(raw_text: str) -> str:
        """Remove artifacts and normalise raw PDF text.

        Args:
            raw_text: Text extracted from a PDF.

        Returns:
            str: Cleaned text.
        """

        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"Page\s+\d+", "", text)  # Remove page numbers
        text = re.sub(r"-\n", "", text)  # Fix hyphenation
        text = re.sub(r"\n([a-z])", r" \1", text)  # Merge broken lines in paragraph
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # Normalize paragraph breaks
        text = re.sub(r"[ \t]+", " ", text)
        text = text.replace("\u00ad", "")  # Remove soft hyphens
        text = text.replace("\u2009", "")  # Remove thin spaces
        text = text.strip()
        return text

    def split(self, txt_file) -> Dict:
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        text = self.clean_pdf_text(text)
        sentences = nltk.sent_tokenize(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        splits = self.get_sentence_dict(sentences)

        return splits

class SemanticSplitter(Splitter):

    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)

        text_splitter = SemanticChunker(OpenAIEmbeddings(),
                                        min_chunk_size=50)

        docs = text_splitter.create_documents([text_content])

        splits = {}
        for i in range(len(docs)):
            splits[str(i)] = docs[i].page_content

        return splits

class SpacySplitter(Splitter):

    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)

        #TODO:

class NLTKSplitter(Splitter):
    def __init__(self, files: List[str]) -> None:
        super().__init__(files)

    def split(self, txt_file) -> Dict:
        text_content = load_text(txt_file)
        # TODO:

# TODO: In general, compare time complexity as well

if __name__ == "__main__":
    sample_text = ["data/texts/14.basf_$42.93 b_industrials/2023/results.txt"]
    sample_file = sample_text[0]

    # ss = SemanticSplitter(sample_text)
    # splits = ss.split(sample_file)

    ns = NaiveSplitter(sample_text)
    splits = ns.split(sample_file)

    if not os.path.exists("results/sample_splits"):
        os.makedirs("results/sample_splits")

    save_json("results/sample_splits/basf_naive_splits.json", splits)
