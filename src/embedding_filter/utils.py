""" Util functions for Filtering"""
import os

from langdetect import detect, DetectorFactory

from src.utils.file_utils import load_json, save_json
from sentence_transformers import SentenceTransformer

DetectorFactory.seed = 0  # Ensures consistent results

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# embeddings = model.encode(sentences)
ai_kw_json = os.path.join("kw_data", "nist_airc.json")

ai_kw_data = load_json(ai_kw_json)

def get_embeddings_ai(terms):
    embedding_data = {}
    for term in terms:
        embedding = model.encode(term)
        embedding_data[term] = embedding
    return embedding_data

def get_embeddings_sdgs(sdg_data):
    embedding_data = {}
    for k,v in sdg_data.items():
        embedding_data[k] = model.encode(v)
    return embedding_data

def load_and_embed(path):
    data = load_json(path)
    return get_embeddings_sdgs(data)

ai_terms = []
ai_terms_de = []

for k,v in ai_kw_data.items():
    ai_terms.append(k)
    ai_terms_de.append(v)

ai_embeddings = get_embeddings_ai(ai_terms)
ai_embeddings_de = get_embeddings_ai(ai_terms_de)

# English
sdg_embeddings = {}
sdg_embeddings.update(load_and_embed(os.path.join("kw_data", "sdgs.json")))
sdg_embeddings.update(load_and_embed(os.path.join("kw_data", "sdgs_detailed.json")))

# German
sdg_embeddings_de = {}
sdg_embeddings_de.update(load_and_embed(os.path.join("kw_data", "sdgs_de.json")))
sdg_embeddings_de.update(load_and_embed(os.path.join("kw_data", "sdgs_detailed_de.json")))


def detect_german(text: str) -> bool:
    """
        Detects if the input text is in German ('de') by analyzing
        only the first 10,000 characters.

        Args:
            text (str): The text to analyze.

        Returns:
            bool: True if the language is German, False otherwise.
        """
    try:
        truncated_text = text[:10000]
        language = detect(truncated_text)
        return language == 'de'
    except:
        return False
