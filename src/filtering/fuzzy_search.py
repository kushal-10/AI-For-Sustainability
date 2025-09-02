from rapidfuzz import fuzz

ai_terms = [
    "Artificial Intelligence",
    "Machine Learning",
    "Reinforcement Learning",
    "Deep Learning",
    "Computer Vision",
    "Natural Language Processing",
]

ai_terms_de = [
    "Künstliche Intelligenz",       # Artificial Intelligence
    "Maschinelles Lernen",          # Machine Learning
    "Bestärkendes Lernen",          # Reinforcement Learning
    "Tiefes Lernen",                # Deep Learning
    "Computer Vision",              # Often untranslated
    "Natürliche Sprachverarbeitung" # Natural Language Processing
]

_TERMS_LOWER = [t.lower() for t in (ai_terms + ai_terms_de)]

def is_ai_related(sentence: str, threshold: int = 90) -> bool:
    """
    Return True if `sentence` contains or fuzzy-matches any AI-related term
    from `ai_terms` or `ai_terms_de` at or above `threshold`, else False.
    """
    if not sentence:
        return False

    s = sentence.lower()

    # Fast path: exact/substring match
    if any(t in s for t in _TERMS_LOWER):
        return True

    # Fuzzy path
    for term in _TERMS_LOWER:
        if fuzz.partial_ratio(term, s) >= threshold:
            return True

    return False
