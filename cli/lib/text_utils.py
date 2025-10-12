import string
from nltk.stem import PorterStemmer
from .search_utils import load_stopwords


def preprocess_text(text: str) -> str:
    text = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = [t for t in text.split() if t]
    stopwords = set(load_stopwords())
    filtered = [w for w in tokens if w not in stopwords]
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in filtered]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    return any(q in t for q in query_tokens for t in title_tokens)
