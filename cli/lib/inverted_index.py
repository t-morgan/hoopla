from collections import defaultdict, Counter
import math
import pickle
import os
from typing import Callable, Dict, Set, List

from .search_utils import load_movies
from .text_utils import tokenize_text


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")


class InvertedIndex:
    def __init__(self, tokenizer: Callable[[str], List[str]] = tokenize_text):
        self.tokenize = tokenizer
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Counter[str] = Counter()
    
    def __tf_key(self, doc_id: int, token: str) -> str:
        return f"{doc_id}|{token}"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[self.__tf_key(doc_id, token)] += 1

    def get_document_ids(self, term: str) -> List[int]:
        term = term.lower()
        if term not in self.index:
            return []
        return sorted(self.index[term])

    def get_documents(self, term: str) -> List[dict]:
        docs: List[dict] = []
        for doc_id in self.get_document_ids(term):
            if doc_id in self.docmap:
                docs.append(self.docmap[doc_id])
        return docs
    
    def get_idf(self, term:str) -> float:
        tokens = tokenize_text(term)
        if len(tokens) == 0 or len(tokens) > 1:
            raise ValueError("Argument for term can only be one token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_document_ids(token))
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        tokens = self.tokenize(term)
        if len(tokens) == 0 or len(tokens) > 1:
            raise ValueError("Argument for term can only be one token")
        token = tokens[0]
        return self.term_frequencies[self.__tf_key(doc_id, token)]
    
    def get_tf_idf(self, doc_id: int, term: str) -> float:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        def save_pickle(path: str, item: object):
            with open(path, "wb") as f:
                pickle.dump(item, f)
        
        os.makedirs(CACHE_DIR, exist_ok=True)
        save_pickle(INDEX_PATH, self.index)
        save_pickle(DOCMAP_PATH, self.docmap)
        save_pickle(TERM_FREQUENCIES_PATH, self.term_frequencies)
    
    def load(self) -> None:
        def load_pickle(path: str):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Required file not found: '{path}'")
            except pickle.UnpicklingError as e:
                raise ValueError(f"Corrupted pickle file: '{path}'") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected error while loading '{path}': {e}") from e

        self.index = load_pickle(INDEX_PATH)
        self.docmap = load_pickle(DOCMAP_PATH)
        self.term_frequencies = load_pickle(TERM_FREQUENCIES_PATH)
