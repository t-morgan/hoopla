from collections import defaultdict
import pickle
import os
from typing import Dict, Set, List

from .search_utils import load_movies
from .text_utils import tokenize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")


class InvertedIndex:
    def __init__(self, tokenizer=tokenize_text):
        self.tokenize = tokenizer
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.docmap: Dict[int, dict] = {}

    def __add_document(self, doc_id: int, text: str) -> None:
        for token in set(self.tokenize(text)):
            self.index[token].add(doc_id)

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

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
