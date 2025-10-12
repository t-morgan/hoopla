from collections import defaultdict
import pickle
import os

from .search_utils import load_movies
from .keyword_search import tokenize_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")

class InvertedIndex():
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
    
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            self.index[token].add(doc_id)
        
    
    def get_documents(self, term):
        term = term.lower()
        if term not in self.index:
            return []
        doc_ids = sorted(self.index[term])
        docs = []
        for id in doc_ids:
            docs.append(self.docmap[id])
        return docs
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie['id']] = movie
    
    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(INDEX_PATH, 'wb') as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, 'wb') as f:
            pickle.dump(self.docmap, f)