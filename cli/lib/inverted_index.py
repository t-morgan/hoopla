from collections import defaultdict, Counter
import math
import pickle
import os
from typing import Callable, Dict, Set, List

from .search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT, load_movies
from .text_utils import tokenize_text


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_DIR, "docmap.pkl")
TERM_FREQUENCIES_PATH = os.path.join(CACHE_DIR, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_DIR, "doc_lengths.pkl")


class InvertedIndex:
    def __init__(self, tokenizer: Callable[[str], List[str]] = tokenize_text):
        self.tokenize = tokenizer
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self.docmap: Dict[int, dict] = {}
        self.term_frequencies: Counter[str] = Counter()
        self.doc_lengths: Dict[int, int] = {}
        self.movies = None

    def get_bm25_idf(self, term:str) -> float:
        tokens = self.tokenize(term)
        if len(tokens) == 0 or len(tokens) > 1:
            raise ValueError("Argument for term can only be one token")
        token = tokens[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.get_document_ids(token))
        return math.log((doc_count - term_doc_count + 0.5) / (term_doc_count + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
    
    def get_bm25(self, doc_id: int, term:str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

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
        tokens = self.tokenize(term)
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
        if self.movies is None:
            self.movies = load_movies()
        for movie in self.movies:
            self.__add_document(movie["id"], self.__get_doc_text(movie))
            self.docmap[movie["id"]] = movie
    
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
        self.doc_lengths = load_pickle(DOC_LENGTHS_PATH)
    
    def save(self) -> None:
        def save_pickle(path: str, item: object):
            with open(path, "wb") as f:
                pickle.dump(item, f)

        os.makedirs(CACHE_DIR, exist_ok=True)
        save_pickle(INDEX_PATH, self.index)
        save_pickle(DOCMAP_PATH, self.docmap)
        save_pickle(TERM_FREQUENCIES_PATH, self.term_frequencies)
        save_pickle(DOC_LENGTHS_PATH, self.doc_lengths)
    
    def bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = self.tokenize(query)

        scores = defaultdict(float)
        for doc in self.docmap.values():
            for token in query_tokens:
                scores[doc["id"]] += self.get_bm25(doc["id"], token)
        sorted_scores = sorted(scores.items(), key=lambda score: score[1], reverse=True)
        
        results = []
        for doc_id, score in sorted_scores[:limit]:
            doc = self.docmap[doc_id].copy()
            doc["score"] = score
            results.append(doc)
        return results
        
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        query_tokens = self.tokenize(query)

        results = {}
        for token in query_tokens:
            matching_docs = self.get_documents(token)
            for doc in matching_docs:
                results[doc["id"]] = doc
                if len(results) >= limit:
                    break
            if len(results) >= limit:
                break
        return list(results.values())

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = self.tokenize(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        for token in tokens:
            self.term_frequencies[self.__tf_key(doc_id, token)] += 1
        self.doc_lengths[doc_id] = len(tokens)
    
    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.docmap)
        if num_docs == 0:
            return 0.0
        return sum(self.doc_lengths.values()) / num_docs
    
    def __get_doc_text(self, doc: Dict) -> str:
        return f"{doc['title']} {doc['description']}"
    
    def __tf_key(self, doc_id: int, token: str) -> str:
        return f"{doc_id}|{token}"
