import os
from typing import Dict

from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import load_movies
from .inverted_index import movie_to_search_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = None

    def build_embeddings(self, documents: list[dict]):
        if not documents:
            raise ValueError("Document list cannot be empty.")
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        doc_strings = [movie_to_search_text(doc) for doc in documents]
        self.embeddings = np.array(self.model.encode(doc_strings, show_progress_bar=True))
        np.save(EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")
        return self.model.encode([text])[0]

    def load_or_create_embeddings(self, documents: list[Dict]):
        if os.path.exists(EMBEDDINGS_PATH):
            self.embeddings = np.load(EMBEDDINGS_PATH)
            self.documents = documents
            self.document_map = {doc['id']: doc for doc in documents}
        if self.embeddings is not None and len(self.embeddings) != len(documents):
            print("Document count has changed, rebuilding embeddings...")
            return self.build_embeddings(documents)
        if self.embeddings is None:
            return self.build_embeddings(documents)
        return self.embeddings

    def search(self, query: str, top_k: int = 5):
        if self.embeddings is None or self.documents is None:
            raise ValueError("Embeddings and documents must be loaded or created before searching. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarities = np.array([cosine_similarity(query_embedding, doc_embedding) for doc_embedding in self.embeddings])
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            results.append({
                'id': doc['id'],
                'title': doc['title'],
                'description': doc['description'],
                'score': float(similarities[idx])
            })
        return results

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must be of the same dimensions")
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One or both vectors are zero-vectors")
    return float(np.dot(vec1, vec2) / (norm1 * norm2))

def embed_query_text(query: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def search_movies(query: str, limit: int = 5, movies: list = None):
    ss = SemanticSearch()
    if movies is None:
        movies = load_movies()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, top_k=limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"   Description: {result['description']}\n")
    return results

def verify_embeddings(movies: list = None):
    ss = SemanticSearch()
    if movies is None:
        movies = load_movies()
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {int(embeddings.shape[0])} vectors in {int(embeddings.shape[1])} dimensions")

def verify_model():
    try:
        ss = SemanticSearch()
        print(f"Model loaded successfully: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None