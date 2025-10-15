import os
from typing import Dict

from sentence_transformers import SentenceTransformer
import numpy as np

from .search_utils import load_movies

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = None

    def build_embeddings(self, documents: list[Dict]):
        if not documents:
            raise ValueError("Document list cannot be empty.")
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}
        doc_strings = [str(f"{doc['title']}: {doc['description']}") for doc in documents]
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

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    ss = SemanticSearch()
    movies = load_movies()
    embeddings = ss.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def verify_model():
    try:
        ss = SemanticSearch()
        print(f"Model loaded successfully: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None