import json
import numpy as np
import os
from .search_utils import load_movies
from .semantic_search import SemanticSearch, cosine_similarity
from .text_chunker import semantic_chunk_text

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2", max_chunk_size=4, overlap=1) -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def build_chunk_embeddings(self, documents: list[dict]):
        if not documents:
            raise ValueError("Document list cannot be empty.")

        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for i, doc in enumerate(documents):
            if doc['description'] == '':
                continue
            chunks = semantic_chunk_text(doc['description'], max_chunk_size=self.max_chunk_size, overlap=self.overlap)
            all_chunks.extend(chunks)
            for j, chunk in enumerate(chunks):
                chunk_metadata.append({
                    'movie_idx': i,
                    'chunk_idx': j,
                    'total_chunks': len(chunks)
                })

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        np.save(EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(METADATA_PATH, 'w') as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(METADATA_PATH):
            self.chunk_embeddings = np.load(EMBEDDINGS_PATH)
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata['chunks']
        else:
            return self.build_chunk_embeddings(documents)
        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError("Chunk embeddings and metadata must be loaded or created before searching.")

        query_embedding = self.model.encode([query])[0]
        similarities = np.array([cosine_similarity(query_embedding, chunk) for chunk in self.chunk_embeddings])
        top_indices = np.argsort(similarities)[-limit * 2:][::-1]  # get more to filter duplicates

        seen_movies = {}
        for idx in top_indices:
            chunk_info = self.chunk_metadata[idx]
            movie_idx = chunk_info['movie_idx']
            score = float(similarities[idx])
            if movie_idx not in seen_movies or score > seen_movies[movie_idx]['score']:
                movie = self.documents[movie_idx]
                result = {
                    'id': movie['id'],
                    'title': movie['title'],
                    'description': movie['description'][:100],
                    'chunk_idx': chunk_info['chunk_idx'],
                    'movie_idx': movie_idx,
                    'score': score
                }
                seen_movies[movie_idx] = result
        # Sort by score and limit
        results = sorted(seen_movies.values(), key=lambda x: x['score'], reverse=True)[:limit]
        return results

def embed_chunks(max_chunk_size=4, overlap=1) -> None:
    searcher = ChunkedSemanticSearch(max_chunk_size=max_chunk_size, overlap=overlap)
    documents = load_movies()
    embeddings = searcher.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked(query: str, limit: int = 5, max_chunk_size=4, overlap=1) -> None:
    searcher = ChunkedSemanticSearch(max_chunk_size=max_chunk_size, overlap=overlap)
    documents = load_movies()
    searcher.load_or_create_chunk_embeddings(documents)
    results = searcher.search_chunks(query, limit=limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
        print(f"\t{result['description']}")