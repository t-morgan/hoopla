from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

from .search_utils import load_movies


class MultimodalSearch:
    def __init__(self, model_name='clip-ViT-B-32', documents=None):
        self.model = SentenceTransformer(model_name)
        self.documents = documents if documents is not None else []
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)


    def encode_text(self, text):
        return self.model.encode([text])[0]

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.model.encode([image])[0]

    def compute_similarity(self, text_embedding, image_embedding):
        return dot(text_embedding, image_embedding) / (norm(text_embedding) * norm(image_embedding))

    def search_with_image(self, image_path):
        image_embedding = self.encode_image(image_path)
        similarities = []
        for idx, text_embedding in enumerate(self.text_embeddings):
            similarity = self.compute_similarity(text_embedding, image_embedding)
            similarities.append({
                'id': self.documents[idx]['id'],
                'title': self.documents[idx]['title'],
                'description': self.documents[idx]['description'],
                'similarity': similarity
            })
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:5]


def image_search_command(image_path):
    documents = load_movies()
    search = MultimodalSearch(documents=documents)
    results = search.search_with_image(image_path)
    return results

def verify_image_embedding(image_path, text_query):
    search = MultimodalSearch()
    text_embedding = search.encode_text(text_query)
    image_embedding = search.encode_image(image_path)
    similarity = search.compute_similarity(text_embedding, image_embedding)
    print(f"Similarity between text and image: {similarity:.4f}")
    return similarity