from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

class MultimodalSearch:
    def __init__(self, model_name='clip-ViT-B-32'):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text):
        return self.model.encode([text])[0]

    def encode_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.model.encode([image])[0]

    def compute_similarity(self, text_embedding, image_embedding):
        return dot(text_embedding, image_embedding) / (norm(text_embedding) * norm(image_embedding))


def verify_image_embedding(image_path, text_query):
    search = MultimodalSearch()
    text_embedding = search.encode_text(text_query)
    image_embedding = search.encode_image(image_path)
    similarity = search.compute_similarity(text_embedding, image_embedding)
    print(f"Similarity between text and image: {similarity:.4f}")
    return similarity