from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        if not text.strip():
            raise ValueError("Input text cannot be empty or whitespace.")
        return self.model.encode([text])[0]

def embed_text(text: str):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    try:
        ss = SemanticSearch()
        print(f"Model loaded successfully: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None