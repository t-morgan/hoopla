from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

def verify_model():
    try:
        ss = SemanticSearch("all-MiniLM-L6-v2")
        print(f"Model loaded successfully: {ss.model}")
        print(f"Max sequence length: {ss.model.max_seq_length}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None