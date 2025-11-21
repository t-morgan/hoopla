import re


def chunk_text(text, chunk_size=200, overlap=0) -> list[str]:
    """Chunk the input text into smaller pieces of specified size.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk in words.
        overlap (int): The number of overlapping words between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap if chunk_size > overlap else 1):
        chunk = " ".join(words[i : min(i + chunk_size, len(words))])
        chunks.append(chunk)
    return chunks

def semantic_chunk_text(text, max_chunk_size=4, overlap=0) -> list[str]:
    """Chunk the input text semantically into sentence groups of specified size.

    Args:
        text (str): The input text to be chunked.
        max_chunk_size (int): The maximum size of each chunk in sentences.
        overlap (int): The number of overlapping sentences between chunks.

    Returns:
        list: A list of semantically chunked sentences.
    """
    text = text.strip()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    chunks = []
    for i in range(0, len(sentences), max_chunk_size - overlap if max_chunk_size > overlap else 1):
        chunk = " ".join(sentences[i: min(i + max_chunk_size, len(sentences))])
        chunks.append(chunk)
    return chunks