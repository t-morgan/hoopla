def chunk_text(text, chunk_size=200, overlap=0):
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
        if i + chunk_size > len(words):
            break
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks