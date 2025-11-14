import mimetypes
from google.genai import types as genai_types

from .llm_utils import execute_llm_prompt, execute_llm_response


def describe_image(image_path: str, query: str):
    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as img_file:
        img = img_file.read()

        prompt = f"""\
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

        parts = [
            prompt,
            genai_types.Part.from_bytes(data=img, mime_type=mime),
            query.strip(),
        ]

        response = execute_llm_response(parts=parts)
        print(f"Rewritten query: {response.text.strip()}")
        if response.usage_metadata is not None:
            print(f"Total tokens:    {response.usage_metadata.total_token_count}")