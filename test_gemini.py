import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)
response = client.models.generate_content(model="gemini-2.0-flash-001", contents="Write a short poem about the sea.")
print("Response from Gemini-2.0-flash-001:")
print(response.text)
print("Usage metadata:")
print(f"Prompt tokens: {response.usage_metadata.prompt_token_count}")
print(f"Response tokens: {response.usage_metadata.candidates_token_count}")
