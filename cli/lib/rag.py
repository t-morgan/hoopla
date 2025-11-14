# The "rag" command should:
# Load the movies and perform an RRF search using the provided query. Search for the top 5 results.
# Prompt the Gemini API with the query, the search results, and instructions for the LLM to generate an answer based on the retrieved documents. Here's the prompt I used:
# prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
#
# Query: {query}
#
# Documents:
# {docs}
#
# Provide a comprehensive answer that addresses the query:"""
#
# Print the results and the generated answer in this format:
# Search Results:
#   - We're Back! A Dinosaur's Story
#   - Jurassic Park
#   - The Lost World
#   - Carnosaur
#   - A Sound of Thunder
#
# RAG Response:
# <RESPONSE HERE>

from .hybrid_search import search_rrf
from .llm_utils import execute_llm_prompt


def perform_rag(query: str):
    # Perform RRF search
    top_k = 5
    search_results = search_rrf(query, k=top_k)

    # Prepare documents for the prompt
    docs = "\n".join([f"- {movie['title']}" for movie in search_results])

    # Create the prompt for Gemini API
    prompt = f"""\
Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}
    
Provide a comprehensive answer that addresses the query:"""

    llm_response = execute_llm_prompt(prompt)
    # Print the results
    print("Search Results:")
    for movie in search_results:
        print(f"  - {movie['title']}")
    print("\nRAG Response:")
    print(llm_response)