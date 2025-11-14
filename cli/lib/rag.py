from .hybrid_search import search_rrf
from .llm_utils import execute_llm_prompt


def answer_question(query: str, limit: int = 5):
    # Perform RRF search
    search_results = search_rrf(query, k=limit)

    # Prepare documents for the prompt
    docs = "\n".join([f"- {movie['title']} --- {movie['description']}" for movie in search_results])

    # Create the prompt for Gemini API
    prompt = f"""\
Answer the user's question based on the provided movies that are available on Hoopla.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Question: {query}

Documents:
{docs}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""

    llm_response = execute_llm_prompt(prompt)
    # Print the results
    print("Search Results:")
    print(docs)
    print("\nAnswer:")
    print(llm_response)

def perform_rag_with_citations(query: str, limit: int = 5):
    # Perform RRF search
    search_results = search_rrf(query, k=limit)

    # Prepare documents for the prompt
    docs = "\n".join([f"- {movie['title']} --- {movie['description']}" for movie in search_results])

    # Create the prompt for Gemini API
    prompt = f"""\
Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- After the answer, include a "Sources" section that maps each citation number to the corresponding document title.
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    llm_response = execute_llm_prompt(prompt)
    # Print the results
    print("Search Results:")
    print(docs)
    print("\nRAG Response with Citations:")
    print(llm_response)


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
    print(docs)
    print("\nRAG Response:")
    print(llm_response)


def get_summary(query: str, limit: int = 5):
    # Perform RRF search
    search_results = search_rrf(query, k=limit)

    # Prepare documents for the prompt
    docs = "\n".join([f"- {movie['title']}" for movie in search_results])

    # Create the prompt for Gemini API
    prompt = f"""\
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
"""

    llm_response = execute_llm_prompt(prompt)
    print("Search Results:")
    print(docs)
    print("\nLLM Summary:")
    print(llm_response)
