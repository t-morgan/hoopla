import json

from .llm_utils import execute_llm_prompt


def evaluate_search(query, results) -> None:
    """Evaluate search results using LLM as a judge.

    Args:
        query (str): The search query.
        results (list): List of search results to evaluate.
    """
    formatted_results = [f"{result['title']} - {result['description']}" for result in results]
    prompt = f"""\
Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = execute_llm_prompt(prompt)
    try:
        scores = json.loads(response)
        if not isinstance(scores, list) or not all(isinstance(s, int) for s in scores):
            raise ValueError("Invalid scores format")
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"LLM response was: {response}")
        return
    for i, (result, score) in enumerate(zip(formatted_results, scores), start=1):
        print(f"{i}. {result}: {score}/3")