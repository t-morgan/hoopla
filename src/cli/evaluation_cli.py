import argparse
import json
import os

from lib.hybrid_search import search_rrf

DATA_PATH = os.path.join("data", "golden_dataset.json")

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    with open(DATA_PATH, "r") as f:
        data = json.load(f)

    print(f"k={limit} Evaluation Results")
    print()
    for case in data["test_cases"]:
        print(f"- Query: {case['query']}")
        results = search_rrf(case['query'], k=60, limit=limit)
        relevant_retrieved = 0
        total_retrieved = len(results)
        total_relevant = len(case['relevant_docs'])
        for result in results:
            if result['title'] in case['relevant_docs']:
                relevant_retrieved += 1
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Recall@{limit}: {recall:.4f}")
        print(f"\t- F1 Score: {f1:.4f}")
        print(f"\t- Retrieved: {[result['title'] for result in results]}")
        print(f"\t- Relevant: {[result['title'] for result in results if result['title'] in case['relevant_docs']]}")
        print()


if __name__ == "__main__":
    main()