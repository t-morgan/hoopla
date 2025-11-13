import argparse
from lib.hybrid_search import normalize_vector, search_hybrid_weighted, search_rrf
from lib.search_utils import enhance_query

ENHANCERS = ["expand", "rewrite", "spell"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        'normalize', help="Normalize scores using vector normalization"
    )
    normalize_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    rrf_search_parser = subparsers.add_parser(
        'rrf_search', help="Perform RRF hybrid search"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("--k", type=int, default=60, help="RRF parameter k")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=ENHANCERS, help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual"], help="Reranking method to apply after initial search")

    weighted_search_parser = subparsers.add_parser(
        'weighted_search', help="Perform weighted hybrid search"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weighting factor for semantic search")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    weighted_search_parser.add_argument("--enhance", type=str, choices=ENHANCERS, help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_vector = normalize_vector(args.scores)
            for score in normalized_vector:
                print(f"* {score:.4f}")
        case "rrf_search":
            q = enhance_query(args.query, args.enhance)
            search_rrf(q, args.k, args.limit, args.rerank_method)
        case "weighted_search":
            q = enhance_query(args.query, args.enhance)
            search_hybrid_weighted(q, args.alpha, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()