import argparse

from lib.rag import perform_rag, get_summary


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Search query and summarize results"
    )
    summarize_parser.add_argument("query", type=str, help="Query to search and summarize")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of top results to summarize")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            perform_rag(query)
        case "summarize":
            query = args.query
            get_summary(query, limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()