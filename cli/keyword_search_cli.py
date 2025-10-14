#!/usr/bin/env python3

import argparse

from lib.keyword_search import (
    build_command,
    idf_command,
    search_command,
    tf_command,
    tfidf_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    subparsers.add_parser("build", help="Build movies inverted index")
    
    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency")
    idf_parser.add_argument("term", type=str, help="IDF term")
    
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    
    tf_parser = subparsers.add_parser("tf", help="Get term frequency")
    tf_parser.add_argument("doc_id", type=int, help="TF document ID")
    tf_parser.add_argument("term", type=str, help="TF term")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Get TD-IDF")
    tfidf_parser.add_argument("doc_id", type=int, help="TD-IDF document ID")
    tfidf_parser.add_argument("term", type=str, help="TD-IDF term")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(
                "Term frequency for:\n",
                f"\t- Document ID = {args.doc_id}",
                "\n",
                f"\t- Term = {args.term}",
                f"\nTF: {tf}"
            )
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}"
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()