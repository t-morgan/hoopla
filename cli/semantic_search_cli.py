#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_movies
from lib.text_chunker import chunk_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk text into smaller pieces")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk_size", type=int, default=200, help="Size of each chunk in words")

    embedding_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embedding_parser.add_argument("text", type=str, help="Text to generate embedding for")

    embedquery_parser = subparsers.add_parser("embedquery", help="Generate embedding for a given query text")
    embedquery_parser.add_argument("text", type=str, help="Query text to generate embedding for")

    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of top results to return")

    subparsers.add_parser("verify_embeddings", help="Verify if the embeddings load correctly")

    subparsers.add_parser("verify", help="Verify if the semantic search model loads correctly")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunks = chunk_text(args.text, chunk_size=args.chunk_size)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks):
                print(f"{i + 1}. {chunk}")
        case "embedquery":
            embed_query_text(args.text)
        case "embed_text":
            embed_text(args.text)
        case "search":
            search_movies(args.query, limit=args.limit)
        case "verify_embeddings":
            verify_embeddings()
        case "verify":
            verify_model()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()