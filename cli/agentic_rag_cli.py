"""CLI for Agentic Recursive RAG."""

import argparse
import json
import logging

from lib.agentic_rag import AgenticRAG, AgenticSearchConfig


def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG - Dynamic multi-tool search system"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Search command
    search_parser = subparsers.add_parser(
        "search",
        help="Perform agentic search that dynamically chooses tools"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of search iterations (default: 5)"
    )
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum results per tool (default: 10)"
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Final number of results to return (default: 5)"
    )
    search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    # Generate command (search + answer)
    generate_parser = subparsers.add_parser(
        "generate",
        help="Perform agentic search and generate an answer"
    )
    generate_parser.add_argument("query", type=str, help="Question or query")
    generate_parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of search iterations (default: 5)"
    )
    generate_parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Maximum results per tool (default: 10)"
    )
    generate_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Final number of results to consider (default: 5)"
    )
    generate_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    match args.command:
        case "search":
            config = AgenticSearchConfig(
                max_iterations=args.max_iterations,
                max_results_per_tool=args.max_results,
                final_result_limit=args.limit,
                debug=args.debug
            )

            agent = AgenticRAG(config)
            result = agent.search(args.query)

            if args.json:
                # Convert search_history to serializable format
                output = {
                    'query': result['query'],
                    'iterations': result['iterations'],
                    'total_unique_results': result['total_unique_results'],
                    'search_history': [
                        {
                            'tool_name': sr.tool_name,
                            'query': sr.query,
                            'num_results': len(sr.results),
                            'reasoning': sr.reasoning
                        }
                        for sr in result['search_history']
                    ],
                    'results': result['results']
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nQuery: {result['query']}")
                print(f"Iterations: {result['iterations']}")
                print(f"Total unique results found: {result['total_unique_results']}")

                # Check if this was an intersection query with no overlap
                tool_names = [sr.tool_name for sr in result['search_history']]
                has_actor = 'actor_search' in tool_names
                has_filter = any(t in tool_names for t in ['genre_search', 'keyword_search', 'regex_search'])
                is_intersection = has_actor and has_filter

                if is_intersection and result['total_unique_results'] > 0:
                    # Check if results have matched_by_count (indicates intersection worked)
                    if result['results'] and 'matched_by_count' in result['results'][0]:
                        print("Merge strategy: INTERSECTION (showing movies matching ALL criteria)")
                    else:
                        print("Merge strategy: INTERSECTION → UNION fallback")
                        print("(No movies matched ALL criteria, showing results from individual searches)")

                print("\n" + "="*60)
                print("SEARCH STRATEGY")
                print("="*60)
                for i, sr in enumerate(result['search_history'], 1):
                    print(f"\n{i}. Tool: {sr.tool_name}")
                    print(f"   Query: {sr.query}")
                    print(f"   Results: {len(sr.results)}")
                    if sr.reasoning:
                        print(f"   Reasoning: {sr.reasoning}")

                print("\n" + "="*60)
                print("TOP RESULTS")
                print("="*60)
                for i, movie in enumerate(result['results'], 1):
                    print(f"\n{i}. {movie['title']}")
                    print(f"   Score: {movie.get('aggregate_score', 0):.4f}")
                    print(f"   Found by: {movie.get('found_by', 'unknown')}")
                    if 'matched_by_count' in movie:
                        print(f"   Matched {movie['matched_by_count']} criteria")
                    print(f"   {movie['description'][:200]}...")

        case "generate":
            config = AgenticSearchConfig(
                max_iterations=args.max_iterations,
                max_results_per_tool=args.max_results,
                final_result_limit=args.limit,
                debug=args.debug
            )

            agent = AgenticRAG(config)

            # Show search strategy
            print("Executing agentic search...\n")
            search_result = agent.search(args.query)

            print("="*60)
            print("SEARCH STRATEGY")
            print("="*60)
            for i, sr in enumerate(search_result['search_history'], 1):
                print(f"{i}. {sr.tool_name}: '{sr.query}' → {len(sr.results)} results")
                if sr.reasoning:
                    print(f"   {sr.reasoning}")

            print("\n" + "="*60)
            print("GENERATING ANSWER")
            print("="*60 + "\n")

            # Generate answer
            answer = agent.search_and_generate(args.query)
            print(answer)

            print("\n" + "="*60)
            print("TOP SOURCES")
            print("="*60)
            for i, movie in enumerate(search_result['results'], 1):
                print(f"[{i}] {movie['title']} (found by {movie.get('found_by', 'unknown')})")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()

