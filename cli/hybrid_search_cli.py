import argparse
from lib.hybrid_search import normalize_vector


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser(
        'normalize', help="Normalize scores using vector normalization"
    )
    normalize_parser.add_argument("scores", nargs="+", type=float, help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_vector = normalize_vector(args.scores)
            for score in normalized_vector:
                print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()