import argparse

from lib.image_search import describe_image


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    describe_parser = subparsers.add_parser(
        "describe", help="Describe an image using RAG"
    )
    describe_parser.add_argument("--image", type=str, required=True, help="Path to the image to describe")
    describe_parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")

    args = parser.parse_args()

    match args.command:
        case "describe":
            image_path = args.image
            query = args.query
            describe_image(image_path, query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()