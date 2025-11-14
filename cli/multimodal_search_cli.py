import argparse

from lib.multimodal_search import verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding against a text query"
    )
    verify_image_embedding_parser.add_argument("--image", type=str, required=True, help="Path to the image to describe")
    verify_image_embedding_parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            image_path = args.image
            query = args.query
            verify_image_embedding(image_path, query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()