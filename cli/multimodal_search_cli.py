import argparse

from lib.multimodal_search import image_search_command, verify_image_embedding


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    image_search_parser = subparsers.add_parser(
        "image_search", help="Search for movies using an image"
    )
    image_search_parser.add_argument("--image", type=str, required=True, help="Path to the image to search with")

    verify_image_embedding_parser = subparsers.add_parser(
        "verify_image_embedding", help="Verify image embedding against a text query"
    )
    verify_image_embedding_parser.add_argument("--image", type=str, required=True, help="Path to the image to describe")
    verify_image_embedding_parser.add_argument("--query", type=str, required=True, help="A text query to rewrite based on the image")

    args = parser.parse_args()

    match args.command:
        # Have image_search call the relevant function and print the results in this format:
        # 1. Paddington (similarity: 0.722)
        #    Deep in the rainforests of Peru, a young bear lives peacefully with his Aunt Lucy and Uncle Pastuzo,...
        #
        # 2. Murder She Said (similarity: 0.686)
        #    This is based on the Agatha Christie book "4:50 from Paddington" and the opening locale is Paddingto...
        #
        # 3. Ted (similarity: 0.685)
        #    In 1985, eight-year-old John Bennett makes a Christmas wish that his teddy bear, Ted, would
        case "image_search":
            image_path = args.image
            results = image_search_command(image_path)
            for i, doc in enumerate(results, start=1):
                print(f"{i}. {doc['title']} (similarity {doc['similarity']:.4f})")
                print(f"\t{doc['description']}")
        case "verify_image_embedding":
            image_path = args.image
            query = args.query
            verify_image_embedding(image_path, query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()