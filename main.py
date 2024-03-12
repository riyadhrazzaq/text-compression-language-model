import argparse

from src.evaluator import evaluate


def main():
    # parser
    parser = argparse.ArgumentParser(description='inference with model.')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument("--decompress", action="store_true", help="decompress the input text")
    parser.add_argument('--vocab', type=str, help='Path to the vocab file')
    parser.add_argument('--text', type=str, help='Text to be tokenized')
    args = parser.parse_args()

    # load model and vocab
    evaluate(args)


if __name__ == "__main__":
    main()
