from src.evaluator import evaluate
import argparse


parser = argparse.ArgumentParser(description='inference test with model.')
parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint file', default='model_lr0.0001_bs256_epoch50.pt')
parser.add_argument("--decompress", action="store_true", help="decompress the input text", default=False)
parser.add_argument('--vocab', type=str, help='Path to the vocab file', default='vocab.pt')
parser.add_argument('--text', type=str, help='Text to be tokenized', default="""dr. tonie mcdonald is a life long levittown resident who taught and rose through the ranks of the district she now leads .
he received his ba in chemistry , magna cum laude , from amherst college in 1 9 8 1 .""")
args = parser.parse_args()

print("--- input ---")
print(args.text)

# compress
print("--- compress ---")
evaluate(args)

# decompress
print("--- decompress ---")
args.decompress = True
args.text = """dr. tonie mcdonald is X life long levittown resident who taught and rose through X ranks of the district she now leads .
he received his ba X chemistry X magna cum laude X from amherst college in X X 8 1 ."""
evaluate(args)
