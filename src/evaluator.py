from src.model import Model2
from src.tokenizer import Tokenizer
from src.util import *


def evaluate(args):
    vocab = torch.load(args.vocab, map_location=torch.device('cpu'))
    model = Model2(len(vocab), 300, 256, vocab['<PAD>'])
    load_from_checkpoint(model, args.checkpoint)

    print()
    if args.decompress:
        print(decompress(args.text, Tokenizer(vocab), model))
    else:
        print(compress(args.text, Tokenizer(vocab), model))
