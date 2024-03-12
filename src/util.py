import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def stringify(array):
    return  '\n'.join([' '.join(inner_list) for inner_list in array])


def compress(text, tokenizer, model):
    """
    tokenizer: Tokenizer.
    text: str.
        Each line represents a single document.
    """
    tokens = [sentence.split() for sentence in text.split("\n")]
    indices, _ = tokenizer(text.split("\n"))

    logits = model(indices)
    next_token_predicted = logits.argmax(dim=2)

    # slices are for skipping edge tokens
    prediction_mask = indices[:, 1:] == next_token_predicted[:, :-1]

    # replace correctly predicted tokens with "X"
    for i, sentence_mask in enumerate(prediction_mask):
        sentence_len = len(tokens[i])
        for j, predicted_successfully in enumerate(sentence_mask):
            # length check is to ignore pad tokens
            if predicted_successfully and j < sentence_len and tokenizer.vocab[tokens[i][j]] != tokenizer.unk_index:
                tokens[i][j] = "X"

    sentences = [" ".join(sentence) for sentence in tokens]
    document = "\n".join(sentences)
    return document


def decompress(text, tokenizer, model):
    """
    text: str.
        Each line represents a single document.
    """
    sentence_tokens = [document.split() for document in text.split("\n")]
    indices, _ = tokenizer(text.split("\n"))

    uncompressed = []
    for i, sentence in enumerate(sentence_tokens):
        prefix = ['<EDGE>']
        for j, token in enumerate(sentence):
            if token != "X":
                prefix.append(token)
            else:
                # only infer when X is found
                indices = torch.tensor([tokenizer.vocab(prefix)],
                                       dtype=torch.int,
                                       device=device)
                logits = model(indices)
                # prediction logit for X
                logit = logits[:, -1, :]
                index = logit.argmax(dim=1)
                prefix.append(tokenizer.vocab.lookup_token(index))

        # reset prefix for new sentence
        uncompressed.append(prefix[1:])

    return stringify(uncompressed)


def load_from_checkpoint(model, checkpoint_path):
    """
    Loads a model from a checkpoint.

    Parameters:
    ----------
    checkpoint_path: The path to the checkpoint.

    Raises:
    ------
    Exception: If no checkpoint is found in the provided path.
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"loaded existing model.")
    else:
        raise Exception("No checkpoint found in the provided path")
