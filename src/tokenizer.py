import os
from typing import List, Tuple

import numpy as np
import torch
from torchtext.vocab import Vocab
from torch import nn, Tensor

from src.util import device


class Tokenizer(nn.Module):
    def __init__(self, vocab: str | Vocab):
        super().__init__()

        # check vocab file exists
        if isinstance(vocab, str):
            assert os.path.exists(vocab)
            self.vocab = torch.load(vocab, map_location=device)
        else:
            self.vocab = vocab

        self.edge_index = vocab['<EDGE>']
        self.pad_index = vocab['<PAD>']
        self.unk_index = vocab['<UNK>']

    def get_tensors(self, data):
        """
        Builds torch.Tensor from a variable length 2D python list. The return value is a tuple of two tensors, one for input and the other for output.

        Parameters
        ----------
        data: Nested list of token indices
            [[1,2,3],
             [4,2,3,4,2],
             [223,4,2]]
            This example has three sentences.

        """
        max_len = max([len(datum) for datum in data]) + 1
        N = len(data)
        X = np.full((N, max_len), self.pad_index, np.int64)
        Y = np.full((N, max_len), self.pad_index, np.int64)

        for i in range(N):
            # prepend the inputs with edge token
            X[i, 0] = self.edge_index
            for j in range(len(data[i])):
                X[i, j + 1] = data[i][j]
                Y[i, j] = data[i][j]

            # finish the outputs with edge token
            Y[i, j] = self.edge_index

        return torch.tensor(X, device=device), torch.tensor(Y, device=device)

    def forward(self, text: List[str]) -> Tuple[Tensor, Tensor]:
        """
        Tokenizes a list of natural text. The return value is a tensor of token ids.

        Parameters
        ----------
        text: List[str]. A list of natural language strings.

        Returns
        -------
        torch.Tensor. A tensor of token ids.
        """

        text = [sentence.split() for sentence in text]
        tokenized = [self.vocab(sentence) for sentence in text]
        return self.get_tensors(tokenized)
