import torch
from torch import nn

from src.util import device


class Transpose(nn.Module):
    def __init__(self, dim0=None, dim1=None):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor):
        if self.dim0 is None:
            self.dim0 = tensor.dim() - 2
            self.dim1 = tensor.dim() - 1

        return torch.transpose(tensor, self.dim0, self.dim1)

class Model2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        state_size,
        pad_index,
    ):
        super().__init__()
        self.state_size = state_size
        self.pad_index = pad_index
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_index,
        )

        self.rnn_layer = nn.LSTMCell(input_size=embedding_dim, hidden_size=state_size)
        self.lin1 = nn.Sequential(
            nn.Linear(state_size, state_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.lin2 = nn.Sequential(
            nn.Linear(state_size * 4, state_size * 8),
            Transpose(),
            nn.BatchNorm1d(state_size * 8),
            Transpose(),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.lin3 = nn.Sequential(
            nn.Linear(state_size * 8, state_size * 16),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.lin4 = nn.Sequential(nn.Linear(state_size * 16, vocab_size))

    def forward(self, X):
        N, T = X.shape
        non_pad_mask = X != self.pad_index
        X = self.embedding_layer(X)

        state = torch.zeros((N, self.state_size), device=device)
        c = torch.zeros((N, self.state_size), device=device)
        states = []
        for t in range(T):
            next_state, next_c = self.rnn_layer(X[:, t, :], (state, c))
            # print(non_pad_mask[:, t].reshape(-1, 1).shape, next_state.shape, state.shape)
            state = torch.where(non_pad_mask[:, t].reshape(-1, 1), next_state, state)
            c = torch.where(non_pad_mask[:, t].reshape(-1, 1), next_c, c)

            states.append(state)

        # (N, T, states)
        states = torch.stack(states, dim=1)
        output = self.lin1(states)
        output = self.lin2(output)
        output = self.lin3(output)
        output = self.lin4(output)

        return output