import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    def __init__(
        self, d_model: int, dropout: float = 0.1, max_sequence_len: int = 5000
    ):
        """
        Arguments:
            d_model: the embedding dimension
            dropout: the dropout rate
            max_len: the maximum length of the sequence
        """
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        position = torch.arange(max_sequence_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        positional_enc = torch.zeros(max_sequence_len, 1, d_model)
        positional_enc[:, 0, 0::2] = torch.sin(position * div_term)
        positional_enc[:, 0, 1::2] = torch.cos(position * div_term)
        self.positional_enc = positional_enc

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.positional_enc[: x.size(0)]
        return self.dropout_layer(x)
