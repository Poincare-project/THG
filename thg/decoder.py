import torch
from torch import Tensor, nn

from thg.positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding(emb_dim, dropout)
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
