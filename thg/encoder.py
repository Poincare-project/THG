from torch import nn

from thg.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.positional_encoding = PositionalEncoding(emb_dim, dropout)
