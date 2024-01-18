import torch

from thg import Decoder, Encoder, HyTransformer, PositionalEncoding


def test_positional():
    d_model = 512
    output_dim = d_model

    pos_enc = PositionalEncoding(d_model=d_model, dropout=0.1, max_sequence_len=5000)
    x = torch.randn(10, 32, 512)
    assert pos_enc(x).shape == x.shape
