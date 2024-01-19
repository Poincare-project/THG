import pytest
import torch

from thg import (Decoder, Encoder, FeedForward, HyMultiHeadAttention,
                 HyTransformer, PositionalEncoding)


def test_positional():
    d_model = 512
    output_dim = d_model

    pos_enc = PositionalEncoding(d_model=d_model, dropout=0.1, max_sequence_len=5000)
    x = torch.randn(10, 32, 512)
    assert pos_enc(x).shape == x.shape


def test_feed_forward():
    d_model = 512
    output_dim = 2048

    ff = FeedForward(d_model=d_model, d_ff=output_dim, dropout=0.1)
    x = torch.randn(10, 512)
    assert ff(x).shape == x.shape

import itertools


@pytest.mark.parametrize(
    "d_model, n_heads, batch_size, seq_len",
    [
        (512, 8, 32, 10),
        (512, 8, 32, 20),
        (512, 8, 32, 30),
        (512, 8, 32, 40),
        (1024, 8, 32, 50),
    ],
)
def test_multihead_attention(d_model, n_heads, batch_size, seq_len):
    d_model = 512
    n_heads = 8

    mha = HyMultiHeadAttention(d_model=d_model, num_heads=n_heads, dropout=0.1)
    batch_size = 1
    seq_len = 10
    x = torch.randn(batch_size, seq_len, d_model)

    Q, K, V = mha.get_qkv(x)
    assert Q.shape == (batch_size, n_heads, seq_len, d_model // n_heads)
    assert K.shape == (batch_size, n_heads, seq_len, d_model // n_heads)
    assert V.shape == (batch_size, n_heads, seq_len, d_model // n_heads)


def test_encoder():
    pass
