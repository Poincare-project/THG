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

d_model_list = [128, 512, 1024]
n_heads_list = [8, 16]
batch_size_list = [32, 64]
seq_len_list = [10, 20]
d_ff_list = [2048, 4096]
dropout_list = [0.1, 0.2, 0.3]
mh_parametrize_set = list(
    itertools.product(d_model_list, n_heads_list, batch_size_list, seq_len_list)
)


@pytest.mark.parametrize(
    "d_model, n_heads, batch_size, seq_len",
    mh_parametrize_set,
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


encoder_parametrize_set = list(
    itertools.product(d_model_list, n_heads_list, d_ff_list, dropout_list)
)


@pytest.mark.parametrize(
    "d_model, n_heads, d_ff, dropout",
    encoder_parametrize_set,
)
def test_encoder(d_model, n_heads, d_ff, dropout):
    x = torch.randn(10, 32, d_model)
    encoder = Encoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
    assert encoder(x).shape == x.shape
