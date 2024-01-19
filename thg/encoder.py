from torch import nn

from thg.feed_forward import FeedForward
from thg.hy_multihead_attention import HyMultiHeadAttention
from thg.positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.self_attention = HyMultiHeadAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x):
        x = self.positional_encoding(x)
        x = self.layer_norm1(x + self.dropout(self.self_attention(x)))
        x = self.layer_norm2(x + self.dropout(self.feed_forward(x)))
        return x
