import math

import hypll
import torch
from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.nn import HLinear
from hypll.tensors import ManifoldTensor
from torch import Tensor, nn

MANIFOLD = PoincareBall(c=Curvature(1.0))


class HyMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Arguments:
            d_model: the embedding dimension
            num_heads: the number of heads
            dropout: the dropout rate
        """
        super().__init__()
        assert (
            d_model % num_heads == 0
        ), "Embedding dimension must be a multiple of num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_Q = HLinear(d_model, d_model, manifold=MANIFOLD)
        self.w_K = HLinear(d_model, d_model, manifold=MANIFOLD)
        self.w_V = nn.Linear(d_model, d_model)

    def attn_prod(self, Q, K, V):
        attention_score = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_probs = torch.softmax(attention_score, dim=-1)

        return torch.matmul(attention_probs, V)

    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        if type(x) == ManifoldTensor:
            x = x.tensor
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)

    def get_qkv(self, x):
        hy_x = ManifoldTensor(x, MANIFOLD)
        Q = self.w_Q(hy_x)
        Q = self.split_heads(Q)
        K = self.w_K(hy_x)
        K = self.split_heads(K)
        V = self.w_V(x)
        V = self.split_heads(V)

        return Q, K, V

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        Q, K, V = self.get_qkv(x)
        x = self.attn_prod(Q, K, V)
        x = x.transpose(1, 2).contiguous().view(x.size(0), -1, self.d_model)

        return x
