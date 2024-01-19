import hypll
import torch
from torch import nn

from thg import (Decoder, Encoder, FeedForward, HyMultiHeadAttention,
                 HyTransformer, PositionalEncoding)

d_model = 512
n_heads = 8
output_dim = d_model

mat = nn.Linear(d_model, d_model * 2)
x = torch.randn(10, d_model)
output = mat(x)

a  =torch.randint(0, 100, (2, 3,4)).to(float)
print(a)
#print(torch.softmax(a, dim=-1))
#print(torch.softmax(a, dim=0))
dim = 1
print(torch.softmax(a, dim=dim))
print(torch.softmax(a, dim=dim).sum(dim=dim))