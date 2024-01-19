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
print(output.shape)
from itertools import product

# Supposons que vous ayez trois listes
liste1 = [1, 2]
liste2 = ['a', 'b']
liste3 = ['x', 'y']

# Utilisez la fonction product pour obtenir toutes les combinaisons possibles
combinaisons = list(product(liste1, liste2, liste3))

# Affichez le résultat
print(combinaisons)
