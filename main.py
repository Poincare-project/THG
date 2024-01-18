import hypll
import torch

from thg import Decoder, Encoder, HyTransformer, PositionalEncoding

pos_enc = PositionalEncoding(d_model=512, dropout=0.1, max_sequence_len=5000)
x = torch.randn(10, 32, 512)
