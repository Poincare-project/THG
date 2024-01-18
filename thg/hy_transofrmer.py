from torch import nn

from thg.decoder import Decoder
from thg.encoder import Encoder
from thg.positional_encoding import PositionalEncoding


class HyTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_seq_length,
        dropout,
    ):
        super(HyTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_seq_length)

        self.encoders = nn.ModuleList(
            [Encoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoders = nn.ModuleList(
            [Decoder(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
