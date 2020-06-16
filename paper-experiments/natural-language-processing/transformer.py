import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinydfa import DFA, DFALayer
import attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        if dropout == 0 or dropout is None:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LMTransformer(nn.Module):
    """
    Transformer for language modelling, like GPT.
    kwargs is passed to CustomTransformerEncoderLayer
    """
    def __init__(self, vocab_size, d_model, n_heads, dim_feedforward, n_layers, dropout=0.1,
                 tie_embeddings=False, dfa='none', no_training=False, dfa_after_vocab=False,
                 dfa_embed=False, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        layer = CustomTransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, dfa=dfa, **kwargs)
        self.transformer_encoder = nn.TransformerEncoder(layer, n_layers)

        self.dfa = None
        self.dfaemb = None
        if dfa in ['simple', 'full']:
            self.dfaemb = [DFALayer()] if dfa_embed else []
            dfas = self.dfaemb + [dfalayer for layer in self.transformer_encoder.layers for dfalayer in layer.dfas]
            self.dfa = DFA(dfas, no_training=no_training, batch_dims=(0, 1))
        self.dfa_after_vocab = dfa_after_vocab

        self.decoder = None
        if not tie_embeddings:
            self.decoder = nn.Linear(d_model, vocab_size)
        self.tie_embeddings = tie_embeddings

        self.d_model = d_model
        self.src_mask = None
        self.init_weights()

    @staticmethod
    def _generate_square_subsequent_mask(size):
        mask = torch.tril(torch.ones(size, size)) == 1
        mask = mask.float()\
            .masked_fill(mask == 0, float('-inf'))\
            .masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        if self.decoder:
            self.decoder.bias.data.zero_()
            self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.embedding(src) * math.sqrt(self.d_model)
        if self.dfaemb:
            src = self.dfaemb[0](src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)

        if self.dfa and not self.dfa_after_vocab:
            output = self.dfa(output)

        if self.tie_embeddings:
            output = F.linear(output, weight=self.embedding.weight)
        else:
            output = self.decoder(output)

        if self.dfa and self.dfa_after_vocab:
            output = self.dfa(output)

        return output


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layernorm=True, attn='standard', dfa='none'):
        super().__init__()

        if attn == 'standard':
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        elif attn == 'fixed':
            self.self_attn = attention.FixedAttention(d_model, language_model=True)
        elif attn in ['dense', 'random']:
            self.self_attn = attention.SynthesizerAttention(d_model, n_heads,
                                                            synth=attn, dropout=dropout)
        else:
            raise ValueError("attn must be in ['standard', 'fixed', 'dense', 'random']")

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            print("WARNING: layer normalization is deactivated")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.dfas = []
        if dfa == 'simple':
            self.dfas = [DFALayer()]
        elif dfa == 'full':
            self.dfas = [DFALayer(), DFALayer(), DFALayer()]

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if isinstance(self.self_attn, nn.MultiheadAttention):
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
        elif isinstance(self.self_attn, attention.FixedAttention):
            src2 = self.self_attn(src)
        elif isinstance(self.self_attn, attention.SynthesizerAttention):
            src2 = self.self_attn(src, attention_mask=src_mask)
        else:
            raise NotImplementedError()

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        if len(self.dfas) == 3:
            src = self.dfas[0](src)

        src2 = self.dropout(self.activation(self.linear1(src)))
        if len(self.dfas) == 3:
            src2 = self.dfas[1](src2)

        src2 = self.linear2(src2)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if len(self.dfas) in [1, 3]:
            src = self.dfas[-1](src)

        return src
