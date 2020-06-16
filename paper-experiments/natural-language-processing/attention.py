import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedAttention(nn.Module):
    """
    Implementation of Fixed Encoder Self-Attention Patterns in Transformer-Based MachineTranslation
    by Alessandro Raganato, Yves Scherrer and J ̈org Tiedemann
    https://arxiv.org/abs/2002.10260
    with an adaptation to language modelling (can't attend to future tokens).
    """
    def __init__(self, d_model, language_model=True):
        super().__init__()
        self.d_model = d_model
        self.language_model = language_model
        if language_model:
            self.n_heads = 4
        else:
            self.n_heads = 8

        # project to d_model but it's actually n_heads * d_model / n_heads
        self.linearV = nn.Linear(d_model, d_model, bias=True)
        self.h4mat = None

    def make_heads(self, n_tokens, device):
        self.h4mat = torch.stack([self.energyfunc(0, i - 2, n_tokens) for i in range(n_tokens)]).to(device)  # past
        if self.language_model:
            self.h2perm = torch.LongTensor([0] + [i for i in range(n_tokens - 1)])
            self.hlm_mat = torch.stack([self.uniform_head(0, i - 2, n_tokens) for i in range(n_tokens)]).to(device)
        else:
            self.h2perm = torch.LongTensor([-1] + [i for i in range(n_tokens - 1)])         # previous token
            self.h3perm = torch.LongTensor([i for i in range(1, n_tokens)] + [-1])          # next token
            self.h5mat = self.h4mat[::-1, ::-1]                                             # future
            self.h6vec = self.energyfunc(0, n_tokens, n_tokens).to(device)                  # all forwards
            self.h7vec = self.h6vec[::-1]                                                   # all backwards
            self.h8idx = torch.LongTensor([n_tokens] for i in range(n_tokens))              # last token

    @staticmethod
    def uniform_head(start, end, n_tokens):
        if end <= start:
            return torch.FloatTensor([1] + [0] * (n_tokens - 1))
        n = end - start
        weights = [1 / n] * n + [0] * (n_tokens - end)
        return torch.FloatTensor(weights)

    @staticmethod
    def energyfunc(start, end, n_tokens):
        if end <= start:
            return torch.FloatTensor([1] + [0] * (n_tokens - 1))
        unnorm = [(i + 1) ** 3 for i in range(start, end)] + [0] * (n_tokens - end)
        unnorm = torch.FloatTensor(unnorm)
        return unnorm / unnorm.sum()

    @staticmethod
    def token_matmul(A, B):
        """
        This was probably implementable with some transpose and torch.bmm
        :param A: attention matrix of shape (n_tokens, n_tokens)
        :param B: features of shape (n_tokens, batch_size, d)
        :return: tensor of the same shape as B
        """
        n = B.size(0)
        return torch.einsum('ij,jkl->ikl', A[:n, :n], B)  # slice is only useful for last batch

    def forward(self, x):
        """Unlike MultiHeadAttention, FixedAttention does not use masking arguments
        as masking is already included in the definition of the fixed heads."""

        if self.h4mat is None:
            device = self.linearV.weight.device
            self.make_heads(x.size(0), device)

        # x is of shape (n_tokens, batch_size, dmodel)
        V = self.linearV(x)
        V = V.view(V.size(0), V.size(1), -1, self.n_heads)

        h1 = V[:, :, :, 0]
        h2 = V[self.h2perm[:V.size(0)], :, :, 1]
        h4 = self.token_matmul(self.h4mat, V[:, :, :, 3])

        if self.language_model:
            h3 = self.token_matmul(self.hlm_mat, V[:, :, :, 2])
            out = torch.cat((h1, h2, h3, h4), dim=2)  # again same shape as x
        else:
            h3 = V[self.h3perm, :, :, 2]
            h5 = self.token_matmul(self.h5mat, V[:, :, :, 4])
            h6 = self.token_matmul(self.h6vec, V[:, :, :, 5])
            h7 = self.token_matmul(self.h7vec, V[:, :, :, 6])
            h8 = V[self.h8idx, :, :, 7]
            out = torch.cat((h1, h2, h3, h4, h5, h6, h7, h8), dim=2)

        return out


class SynthesizerAttention(nn.Module):
    def __init__(self, d_model, n_heads, synth='dense', dropout=0.1):
        super().__init__()
        self.d_model = d_model
        if synth == 'dense':
            self.linear_in = nn.Linear(d_model, 2 * d_model, bias=True)
        elif synth == 'random':
            self.linear_in = nn.Linear(d_model, d_model, bias=True)
        else:
            raise ValueError("synth must be 'dense' or 'random'")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.attn = None
        self.synth = synth

        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_model, d_model, bias=True)

    def forward(self, x, attention_mask=None):
        """
        :param x: array of shape (n_tokens, batch_size, d_model)
        :param attention_mask: array of shape (n_tokens, n_tokens)
        :return: attention output of shape (n_tokens, batch_size, d_model)
        """

        n_tokens, batch_size, embed_dim = x.size()
        assert embed_dim == self.d_model, "Embedding dimension is wrong"
        if self.attn is None:
            if self.synth == 'dense':
                self.attn = DenseSynthesizer(self.head_dim, self.n_heads, n_tokens).to(x.device)
            elif self.synth == 'random':
                self.attn = RandomSynthesizer(n_tokens, self.n_heads).to(x.device)

        if self.synth == 'dense':
            x, v = self.linear_in(x).chunk(2, dim=-1)  # proj to allow multihead
        elif self.synth == 'random':
            v = self.linear_in(x)
        x = x.contiguous().view(n_tokens, batch_size, self.n_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(n_tokens, batch_size * self.n_heads, self.head_dim).transpose(0, 1)

        attention_logits = self.attn(x)  # shape (batch_size * n_heads, n_tokens, n_tokens)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)
            try:
                attention_logits += attention_mask
            except RuntimeError:
                print('Query size:', x.size())
                print('Value size:', v.size())
                print('Attention weights size:', attention_logits.size())
                print('Attention mask size:', attention_mask.size())
                raise RuntimeError("Number of tokens not compatible. Check your handling of the last batch.")

        attention_weights = F.softmax(attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention_output = torch.bmm(attention_weights, v)  # (batch_size * n_heads, n_tokens, head_dim)
        attention_output = attention_output.transpose(0, 1).contiguous().view(n_tokens, batch_size, embed_dim)
        attention_output = self.linear_out(attention_output)
        return attention_output


class DenseSynthesizer(nn.Module):
    def __init__(self, head_dim, n_heads, n_tokens, big=True):
        super().__init__()
        h = max(head_dim, n_tokens) if big else min(head_dim, n_tokens)
        w1 = torch.empty(n_heads, head_dim, h)
        b1 = torch.empty(n_heads, h)
        w2 = torch.empty(n_heads, h, n_tokens)
        b2 = torch.empty(n_heads, n_tokens)

        nn.init.kaiming_uniform_(w1)
        nn.init.kaiming_uniform_(w2)
        nn.init.zeros_(b1)
        nn.init.zeros_(b2)

        self.register_parameter('w1', nn.Parameter(w1))
        self.register_parameter('b1', nn.Parameter(b1))
        self.register_parameter('w2', nn.Parameter(w2))
        self.register_parameter('b2', nn.Parameter(b2))

        self.activation = nn.ReLU()

    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, n_tokens, n_heads, head_dim)
        :return: tensor of shape (batch_size * n_heads, n_tokens, n_tokens)
        """
        bs, l, nh, dh = x.size()  # l is n_tokens from the init except for the last batch
        x = torch.einsum('ijkl,klm->ijkm', x, self.w1) + self.b1
        x = self.activation(x)
        x = torch.einsum('ijkl,klm->ijkm', x, self.w2) + self.b2  # (batch_size, l, n_heads, n_tokens)
        x = x[:, :, :, :l]
        x = x.transpose(0, 3).contiguous().view(l, l, bs * nh).transpose(0, 2)
        return x


class RandomSynthesizer(nn.Module):
    def __init__(self, n_tokens, n_heads=8, fixed=False):
        super().__init__()
        self.n_heads = n_heads
        R = torch.empty(n_heads, n_tokens, n_tokens)
        nn.init.kaiming_uniform_(R)
        if fixed:
            self.register_buffer('attention_matrix', R)
        else:
            self.register_parameter('attention_matrix', nn.Parameter(R))

    def forward(self, x):
        """
        Returns attention logits that don't depend on x, except for outputting the correct shape.
        :param x: tensor of shape (batch_size,  n_tokens, n_heads, head_dim)
        :return: tensor of shape (batch_size * n_heads, n_tokens, n_tokens)
        """
        bs, l, h, dh = x.size()
        out = self.attention_matrix[:, :l, :l]  # slicing for the last batch
        return out.repeat((bs, 1, 1))
