import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, n_embd):
        super().__init__()

        pe = torch.zeros(seq_len, n_embd)
        pos = torch.arange(0, seq_len).type(torch.float32).unsqueeze(1)  # Shape of (seq_len, 1)
        # div = 10_000 ** (torch.arange(0, n_embd, 2) / n_embd)  # 'Original version'
        # 'Cleaner' method, mostly for numerical stability
        div = torch.exp(torch.arange(0, n_embd, 2).type(torch.float32) * (-math.log(10_000.0) / n_embd))  # Shape of (n_embd//2)
        pe[:, 0::2] = torch.sin(pos * div)  # Original div would be /, but this version uses * due to rearrangement
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        return self.pe[:x.shape[1], :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, n_heads, device):
        super().__init__()
        assert n_embd % n_heads == 0, f"n_embd must be divisible by n_heads! {n_embd=}, {n_heads=}"

        self.n_heads = n_heads
        self.h_dim = n_embd // n_heads
        self.device = device

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)


    def forward(self, x: torch.Tensor, mask: bool):
        assert len(x.shape) == 3, f"x should be a tensor of shape (batch, seq_len, n_embd), but instead got {x.shape=}"
        batch_size, seq_len, n_embd = x.shape

        # (batch, seq_len, n_embd) -> (batch, seq_len, n_embd * 3) -> (batch, seq_len, n_heads, h_dim * 3)
        qkv = self.qkv(x).reshape(batch_size, seq_len, self.n_heads, self.h_dim * 3)

        # (batch, seq_len, n_heads, h_dim * 3) -> (batch, n_heads, seq_len, h_dim * 3)
        # -> (batch, n_heads, seq_len, h_dim) split among q, k, v
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)


        # Apply the attention mechanism
        # (batch, n_heads, seq_len, h_dim) @ (batch, n_heads, h_dim, seq_len) -> (batch, n_heads, seq_len, seq_len)
        h = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)

        if mask:  # Apply causal mask when in Decoder Self Attention
            m = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
            h = h.masked_fill(~m, -1e9)

        h = nn.functional.softmax(h, dim=-1)

        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, h_dim) -> (batch, n_heads, seq_len, h_dim)
        h = h @ v


        # Now we permute/reshape back into original input shape
        # (batch, n_heads, seq_len, h_dim) -> (batch, seq_len, n_heads, h_dim) -> (batch, seq_len, n_embd)
        h = h.permute(0, 2, 1, 3).reshape(batch_size, seq_len, n_embd)

        # Apply W_o and return
        return self.out_proj(h)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        assert n_embd % n_heads == 0, f"n_embd must be divisible by n_heads! {n_embd=}, {n_heads=}"

        self.n_heads = n_heads
        self.h_dim = n_embd // n_heads

        self.q = nn.Linear(n_embd, n_embd)
        self.k = nn.Linear(n_embd, n_embd)
        self.v = nn.Linear(n_embd, n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)

    def forward(self, q, k, v):
        # In cross attention, query comes from decoder and kv comes from encoder
        # Batch size and n_embd of these should be the same, though seq_len would vary

        assert len(q.shape) == len(k.shape) == len(v.shape) == 3, f"{q.shape=}, {k.shape=}, {v.shape=}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"{q.shape[0]=}, {k.shape[0]=}, {v.shape[0]=}"
        assert q.shape[2] == k.shape[2] == v.shape[2], f"{q.shape[2]=}, {k.shape[2]=}, {v.shape[2]=}"

        batch, seq_len_q, n_embd = q.shape
        seq_len_kv = k.shape[1]


        # q.shape == (batch, seq_len_q, n_embd)
        # kv.shape == (batch, seq_len_kv, n_embd)
        q, k, v = self.q(q), self.k(k), self.v(v)

        # (batch, seq_len_q, n_embd) -> (batch, seq_len_q, n_heads, h_dim) -> (batch, n_heads, seq_len_q, h_dim)
        q = q.reshape(batch, seq_len_q, self.n_heads, self.h_dim).permute(0, 2, 1, 3)

        # (batch, seq_len_q, n_embd) -> (batch, seq_len_kv, n_heads, h_dim) -> (batch, n_heads, seq_len_kv, h_dim)
        k = k.reshape(batch, seq_len_kv, self.n_heads, self.h_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch, seq_len_kv, self.n_heads, self.h_dim).permute(0, 2, 1, 3)

        # Apply attention mechanism
        # (batch, n_heads, seq_len_q, h_dim) @ (batch, n_heads, h_dim, seq_len_kv) -> (batch, n_heads, seq_len_q, seq_len_kv)
        h = q @ k.transpose(-2, -1) / (self.h_dim ** 0.5)

        # No mask needed for the CA portion, straight to softmax
        h = nn.functional.softmax(h, dim=-1)

        # (batch, n_heads, seq_len_q, seq_len_kv) @ (batch, n_heads, seq_len_kv, h_dim) -> (batch, n_heads, seq_len_q, h_dim)
        h = h @ v

        # Permute and Reshape back to original q shape
        # (batch, n_heads, seq_len_q, h_dim) -> (batch, seq_len_q, n_heads, h_dim) -> (batch, seq_len_q, n_embd)
        h = h.permute(0, 2, 1, 3).reshape(batch, seq_len_q, n_embd)

        return self.out_proj(h)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        # A simple feedforward layer based on the original paper
        self.seq = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.GELU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_heads, dropout, device):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(n_embd, n_heads, device)
        self.ln1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffwd = FeedForward(n_embd, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.dropout1(self.mhsa(self.ln1(x), mask=False))
        x = x + self.dropout2(self.ffwd(self.ln2(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str):
        super().__init__()
        # Here's the entire Encoder Block, comprised of n_layers
        self.enc_layers = nn.Sequential(*[EncoderLayer(n_embd=n_embd, n_heads=n_heads, dropout=dropout, device=device) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor):
        # Pass it through sequentially
        x = self.enc_layers(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_heads, dropout, device):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(n_embd, n_heads, device)
        self.ln1 = nn.LayerNorm(n_embd)
        self.dropout1 = nn.Dropout(p=dropout)

        self.mhca = MultiHeadCrossAttention(n_embd, n_heads)
        self.ln2_1 = nn.LayerNorm(n_embd)
        self.ln2_2 = nn.LayerNorm(n_embd)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffwd = FeedForward(n_embd, dropout)
        self.ln3 = nn.LayerNorm(n_embd)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, enc_output: torch.tensor, x: torch.tensor):
        x = x + self.dropout1(self.mhsa(self.ln1(x), mask=True))
        x = x + self.dropout2(self.mhca(self.ln2_1(x), self.ln2_2(enc_output), self.ln2_2(enc_output)))
        x = x + self.dropout3(self.ffwd(self.ln3(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers, dropout, device):
        super().__init__()
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(n_embd=n_embd, n_heads=n_heads, dropout=dropout, device=device) for _ in range(n_layers)]
        )

    def forward(self, enc: torch.Tensor, dec: torch.Tensor):
        for i in range(len(self.dec_layers)):
            dec = self.dec_layers[i](enc, dec)
        return dec


class Transformer(nn.Module):
    # Define the overall Transformer class
    def __init__(self, vocab_size: int, seq_len: int, n_embd: int, n_heads: int, n_layers: int, dropout: float, device: str):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.device = device

        self.emb = nn.Embedding(vocab_size, n_embd)
        self.pos_enc = PositionalEncoding(seq_len, n_embd)

        self.enc_block = EncoderBlock(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, dropout=dropout, device=device)
        self.dec_block = DecoderBlock(n_embd=n_embd, n_heads=n_heads, n_layers=n_layers, dropout=dropout, device=device)
        self.projection = nn.Linear(n_embd, vocab_size)

        self.projection.weight = self.emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # This is from Dr. Andrej Karpthy's ng_video_lecture repo on github
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, enc_input: torch.Tensor, dec_input: torch.Tensor):
        enc_input = self.emb(enc_input) * (self.n_embd ** 0.5)
        enc_input += self.pos_enc(enc_input)
        enc_output = self.enc_block(enc_input)

        dec_input = self.emb(dec_input) * (self.n_embd ** 0.5)
        dec_input += self.pos_enc(dec_input)
        dec_output = self.dec_block(enc_output, dec_input)

        output = self.projection(dec_output)
        return output

    @torch.no_grad()
    def translate(self, src: torch.Tensor, sos_token: int, eos_token: int, k=10) -> torch.Tensor:
        self.eval()
        tgt = torch.tensor([sos_token], device=self.device).unsqueeze(0)

        for i in range(self.seq_len):  # Iterate until EOS token is sampled or at seq_len
            logits = self(src, tgt)  # Output would be of shape (batch, seq_len, vocab_size)
            logits = logits[0, -1, :]  # First batch last token

            probs = F.softmax(logits, dim=-1)
            top_k_probs, top_k_idx = torch.topk(probs, k=k, dim=-1)  # Using top-k method for sampling tokens

            token = torch.multinomial(top_k_probs, num_samples=1)
            token_index = top_k_idx[token]

            tgt = torch.cat((tgt, token_index.unsqueeze(0)), dim=1)

            if token_index.item() == eos_token:  # Break out of loop if sampled EOS Token
                break

        self.train()
        return tgt  # If we reach this point, meaning len(tgt) == seq_len, return