"""UniTraj network (Zhu et al., NeurIPS 2025), adapted from github.com/Yasoz/UniTraj
(Apache License 2.0). Changes: no timm/einops dependency, deterministic masking via an
explicit generator, padding-aware helpers. Parameter names are unchanged, so the public
`model.pt` and checkpoints trained with the original repository load directly."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        s = torch.einsum("i,j->ij", torch.arange(max_seq_len).float(), inv_freq)
        self.register_buffer("sin", s.sin(), persistent=False)
        self.register_buffer("cos", s.cos(), persistent=False)

    def forward(self, n):
        return self.sin[:n][None, None], self.cos[:n][None, None]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=64, dropout=0.0, max_seq_len=512):
        super().__init__()
        inner = head_dim * num_heads
        self.h, self.d, self.scale = num_heads, head_dim, head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = (nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
                       if not (num_heads == 1 and head_dim == dim) else nn.Identity())
        self.rotary_emb = RotaryEmbedding(head_dim, max_seq_len)

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = (t.view(b, n, self.h, self.d).transpose(1, 2) for t in self.to_qkv(self.norm(x)).chunk(3, -1))
        sin, cos = self.rotary_emb(n)
        half = self.d // 2
        rot = lambda t: torch.cat([t[..., :half] * cos - t[..., half:] * sin, t[..., half:] * cos + t[..., :half] * sin], -1)
        attn = self.dropout(torch.softmax(rot(q) @ rot(k).transpose(-1, -2) * self.scale, -1))
        return self.to_out((attn @ v).transpose(1, 2).reshape(b, n, self.h * self.d))


class Transformer(nn.Module):
    def __init__(self, dim, depth, num_heads, head_dim, ff_dim, dropout=0.0, max_seq_len=512):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([Attention(dim, num_heads, head_dim, dropout, max_seq_len),
                                                    FeedForward(dim, ff_dim, dropout)]) for _ in range(depth)])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def take_indices(seq, idx):          # seq [T,B,C], idx [T,B]
    return torch.gather(seq, 0, idx.unsqueeze(-1).expand(-1, -1, seq.shape[-1]))


class Encoder(nn.Module):
    def __init__(self, trajectory_length=200, patch_size=1, embedding_dim=128, num_layers=8, num_heads=4):
        super().__init__()
        self.num_tokens = trajectory_length // patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.tokenizer = nn.Conv1d(2, embedding_dim, patch_size, patch_size)
        self.transformer = Transformer(embedding_dim, num_layers, num_heads, embedding_dim // num_heads, embedding_dim * 4)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, trajectory, interval_embedding, forward_idx, n_visible):
        """forward_idx [T,B]: permutation putting visible tokens first (masked last)."""
        tokens = self.tokenizer(trajectory).permute(2, 0, 1) + interval_embedding.transpose(0, 1)
        tokens = take_indices(tokens, forward_idx)[:n_visible]
        tokens = torch.cat([self.cls_token.expand(-1, tokens.shape[1], -1), tokens], 0)
        feats = self.layer_norm(self.transformer(tokens.transpose(0, 1)))
        return feats.transpose(0, 1)                  # [n_visible+1, B, C]


class Decoder(nn.Module):
    def __init__(self, trajectory_length=200, patch_size=1, embedding_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.patch_size, self.num_tokens = patch_size, trajectory_length // patch_size
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.transformer = Transformer(embedding_dim, num_layers, num_heads, embedding_dim // num_heads, embedding_dim * 4)
        self.head = nn.Linear(embedding_dim, 2 * patch_size)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.time_token, std=0.02)

    def forward(self, features, backward_idx, interval_embedding):
        B = features.shape[1]
        backward_idx = torch.cat([torch.zeros(1, B, dtype=backward_idx.dtype, device=backward_idx.device),
                                  backward_idx + 1], 0)
        n_masked = backward_idx.shape[0] - features.shape[0]
        features = torch.cat([features, self.mask_token.expand(n_masked, B, -1)], 0)
        features = take_indices(features, backward_idx)
        emb = torch.cat([self.time_token.expand(B, 1, -1), interval_embedding], 1).transpose(0, 1)
        feats = self.transformer((features + emb).transpose(0, 1)).transpose(0, 1)[1:]
        patches = self.head(feats)                    # [T, B, 2p]
        T = patches.shape[0]
        return patches.permute(1, 2, 0).reshape(B, 2, self.patch_size, T).permute(0, 1, 3, 2).reshape(B, 2, T * self.patch_size)


class UniTraj(nn.Module):
    def __init__(self, trajectory_length=200, patch_size=1, embedding_dim=128, encoder_layers=8, encoder_heads=4,
                 decoder_layers=4, decoder_heads=4, **_):
        super().__init__()
        self.trajectory_length, self.patch_size = trajectory_length, patch_size
        self.encoder = Encoder(trajectory_length, patch_size, embedding_dim, encoder_layers, encoder_heads)
        self.decoder = Decoder(trajectory_length, patch_size, embedding_dim, decoder_layers, decoder_heads)
        self.interval_embedding = nn.Linear(1, embedding_dim)

    @staticmethod
    def permutations(hidden: np.ndarray, rng: np.random.Generator):
        """hidden [B,T] bool (equal count per row) -> forward/backward index tensors [T,B].
        Visible tokens are shuffled (as in pre-training), hidden ones are placed last."""
        fw = []
        for row in hidden:
            vis, hid = np.where(~row)[0], np.where(row)[0]
            fw.append(np.concatenate([rng.permutation(vis), hid]))
        fw = np.stack(fw, 1)
        return torch.as_tensor(fw), torch.as_tensor(np.argsort(fw, 0)), int((~hidden[0]).sum())

    def forward(self, trajectory, intervals, hidden: np.ndarray, rng: np.random.Generator = None):
        rng = rng or np.random.default_rng()
        ie = self.interval_embedding(intervals.unsqueeze(-1))
        fw, bw, n_vis = self.permutations(hidden, rng)
        fw, bw = fw.to(trajectory.device), bw.to(trajectory.device)
        feats = self.encoder(trajectory, ie, fw, n_vis)
        return self.decoder(feats, bw, ie), feats

    def embed(self, trajectory, intervals, hidden: np.ndarray, pooling: str = "cls"):
        ie = self.interval_embedding(intervals.unsqueeze(-1))
        fw, _, n_vis = self.permutations(hidden, np.random.default_rng(0))
        feats = self.encoder(trajectory, ie, fw.to(trajectory.device), n_vis)     # [n_vis+1, B, C]
        if pooling == "cls":
            return feats[0]
        if pooling == "mean":
            return feats[1:].mean(0)
        return torch.cat([feats[0], feats[1:].mean(0)], -1)
