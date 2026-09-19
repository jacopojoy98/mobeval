"""CLIP-style dual-view mobility model (trajectory view + visit view), adapted from
Model/Models/clip_mobility_model.py and mobility_transformer_vector.py. Parameter names
are unchanged so existing checkpoints load. Additions: `hidden_states` helpers used by
the adapter, and a bounded contrastive loss (see `clip_loss`)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000, dropout=0.1, learned=False):
        super().__init__()
        self.dropout, self.learned = nn.Dropout(dropout), learned
        if learned:
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        else:
            pe = torch.zeros(max_seq_length, d_model)
            pos = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
            div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if self.learned:
            x = x + self.pos_embedding(torch.arange(x.size(1), device=x.device))[None]
        else:
            x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def causal_mask(n, device):
    return torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()


class MobilityTransformerVector(nn.Module):
    def __init__(self, token_dim, output_dim, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024,
                 dropout=0.1, max_seq_length=5000, learned_pos_encoding=False):
        super().__init__()
        self.d_model, self.token_dim, self.output_dim = d_model, token_dim, output_dim
        self.token_projection = nn.Linear(token_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout, learned_pos_encoding)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.output_projection = nn.Linear(d_model, output_dim)
        r = 0.1
        self.token_projection.weight.data.uniform_(-r, r); self.token_projection.bias.data.zero_()
        self.output_projection.weight.data.uniform_(-r, r); self.output_projection.bias.data.zero_()

    def generate_causal_mask(self, n, device):
        return causal_mask(n, device)

    def hidden_states(self, src, pad=None):
        h = self.pos_encoder(self.token_projection(src) * math.sqrt(self.d_model))
        return self.transformer_encoder(h, mask=causal_mask(src.size(1), src.device), src_key_padding_mask=pad)

    def forward(self, src, src_padding_mask=None):
        return self.output_projection(self.hidden_states(src, src_padding_mask))


class LocationVisitTransformer(nn.Module):
    def __init__(self, visit_token_dim, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1,
                 max_seq_length=5000, learned_pos_encoding=False):
        super().__init__()
        self.d_model, self.visit_token_dim = d_model, visit_token_dim
        self.visit_projection = nn.Linear(visit_token_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout, learned_pos_encoding)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="gelu", batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.visit_projection.weight.data.uniform_(-0.1, 0.1); self.visit_projection.bias.data.zero_()

    def hidden_states(self, src, pad=None):
        if pad is not None and bool(pad[:, 0].any()):
            raise ValueError("visit padding must be on the RIGHT: with a causal mask, leading padding leaves "
                             "positions with nothing to attend to, which PyTorch's fast path turns into NaN")
        h = self.pos_encoder(self.visit_projection(src) * math.sqrt(self.d_model))
        return self.transformer_encoder(h, mask=causal_mask(src.size(1), src.device), src_key_padding_mask=pad)

    def forward(self, src, src_padding_mask=None, return_all_tokens=False):
        out = self.hidden_states(src, src_padding_mask)
        return out if return_all_tokens else masked_mean(out, src_padding_mask)


def masked_mean(h, pad=None):
    """Mean over valid positions. Uses where() rather than multiplication so NaN at padded
    positions (possible in PyTorch's fast inference path) cannot propagate: NaN * 0 = NaN."""
    if pad is None:
        return h.mean(1)
    valid = (~pad).unsqueeze(-1)
    return torch.where(valid, h, torch.zeros_like(h)).sum(1) / valid.sum(1).clamp(min=1)


class CLIPMobilityModel(nn.Module):
    def __init__(self, trajectory_token_dim, trajectory_max_seq_length, visit_token_dim, visit_max_seq_length,
                 d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, learned_pos_encoding=False,
                 embedding_dim=256, temperature=0.07):
        super().__init__()
        self.embedding_dim, self.temperature = embedding_dim, temperature
        self.trajectory_transformer = MobilityTransformerVector(
            trajectory_token_dim, trajectory_token_dim, d_model, nhead, num_layers, dim_feedforward, dropout,
            trajectory_max_seq_length, learned_pos_encoding)
        self.visit_transformer = LocationVisitTransformer(
            visit_token_dim, d_model, nhead, num_layers, dim_feedforward, dropout, visit_max_seq_length, learned_pos_encoding)
        self.trajectory_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, embedding_dim))
        self.visit_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, embedding_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / temperature))

    def encode_trajectory(self, tokens, pad=None, normalize=True):
        z = self.trajectory_projection(masked_mean(self.trajectory_transformer.hidden_states(tokens, pad), pad))
        return F.normalize(z, dim=-1) if normalize else z

    def encode_visits(self, tokens, pad=None, normalize=True):
        z = self.visit_projection(self.visit_transformer(tokens, pad))
        return F.normalize(z, dim=-1) if normalize else z

    def clip_loss(self, zt, zv):
        """Symmetric InfoNCE. In-batch pairs already provide negatives; the logit scale is
        clamped (as in CLIP) so the temperature cannot collapse."""
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        logits = scale * zt @ zv.t()
        labels = torch.arange(len(zt), device=zt.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def forward(self, trajectory_tokens, visit_tokens, trajectory_padding_mask=None, visit_padding_mask=None,
                compute_alignment=True, **_):
        out = {"trajectory_predictions": self.trajectory_transformer(trajectory_tokens, trajectory_padding_mask)}
        if compute_alignment:
            zt = self.encode_trajectory(trajectory_tokens, trajectory_padding_mask)
            zv = self.encode_visits(visit_tokens, visit_padding_mask)
            out.update(trajectory_embeddings=zt, visit_embeddings=zv, clip_loss=self.clip_loss(zt, zv))
        return out
