"""TrajGPT network (Hsu et al., SIGSPATIAL 2024), adapted from github.com/ktxlh/TrajGPT (MIT).

Parameter names are unchanged, so checkpoints from the original repository load with
`input_order="legacy"`.

Leakage fix. In the original SourceInput the visit embedding is concatenated as
[location, arrival, departure, region], while the travel-time decoder reads the first
2 blocks of the TARGET visit and the duration decoder the first 3. The travel decoder
therefore sees the target's arrival time (travel = arrival - previous departure) and the
duration decoder sees its arrival AND departure time (duration = departure - arrival).
The code comments ("region_id, location" / "+ arrival_time") and the paper's
factorisation p(region) p(travel | region) p(duration | region, travel) indicate the
intended order is [region, location, arrival, departure], implemented as
input_order="fixed" (default for new training). Legacy checkpoints still load; the
mobeval adapter never feeds them the true target times, so they cannot leak there.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

N_SPECIAL_TOKENS = 4
PAD, BLANK, SEP, ANS = range(N_SPECIAL_TOKENS)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe[None], persistent=False)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])       # (not in-place, unlike the original)


class Space2Vec(nn.Module):
    def __init__(self, d_embed, lambda_min, lambda_max, num_scales=64):
        super().__init__()
        self.lambda_min, self.g, self.S = lambda_min, lambda_max / lambda_min, num_scales
        self.register_buffer("scales", torch.arange(num_scales).reshape(1, num_scales), persistent=False)
        a = torch.tensor([[1.0, 0.0], [-0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2]])
        self.register_buffer("a", a, persistent=False)
        self.location_embedding = nn.Sequential(nn.Linear(num_scales * 6, d_embed), nn.ReLU())

    def forward(self, x):
        frac = (x @ self.a.T).unsqueeze(-1) / (self.lambda_min * torch.pow(self.g, self.scales / (self.S - 1)))
        frac = frac.reshape(*frac.shape[:-2], -1)
        return self.location_embedding(torch.cat([torch.cos(frac), torch.sin(frac)], -1))


class Time2Vec(nn.Module):
    def __init__(self, d_embed):
        super().__init__()
        self.linear = nn.Linear(1, d_embed)

    def forward(self, x):
        h = self.linear(x.unsqueeze(-1))
        return torch.cat([h[..., :1], torch.sin(h[..., 1:])], -1)


class SourceInput(nn.Module):
    def __init__(self, num_regions, d_embed, lambda_min, lambda_max, input_order="fixed"):
        super().__init__()
        self.order = input_order
        self.space2vec = Space2Vec(d_embed, lambda_min, lambda_max)
        self.time2vec = Time2Vec(d_embed)
        self.region_embedding = nn.Embedding(N_SPECIAL_TOKENS + num_regions, d_embed, padding_idx=PAD)

    def forward(self, region_id, x, y, arrival_time, departure_time):
        loc = self.space2vec(torch.stack([x, y], -1))
        arr, dep = self.time2vec(arrival_time), self.time2vec(departure_time)
        reg = self.region_embedding(region_id)
        parts = [loc, arr, dep, reg] if self.order == "legacy" else [reg, loc, arr, dep]
        return torch.cat(parts, -1)


class CausalEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, sequence_len):
        super().__init__()
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, num_heads, batch_first=True),
                                                         num_layers, enable_nested_tensor=False)
        self.pos_encoder = PositionalEncoding(d_model)
        self.register_buffer("src_mask", nn.Transformer.generate_square_subsequent_mask(sequence_len), persistent=False)

    def forward(self, src):
        n = src.shape[1]
        return self.transformer_encoder(self.pos_encoder(src), mask=self.src_mask[:n, :n], is_causal=True)

    def forward_no_pe(self, src_with_pe):
        n = src_with_pe.shape[1]
        return self.transformer_encoder(self.pos_encoder.dropout(src_with_pe), mask=self.src_mask[:n, :n], is_causal=True)


class CausalMemoryDecoder(nn.Module):
    """Cross-attention-only decoder layer (same parameter names as the original subclass)."""

    def __init__(self, d_model, num_heads, sequence_len, d_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d_model, d_feedforward), nn.Linear(d_feedforward, d_model)
        self.dropout, self.dropout2, self.dropout3 = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)
        self.norm2, self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps), nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.register_buffer("causal_memory_mask", nn.Transformer.generate_square_subsequent_mask(sequence_len),
                             persistent=False)

    def forward(self, tgt, memory):
        n = memory.shape[1]
        a = self.multihead_attn(tgt, memory, memory, attn_mask=self.causal_memory_mask[:n, :n], need_weights=False)[0]
        x = self.norm2(tgt + self.dropout2(a))
        return self.norm3(x + self.dropout3(self.linear2(self.dropout(F.relu(self.linear1(x))))))


class GMM(nn.Module):
    def __init__(self, d_model, num_gaussians):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(d_model, num_gaussians), nn.Softplus())
        self.loc = nn.Sequential(nn.Linear(d_model, num_gaussians))
        self.scale = nn.Sequential(nn.Linear(d_model, num_gaussians), nn.Softplus())

    def forward(self, x, eps=1e-6):
        return {"weight": self.weight(x) + eps, "loc": self.loc(x), "scale": self.scale(x) + eps}


def gmm_nll(out, y, mask, min_scale: float = 0.0):
    """Per-element NLL. `min_scale` (same units as y) bounds the density: the original 1e-6 floor
    lets a single out-of-range validation value dominate the loss and break early stopping."""
    w, mu, s = out["weight"][mask], out["loc"][mask], out["scale"][mask].clamp(min=min_scale)
    w = w / w.sum(-1, keepdim=True)
    comp = torch.log(w) - torch.log(s) - 0.5 * math.log(2 * math.pi) - 0.5 * ((y[mask][:, None] - mu) / s) ** 2
    return -torch.logsumexp(comp, -1)


class TrajGPT(nn.Module):
    def __init__(self, num_regions, sequence_len, lambda_max, num_heads=2, num_layers=4, num_gaussians=3,
                 d_feedforward=32, d_embed=32, lambda_min=1e0, input_order="fixed"):
        super().__init__()
        self.num_regions, self.d_model, self.input_order = num_regions, d_embed * 4, input_order
        self.input = SourceInput(num_regions, d_embed, lambda_min, lambda_max, input_order)
        self.encoder = CausalEncoder(self.d_model, num_heads, num_layers, sequence_len)
        self.region_id_decoder = CausalEncoder(self.d_model, num_heads, 1, sequence_len)
        self.region_id_head = nn.Linear(self.d_model, num_regions + N_SPECIAL_TOKENS)
        self.d_travel, self.d_duration = d_embed * 2, d_embed * 3
        self.travel_decoder = CausalMemoryDecoder(self.d_travel, num_heads, sequence_len, d_feedforward)
        self.travel_head = GMM(self.d_travel, num_gaussians)
        self.duration_decoder = CausalMemoryDecoder(self.d_duration, num_heads, sequence_len, d_feedforward)
        self.duration_head = GMM(self.d_duration, num_gaussians)

    def forward(self, inputs):
        seq = self.input(**inputs)
        if self.input_order == "legacy":
            # The original PositionalEncoding adds in place to the view seq[:, :-1], which also
            # shifts encodings into tgt = seq[:, 1:]. Reproduced for checkpoint fidelity.
            n = seq.shape[1] - 1
            seq = torch.cat([seq[:, :-1] + self.encoder.pos_encoder.pe[:, :n], seq[:, -1:]], 1)
            memory = self.encoder.forward_no_pe(seq[:, :-1])
        else:
            memory = self.encoder(seq[:, :-1])
        tgt = seq[:, 1:]
        region_out = self.region_id_head(self.region_id_decoder(memory))
        if self.input_order == "legacy":
            # same in-place side effect: the region decoder's PositionalEncoding modifies `memory`
            memory = memory + self.region_id_decoder.pos_encoder.pe[:, :memory.shape[1]]
        return {"region_id": region_out,
                "travel_time": self.travel_head(self.travel_decoder(tgt[..., :self.d_travel], memory[..., :self.d_travel])),
                "duration": self.duration_head(self.duration_decoder(tgt[..., :self.d_duration], memory[..., :self.d_duration])),
                "memory": memory}


def init_gmm_head(head: GMM, y: torch.Tensor):
    """Start a GMM head at the data: component locations at train quantiles, spread ~ data std.
    Without this, targets of tens/hundreds of hours against an initial loc~0, scale~0.7 give huge
    NLL gradients that (after clipping) drown the region cross-entropy of the shared encoder."""
    y = y[torch.isfinite(y)].detach().cpu().numpy()
    k = head.loc[0].out_features
    qs = np.quantile(y, np.linspace(0.2, 0.8, k)) if len(y) else np.zeros(k)
    spread = max(float(np.std(y)) if len(y) else 1.0, 1e-2)
    with torch.no_grad():
        for lin in (head.loc[0], head.scale[0], head.weight[0]):
            lin.weight.mul_(0.01)
        head.loc[0].bias.copy_(torch.as_tensor(qs, dtype=torch.float32))
        head.scale[0].bias.fill_(math.log(math.expm1(spread / k)))
        head.weight[0].bias.fill_(math.log(math.expm1(1.0)))
