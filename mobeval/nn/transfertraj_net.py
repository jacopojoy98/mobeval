"""TransferTraj network (Wang et al.), adapted from github.com/wtl52656/TransferTraj.

Parameter names are unchanged, so checkpoints trained with the original repository load.
Changes: no einops dependency; POI/road inputs are optional (a dummy, never-matching entry is
used when no POI or road-network data is available, which keeps the architecture and the
parameter shapes intact); the POI/road matrices are non-persistent buffers so they follow
`.to(device)` without entering the state dict.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# token vocabulary of the original data pipeline
KNOWN_TOKEN, MASK_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, PAD_TOKEN = range(6)
FEATURE_PAD = 0
ST_MAP = {"spatial": [0, 1], "temporal": [2, 3]}
S_COLS, T_COLS = ST_MAP["spatial"], ST_MAP["temporal"]


# --------------------------------------------------------------------------- encoders
class PositionalEncode(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        inv_freq = 1 / (10000 ** (torch.arange(0.0, hidden_size, 2.0) / hidden_size))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        B, L = pos_seq.shape
        sinusoid = torch.ger(pos_seq.reshape(B * L).float(), self.inv_freq)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], -1).reshape(B, L, -1)


class FourierEncode(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter(torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size)).float())
        self.bias = nn.Parameter(torch.zeros(embed_size).float())
        self.div_term = math.sqrt(1.0 / embed_size)

    def forward(self, x):
        if x.dim() < 3:
            x = x.unsqueeze(-1)
        return self.div_term * torch.cos(x * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1))


class NoisyTopkRouter(nn.Module):
    """Noisy top-k gating. The original adds routing noise on every forward pass, including
    evaluation, which makes predictions non-reproducible. Here the noise is a training-time
    regulariser (as in Shazeer et al.); set `noise_in_eval = True` to reproduce the original."""
    noise_in_eval = False

    def __init__(self, n_embed, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        if self.training or self.noise_in_eval:
            logits_noise = torch.randn_like(logits) * F.softplus(self.noise_linear(mh_output))
        else:
            logits_noise = torch.zeros_like(logits)
        noisy = logits + logits_noise
        top_k_logits, indices = noisy.topk(self.top_k, dim=-1)
        sparse = torch.full_like(noisy, float("-inf")).scatter(-1, indices, top_k_logits)
        return F.softmax(sparse, dim=-1), indices, F.softmax(logits, dim=-1)


class Expert(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class SCMoE(nn.Module):
    def __init__(self, n_embed, num_experts=8, top_k=4):
        super().__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating, indices, softmax_gating = self.router(x)
        out = torch.zeros_like(x)
        flat_x, flat_gate = x.view(-1, x.size(-1)), gating.view(-1, gating.size(-1))
        for i, expert in enumerate(self.experts):
            m = (indices == i).any(dim=-1)
            flat_m = m.view(-1)
            if flat_m.any():
                out[m] += (expert(flat_x[flat_m]) * flat_gate[flat_m, i].unsqueeze(1)).squeeze(1)
        return out, softmax_gating


class RoPE_Attention_float(nn.Module):
    """Self-attention whose rotary embedding is driven by the COORDINATES, not the index."""

    def __init__(self, hidd_dim):
        super().__init__()
        self.hidd_dim = hidd_dim
        self.wq, self.wk, self.wv = (nn.Linear(hidd_dim, hidd_dim) for _ in range(3))
        self.Wr = nn.Linear(2, hidd_dim // 2, bias=False)

    def forward(self, x, norm_coord, causal_mask, batch_mask):
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq, xk = self.apply_rotary_emb(xq, xk, norm_coord)
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(self.hidd_dim)
        scores = scores.masked_fill(causal_mask, float("-inf")).masked_fill(batch_mask, float("-inf"))
        return torch.matmul(F.softmax(scores.float(), dim=-1), xv)

    def apply_rotary_emb(self, xq, xk, norm_coord):
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs = self.Wr(norm_coord)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return (torch.view_as_real(xq_ * freqs_cis).flatten(2).type_as(xq),
                torch.view_as_real(xk_ * freqs_cis).flatten(2).type_as(xk))


class RTTE_Encoder_layer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.wq, self.wk, self.wv = (nn.Linear(dim, dim) for _ in range(3))
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.MoE_fc = SCMoE(dim)
        self.fc_norm = nn.LayerNorm(dim)
        self.RoPE_Attention_float = RoPE_Attention_float(dim)

    def forward(self, x, norm_coord, causal_mask, batch_mask):
        x = self.norm(x + self.RoPE_Attention_float(x, norm_coord, causal_mask, batch_mask))
        out, _ = self.MoE_fc(x)
        return self.fc_norm(x + out)


class RTTE_Encoder(nn.Module):
    def __init__(self, dim, layers, max_seq_len=10000):
        super().__init__()
        self.dim, self.max_seq_len = dim, max_seq_len
        self.layers = nn.ModuleList([RTTE_Encoder_layer(dim) for _ in range(layers)])

    def forward(self, x, norm_coord, mask, src_key_padding_mask):
        src_key_padding_mask = src_key_padding_mask.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, norm_coord, mask, src_key_padding_mask)
        return x


# --------------------------------------------------------------------------- helpers
def gen_causal_mask(seq_len, include_self=True):
    m = torch.ones(seq_len, seq_len)
    return (1 - (torch.triu(m) if include_self else torch.tril(m)).transpose(0, 1)).bool()


def tokenize_timestamp(t):
    """(..., 2) [timestamp, delta seconds] -> (..., 4) [week, hour, minute, delta minutes]."""
    week = t[..., 0] % (7 * 24 * 3600) / (24 * 3600)
    hour = t[..., 0] % (24 * 3600) / 3600
    minute = t[..., 0] % 3600 / 60
    return torch.stack([week, hour, minute, t[..., 1] / 60], -1)


def masked_mean(values, mask):
    total = values.masked_fill(mask, 0).sum()
    count = (~mask).long().sum()
    return 0 if count == 0 else total / count


# --------------------------------------------------------------------------- model
class TransferTraj(nn.Module):
    def __init__(self, embed_size, d_model, poi_embed=None, poi_coors=None, road_embed=None, road_coors=None,
                 rafee_layer=2, UTM_region=None, poi_dist=100, rn_dist=100):
        super().__init__()
        poi_embed, poi_coors = _default_context(poi_embed, poi_coors)
        road_embed, road_coors = _default_context(road_embed, road_coors)
        self.UTM_region, self.poi_dist, self.rn_dist = UTM_region, poi_dist, rn_dist
        self.register_buffer("poi_coors", poi_coors, persistent=False)
        self.register_buffer("road_coors", road_coors, persistent=False)
        self.register_buffer("poi_embed_mat", poi_embed, persistent=False)
        self.register_buffer("road_embed_mat", road_embed, persistent=False)

        self.spatial_embed_layer = nn.Sequential(nn.Linear(2, embed_size), nn.LeakyReLU(),
                                                 nn.Linear(embed_size, d_model))
        self.temporal_embed_modules = nn.ModuleList([FourierEncode(embed_size) for _ in range(4)])
        self.temporal_embed_layer = nn.Sequential(nn.LeakyReLU(), nn.Linear(embed_size * 4, d_model))
        self.poi_embed_layer = nn.Sequential(nn.LayerNorm(poi_embed.shape[1]), nn.Linear(poi_embed.shape[1], d_model))
        self.road_embed_layer = nn.Sequential(nn.LayerNorm(road_embed.shape[1]), nn.Linear(road_embed.shape[1], d_model))
        self.token_embed_layer = nn.Sequential(nn.Embedding(6, embed_size, padding_idx=5), nn.LayerNorm(embed_size),
                                               nn.Linear(embed_size, d_model))
        self.pos_encode_layer = PositionalEncode(d_model)
        self.modal_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=256, batch_first=True), num_layers=1)
        self.seq_model = RTTE_Encoder(d_model, layers=rafee_layer)
        self.spatial_pred_layer = nn.Sequential(nn.Linear(d_model, 2))
        self.temporal_pred_layer = nn.Sequential(nn.Linear(d_model, 4), nn.Softplus())
        self.token_pred_layers = nn.ModuleList([nn.Linear(d_model, 5) for _ in range(2)])

    def forward(self, input_seq, positions, first_point):
        L = input_seq.size(1)
        token = input_seq[..., [S_COLS[0], T_COLS[0]], 1].long()
        spatial = input_seq[:, :, S_COLS, 0]
        temporal_token = tokenize_timestamp(input_seq[:, :, T_COLS, 0])
        modal_h, norm_coord = self.cal_modal_h(spatial, temporal_token, token, positions, first_point)
        causal_mask = gen_causal_mask(L).to(input_seq.device)
        batch_mask = token[..., 0] == PAD_TOKEN
        return modal_h, self.seq_model(modal_h, norm_coord, mask=causal_mask, src_key_padding_mask=batch_mask)

    def _context_embed(self, layer, embed_mat, coors, spatial, first_point, thresh, feature_mask, token_e,
                       zero_masked=True):
        """zero_masked mirrors the original: the POI pathway zeroes masked positions
        (`masked_fill_`), while the road pathway calls the NON in-place `masked_fill` and
        therefore leaves them untouched. Kept as-is for checkpoint fidelity."""
        dist = ((coors.unsqueeze(0).unsqueeze(0) - (spatial + first_point.unsqueeze(1)).unsqueeze(2)) ** 2).sum(-1)
        sel = layer(embed_mat).unsqueeze(0).unsqueeze(0).expand(dist.shape[0], dist.shape[1], -1, -1)
        mask = dist < thresh
        e = (sel * mask.unsqueeze(-1)).sum(dim=2) / mask.sum(-1, keepdim=True).clamp(min=1)
        if zero_masked:
            e = e.masked_fill(feature_mask.unsqueeze(-1), 0)
        return e + token_e

    def cal_modal_h(self, spatial, temporal_token, token, positions, first_point):
        B = spatial.size(0)
        token_e = self.token_embed_layer(token)
        feature_e_mask = ~torch.isin(token, torch.tensor([KNOWN_TOKEN, UNKNOWN_TOKEN], device=token.device))
        norm_coord = spatial

        spatial_e = self.spatial_embed_layer(norm_coord)
        spatial_e.masked_fill(feature_e_mask[..., 0].unsqueeze(-1), 0)          # (original: not in place)
        spatial_e = spatial_e + token_e[:, :, 0]
        poi_e = self._context_embed(self.poi_embed_layer, self.poi_embed_mat, self.poi_coors, spatial, first_point,
                                    self.poi_dist, feature_e_mask[..., 0], token_e[:, :, 0])
        road_e = self._context_embed(self.road_embed_layer, self.road_embed_mat, self.road_coors, spatial, first_point,
                                     self.rn_dist, feature_e_mask[..., 0], token_e[:, :, 0], zero_masked=False)
        temporal_e = self.temporal_embed_layer(
            torch.cat([m(temporal_token[..., i]) for i, m in enumerate(self.temporal_embed_modules)], -1))
        temporal_e.masked_fill(feature_e_mask[..., 1].unsqueeze(-1), 0)          # (original: not in place)
        temporal_e = temporal_e + token_e[:, :, 1]

        modal_e = torch.stack([spatial_e, temporal_e, poi_e, road_e], 2).reshape(B * spatial.size(1), 4, -1)
        modal_h = self.modal_mixer(modal_e).reshape(B, spatial.size(1), 4, -1).mean(axis=2)
        return modal_h + self.pos_encode_layer(positions), norm_coord

    def pred(self, mem_seq, return_raw=True):
        pred_spatial = self.spatial_pred_layer(mem_seq)
        pred_temporal_token = self.temporal_pred_layer(mem_seq)
        pred_token = torch.stack([layer(mem_seq) for layer in self.token_pred_layers], 2)
        return pred_spatial, pred_temporal_token, pred_token, torch.argmax(pred_token, -1)

    def loss(self, input_seq, target_seq, positions, first_point, teacher_ratio=0.5):
        target_spatial = target_seq[..., S_COLS, 0]
        target_temp_token = tokenize_timestamp(target_seq[:, :, T_COLS, 0])
        target_token = target_seq[..., [S_COLS[0], T_COLS[0]], 1].long()
        feature_mask = target_token != UNKNOWN_TOKEN
        token_mask = target_token == PAD_TOKEN
        B, L, _, _ = target_seq.shape
        L_in = input_seq.size(1)
        _, mem_seq = self.forward(input_seq, positions[:, :L_in], first_point)
        pred_spatial, pred_temporal_token, pred_token_dist, _ = self.pred(mem_seq)
        spatial_loss = masked_mean(F.mse_loss(pred_spatial, target_spatial, reduction="none"),
                                   feature_mask[..., 0].unsqueeze(-1))
        temporal_loss = masked_mean(F.mse_loss(pred_temporal_token, target_temp_token, reduction="none"),
                                    feature_mask[..., 1].unsqueeze(-1))
        token_loss = torch.stack([F.cross_entropy(pred_token_dist[:, :, i].reshape(B * L, -1),
                                                  torch.clamp(target_token[..., i], max=4).reshape(B * L),
                                                  reduction="none") for i in range(2)], -1).reshape(B, L, 2)
        return spatial_loss + temporal_loss + masked_mean(token_loss, token_mask)


def _default_context(embed, coors):
    """A single dummy entry placed far away: never selected, so the POI/road pathway contributes
    only the token embedding. Used when no POI or road-network data is available."""
    if embed is None or coors is None:
        return torch.zeros(1, 1), torch.full((1, 2), 1e12)
    embed = embed if torch.is_tensor(embed) else torch.as_tensor(np.asarray(embed), dtype=torch.float32)
    coors = coors if torch.is_tensor(coors) else torch.as_tensor(np.asarray(coors), dtype=torch.float32)
    return embed.float(), coors.float()
