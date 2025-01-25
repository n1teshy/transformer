from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.utils.configs import DecoderConfig, EncoderConfig, ModelConfig


class Embedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.tkn_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_embedding = nn.Embedding(config.context, config.model_dim)
        # self.pos_embedding = torch.randn((config.context, config.model_dim))
        # positions = torch.arange(0, config.context).unsqueeze(1)
        # _2i = torch.arange(0, config.model_dim, 2)
        # self.pos_embedding[:, 0::2] = torch.sin(
        #     positions / 10_000 ** (_2i / config.model_dim)
        # )
        # self.pos_embedding[:, 1::2] = torch.cos(
        #     positions / 10_000 ** (_2i / config.model_dim)
        # )
        # self.pos_embedding = self.pos_embedding.to(config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
        tkn_emb = self.tkn_embedding(x)
        pos_emb = self.pos_embedding(pos)
        return self.dropout(tkn_emb + pos_emb)


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w_attn = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.proj = nn.Linear(config.model_dim, config.model_dim)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        dropout = self.config.dropout if self.config.train_mode else 0
        no_heads = self.config.no_heads
        B, T, C = x.shape
        # qkv: (B, T, 3 * C)
        qkv = self.w_attn(x)
        # q, k, v: (B, T, C) each
        q, k, v = qkv.split(C, dim=2)
        # q: (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, no_heads, C // no_heads).transpose(1, 2)
        k = k.view(B, T, no_heads, C // no_heads).transpose(1, 2)
        v = v.view(B, T, no_heads, C // no_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout,
        )
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj_dropout(self.proj(y))


class MultiheadCrossAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w_kv = nn.Linear(config.model_dim, 2 * config.model_dim)
        self.w_q = nn.Linear(config.model_dim, config.model_dim)
        self.proj = nn.Linear(config.model_dim, config.model_dim)
        self.proj_dropout = nn.Dropout(config.dropout)
        self.config = config

    def forward(
        self, encoded: Tensor, decoded: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        dropout = self.config.dropout if self.config.train_mode else 0
        no_heads = self.config.no_heads
        B, T, C = encoded.shape
        kv = self.w_kv(encoded)
        q = self.w_q(decoded)
        k, v = kv.split(C, dim=2)
        q = q.view(B, q.shape[1], no_heads, C // no_heads).transpose(1, 2)
        k = k.view(B, T, no_heads, C // no_heads).transpose(1, 2)
        v = v.view(B, T, no_heads, C // no_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=dropout,
        )
        B, H, T, C = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, H * C)
        return self.proj_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.GELU("tanh"),
            nn.Linear(config.model_dim * 4, config.model_dim),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.attn = MultiheadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.no_blocks)]
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.self_attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.cross_attn = MultiheadCrossAttention(config)
        self.ln2 = nn.LayerNorm(config.model_dim)
        self.mlp = MLP(config)
        self.ln3 = nn.LayerNorm(config.model_dim)

    def forward(
        self,
        encoded: Tensor,
        decoded: Tensor,
        *,
        enc_mask: Optional[Tensor] = None,
        dec_mask: Optional[Tensor] = None,
    ) -> Tensor:
        decoded = self.ln1(self.self_attn(decoded, dec_mask) + decoded)
        decoded = self.ln2(
            self.cross_attn(encoded=encoded, decoded=decoded, mask=enc_mask)
            + decoded
        )
        return self.ln3(self.mlp(decoded) + decoded)


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.no_blocks)]
        )
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size)

    def forward(
        self,
        encoded: Tensor,
        decoded: Tensor,
        *,
        enc_mask: Optional[Tensor] = None,
        dec_mask: Optional[Tensor] = None,
    ) -> Tensor:
        decoded = self.embedding(decoded)
        for block in self.blocks:
            decoded = block(
                encoded=encoded,
                decoded=decoded,
                enc_mask=enc_mask,
                dec_mask=dec_mask,
            )
        return self.lm_head(decoded)


class GeneratorBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.model_dim)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        x = self.ln1(self.attn(x, mask) + x)
        return self.ln2(self.mlp(x) + x)
