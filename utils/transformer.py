import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from dataclasses import dataclass
from utils.globals import DEVICE

OptionalTensor = Optional[Tensor]
OptionalDevice = Optional[torch.device]


@dataclass
class Config:
    no_blocks: int
    no_heads: int
    model_dim: int
    vocab_size: int
    pad_id: int
    context: int


@dataclass
class EncoderConfig(Config):
    device: OptionalDevice = DEVICE


@dataclass
class DecoderConfig(Config):
    sos_id: int
    eos_id: int
    device: OptionalDevice = DEVICE


class Embedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.tkn_embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.pos_encoding = torch.randn((config.context, config.model_dim))
        positions = torch.arange(0, config.context).unsqueeze(1)
        _2i = torch.arange(0, config.model_dim, 2)
        self.pos_encoding[:, 0::2] = torch.sin(positions / 10000 ** (_2i / config.model_dim))
        self.pos_encoding[:, 1::2] = torch.cos(positions / 10000 ** (_2i / config.model_dim))
        self.pos_encoding = self.pos_encoding.to(config.device)

    def forward(self, x: Tensor) -> Tensor:
        B, T = x.shape
        _ = self.tkn_embedding(x)
        return _ + self.pos_encoding[:T, :]


class MultiheadSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w_attn = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.proj = nn.Linear(config.model_dim, config.model_dim)
        self.model_dim = config.model_dim
        self.no_heads = config.no_heads

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        B, T, C = x.shape
        # qkv: (B, T, 3 * C)
        qkv = self.w_attn(x)
        # q, k, v: (B, T, C) each
        q, k, v = qkv.split(C, dim=2)
        # q: (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.no_heads, C // self.no_heads).transpose(1, 2)
        k = k.view(B, T, self.no_heads, C // self.no_heads).transpose(1, 2)
        v = v.view(B, T, self.no_heads, C // self.no_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
    

class MultiheadCrossAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.w_kv = nn.Linear(config.model_dim, 2 * config.model_dim)
        self.w_q = nn.Linear(config.model_dim, config.model_dim)
        self.proj = nn.Linear(config.model_dim, config.model_dim)
        self.no_heads = config.no_heads
        
    def forward(self, encoded: Tensor, decoded: Tensor, mask: OptionalTensor) -> Tensor:
        B, T, C = encoded.shape
        kv = self.w_kv(encoded)
        q = self.w_q(decoded)
        k, v = kv.split(C, dim=2)
        q = q.view(B, q.shape[1], self.no_heads, C // self.no_heads).transpose(1, 2)
        k = k.view(B, T, self.no_heads, C // self.no_heads).transpose(1, 2)
        v = v.view(B, T, self.no_heads, C // self.no_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        B, H, T, C = y.shape
        y = y.transpose(1, 2).contiguous().view(B, T, H * C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim * 4),
            nn.GELU("tanh"),
            nn.Linear(config.model_dim * 4, config.model_dim)
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

    def forward(self, x: Tensor, mask: OptionalTensor) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))
    

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.no_blocks)]
        )
    
    def forward(self, x: Tensor, mask: OptionalTensor = None) -> Tensor:
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

    def forward(self, encoded: Tensor, decoded: Tensor, *, enc_mask: OptionalTensor, dec_mask: OptionalTensor) -> Tensor:
        decoded = self.ln1(self.self_attn(decoded, dec_mask) + decoded)
        decoded = self.ln2(self.cross_attn(encoded=encoded, decoded=decoded, mask=enc_mask) + decoded)
        return self.ln3(self.mlp(decoded) + decoded)


class Decoder(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.embedding = Embedding(config)
        self.blocks = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.no_blocks)
        ])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size)

    def forward(self, encoded: Tensor, decoded: Tensor, *, enc_mask: OptionalTensor = None, dec_mask: OptionalTensor = None) -> Tensor:
        decoded = self.embedding(decoded)
        for block in self.blocks:
            decoded = block(encoded=encoded, decoded=decoded, enc_mask=enc_mask, dec_mask=dec_mask)
        return self.lm_head(decoded)


class Transformer(nn.Module):
    def __init__(self, enc_config: EncoderConfig, dec_config: DecoderConfig):
        super().__init__()
        self.encoder = Encoder(enc_config)
        self.decoder = Decoder(dec_config)
        self.decoder_context = dec_config.context
        self.enc_pad_id = enc_config.pad_id
        self.dec_pad_id = dec_config.pad_id
        self.sos_id = dec_config.sos_id
        self.eos_id = dec_config.eos_id
        self.device = dec_config.device
    
    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor]:
        _y = y[:, :-1]
        enc_mask = self.get_enc_mask(x)
        dec_mask = self.get_dec_mask(_y)
        x = self.encoder(x, enc_mask)
        logits = self.decoder(encoded=x, decoded=_y, enc_mask=enc_mask, dec_mask=dec_mask)
        logits, y = logits.reshape(-1, logits.shape[-1]), y[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits, y, reduction="none")
        mask = (y != self.dec_pad_id).float().view(-1)
        loss = loss * mask.view(-1)
        return logits, loss.sum() / mask.sum()
    
    def get_enc_mask(self, x: Tensor) -> Tensor:
        # attn shape: (B, H, T, Te)
        # mask shape: (B, 1, 1, Te)
        mask = torch.zeros_like(x, dtype=torch.float).masked_fill((x == self.enc_pad_id), -10000)
        return mask.unsqueeze(1).unsqueeze(2).to(self.device)

    def get_dec_mask(self, y: Tensor) -> Tensor:
        # attn shape: (B, H, Td, Td)
        # mask shape: (B, 1, Td, Td)
        B, T = y.shape
        pad_mask = (y != self.dec_pad_id).unsqueeze(1).unsqueeze(3)
        causal_mask = torch.tril(torch.ones(T, T)).type(torch.ByteTensor)
        causal_pad_mask = (pad_mask & causal_mask)
        mask = torch.zeros_like(causal_pad_mask, dtype=torch.float)
        mask.masked_fill_(causal_pad_mask.logical_not(), -10000)
        return mask.to(self.device)
    
    @torch.no_grad()
    def generate(self, x: Tensor, *, context: OptionalTensor = None, max_tokens: Optional[int] = None):
        assert x.shape[0] == 1
        x = self.encoder(x)
        context = context or torch.full((x.shape[0], 1), self.sos_id)
        yield self.sos_id
        while True:
            context = context[:, -self.decoder_context:]
            logits = self.decoder(x, context)
            probs = F.softmax(logits, dim=2)[:, -1, :]
            next_token = torch.multinomial(probs, num_samples=1)
            scalar_token_id = next_token.item()
            yield scalar_token_id
            if scalar_token_id == self.eos_id:
                return
            elif max_tokens is not None:
                max_tokens -= 1
                if max_tokens == 0:
                    return
            context = torch.cat((context, next_token), dim=1)
        

class GeneratorBlock:
    def __init__(self, config: Config):
        self.attn = MultiheadSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.model_dim)
        self.mlp = MLP(config)
        self.ln2 = nn.LayerNorm(config.model_dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.ln1(self.attn(x, mask) + x)
        return self.ln2(self.mlp(x) + x)


class Generator:
    def __init__(self, config: Config):
        self.embeddings = Embedding(config)
        self.blocks = nn.Sequential(
            *[GeneratorBlock(config) for _ in range(config.no_blocks)]
        )
        self.proj = nn.Linear(config.model_dim, config.vocab_size)
        self.device = config.device
    
    def forward(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor]:
        mask = self.get_mask(x)
        x = self.embeddings(x)
        logits = self.proj(self.blocks(x, mask))
        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        return logits, loss

    def get_mask(self, x: Tensor) -> Tensor:
        # attn shape: (B, H, Td, Td)
        # mask shape: (Td, Td)
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T))
        mask = torch.zeros_like(mask).masked_fill_(mask.logical_not(), -10000)
        return mask.to(dtype=self.device)
