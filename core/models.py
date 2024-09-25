import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
from core.utils.types import OptionalTensor
from core.components import (
    MultiheadSelfAttention,
    Config,
    EncoderConfig,
    Encoder,
    DecoderConfig,
    Decoder,
    MLP,
    Embedding
)


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
        causal_mask = torch.tril(torch.ones(T, T)).type(torch.ByteTensor).to(y.device)
        causal_pad_mask = (pad_mask & causal_mask)
        mask = torch.zeros_like(causal_pad_mask, dtype=torch.float)
        return mask.masked_fill_(causal_pad_mask.logical_not(), -10000)
    
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
        return mask.to(self.device)
