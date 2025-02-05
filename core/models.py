from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.components import (
    Decoder,
    DecoderConfig,
    Embedding,
    Encoder,
    EncoderConfig,
    GeneratorBlock,
)


class Transformer(nn.Module):
    def __init__(self, enc_config: EncoderConfig, dec_config: DecoderConfig):
        super().__init__()
        self.encoder = Encoder(enc_config)
        self.decoder = Decoder(dec_config)
        self.enc_conf = enc_config
        self.dec_conf = dec_config

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor]:
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)
        dec_inp = y[:, :-1]
        enc_mask = self.get_enc_mask(x).to(device)
        dec_mask = self.get_dec_mask(dec_inp).to(device)
        x = self.encoder(x, enc_mask)
        logits = self.decoder(
            encoded=x, decoded=dec_inp, enc_mask=enc_mask, dec_mask=dec_mask
        )
        logits, tgt = (
            logits.reshape(-1, logits.shape[-1]),
            y[:, 1:].reshape(-1),
        )
        loss = F.cross_entropy(logits, tgt, reduction="none")
        mask = (tgt != self.dec_conf.pad_id).float().view(-1)
        loss = (loss * mask).sum() / mask.sum()
        return logits, loss

    def get_enc_mask(self, x: Tensor) -> Tensor:
        # attn shape: (B, H, T, T)
        # mask shape: (B, 1, 1, T)
        mask = torch.zeros_like(x, dtype=torch.float).masked_fill(
            (x == self.enc_conf.pad_id), float("-inf")
        )
        return mask.unsqueeze(1).unsqueeze(2)

    def get_dec_mask(self, y: Tensor) -> Tensor:
        # attn shape: (B, H, T, T)
        # mask shape: (B, 1, T, T)
        B, T = y.shape
        # pad_mask: (B, 1, T, 1)
        pad_mask = (y != self.dec_conf.pad_id).unsqueeze(1).unsqueeze(3)
        # causal_mask: (T, T)
        causal_mask = (
            torch.tril(torch.ones(T, T)).type(torch.ByteTensor).to(y.device)
        )
        # causal_pad_mask: (B, 1, T, 1) & (T, T) -> (B, 1, T, T)
        causal_pad_mask = pad_mask & causal_mask
        mask = torch.zeros_like(causal_pad_mask, dtype=torch.float)
        # TODO: should fill -inf instead of 10,000?
        return mask.masked_fill_(causal_pad_mask.logical_not(), float("-inf"))

    @torch.no_grad()
    def generate(
        self,
        x: Tensor,
        *,
        context: Optional[Tensor] = None,
        max_tokens: Optional[int] = None,
        topk: int | None = None,
    ):
        assert x.shape[0] == 1
        device = next(self.parameters()).device
        x = x.to(device)
        x = self.encoder(x)
        context = context or torch.full(
            (x.shape[0], 1), self.dec_conf.sos_id
        ).to(device)
        yield self.dec_conf.sos_id
        while True:
            context = context[:, -self.dec_conf.context :]
            logits = self.decoder(x, context)
            probs = F.softmax(logits, dim=2)[:, -1, :]
            if topk is None:
                next_idx = torch.multinomial(probs, 1)
            else:
                k_probs, k_indices = torch.topk(probs, k=topk)
                sel_idx = torch.multinomial(k_probs, 1)
                next_idx = torch.gather(k_indices, -1, sel_idx)
            token_id = next_idx.item()
            yield token_id
            if token_id == self.dec_conf.eos_id and max_tokens is None:
                return
            elif max_tokens is not None:
                max_tokens -= 1
                if max_tokens == 0:
                    return
            context = torch.cat((context, next_idx), dim=1)


class Generator(nn.Module):
    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.embeddings = Embedding(config)
        self.blocks = nn.ModuleList([GeneratorBlock(config) for _ in range(config.no_blocks)])
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size)
        self.config = config

    def forward(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        device = next(self.parameters()).device
        x, y = x.to(device), y.to(device)
        mask = self.get_mask(x).to(device)
        x = self.embeddings(x)
        for block in self.blocks:
            x = block(x, mask)
        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            y.view(-1),
            reduction="none"
        )
        mask = (y != self.config.pad_id).float().view(-1)
        loss = (loss * mask).sum() / mask.sum()
        return logits, loss

    def get_mask(self, x: Tensor) -> Tensor:
        # attn shape: (B, H, Td, Td)
        # mask shape: (Td, Td)
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T))
        return torch.zeros_like(mask).masked_fill_(mask.logical_not(), float("-inf"))

    @torch.no_grad()
    def generate(
        self,
        context: Optional[Tensor] = None,
        max_tokens: Optional[int] = None,
        topk: int | None = None,
    ):
        context = context
        if context is None:
            device = next(self.parameters()).device
            context = torch.tensor([[self.config.sos_id]], device=device)
        assert context.shape[0] == 1
        yield self.config.sos_id
        while True:
            context = context[:, -self.config.context :]
            inp = self.embeddings(context.clone())
            for block in self.blocks:
                inp = block(inp)
            logits = self.lm_head(inp)
            probs = F.softmax(logits, dim=2)[:, -1, :]
            if topk is None:
                next_idx = torch.multinomial(probs, 1)
            else:
                k_probs, k_indices = torch.topk(probs, k=topk)
                sel_idx = torch.multinomial(k_probs, 1)
                next_idx = torch.gather(k_indices, -1, sel_idx)
            token_id = next_idx.item()
            yield token_id
            if token_id == self.config.eos_id and max_tokens is None:
                return
            elif max_tokens is not None:
                max_tokens -= 1
                if max_tokens == 0:
                    return
            context = torch.cat((context, next_idx), dim=1)
