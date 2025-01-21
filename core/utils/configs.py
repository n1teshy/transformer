import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from core.globals import DEVICE
from core.utils.types import OptionalDevice


@dataclass
class SeqToSeqDataConfig:
    source: Path
    target: Path
    encoder_context: int
    decoder_context: int
    encode_source: Callable
    encode_target: Callable
    source_pad_id: int
    target_pad_id: int
    sos_id: int
    eos_id: int
    batch_size: int
    shuffle_shards: bool = False
    shuffle_samples: bool = False
    pseudo_newline: str | None = None
    stride: int | None = None
    device: torch.device = DEVICE
    cache_dir: Path | None = None

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.decoder_context // 2


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
