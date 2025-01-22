from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch

from core.globals import DEVICE


@dataclass
class GeneratorDataConfig:
    source: Union[Path, str]
    context: int
    sos_id: int
    eos_id: int
    pad_id: int
    batch_size: int
    encode: Callable[[str], List[int]]
    pseudo_newline: Optional[str] = None
    stride: Optional[int] = None
    device: torch.device = DEVICE
    cache_dir: Optional[Union[Path, str]] = None
    shuffle_shards: bool = False
    shuffle_samples: bool = False

    def __post_init__(self):
        self.source = Path(self.source)
        if self.stride is None:
            self.stride = self.context // 2


@dataclass
class SeqToSeqDataConfig:
    source: Path
    target: Path
    encoder_context: int
    decoder_context: int
    source_pad_id: int
    target_pad_id: int
    sos_id: int
    eos_id: int
    batch_size: int
    encode_source: Callable[[str], List[int]]
    encode_target: Callable[[str], List[int]]
    shuffle_shards: bool = False
    shuffle_samples: bool = False
    pseudo_newline: Optional[str] = None
    stride: Optional[int] = None
    cache_dir: Optional[Union[Path, str]] = None

    def __post_init__(self):
        self.source = Path(self.source)
        self.target = Path(self.target)
        if self.stride is None:
            self.stride = self.decoder_context // 2


@dataclass
class ModelConfig:
    no_blocks: int
    no_heads: int
    model_dim: int
    vocab_size: int
    pad_id: int
    context: int


@dataclass
class EncoderConfig(ModelConfig):
    pass


@dataclass
class DecoderConfig(ModelConfig):
    sos_id: int
    eos_id: int
