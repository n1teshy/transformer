from dataclasses import dataclass
from core.globals import DEVICE
from core.utils.types import OptionalDevice


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