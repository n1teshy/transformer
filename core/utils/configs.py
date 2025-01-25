from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union


@dataclass
class GeneratorDataConfig:
    batch_size: int
    pad_id: int

    source: Optional[Union[Path, str]] = None
    context: Optional[int] = None
    sos_id: Optional[int] = None
    eos_id: Optional[int] = None
    encode: Optional[Callable[[str], List[int]]] = None

    sample_delimiter: Optional[str] = None
    stride: Optional[int] = None
    cache_dir: Optional[Union[Path, str]] = None
    shuffle_shards: bool = False
    shuffle_samples: bool = False

    def __post_init__(self):
        if self.cache_dir is None:
            required_params = [
                "source",
                "context",
                "sos_id",
                "eos_id",
                "encode",
            ]
            assert all(
                getattr(self, param) is not None for param in required_params
            ), "arguments %s are required when 'cache_dir' is not provided" % (
                ", ".join(required_params),
            )
            self.source = Path(self.source)
            assert self.source.exists(), "%s doesn't exist" % (self.source,)
            if self.stride is None:
                self.stride = self.context // 2
        else:
            self.cache_dir = Path(self.cache_dir)
            assert self.cache_dir.exists(), "%s doesn't exist" % (self.cache_dir,)


@dataclass
class TransformerDataConfig:
    batch_size: int
    source_pad_id: int
    target_pad_id: int

    source: Optional[Union[Path, str]] = None
    target: Optional[Union[Path, str]] = None
    sos_id: Optional[int] = None
    eos_id: Optional[int] = None
    encoder_context: Optional[int] = None
    decoder_context: Optional[int] = None
    encode_source: Optional[Callable[[str], List[int]]] = None
    encode_target: Optional[Callable[[str], List[int]]] = None

    sample_delimiter: Optional[str] = None
    stride: Optional[int] = None
    cache_dir: Optional[Union[Path, str]] = None
    shuffle_shards: bool = False
    shuffle_samples: bool = False

    def __post_init__(self):
        if self.cache_dir is None:
            required_params = [
                "source",
                "target",
                "sos_id",
                "eos_id",
                "encoder_context",
                "decoder_context",
                "encode_source",
                "encode_target",
            ]
            assert all(
                getattr(self, param) is not None for param in required_params
            ), "arguments %s are required when 'cache_dir' is not provided" % (
                ", ".join(required_params),
            )
            self.source = Path(self.source)
            self.target = Path(self.target)
            assert (
                self.source.exists() and self.target.exists()
            ), "non-existent source or target path"
            if self.stride is None:
                self.stride = self.decoder_context // 2
        else:
            self.cache_dir = Path(self.cache_dir)
            assert (self.cache_dir / "source").exists() and (
                self.cache_dir / "target"
            ).exists(), "invalid cache directory"


@dataclass
class ModelConfig:
    no_blocks: int
    no_heads: int
    model_dim: int
    vocab_size: int
    pad_id: int
    context: int
    dropout: float
    train_mode: bool


@dataclass
class EncoderConfig(ModelConfig):
    pass


@dataclass
class DecoderConfig(ModelConfig):
    sos_id: int
    eos_id: int
