import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from core.globals import DEVICE


@dataclass
class GeneratorDataConfig:
    source: Path
    context: int
    encode: Callable
    sos_id: int
    eos_id: int
    pseudo_newline: str | None = None
    stride: int | None = None
    device: torch.device = DEVICE
    cache_dir: Path | None = None

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.target_context // 2


class GeneratorDataset:
    def __init__(self, config: GeneratorDataConfig):
        self.config = config
        self.source_tokens = []
        self.target_tokens = []
        self.src_pad = config.source_pad_id
        self.tgt_pad = config.target_pad_id
        self._prepare_tokens(config)

    def encode_sample(
        self, src: str, tgt: str, config: GeneratorDataConfig
    ) -> tuple[list[list[int]]]:
        src = (
            src.replace(config.pseudo_newline, "\n")
            if config.pseudo_newline
            else src
        )
        tgt = (
            tgt.replace(config.pseudo_newline, "\n")
            if config.pseudo_newline
            else tgt
        )
        src_tokens = config.encode_source(src)
        assert (
            len(src_tokens) <= config.source_context
        ), f'"{src}" exceeds maximum context'
        tgt_tokens = (
            [config.sos_id] + config.encode_target(tgt) + [config.eos_id]
        )
        if len(tgt_tokens) > config.target_context + 1:
            it = range(
                config.target_context + 1,
                len(tgt_tokens) + config.stride,
                config.stride,
            )
            tgt_tokens = [
                tgt_tokens[i - (config.target_context + 1) : i] for i in it
            ]
        else:
            tgt_tokens = [tgt_tokens]
        return [src_tokens] * len(tgt_tokens), tgt_tokens

    def _prepare_tokens(self, config: GeneratorDataConfig):
        source = open(config.source, encoding="utf-8")
        target = open(config.target, encoding="utf-8")
        line_count = sum(1 for _ in source)
        assert sum(1 for _ in target) == line_count
        source.seek(0)
        target.seek(0)
        for _ in range(line_count):
            src = source.readline().rstrip("\n")
            tgt = target.readline().rstrip("\n")
            src, tgt = self.encode_sample(src, tgt, config)
            self.source_tokens.extend(src)
            self.target_tokens.extend(tgt)
        source.close()
        target.close()
