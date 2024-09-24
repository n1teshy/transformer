import torch
from torch.utils.data import Dataset as _Dataset
from torch.nn.utils.rnn import pad_sequence
from os import PathLike
from dataclasses import dataclass
from typing import Callable
from core.globals import DEVICE


@dataclass
class Config:
    source: PathLike
    target: PathLike
    source_context: int
    target_context: int
    encode_source: Callable
    encode_target: Callable
    source_pad_id: int
    target_pad_id: int
    sos_id: int
    eos_id: int
    pseudo_newline: str | None = None
    stride: int | None = None
    device: torch.device = DEVICE

    def __post_init__(self):
        if self.stride is None:
            self.stride = self.target_context // 2
    

class Dataset(_Dataset):
    def __init__(self, config: Config):
        super().__init__()
        self.source_tokens = []
        self.target_tokens = []
        self.src_pad = config.source_pad_id
        self.tgt_pad = config.target_pad_id
        self._prepare_tokens(config)

    def __len__(self) -> int:
        return len(self.source_tokens)
    
    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.source_tokens[idx], self.target_tokens[idx]

    def collate(self, batch: list[tuple[list[int]]]) -> tuple[list[list[int]]]:
        Xs, Ys = zip(*batch)
        Xs, Ys = [torch.tensor(x) for x in Xs], [torch.tensor(y) for y in Ys]
        Xs = pad_sequence(Xs, batch_first=True, padding_value=self.src_pad)
        Ys = pad_sequence(Ys, batch_first=True, padding_value=self.tgt_pad)
        return Xs, Ys

    def encode_sample(self, src: str, tgt: str, config: Config) -> tuple[list[list[int]]]:
        src = src.replace(config.pseudo_newline, "\n") if config.pseudo_newline else src
        tgt = tgt.replace(config.pseudo_newline, "\n") if config.pseudo_newline else tgt
        src_tokens = config.encode_source(src)
        assert len(src_tokens) <= config.source_context, f"\"{src}\" exceeds maximum context"
        tgt_tokens = [config.sos_id] + config.encode_target(tgt) + [config.eos_id]
        if len(tgt_tokens) > config.target_context + 1:
            it = range(config.target_context + 1, len(tgt_tokens) + config.stride, config.stride)
            tgt_tokens = [tgt_tokens[i - (config.target_context + 1): i] for i in it]
        else:
            tgt_tokens = [tgt_tokens]
        return [src_tokens] * len(tgt_tokens), tgt_tokens

    def _prepare_tokens(self, config: Config):
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
