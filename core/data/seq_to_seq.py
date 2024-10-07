import torch
from typing import Generator
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from core.utils.configs import SeqToSeqDataConfig


class SeqToSeqDataset():
    def __init__(self, config: SeqToSeqDataConfig):
        assert config.source.is_file() == config.target.is_file()
        if config.source.is_file():
            self.source_shards = [config.source]
            self.target_shards = [config.target]
        else:
            self.source_shards = [f for f in config.source.iterdir() if f.is_file()]
            self.target_shards = [f for f in config.target.iterdir() if f.is_file()]
        assert set([f.name for f in self.source_shards]) == set([f.name for f in self.target_shards])
        self.source_shards.sort(key=lambda addr: addr.stem)
        self.target_shards.sort(key=lambda addr: addr.stem)
        for idx in range(len(self.source_shards)):
            with open(self.source_shards[idx], encoding="utf-8") as f:
                source_line_count = sum(1 for _ in f)
            with open(self.target_shards[idx], encoding="utf-8") as f:
                target_line_count = sum(1 for _ in f)
            assert source_line_count == target_line_count != 0
        self.source_sequences = []
        self.target_sequences = []
        self.config = config
        self._reset()

    def _reset(self):
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._prepare_current_shard()

    def _encode_sample(self, src: str, tgt: str) -> tuple[list[list[int]]]:
        if self.config.pseudo_newline is not None:
            src = src.replace(self.config.pseudo_newline, "\n")
            tgt = tgt.replace(self.config.pseudo_newline, "\n")
        src_tokens = self.config.encode_source(src)
        assert len(src_tokens) <= self.config.encoder_context, f"\"{src}\" exceeds encoder context"
        tgt_tokens = [self.config.sos_id] + self.config.encode_target(tgt) + [self.config.eos_id]
        if len(tgt_tokens) - 1 > self.config.decoder_context:
            it = range(self.config.decoder_context + 1, len(tgt_tokens) + self.config.stride, self.config.stride)
            tgt_tokens = [tgt_tokens[i - (self.config.decoder_context + 1): i] for i in it]
        else:
            tgt_tokens = [tgt_tokens]
        return [src_tokens] * len(tgt_tokens), tgt_tokens

    def _prepare_current_shard(self):
        self.source_sequences, self.target_sequences = [], []
        source = self.source_shards[self.current_shard_idx]
        target = self.target_shards[self.current_shard_idx]
        with open(source, encoding="utf-8") as source_f:
            with open(target, encoding="utf-8") as target_f:
                while source_line := source_f.readline():
                    source_line = source_line.rstrip("\n")
                    target_line = target_f.readline().rstrip("\n")
                    src, tgt = self._encode_sample(source_line, target_line)
                    self.source_sequences.extend(src)
                    self.target_sequences.extend(tgt)

    def batch_generator(self) -> Generator[tuple[Tensor], None, None]:
        self._reset()
        while True:
            batch_till = self.sequences_processed + self.config.batch_size
            Xs = self.source_sequences[self.sequences_processed: batch_till]
            Ys = self.target_sequences[self.sequences_processed: batch_till]
            if batch_till > len(self.source_sequences):
                self.current_shard_idx += 1
                if self.current_shard_idx == len(self.source_shards):
                    return
                self._prepare_current_shard()
                batch_till = self.config.batch_size - len(Xs)
                Xs += self.source_sequences[:batch_till]
                Ys += self.target_sequences[:batch_till]
            Xs, Ys = [torch.tensor(x) for x in Xs], [torch.tensor(y) for y in Ys]
            Xs = pad_sequence(Xs, batch_first=True, padding_value=self.config.source_pad_id)
            Ys = pad_sequence(Ys, batch_first=True, padding_value=self.config.target_pad_id)
            self.sequences_processed = batch_till
            yield Xs.to(device=self.config.device), Ys.to(device=self.config.device)
