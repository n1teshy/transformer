import os
import pickle
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.utils.configs import SeqToSeqDataConfig


class SeqToSeqDataset:
    def __init__(self, config: SeqToSeqDataConfig):
        self.config = config
        self._prepare_shards()
        self.source_sequences = []
        self.target_sequences = []
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._load_current_shard()

    def _prepare_shards(self):
        config = self.config
        if config.cache_dir is not None:
            assert (config.cache_dir / "source").exists()
            assert (config.cache_dir / "target").exists()
            self.source_shards = [
                f for f in (config.cache_dir / "source").iterdir()
            ]
            self.target_shards = [
                f for f in (config.cache_dir / "target").iterdir()
            ]
        else:
            assert config.source.is_file() == config.target.is_file()
            if config.source.is_file():
                self.source_shards = [config.source]
                self.target_shards = [config.target]
            else:
                self.source_shards = [
                    f for f in config.source.iterdir() if f.is_file()
                ]
                self.target_shards = [
                    f for f in config.target.iterdir() if f.is_file()
                ]
                # make sure there is a 1:1 mapping between source and target files
                assert set(f.name for f in self.source_shards) == set(
                    f.name for f in self.target_shards
                )
            for idx in range(len(self.source_shards)):
                with open(self.source_shards[idx], encoding="utf-8") as f:
                    source_line_count = sum(1 for _ in f)
                with open(self.target_shards[idx], encoding="utf-8") as f:
                    target_line_count = sum(1 for _ in f)
                assert source_line_count == target_line_count != 0
        if self.config.shuffle_shards:
            pairs = list(zip(self.source_shards, self.target_shards))
            random.shuffle(pairs)
            self.source_shards, self.target_shards = zip(*pairs)

    def reset(self):
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._load_current_shard()

    def _encode_sample(self, src: str, tgt: str) -> tuple[list[list[int]]]:
        if self.config.pseudo_newline is not None:
            src = src.replace(self.config.pseudo_newline, "\n")
            tgt = tgt.replace(self.config.pseudo_newline, "\n")
        src_tokens = self.config.encode_source(src)
        assert (
            len(src_tokens) <= self.config.encoder_context
        ), f'"{src}" exceeds encoder context'
        tgt_tokens = (
            [self.config.sos_id]
            + self.config.encode_target(tgt)
            + [self.config.eos_id]
        )
        # the target sequences are stored as "<sos> + xxxx + <eos>"
        # only "<sos> + xxxx" is fed to the decoder
        # i.e "full_target_sequence - <eos_token>"
        if len(tgt_tokens) - 1 > self.config.decoder_context:
            it = range(
                self.config.decoder_context + 1,
                len(tgt_tokens) + self.config.stride,
                self.config.stride,
            )
            tgt_tokens = [
                tgt_tokens[i - (self.config.decoder_context + 1) : i]
                for i in it
            ]
        else:
            tgt_tokens = [tgt_tokens]
        return [src_tokens] * len(tgt_tokens), tgt_tokens

    def _load_current_shard(self):
        if self.config.cache_dir is not None:
            self.source_sequences = pickle.load(
                open(self.source_shards[self.current_shard_idx], "rb")
            )
            self.target_sequences = pickle.load(
                open(self.target_shards[self.current_shard_idx], "rb")
            )
        else:
            self.source_sequences, self.target_sequences = [], []
            source = self.source_shards[self.current_shard_idx]
            target = self.target_shards[self.current_shard_idx]
            with open(source, encoding="utf-8") as source_f:
                with open(target, encoding="utf-8") as target_f:
                    while source_line := source_f.readline():
                        target_line = target_f.readline()
                        src, tgt = self._encode_sample(
                            source_line, target_line
                        )
                        self.source_sequences.extend(src)
                        self.target_sequences.extend(tgt)
        if self.config.shuffle_samples:
            pairs = list(zip(self.source_sequences, self.target_sequences))
            random.shuffle(pairs)
            self.source_sequences, self.target_sequences = zip(*pairs)

    def cache(self, path: os.PathLike, verbose: bool = False):
        path = Path(path)
        source_dir, target_dir = path / "source", path / "target"
        source_dir.mkdir(exist_ok=True), target_dir.mkdir(exist_ok=True)
        for shard_idx in range(len(self.source_shards)):
            self.current_shard_idx = shard_idx
            self._load_current_shard()
            source_file = (
                source_dir / f"{self.source_shards[shard_idx].name}.pkl"
            )
            target_file = (
                target_dir / f"{self.target_shards[shard_idx].name}.pkl"
            )
            with open(source_file, "wb") as sf, open(target_file, "wb") as tf:
                pickle.dump(self.source_sequences, sf)
                pickle.dump(self.target_sequences, tf)
            if verbose:
                print(
                    f"saved {shard_idx + 1}/{len(self.source_shards)} shards"
                )

    def next_batch(self) -> tuple[Tensor] | None:
        batch_till = self.sequences_processed + self.config.batch_size
        Xs = self.source_sequences[self.sequences_processed : batch_till]
        Ys = self.target_sequences[self.sequences_processed : batch_till]
        if batch_till > len(self.source_sequences):
            self.current_shard_idx += 1
            if self.current_shard_idx >= len(self.source_shards):
                return
            self._load_current_shard()
            batch_till = self.config.batch_size - len(Xs)
            Xs += self.source_sequences[:batch_till]
            Ys += self.target_sequences[:batch_till]
        Xs, Ys = [torch.tensor(x) for x in Xs], [torch.tensor(y) for y in Ys]
        Xs = pad_sequence(
            Xs, batch_first=True, padding_value=self.config.source_pad_id
        )
        Ys = pad_sequence(
            Ys, batch_first=True, padding_value=self.config.target_pad_id
        )
        self.sequences_processed += self.config.batch_size
        return Xs, Ys
