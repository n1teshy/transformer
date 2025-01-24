import os
import pickle
import random
from pathlib import Path
from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence

from core.utils.configs import GeneratorDataConfig


class GeneratorDataset:
    def __init__(self, config: GeneratorDataConfig):
        self.config = config
        self.shards = []
        self.token_sequences = []
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._prepare_shards()
        self._load_current_shard()

    def _prepare_shards(self):
        if self.config.cache_dir is not None:
            self.shards = list(self.config.cache_dir.glob("*.pkl"))
        else:
            if self.config.source.is_file():
                self.shards = [self.config.source]
            else:
                self.shards = list(self.config.source.rglob("*.txt"))
        if self.config.shuffle_shards:
            random.shuffle(self.shards)

    def reset(self):
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._load_current_shard()

    def _encode_sample(self, text: str) -> list[list[int]]:
        tokens = (
            [self.config.sos_id]
            + self.config.encode(text)
            + [self.config.eos_id]
        )
        if len(tokens) - 1 > self.config.context:
            it = range(
                self.config.context + 1,
                len(tokens) + self.config.stride,
                self.config.stride,
            )
            return [tokens[i - (self.config.context + 1) : i] for i in it]
        return [tokens]

    def _read_shard(self, idx: int) -> list[str]:
        with open(self.shards[idx], encoding="utf-8") as f:
            if self.config.sample_delimiter is None:
                return f.read().splitlines()
            return f.read().split(self.config.sample_delimiter)

    def _load_current_shard(self):
        if self.config.cache_dir is not None:
            with open(self.shards[self.current_shard_idx], "rb") as f:
                self.token_sequences = pickle.load(f)
        else:
            self.token_sequences = []
            for sample in self._read_shard(self.current_shard_idx):
                self.token_sequences.extend(self._encode_sample(sample))
        if self.config.shuffle_samples:
            random.shuffle(self.token_sequences)

    def cache(self, path: os.PathLike, verbose: bool = False):
        path = Path(path)
        assert path.exists(), "%s doesn't exist" % (path,)
        for shard_idx in range(len(self.shards)):
            self.current_shard_idx = shard_idx
            self._load_current_shard()
            cache_file = path / f"{shard_idx}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self.token_sequences, f)
            if verbose:
                print(f"saved {shard_idx + 1}/{len(self.shards)} shards")

    def next_batch(self) -> Optional[tuple[torch.Tensor]]:
        batch_till = self.sequences_processed + self.config.batch_size
        tokens = self.token_sequences[self.sequences_processed : batch_till]
        if batch_till > len(self.token_sequences):
            self.current_shard_idx += 1
            if self.current_shard_idx < len(self.shards):
                self._load_current_shard()
                batch_till = self.config.batch_size - len(tokens)
                tokens += self.token_sequences[:batch_till]
            elif len(tokens) == 0:
                return
        Xs = [torch.tensor(row[:-1]) for row in tokens]
        Ys = [torch.tensor(row[1:]) for row in tokens]
        Xs = pad_sequence(
            Xs, batch_first=True, padding_value=self.config.pad_id
        )
        Ys = pad_sequence(
            Ys, batch_first=True, padding_value=self.config.pad_id
        )
        self.sequences_processed += Xs.shape[0]
        return Xs, Ys
