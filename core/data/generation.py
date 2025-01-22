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
        self._prepare_shards(config)

    def _encode_sample(self, text: str) -> list[list[int]]:
        if self.config.pseudo_newline is not None:
            text = text.replace(self.config.pseudo_newline, "\n")
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

    def _prepare_shards(self):
        if self.config.cache_dir is not None:
            self.shards = [
                file
                for file in self.config.cache_dir.iterdir()
                if file.suffix == ".pkl"
            ]
        if self.config.source.is_dir():
            self.shards = [
                file
                for file in self.config.source.iterdir()
                if file.suffix == ".txt"
            ]
        else:
            self.shards = [self.config.source]
        if self.config.shuffle_shards:
            random.shuffle(self.shards)

    def _load_current_shard(self):
        if self.config.cache_dir is not None:
            with open(self.shards[self.current_shard_idx], "rb") as f:
                self.token_sequences = pickle.load(f)
        else:
            shard_file = self.shards[self.current_shard_idx]
            with open(shard_file, encoding="utf-8") as f:
                for sample in f:
                    self.token_sequences.extend(self._encode_sample(sample))
        if self.config.shuffle_samples:
            random.shuffle(self.token_sequences)

    def cache(self, path: os.PathLike, verbose: bool = False):
        path = Path(path)
        assert path.exists()
        for shard_idx in range(len(self.shards)):
            self.current_shard_idx = shard_idx
            self._load_current_shard()
            cache_file = path / f"{self.shards[shard_idx].name}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(self.source_sequences, f)
            if verbose:
                print(
                    f"saved {shard_idx + 1}/{len(self.source_shards)} shards"
                )

    def next_batch(self) -> Optional[tuple[torch.Tensor]]:
        batch_till = self.sequences_processed + self.config.batch_size
        Xs = self.token_sequences[self.sequences_processed : batch_till]
        if batch_till > len(self.token_sequences):
            self.current_shard_idx += 1
            if self.current_shard_idx >= len(self.shards):
                return
            self._load_current_shard()
            batch_till = self.config.batch_size - len(Xs)
            Xs += self.token_sequences[:batch_till]
        Xs, Ys = [torch.tensor(x) for x in Xs], [torch.tensor(y) for y in Ys]
        Xs = pad_sequence(
            Xs, batch_first=True, padding_value=self.config.pad_id
        )
        Xs, Ys = Xs[:, :-1], Xs[:, 1:]
        self.sequences_processed += self.config.batch_size
        return Xs, Ys
