import os
import pickle
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from core.utils.configs import TransformerDataConfig


class TransformerDataset:
    def __init__(self, config: TransformerDataConfig):
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
            src_dir, tgt_dir = (
                config.cache_dir / "source",
                config.cache_dir / "target",
            )
            self.source_shards = list(src_dir.glob("*.pkl"))
            self.target_shards = list(tgt_dir.glob("*.pkl"))
            assert set(f.name for f in self.source_shards) >= set(
                f.name for f in self.target_shards
            ), "invalid cache directory"
        else:
            assert config.source.is_file() == config.target.is_file()
            if config.source.is_file():
                self.source_shards = [config.source]
                self.target_shards = [config.target]
            else:
                self.source_shards = list(config.source.rglob("*.txt"))
                self.target_shards = list(config.target.rglob("*.txt"))
                # ensure there is a target file for every source file
                assert all(
                    (config.target / shard.relative_to(config.source)).exists()
                    for shard in self.source_shards
                ), "some source shards are missing corresponding target shards"
            for idx in range(len(self.source_shards)):
                src_samples, tgt_samples = self._read_shard(idx)
                assert len(src_samples) == len(tgt_samples), (
                    "source shard %s(%d) is incompatible with target shard %s(%d)"
                    % (
                        self.source_shards[idx],
                        len(src_samples),
                        self.target_shards[idx],
                        len(tgt_samples),
                    )
                )
        if self.config.shuffle_shards:
            pairs = list(zip(self.source_shards, self.target_shards))
            random.shuffle(pairs)
            self.source_shards, self.target_shards = zip(*pairs)

    def reset(self):
        self.current_shard_idx = 0
        self.sequences_processed = 0
        self._load_current_shard()

    def _encode_sample(
        self, src: str, tgt: str
    ) -> tuple[list[list[int]], list[list[int]]]:
        src_tokens = self.config.encode_source(src)
        assert len(src_tokens) <= self.config.encoder_context, (
            '"%s" exceeds encoder context' % (src,)
        )
        tgt_tokens = (
            [self.config.sos_id]
            + self.config.encode_target(tgt)
            + [self.config.eos_id]
        )
        # "len(tgt_tokens) - 1" because the decoder won't be fed EOS
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

    def _read_shard(self, idx: int) -> tuple[list[str], list[str]]:
        src, tgt = self.source_shards[idx], self.target_shards[idx]
        with open(src, encoding="utf-8") as sf, open(
            tgt, encoding="utf-8"
        ) as tf:
            if self.config.sample_delimiter is not None:
                src_samples = sf.read().split(self.config.sample_delimiter)
                tgt_samples = tf.read().split(self.config.sample_delimiter)
            else:
                src_samples = sf.read().splitlines()
                tgt_samples = tf.read().splitlines()
            return src_samples, tgt_samples

    def _load_current_shard(self):
        if self.config.cache_dir is not None:
            with open(
                self.source_shards[self.current_shard_idx], "rb"
            ) as sf, open(
                self.target_shards[self.current_shard_idx], "rb"
            ) as tf:
                self.source_sequences = pickle.load(sf)
                self.target_sequences = pickle.load(tf)
        else:
            self.source_sequences, self.target_sequences = [], []
            for src, tgt in zip(*self._read_shard(self.current_shard_idx)):
                src, tgt = self._encode_sample(src, tgt)
                self.source_sequences.extend(src)
                self.target_sequences.extend(tgt)
        if self.config.shuffle_samples:
            pairs = list(zip(self.source_sequences, self.target_sequences))
            random.shuffle(pairs)
            self.source_sequences, self.target_sequences = zip(*pairs)

    def cache(self, path: os.PathLike, verbose: bool = True):
        path = Path(path)
        assert path.exists(), "%s doesn't exist" % (path,)
        source_dir, target_dir = path / "source", path / "target"
        source_dir.mkdir(exist_ok=True), target_dir.mkdir(exist_ok=True)
        for shard_idx in range(len(self.source_shards)):
            self.current_shard_idx = shard_idx
            self._load_current_shard()
            source_file = source_dir / f"{shard_idx}.pkl"
            target_file = target_dir / f"{shard_idx}.pkl"
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
            if self.current_shard_idx < len(self.source_shards):
                self._load_current_shard()
                batch_till = self.config.batch_size - len(Xs)
                Xs += self.source_sequences[:batch_till]
                Ys += self.target_sequences[:batch_till]
            elif len(Xs) == 0:
                return
        Xs, Ys = [torch.tensor(x) for x in Xs], [torch.tensor(y) for y in Ys]
        Xs = pad_sequence(
            Xs, batch_first=True, padding_value=self.config.source_pad_id
        )
        Ys = pad_sequence(
            Ys, batch_first=True, padding_value=self.config.target_pad_id
        )
        self.sequences_processed += Xs.shape[0]
        return Xs, Ys
