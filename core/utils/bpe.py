import unicodedata

import regex as re

import core.constants as c

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def get_stats(ids: list[int], cache: dict | None = None) -> dict:
    cache = {} if cache is None else cache
    for pair in zip(ids, ids[1:]):
        cache[pair] = cache.get(pair, 0) + 1
    return cache


def merge_tokens(ids: list[int], pair: tuple[int], new_id: int) -> list[int]:
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


def render_token(parts: bytes) -> str:
    string = parts.decode("utf-8", errors="replace")
    chars = []
    for c in string:
        if unicodedata.category(c)[0] == "C":
            chars.append(f"\\u{ord(c):04x}")
        else:
            chars.append(c)
    return "".join(chars)


class Tokenizer:
    def __init__(self):
        self.pattern = re.compile(GPT4_SPLIT_PATTERN)
        self.vocab = {}
        self.merges = {}
        self.specials = {}  # str: int
        self.inverse_specials = {}

    @property
    def size(self):
        return len(self.vocab) + len(self.specials)

    @property
    def sos_id(self):
        return self.specials.get(c.TOKEN_SOS)

    @property
    def eos_id(self):
        return self.specials.get(c.TOKEN_EOS)

    @property
    def pad_id(self):
        return self.specials.get(c.TOKEN_PAD)

    def train(self, file: str, max_tokens=256, verbose=False) -> None:
        with open(file, encoding="utf-8") as f:
            text = f.read()
        self.vocab = {i: bytes([i]) for i in range(256)}
        splits = re.findall(self.pattern, text)
        splits = [list(split.encode("utf-8")) for split in splits]
        for round in range(max_tokens - 256):
            frequencies = {}
            for split in splits:
                frequencies = get_stats(split, frequencies)
            pair = max(frequencies, key=frequencies.get)
            splits = [
                merge_tokens(split, pair, 256 + round) for split in splits
            ]
            self.merges[pair] = 256 + round
            self.vocab[256 + round] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(
                    f"token {256 + round}: {self.vocab[pair[0]]} + {self.vocab[pair[1]] } -> {self.vocab[256 + round]}"
                )

    def add_special_tokens(self, *specials) -> None:
        assert all([isinstance(token, str) for token in specials])
        self.specials = {
            token: self.size + i for i, token in enumerate(specials)
        }
        self.inverse_specials = {v: k for k, v in self.specials.items()}

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge_tokens(ids, pair, self.merges[pair])
        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        splits = re.findall(self.pattern, text)
        ids = []
        for split in splits:
            ids.extend(self._encode_chunk(split.encode("utf-8")))
        return ids

    def encode(self, text: str, allowed_specials="none_raise") -> list[int]:
        specials = None
        if allowed_specials == "all":
            specials = self.specials
        elif allowed_specials == "none":
            specials = {}
        elif allowed_specials == "none_raise":
            specials = {}
            assert all(tkn not in text for tkn in self.specials)
        elif isinstance(allowed_specials, set):
            specials = {
                k: v for k, v in self.specials.item() if k in allowed_specials
            }
        if not specials:
            return self.encode_ordinary(text)
        specials_pattern = (
            "(" + "|".join(re.escape(tkn) for tkn in specials) + ")"
        )
        chunks = re.split(specials_pattern, text)
        ids = []
        for chunk in chunks:
            if chunk in specials:
                ids.append(specials[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_parts = []
        for id in ids:
            if id in self.vocab:
                byte_parts.append(self.vocab[id])
            elif id in self.inverse_specials:
                byte_parts.append(self.inverse_specials[id].encode("utf-8"))
            else:
                raise ValueError(f"{id} is not a valid token id")
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, addr: str) -> None:
        with open(addr + ".model", "w") as f:
            f.write(f"{len(self.specials)}\n")
            for special, special_idx in self.specials.items():
                f.write(f"{special} {special_idx}\n")
            for p0, p1 in self.merges:
                f.write(f"{p0} {p1}\n")
        merged_tokens = {v: k for k, v in self.merges.items()}
        with open(addr + ".vocab", "w", encoding="utf-8") as f:
            for idx, word in self.vocab.items():
                word = render_token(word)
                if idx not in merged_tokens:
                    f.write(f"{word} [{idx}]\n")
                else:
                    p0, p1 = merged_tokens[idx]
                    p0, p1 = (
                        render_token(self.vocab[p0]),
                        render_token(self.vocab[p1]),
                    )
                    f.write(f"[{p0}] [{p1}] -> {word} [{idx}]\n")

    def load(self, file: str) -> None:
        with open(file, encoding="utf-8") as f:
            num_specials = int(f.readline().strip())
            self.vocab = {i: bytes([i]) for i in range(256)}
            for _ in range(num_specials):
                special, idx = f.readline().strip().split()
                self.specials[special] = int(idx)
                self.inverse_specials[int(idx)] = special
            for round, line in enumerate(f):
                p0, p1 = map(int, line.strip().split())
                self.merges[(p0, p1)] = 256 + round
                self.vocab[256 + round] = self.vocab[p0] + self.vocab[p1]
