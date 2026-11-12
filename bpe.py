#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3 byte-level BPE codec, pure standard library.

Reads the real ``vocab.json`` / ``merges.txt`` shipped with the teacher repo.
A controllable prefix of the vocabulary is used so that a CPU-only sandbox can
train a real-token language model; tokens outside the prefix fall back to their
byte-level single-character ids (which are always inside the prefix).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

# GPT-2 / Qwen byte-to-unicode tables -----------------------------------------
def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

B2U = _bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


class QwenBPESubset:
    def __init__(self, vocab_json: str, merges_txt: str, vocab_size: int = 2048):
        with open(vocab_json, encoding="utf-8") as f:
            full = json.load(f)
        tok2id_full = full
        id2tok_full = {v: k for k, v in tok2id_full.items()}
        self.V = min(vocab_size, len(id2tok_full) + 1)
        # prefix of the real vocabulary, plus one reserved UNK slot at the end
        self.id2tok = {i: id2tok_full.get(i, "!") for i in range(self.V - 1)}
        self.tok2id = {t: i for i, t in self.id2tok.items()}
        self.unk_id = self.V - 1
        self.id2tok[self.unk_id] = "<unk>"
        self.tok2id["<unk>"] = self.unk_id
        self.merges = {}
        lines = open(merges_txt, encoding="utf-8").read().splitlines()
        if lines and lines[0].startswith("#version"):
            lines = lines[1:]
        for rank, line in enumerate(lines):
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                self.merges[tuple(parts)] = rank
        self._cache = {}

    def _byte_fallback(self, text: str):
        out = []
        for ch in text:
            for b in ch.encode("utf-8"):
                u = B2U[b]
                out.append(self.tok2id.get(u, self.unk_id))
        return out

    def _bpe(self, token: str):
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        while len(word) > 1:
            pairs = [(self.merges.get((word[i], word[i + 1]), 1 << 30), i,
                      (word[i], word[i + 1])) for i in range(len(word) - 1)]
            best = min(pairs, key=lambda x: (x[0], x[1]))
            if best[0] == 1 << 30:
                break
            rank, i, pair = best
            merged = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and (word[j], word[j + 1]) == pair:
                    merged.append(word[j] + word[j + 1])
                    j += 2
                else:
                    merged.append(word[j])
                    j += 1
            word = merged
        self._cache[token] = word
        return word

    def encode(self, text: str):
        ids = []
        # byte-level pre-tokenisation with a simple whitespace-aware split
        buf = ""
        chunks = []
        for ch in text:
            buf += ch
            if ch.isspace() or ch in ".,;:!?\"'()[]{}":
                chunks.append(buf)
                buf = ""
        if buf:
            chunks.append(buf)
        for chunk in chunks:
            raw = "".join(B2U[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe(raw):
                if piece in self.tok2id:
                    ids.append(self.tok2id[piece])
                else:
                    ids.extend(self._byte_fallback("".join(chr(U2B[c]) if c in U2B else c for c in piece)))
        return ids

    def decode(self, ids) -> str:
        s = "".join(self.id2tok.get(int(i), "<unk>") for i in ids)
        s = s.replace("<unk>", "")
        try:
            return bytes(U2B.get(c, ord(c)) for c in s).decode("utf-8", errors="replace")
        except Exception:
            return s

    def token(self, i: int) -> str:
        return self.id2tok.get(int(i), "<unk>")
