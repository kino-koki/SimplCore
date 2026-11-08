#!/usr/bin/env python3
"""
speak_test.py -- a REAL, dependency-free "can the base model speak?" probe.

Runs on this machine: 2 CPUs, 2.3 GiB RAM, 370 MiB free disk, NO torch,
NO transformers, NO safetensors library.  Only numpy + stdlib.

It streams the real MiniCPM5-1B BF16 weights from the private GitHub repo
through a virtual-file reader (never materialises the 2.16 GB model on disk),
runs ONE full 24-layer Llama forward pass on a batch of Chinese prompts, and
reports the top-5 next tokens decoded with the model's real tokenizer.

This is a genuine next-token distribution test, NOT autoregressive generation
and NOT a claim about downstream task ability.
"""
from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import struct
import sys
import threading
import time
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

W = "/data/workspace"
D = W + "/ts2sc_vl47"
REPO = "MiniCPM5-1B"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set (env only, never written to disk).")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-speak-probe",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, timeout=180):
    req = urllib.request.Request("https://api.github.com" + path, headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


# --------------------------------------------------------------------------
# 1. Real byte-level BPE tokenizer (parsed from the repo's 9.9 MB tokenizer.json)
# --------------------------------------------------------------------------
def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("¡"), ord("¬") + 1)) +
          list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class BPETokenizer:
    def __init__(self, path):
        d = json.load(open(path, encoding="utf-8"))
        m = d["model"]
        self.vocab = m["vocab"]                       # token-string -> id
        self.id2tok = {v: k for k, v in self.vocab.items()}
        self.merges = {}
        for i, pair in enumerate(m["merges"]):
            self.merges[tuple(pair if isinstance(pair, list) else pair.split())] = i
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.added = {a["content"]: a["id"] for a in d.get("added_tokens", [])}
        self.bos_id = self.added.get("<s>", 0)
        self.pad_id = self.added.get("</s>", 1)
        # simplified pre-tokenizer: letters / digits / punctuation / whitespace
        self.pat = re.compile(r"[^\W\d_]+|\d+|[^\s\w]+|\s+", re.UNICODE)

    def _bpe(self, chars):
        word = list(chars)
        while len(word) >= 2:
            best, best_i = None, -1
            for i in range(len(word) - 1):
                r = self.merges.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, best_i = r, i
            if best is None:
                break
            a, b = word[best_i], word[best_i + 1]
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            word = out
        return word

    def encode(self, text):
        ids = [self.bos_id]
        for m in self.pat.finditer(text):
            piece = m.group(0)
            # special tokens
            if piece in self.added:
                ids.append(self.added[piece])
                continue
            bl = "".join(self.b2u[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(list(bl)):
                ids.append(self.vocab.get(tok, self.vocab.get("<unk>", 1)))
        return ids

    def decode(self, ids):
        out = bytearray()
        for i in ids:
            s = self.id2tok.get(i)
            if s is None or s in self.added:
                continue
            for ch in s:
                if ch in self.u2b:
                    out.append(self.u2b[ch])
        return out.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# 2. Virtual safetensors file: model_part_00 carries the full header, the rest
#    are pure byte slices.  Data offsets in the header refer to the ORIGINAL
#    file's data region.  Read by offset, cache at most 4 parts, prefetch ahead.
# --------------------------------------------------------------------------
class VirtualSafetensors:
    def __init__(self, names, shas, sizes):
        self.names, self.shas, self.sizes = names, shas, sizes
        self.n = len(names)
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.bytes_fetched = 0
        self.n_fetch = 0
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.fut = {}
        self.next_idx = 0
        self.header = None
        self.hlen = 0
        self.starts = []
        self.dlens = []
        self._init_layout()

    def _fetch(self, i):
        last = None
        for attempt in range(4):
            try:
                d = api("/repos/kino-koki/%s/git/blobs/%s" % (REPO, self.shas[i]),
                        timeout=240)
                raw = base64.b64decode(d["content"])
                if len(raw) != self.sizes[i]:
                    raise IOError("size mismatch %d != %d" % (len(raw), self.sizes[i]))
                with self.lock:
                    self.bytes_fetched += len(raw)
                    self.n_fetch += 1
                return raw
            except Exception as e:                    # noqa: BLE001
                last = e
                print("   retry part_%02d attempt %d: %s" %
                      (i, attempt + 1, str(e)[:90]), flush=True)
                time.sleep(2 + 2 * attempt)
        raise RuntimeError("part %d failed after 4 attempts: %s" % (i, last))

    def _submit(self, i):
        if 0 <= i < self.n and i not in self.fut:
            self.fut[i] = self.pool.submit(self._fetch, i)

    def _get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        # Strictly sequential drain.  At most 3 futures are in flight at any
        # time, and no completed-but-unconsumed future can accumulate: that was
        # the OOM cause in the first full-depth run.
        for j in range(self.next_idx, i + 1):
            for k in range(j, min(self.n, j + 2)):
                self._submit(k)
            raw = self.fut.pop(j).result()
            self.cache[j] = raw
            self.cache.move_to_end(j)
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
        self.next_idx = max(self.next_idx, i + 1)
        if i not in self.cache:                 # evicted earlier: refetch once
            raw = self._fetch(i)
            self.cache[i] = raw
            self.cache.move_to_end(i)
            while len(self.cache) > 3:
                self.cache.popitem(last=False)
            self.n_fetch += 1
        return self.cache[i]

    def _init_layout(self):
        # part 00: 8-byte header length + JSON header + data
        raw0 = self._get(0)
        n = struct.unpack("<Q", raw0[:8])[0]
        self.header = json.loads(raw0[8:8 + n])
        self.hlen = 8 + n
        self.tensors = {k: v for k, v in self.header.items() if k != "__metadata__"}
        d0 = len(raw0) - self.hlen
        self.starts.append(0)
        self.dlens.append(d0)
        acc = d0
        for i in range(1, self.n):
            self.starts.append(acc)
            self.dlens.append(self.sizes[i])   # pure data slices
            acc += self.sizes[i]
        self.total = acc

    def read(self, off, length):
        buf = bytearray()
        pos = off
        end = off + length
        while pos < end:
            idx = None
            for i in range(self.n):
                s, l = self.starts[i], self.dlens[i]
                if s <= pos < s + l:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError("offset %d outside virtual file (total %d)" % (pos, self.total))
            local = pos - self.starts[idx] + (self.hlen if idx == 0 else 0)
            raw = self._get(idx)
            take = min(end - pos, self.dlens[idx] - (pos - self.starts[idx]))
            buf += raw[local:local + take]
            pos += take
        return bytes(buf)

    def tensor(self, name, dtype="BF16"):
        meta = self.tensors[name]
        a, b = meta["data_offsets"]
        raw = self.read(a, b - a)
        if dtype == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).reshape(meta["shape"])
            return (u.astype(np.uint32) << 16).view(np.float32).copy()
        raise ValueError(dtype)


# --------------------------------------------------------------------------
# 3. Llama forward (RMSNorm / RoPE / GQA / SiLU), pure numpy, fp32
# --------------------------------------------------------------------------
def rmsnorm(x, w, eps=1e-6):
    return x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * w


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def rope(x, cos, sin):
    # cos/sin: [B,T,128] -> broadcast over heads as [B,1,T,128]
    x1, x2 = x[..., :64], x[..., 64:]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos[:, None, :, :] + rot * sin[:, None, :, :]


class Llama:
    def __init__(self, vf, cfg):
        self.vf, self.cfg = vf, cfg
        self.h = cfg["hidden_size"]
        self.nl = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg["num_key_value_heads"]
        self.hd = cfg["head_dim"]
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.names = sorted(vf.tensors.keys())
        self.has_qknorm = any("q_norm" in n for n in self.names)
        self.lm_head_u16 = None

    def embed_rows(self, ids):
        # read ONLY the needed rows of the [V, h] BF16 embedding.
        # Its data region does NOT start at virtual offset 0 -- lm_head occupies
        # the first 401 MB, so we must add the tensor's own data offset.
        base = self.vf.tensors["model.embed_tokens.weight"]["data_offsets"][0]
        out = []
        for i in ids:
            off = base + int(i) * self.h * 2
            raw = self.vf.read(off, self.h * 2)
            out.append(np.frombuffer(raw, dtype=np.uint16).copy())
        return np.stack(out)

    def embed(self, ids):
        u = self.embed_rows(ids)                       # [T,h] uint16 (BF16 bits)
        return (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)

    def layer(self, L, x, cos, sin, mask):
        p = "model.layers.%d." % L
        q = self.vf.tensor(p + "self_attn.q_proj.weight")
        k = self.vf.tensor(p + "self_attn.k_proj.weight")
        v = self.vf.tensor(p + "self_attn.v_proj.weight")
        o = self.vf.tensor(p + "self_attn.o_proj.weight")
        n1 = self.vf.tensor(p + "input_layernorm.weight")
        g = self.vf.tensor(p + "mlp.gate_proj.weight")
        up = self.vf.tensor(p + "mlp.up_proj.weight")
        dn = self.vf.tensor(p + "mlp.down_proj.weight")
        n2 = self.vf.tensor(p + "post_attention_layernorm.weight")
        B, T, _ = x.shape
        h = rmsnorm(x, n1, self.eps)
        qh = (h @ q.T).reshape(B, T, self.nh, self.hd).transpose(0, 2, 1, 3)
        kh = (h @ k.T).reshape(B, T, self.nkv, self.hd).transpose(0, 2, 1, 3)
        vh = (h @ v.T).reshape(B, T, self.nkv, self.hd).transpose(0, 2, 1, 3)
        qn_name, kn_name = p + "self_attn.q_norm.weight", p + "self_attn.k_norm.weight"
        if qn_name in self.vf.tensors:
            qh = rmsnorm(qh, self.vf.tensor(qn_name), self.eps)
        if kn_name in self.vf.tensors:
            kh = rmsnorm(kh, self.vf.tensor(kn_name), self.eps)
        qh = rope(qh, cos, sin)
        kh = rope(kh, cos, sin)
        rep = self.nh // self.nkv
        kh = np.repeat(kh, rep, axis=1)
        vh = np.repeat(vh, rep, axis=1)
        sc = qh @ kh.transpose(0, 1, 3, 2) / math.sqrt(self.hd)
        sc = sc + mask[:, None, :, :]
        a = softmax(sc) @ vh
        a = a.transpose(0, 2, 1, 3).reshape(B, T, self.nh * self.hd)
        x = x + a @ o.T
        h2 = rmsnorm(x, n2, self.eps)
        gg = h2 @ g.T
        uu = h2 @ up.T
        act = gg / (1.0 + np.exp(-gg))
        x = x + (act * uu) @ dn.T
        return x

    def forward(self, ids_batch, pad_id, lm_u16=None):
        B = len(ids_batch)
        T = max(len(x) for x in ids_batch)
        pad = np.full((B, T), pad_id, dtype=np.int64)
        for i, seq in enumerate(ids_batch):
            pad[i, T - len(seq):] = seq           # LEFT padding -> last pos is real
        # embedding: read unique rows once
        uniq = sorted(set(int(i) for i in pad.ravel()))
        cache = {i: self.embed([i])[0] for i in uniq}
        x = np.stack([[cache[int(i)] for i in row] for row in pad]).astype(np.float32)
        inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2, dtype=np.float32) / self.hd))
        pos_mat = np.stack([np.arange(T, dtype=np.float32) - (T - len(seq))
                            for seq in ids_batch]).clip(0)        # [B,T]
        fr = pos_mat[:, :, None] * inv[None, None, :]              # [B,T,64]
        emb = np.concatenate([fr, fr], axis=-1)                    # [B,T,128]
        cos, sin = np.cos(emb), np.sin(emb)
        # causal + left-padding mask, shape [B,T,T] (query, key)
        key_ok = np.array([[j >= T - len(seq) for j in range(T)]
                           for seq in ids_batch])[:, None, :]      # [B,1,T]
        causal = np.tril(np.ones((T, T), dtype=bool))[None, :, :]  # [1,T,T]
        mask = np.where(key_ok & causal, 0.0, -1e9).astype(np.float32)
        t0 = time.time()
        n_layers = int(os.environ.get("SMOKE_LAYERS", self.nl))
        for L in range(n_layers):
            x = self.layer(L, x, cos, sin, mask)
            if L % 6 == 0:
                print("   layer %2d/%d  %.1fs" % (L, self.nl, time.time() - t0),
                      flush=True)
        fn = self.vf.tensor("model.norm.weight")
        x = rmsnorm(x, fn, self.eps)
        hidden = x[:, -1, :]                       # [B,h]
        # logits with chunked BF16 lm_head
        logits = np.empty((B, self.cfg["vocab_size"]), dtype=np.float32)
        V, h = self.cfg["vocab_size"], self.h
        CH = 8192
        for s in range(0, V, CH):
            e = min(V, s + CH)
            u = lm_u16[s:e]                       # uint16 BF16 bit patterns
            w = (u.astype(np.uint32) << 16).view(np.float32)
            logits[:, s:e] = hidden @ w.T
        return logits, pad


# --------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("=== REAL BASE-MODEL SPEAK PROBE (numpy only, no torch) ===", flush=True)
    tok = BPETokenizer(D + "/probe/m5_tokenizer.json")
    cfg = json.load(open(D + "/probe/m5_config.json", encoding="utf-8"))
    print("config: %d layers, h=%d, %d heads/%d kv, V=%d" %
          (cfg["num_hidden_layers"], cfg["hidden_size"],
           cfg["num_attention_heads"], cfg["num_key_value_heads"],
           cfg["vocab_size"]), flush=True)

    prompts = ["中国的首都是", "今天天气很好，", "水的化学式是"]
    enc = [tok.encode(p) for p in prompts]
    roundtrip = [tok.decode(e[1:]) for e in enc]        # strip BOS
    print("tokenizer roundtrip:", [r == p for r, p in zip(roundtrip, prompts)],
          flush=True)
    for p, e, r in zip(prompts, enc, roundtrip):
        print("  %-14s -> %2d tokens | roundtrip=%r" % (p, len(e), r), flush=True)

    # list real weight parts
    listing = api("/repos/kino-koki/%s/contents/?ref=main" % REPO)
    parts = sorted([x for x in listing if x["name"].startswith("model_part_")],
                   key=lambda x: x["name"])
    names = [x["name"] for x in parts]
    shas = [x["sha"] for x in parts]
    sizes = [x["size"] for x in parts]
    print("weight parts: %d, total %.3f GB" %
          (len(parts), sum(sizes) / 1e9), flush=True)

    vf = VirtualSafetensors(names, shas, sizes)
    print("virtual file: header tensors=%d, data=%.3f GB, hlen=%d" %
          (len(vf.tensors), vf.total / 1e9, vf.hlen), flush=True)
    print("has q/k norm:", any("q_norm" in n for n in vf.tensors), flush=True)
    # sanity: first tensor offset must be 0 and lm_head shape must match config
    lh = vf.tensors["lm_head.weight"]
    print("lm_head:", lh["dtype"], lh["shape"], lh["data_offsets"], flush=True)
    assert lh["data_offsets"][0] == 0
    assert list(lh["shape"]) == [cfg["vocab_size"], cfg["hidden_size"]]

    # Materialise lm_head FIRST, in strictly increasing virtual-offset order,
    # so the stream never has to seek backwards later.
    V, h = cfg["vocab_size"], cfg["hidden_size"]
    lh_base = vf.tensors["lm_head.weight"]["data_offsets"][0]
    lm_raw = vf.read(lh_base, V * h * 2)
    lm_u16 = np.frombuffer(lm_raw, dtype=np.uint16).reshape(V, h).copy()
    print("lm_head cached in RAM: %d MB (uint16 BF16 bits)" %
          (lm_u16.nbytes // 1000000), flush=True)
    del lm_raw

    model = Llama(vf, cfg)
    logits, pad = model.forward(enc, tok.pad_id, lm_u16=lm_u16)
    print("forward done in %.1fs | bytes fetched %.3f GB in %d API calls" %
          (time.time() - t_start, vf.bytes_fetched / 1e9, vf.n_fetch), flush=True)

    out = []
    for i, p in enumerate(prompts):
        top = np.argsort(-logits[i])[:8]
        cands = [{"id": int(j), "logit": float(logits[i, j]),
                  "text": tok.decode([int(j)]),
                  "raw": tok.id2tok.get(int(j), "")} for j in top]
        print("\n  PROMPT %r" % p, flush=True)
        for c in cands[:5]:
            print("     %+9.3f  id=%-7d %r" %
                  (c["logit"], c["id"], c["text"]), flush=True)
        out.append({"prompt": p,
                    "n_tokens": len(enc[i]),
                    "ids": [int(v) for v in enc[i]],
                    "roundtrip": roundtrip[i],
                    "top": cands})
    res = {
        "status": "REAL_FORWARD_PASS",
        "claim": "single full-depth forward pass; next-token distribution only",
        "not": ["autoregressive generation", "downstream task evaluation",
                "LM parity", "throughput benchmark"],
        "model": {"repo": REPO, "architecture": cfg["architectures"],
                  "layers": cfg["num_hidden_layers"], "hidden": cfg["hidden_size"],
                  "vocab": cfg["vocab_size"], "dtype": "BF16->fp32"},
        "device": {"torch": False, "transformers": False, "gpu": False,
                   "numpy": np.__version__, "cpu_count": os.cpu_count()},
        "weights": {"parts": len(parts), "bytes_fetched": vf.bytes_fetched,
                    "api_calls": vf.n_fetch, "virtual_data_bytes": vf.total,
                    "disk_model_materialised": False},
        "elapsed_s": round(time.time() - t_start, 1),
        "prompts": out,
    }
    json.dump(res, open(W + "/ts2sc_vl47/speak_results.json", "w"),
              ensure_ascii=False, indent=2)
    print("\nWROTE ts2sc_vl47/speak_results.json", flush=True)


if __name__ == "__main__":
    main()
