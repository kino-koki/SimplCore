#!/usr/bin/env python3
"""
speak_staged.py -- staged, resumable REAL full-depth forward pass.

The sandbox kills long, high-traffic commands, so the 24-layer MiniCPM5-1B
forward is split into 5 short invocations.  Each stage downloads ONLY the
weight slices it needs, saves the tiny hidden state (B,T,h) to disk, exits,
and the next stage resumes from it.  Header/layout is cached once (25 KB).

Usage:
  GH_TOKEN=... python3 speak_staged.py layout
  GH_TOKEN=... python3 speak_staged.py 0     # init + layers  0..5
  GH_TOKEN=... python3 speak_staged.py 1     #        layers  6..11
  GH_TOKEN=... python3 speak_staged.py 2     #        layers 12..17
  GH_TOKEN=... python3 speak_staged.py 3     #        layers 18..23 + norm
  GH_TOKEN=... python3 speak_staged.py 4     # lm_head logits + top-5
"""
from __future__ import annotations

import base64
import json
import os
import struct
import sys
import time
import urllib.request
from collections import OrderedDict

import numpy as np

sys.path.insert(0, "/data/workspace/ts2sc_vl47")
from speak_test import BPETokenizer, rmsnorm, rope, softmax, Llama  # noqa: E402

W = "/data/workspace"
D = W + "/ts2sc_vl47"
REPO = "MiniCPM5-1B"
LAYOUT = D + "/probe/m5_layout.json"
CKPT = D + "/probe/forward_state.npz"
RESULT = D + "/speak_results.json"
TOK = os.environ.get("GH_TOKEN")
if not TOK:
    sys.exit("GH_TOKEN is not set (env only).")

H = {"Authorization": "Bearer " + TOK,
     "Accept": "application/vnd.github+json",
     "User-Agent": "simplcore-speak-staged",
     "X-GitHub-Api-Version": "2022-11-28"}


def api(path, timeout=240):
    req = urllib.request.Request("https://api.github.com" + path, headers=H)
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())


class StagedVF:
    """On-demand virtual safetensors reader; at most 3 parts in RAM."""

    def __init__(self, layout, shas, sizes):
        self.layout, self.shas, self.sizes = layout, shas, sizes
        self.tensors = layout["tensors"]
        self.starts, self.dlens = layout["starts"], layout["dlens"]
        self.hlen, self.total = layout["hlen"], layout["total"]
        self.n = len(shas)
        self.cache = OrderedDict()
        self.bytes_fetched = 0
        self.n_fetch = 0

    def _fetch(self, i):
        last = None
        for a in range(4):
            try:
                d = api("/repos/kino-koki/%s/git/blobs/%s" % (REPO, self.shas[i]))
                raw = base64.b64decode(d["content"])
                if len(raw) != self.sizes[i]:
                    raise IOError("size %d != %d" % (len(raw), self.sizes[i]))
                self.bytes_fetched += len(raw)
                self.n_fetch += 1
                return raw
            except Exception as e:
                last = e
                print("   retry part_%02d #%d: %s" % (i, a + 1, str(e)[:80]),
                      flush=True)
                time.sleep(2 + 2 * a)
        raise RuntimeError("part %d: %s" % (i, last))

    def _get(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        raw = self._fetch(i)
        self.cache[i] = raw
        self.cache.move_to_end(i)
        while len(self.cache) > 3:
            self.cache.popitem(last=False)
        return raw

    def read(self, off, length):
        buf = bytearray()
        pos, end = off, off + length
        while pos < end:
            idx = None
            for i in range(self.n):
                if self.starts[i] <= pos < self.starts[i] + self.dlens[i]:
                    idx = i
                    break
            if idx is None:
                raise RuntimeError("offset %d outside virtual file %d" %
                                   (pos, self.total))
            local = pos - self.starts[idx] + (self.hlen if idx == 0 else 0)
            raw = self._get(idx)
            take = min(end - pos, self.dlens[idx] - (pos - self.starts[idx]))
            buf += raw[local:local + take]
            pos += take
        return bytes(buf)

    def tensor(self, name):
        m = self.tensors[name]
        a, b = m["data_offsets"]
        raw = self.read(a, b - a)
        u = np.frombuffer(raw, dtype=np.uint16).reshape(m["shape"])
        return (u.astype(np.uint32) << 16).view(np.float32).copy()


class StagedLlama(Llama):
    def __init__(self, vf, cfg):
        super().__init__(vf, cfg)
        self.vf = vf

    def init_state(self, ids_batch, pad_id):
        B = len(ids_batch)
        T = max(len(s) for s in ids_batch)
        pad = np.full((B, T), pad_id, dtype=np.int64)
        for i, s in enumerate(ids_batch):
            pad[i, T - len(s):] = s
        base = self.vf.tensors["model.embed_tokens.weight"]["data_offsets"][0]
        cache = {}
        for i in sorted(set(int(v) for v in pad.ravel())):
            raw = self.vf.read(base + i * self.h * 2, self.h * 2)
            u = np.frombuffer(raw, dtype=np.uint16).copy()
            cache[i] = (u.astype(np.uint32) << 16).view(np.float32).astype(np.float32)
        x = np.stack([[cache[int(v)] for v in row] for row in pad])
        inv = 1.0 / (self.theta ** (np.arange(0, self.hd, 2, dtype=np.float32) / self.hd))
        pos = np.stack([np.arange(T, dtype=np.float32) - (T - len(s))
                        for s in ids_batch]).clip(0)
        fr = pos[:, :, None] * inv[None, None, :]
        emb = np.concatenate([fr, fr], axis=-1)
        cos, sin = np.cos(emb), np.sin(emb)
        key_ok = np.array([[j >= T - len(s) for j in range(T)]
                           for s in ids_batch])[:, None, :]
        causal = np.tril(np.ones((T, T), dtype=bool))[None, :, :]
        mask = np.where(key_ok & causal, 0.0, -1e9).astype(np.float32)
        return x, cos, sin, mask

    def run_layers(self, a, b, x, cos, sin, mask):
        t0 = time.time()
        for L in range(a, b):
            x = self.layer(L, x, cos, sin, mask)
            if (L - a) % 3 == 0:
                print("   layer %2d  %.1fs" % (L, time.time() - t0), flush=True)
        return x

    def finalize(self, x):
        fn = self.vf.tensor("model.norm.weight")
        return rmsnorm(x, fn, self.eps)[:, -1, :]

    def logits(self, hidden):
        V, h = self.cfg["vocab_size"], self.h
        lh_base = self.vf.tensors["lm_head.weight"]["data_offsets"][0]
        out = np.empty((hidden.shape[0], V), dtype=np.float32)
        CH = 8192
        for s in range(0, V, CH):
            e = min(V, s + CH)
            raw = self.vf.read(lh_base + s * h * 2, (e - s) * h * 2)
            u = np.frombuffer(raw, dtype=np.uint16).reshape(e - s, h)
            w = (u.astype(np.uint32) << 16).view(np.float32)
            out[:, s:e] = hidden @ w.T
        return out


def load_parts():
    listing = api("/repos/kino-koki/%s/contents/?ref=main" % REPO)
    parts = sorted([x for x in listing if x["name"].startswith("model_part_")],
                   key=lambda x: x["name"])
    return [x["name"] for x in parts], [x["sha"] for x in parts], [x["size"] for x in parts]


def build_layout():
    names, shas, sizes = load_parts()
    d = api("/repos/kino-koki/%s/git/blobs/%s" % (REPO, shas[0]))
    raw0 = base64.b64decode(d["content"])
    n = struct.unpack("<Q", raw0[:8])[0]
    hdr = json.loads(raw0[8:8 + n])
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    hlen = 8 + n
    d0 = len(raw0) - hlen
    starts, dlens, acc = [0], [d0], d0
    for i in range(1, len(sizes)):
        starts.append(acc)
        dlens.append(sizes[i])
        acc += sizes[i]
    layout = {"names": names, "shas": shas, "sizes": sizes,
              "hlen": hlen, "tensors": tensors, "starts": starts,
              "dlens": dlens, "total": acc}
    json.dump(layout, open(LAYOUT, "w"), ensure_ascii=False)
    print("layout cached: %d parts, %d tensors, data %.3f GB, hlen %d" %
          (len(names), len(tensors), acc / 1e9, hlen), flush=True)
    print("lm_head:", tensors["lm_head.weight"]["shape"],
          tensors["lm_head.weight"]["data_offsets"], flush=True)
    print("has q/k norm:", any("q_norm" in k for k in tensors), flush=True)
    return layout


def main():
    stage = sys.argv[1]
    if stage == "layout":
        build_layout()
        return
    layout = json.load(open(LAYOUT, encoding="utf-8"))
    cfg = json.load(open(D + "/probe/m5_config.json", encoding="utf-8"))
    tok = BPETokenizer(D + "/probe/m5_tokenizer.json")
    prompts = ["中国的首都是", "今天天气很好，", "水的化学式是"]
    enc = [tok.encode(p) for p in prompts]
    vf = StagedVF(layout, layout["shas"], layout["sizes"])
    model = StagedLlama(vf, cfg)
    t0 = time.time()
    print("=== STAGE %s ===" % stage, flush=True)

    if stage == "0":
        x, cos, sin, mask = model.init_state(enc, tok.pad_id)
        x = model.run_layers(0, 6, x, cos, sin, mask)
        np.savez(CKPT, x=x, cos=cos, sin=sin, mask=mask)
        print("saved state -> stage 1", flush=True)
    elif stage in ("1", "2", "3"):
        z = np.load(CKPT)
        x, cos, sin, mask = z["x"], z["cos"], z["sin"], z["mask"]
        a = {"1": 6, "2": 12, "3": 18}[stage]
        x = model.run_layers(a, a + 6, x, cos, sin, mask)
        if stage == "3":
            np.savez(D + "/probe/final_hidden.npz", hidden=model.finalize(x))
            print("saved final hidden -> stage 4", flush=True)
        else:
            np.savez(CKPT, x=x, cos=cos, sin=sin, mask=mask)
            print("saved state -> stage %s" % (int(stage) + 1), flush=True)
    elif stage == "4":
        z = np.load(D + "/probe/final_hidden.npz")
        hidden = z["hidden"]
        logits = model.logits(hidden)
        out = []
        for i, p in enumerate(prompts):
            top = np.argsort(-logits[i])[:8]
            cands = [{"id": int(j), "logit": float(logits[i, j]),
                      "text": tok.decode([int(j)]),
                      "raw": tok.id2tok.get(int(j), "")} for j in top]
            print("\n  PROMPT %r" % p, flush=True)
            for c in cands[:5]:
                print("     %+9.3f  id=%-7d %r  raw=%r" %
                      (c["logit"], c["id"], c["text"], c["raw"]), flush=True)
            out.append({"prompt": p, "n_tokens": len(enc[i]),
                        "ids": [int(v) for v in enc[i]],
                        "roundtrip": tok.decode(enc[i][1:]), "top": cands})
        res = {
            "status": "REAL_FULL_DEPTH_FORWARD",
            "claim": "complete 24-layer forward on real BF16 weights; next-token distribution only",
            "not": ["autoregressive generation", "downstream evaluation",
                    "LM parity", "throughput benchmark", "multimodal inference"],
            "model": {"repo": REPO, "architecture": cfg["architectures"],
                      "layers": cfg["num_hidden_layers"], "hidden": cfg["hidden_size"],
                      "vocab": cfg["vocab_size"], "dtype": "BF16->fp32"},
            "device": {"torch": False, "transformers": False, "gpu": False,
                       "numpy": np.__version__, "cpu_count": os.cpu_count()},
            "weights": {"parts": len(layout["names"]),
                        "virtual_data_bytes": layout["total"],
                        "disk_model_materialised": False,
                        "stage4_bytes_fetched": vf.bytes_fetched,
                        "stage4_api_calls": vf.n_fetch},
            "prompts": out,
        }
        json.dump(res, open(RESULT, "w"), ensure_ascii=False, indent=2)
        print("\nWROTE", RESULT, flush=True)
    else:
        sys.exit("unknown stage")
    print("stage %s done in %.1fs | fetched %.3f GB in %d calls" %
          (stage, time.time() - t0, vf.bytes_fetched / 1e9, vf.n_fetch),
          flush=True)


if __name__ == "__main__":
    main()
