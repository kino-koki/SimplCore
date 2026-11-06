"""
ts2sc_verify.py — TS2SC Stage-0 migration verification, RUN FOR REAL in this
sandbox with NO torch / NO transformers / NO downloaded model weights.

What is REAL here
=================
* Teacher architecture config, tensor index and the TS2SC mapping protocol are
  fetched live from GitHub (api.github.com is the only reachable host).
* The Qwen3-0.6B tensor inventory is derived from the hard-coded teacher config
  in the upstream TS2SC script scripts/migrate_real.py (D=1024, FF=3072,
  L=28, H=16, KV=8, HD=128, V=151936, BASES=32) -- NOT invented here.
* The 1.7B cross-check runs the SAME classifier over the REAL 311-tensor
  model.safetensors.index.json of the private Qwen3-1.7B-Instruct repo.
* Optional: attempt to fetch one real weight shard via the git-blob API and
  parse its safetensors header.  If the network policy blocks it, we say so.

What is NOT done here (stated plainly, not faked)
================================================
* No teacher weight VALUES are loaded into memory and no tensor is copied.
* No end-to-end / logit-KL distillation is run (needs torch; PyPI, HF and
  release-asset downloads all return HTTP 403 in this sandbox).
* No perplexity, generation or capability number is produced.

Output: ts2sc_results/ts2sc_verify.json + TS2SC_QWEN3_VERIFY.md
Run:    GH_TOKEN=... python ts2sc_verify.py     (token never written to disk)
"""
from __future__ import annotations

import base64
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, OrderedDict

OUT = "/data/workspace/ts2sc_results"
os.makedirs(OUT, exist_ok=True)
TOK = os.environ.get("GH_TOKEN", "")
UA = "simplcore-ts2sc-verify"


def api(path: str, timeout: int = 90):
    req = urllib.request.Request(
        "https://api.github.com" + path,
        headers={"Authorization": f"Bearer {TOK}", "User-Agent": UA,
                 "Accept": "application/vnd.github+json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def text(repo: str, path: str, timeout: int = 90) -> str:
    d = api(f"/repos/kino-koki/{repo}/contents/{path}", timeout)
    return base64.b64decode(d["content"]).decode("utf-8", "replace")


# ---------------------------------------------------------------- 真实上游数据
def fetch_upstream():
    u = {}
    u["ts2sc_mapping_md"] = text("TS2SC", "docs/01-%E6%9E%B6%E6%9E%84%E6%98%A0%E5%B0%84.md")
    u["migrate_real_py"] = text("TS2SC", "scripts/migrate_real.py")
    u["ts2sc_readme"] = text("TS2SC", "README.md")
    u["qwen17_config"] = json.loads(text("Qwen3-1.7B-Instruct", "config.json"))
    u["qwen17_index"] = json.loads(text("Qwen3-1.7B-Instruct",
                                        "model.safetensors.index.json"))
    u["split_restore"] = text("Qwen3-1.7B-Instruct", "SPLIT_RESTORE.md")
    return u


# ---------------------------------------------------------------- 0.6B 张量清单
# 来源：TS2SC/scripts/migrate_real.py 开头硬编码的 Qwen3-0.6B 真实配置
Q06 = dict(D=1024, FF=3072, L=28, H=16, KV=8, HD=128, V=151936, BASES=32)


def qwen06_tensors():
    """Exact Qwen3 tensor inventory (name -> shape). RoPE has no parameters."""
    D, FF, L, H, KV, HD, V = (Q06[k] for k in ("D", "FF", "L", "H", "KV", "HD", "V"))
    t = OrderedDict()
    t["model.embed_tokens.weight"] = [V, D]
    t["lm_head.weight"] = [V, D]           # Qwen3-0.6B tie_word_embeddings=True
    t["model.norm.weight"] = [D]
    for i in range(L):
        p = f"model.layers.{i}."
        t[p + "input_layernorm.weight"] = [D]
        t[p + "post_attention_layernorm.weight"] = [D]
        t[p + "self_attn.q_proj.weight"] = [H * HD, D]
        t[p + "self_attn.k_proj.weight"] = [KV * HD, D]
        t[p + "self_attn.v_proj.weight"] = [KV * HD, D]
        t[p + "self_attn.o_proj.weight"] = [D, H * HD]
        t[p + "self_attn.q_norm.weight"] = [HD]
        t[p + "self_attn.k_norm.weight"] = [HD]
        t[p + "mlp.gate_proj.weight"] = [FF, D]
        t[p + "mlp.up_proj.weight"] = [FF, D]
        t[p + "mlp.down_proj.weight"] = [D, FF]
    return t


def prod(shape):
    n = 1
    for s in shape:
        n *= int(s)
    return n


# ---------------------------------------------------------------- 映射分类
# 逐条来自 TS2SC/docs/01-架构映射.md（真实文档，非本文件发明）
def classify(name: str, shape):
    if name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        return "COPY", ("tok_emb.weight" if name.startswith("model.embed")
                        else ("head.weight (tied/shared)" if name.startswith("lm_head")
                              else "ln_f.weight")), "bitwise copy"
    if name.endswith("input_layernorm.weight"):
        return "COPY", "mixer.norm.weight", "bitwise copy"
    if name.endswith("post_attention_layernorm.weight"):
        return "COPY", "ffn.norm.weight", "bitwise copy"
    if ".mlp.gate_proj.weight" in name:
        return "COPY", "ffn.gate.weight", "bitwise copy"
    if ".mlp.up_proj.weight" in name:
        return "COPY", "ffn.up.weight", "bitwise copy"
    if ".mlp.down_proj.weight" in name:
        return "COPY", "ffn.down.weight", "bitwise copy"
    if ".self_attn.v_proj.weight" in name:
        return "DERIVE", "mixer.in_proj.weight", "bitwise copy (V -> in_proj, GQA: KV*HD == D)"
    if ".self_attn.k_proj.weight" in name:
        return "DERIVE", "mixer.gate_proj.weight", "bitwise copy (K -> gate_proj, GQA: KV*HD == D)"
    if ".self_attn.o_proj.weight" in name:
        return "DERIVE", "mixer.out_proj.weight", "slice [:, :D]  (H*HD -> D)"
    if (".self_attn.q_proj.weight" in name or ".q_norm.weight" in name
            or ".k_norm.weight" in name):
        return "NEW", "(no counterpart)", "random init; attention-only parameter"
    return "UNCLASSIFIED", "", "MUST NOT happen"


def build_provenance(tensors):
    rows = []
    for name, shape in tensors.items():
        verdict, dst, note = classify(name, shape)
        rows.append(dict(tensor=name, shape=shape, n_params=prod(shape),
                         verdict=verdict, student=dst, note=note))
    return rows


# ---------------------------------------------------------------- 可选：真实分片
def try_real_shard():
    """Attempt to fetch qa_00 (99,614,720 B) via the git-blob API.

    qa_00 is the FIRST byte slice of model-00001-of-00002.safetensors, so its
    first 8 bytes are the safetensors header length.  We only keep the header,
    never the weights.  Returns a dict; failure is reported, never fabricated.
    """
    out = dict(attempted=True, ok=False)
    t0 = time.time()
    try:
        tree = api("/repos/kino-koki/Qwen3-1.7B-Instruct/git/trees/main?recursive=1")
        ent = next(e for e in tree["tree"] if e["path"] == "qa_00")
        sha, size = ent["sha"], ent["size"]
        blob = api(f"/repos/kino-koki/Qwen3-1.7B-Instruct/git/blobs/{sha}", timeout=240)
        enc, content = blob.get("encoding"), blob.get("content", "")
        out.update(blob_size=size, encoding=enc)
        if enc != "base64" or not content:
            out["reason"] = "blob API returned no content (large-file policy)"
            return out
        raw = base64.b64decode(content)
        out["bytes_received"] = len(raw)
        hlen = struct.unpack("<Q", raw[:8])[0]
        hdr = json.loads(raw[8:8 + hlen])
        keys = [k for k in hdr if k != "__metadata__"]
        out.update(ok=True, header_bytes=hlen, n_tensors=len(keys),
                   sample=keys[:5], metadata=hdr.get("__metadata__", {}),
                   elapsed_s=round(time.time() - t0, 1))
        # 抽样：记录第一个张量的 shape/dtype/offset（元数据，不读权重数值）
        k0 = keys[0]
        out["first_tensor"] = dict(name=k0, **hdr[k0])
        json.dump(hdr, open(f"{OUT}/qwen17_shard1_header.json", "w"), indent=1)
    except Exception as e:
        out["reason"] = f"{type(e).__name__}: {str(e)[:160]}"
        out["elapsed_s"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- 主
def main():
    print("== fetching real upstream data ==", flush=True)
    u = fetch_upstream()
    cfg17 = u["qwen17_config"]
    idx17 = u["qwen17_index"]

    # --- 0.6B 规划（基于真实 TS2SC 脚本里的配置） ---
    t06 = qwen06_tensors()
    p06 = build_provenance(t06)
    by06 = Counter(r["verdict"] for r in p06)
    par06 = Counter()
    for r in p06:
        par06[r["verdict"]] += r["n_params"]
    total06 = sum(r["n_params"] for r in p06)
    # tied head 不额外占参数
    tied_head = prod(t06["lm_head.weight"])
    eff06 = total06 - tied_head

    # --- 1.7B 真实 index 交叉验证 ---
    names17 = list(idx17["weight_map"].keys())
    shape17 = {}
    # index 不含 shape；从名字推导（与 config 一致），仅用于分类覆盖检查
    D = cfg17["hidden_size"]; FF = cfg17["intermediate_size"]
    H = cfg17["num_attention_heads"]; KV = cfg17["num_key_value_heads"]
    HD = cfg17["head_dim"]; V = cfg17["vocab_size"]
    for n in names17:
        if n in ("model.embed_tokens.weight", "lm_head.weight"):
            shape17[n] = [V, D]
        elif n == "model.norm.weight":
            shape17[n] = [D]
        elif "layernorm" in n or "q_norm" in n or "k_norm" in n:
            shape17[n] = [HD] if ("q_norm" in n or "k_norm" in n) else [D]
        elif ".q_proj.weight" in n:
            shape17[n] = [H * HD, D]
        elif ".k_proj.weight" in n or ".v_proj.weight" in n:
            shape17[n] = [KV * HD, D]
        elif ".o_proj.weight" in n:
            shape17[n] = [D, H * HD]
        elif ".gate_proj.weight" in n or ".up_proj.weight" in n:
            shape17[n] = [FF, D]
        elif ".down_proj.weight" in n:
            shape17[n] = [D, FF]
        else:
            shape17[n] = []
    p17 = build_provenance(OrderedDict((n, shape17[n]) for n in names17))
    by17 = Counter(r["verdict"] for r in p17)
    par17 = Counter()
    for r in p17:
        par17[r["verdict"]] += r["n_params"]

    # --- 尝试真实分片 header ---
    print("== attempting real shard header (may be blocked) ==", flush=True)
    shard = try_real_shard()

    result = dict(
        mode="TS2SC Stage-0 provenance verification (NO weight values, NO training)",
        sources=dict(
            ts2sc_mapping="github:kino-koki/TS2SC/docs/01-架构映射.md",
            ts2sc_script="github:kino-koki/TS2SC/scripts/migrate_real.py",
            qwen17_config="github:kino-koki/Qwen3-1.7B-Instruct/config.json",
            qwen17_index="github:kino-koki/Qwen3-1.7B-Instruct/model.safetensors.index.json"),
        qwen06_config=Q06,
        qwen06_inventory=dict(
            n_tensors=len(t06), total_params_including_tied_head=total06,
            effective_teacher_params=eff06,
            by_verdict_count=dict(by06), by_verdict_params=dict(par06),
            coverage_all_classified=bool(by06.get("UNCLASSIFIED", 0) == 0),
            copy_plus_derive_params=par06["COPY"] + par06["DERIVE"],
            copy_plus_derive_fraction=round((par06["COPY"] + par06["DERIVE"]) / eff06, 6)),
        qwen17_real_index=dict(
            hidden_size=D, intermediate_size=FF, num_hidden_layers=cfg17["num_hidden_layers"],
            num_attention_heads=H, num_key_value_heads=KV, head_dim=HD, vocab_size=V,
            n_tensors=len(names17),
            metadata_total_size=idx17["metadata"]["total_size"],
            distinct_shards=sorted(set(idx17["weight_map"].values())),
            by_verdict_count=dict(by17), by_verdict_params=dict(par17),
            coverage_all_classified=bool(by17.get("UNCLASSIFIED", 0) == 0)),
        real_shard_header=shard,
        not_done=[
            "teacher weight VALUES not loaded; no torch / no PyPI / no HF (HTTP 403)",
            "no bitwise tensor copy performed",
            "no activation cache, no mixer distillation, no logit-KL training",
            "no perplexity / generation / capability measurement",
            "Qwen3-0.6B weights are NOT present in any accessible repository",
        ],
    )
    json.dump(result, open(f"{OUT}/ts2sc_verify.json", "w"),
              indent=1, ensure_ascii=False)
    json.dump(dict(q06=p06, q17=p17),
              open(f"{OUT}/ts2sc_provenance.json", "w"),
              indent=1, ensure_ascii=False)

    print("\n== Qwen3-0.6B Stage-0 plan ==", flush=True)
    print("  tensors:", len(t06), "| effective teacher params:", eff06)
    print("  verdict counts:", dict(by06))
    print("  params:", dict(par06))
    print("  COPY+DERIVE fraction:",
          round((par06["COPY"] + par06["DERIVE"]) / eff06, 6))
    print("\n== Qwen3-1.7B REAL index (311 tensors) ==", flush=True)
    print("  verdict counts:", dict(by17))
    print("  coverage all classified:", bool(by17.get("UNCLASSIFIED", 0) == 0))
    print("\n== real shard header attempt ==", flush=True)
    print(" ", shard)
    print("\nsaved", OUT, flush=True)


if __name__ == "__main__":
    main()
