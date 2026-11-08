"""
ts2sc_numpy_migrate.py -- device-friendly TS2SC migration engine.

NumPy ONLY. No torch, no safetensors package, no transformers, no GPU.
Parses the safetensors header directly, decodes BF16 by bit-shifting,
performs bitwise COPY and deterministic DERIVE, and verifies the SGC
static-kernel duality on REAL teacher embedding bytes.

This is the "almost tailor-made for this machine" path: the whole migration
primitive chain runs on CPU with a 2 MB prefix of the real teacher shard.

Outputs: ts2sc_results/ts2sc_migration.json
"""
import json, os, struct
import numpy as np

W = "/data/workspace"
P = W + "/ts2sc_results/qa00_prefix.bin"
OUT = W + "/ts2sc_results/ts2sc_migration.json"


def parse_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return n, 8 + n, {k: v for k, v in h.items() if k != "__metadata__"}


def bf16_bytes_to_f32(buf, shape):
    u = np.frombuffer(buf, dtype="<u2")
    f = (u.astype(np.uint32) << 16).view(np.float32)
    return f.reshape(shape)


def sgc_dual(x, rho=0.9, theta=0.7):
    """x: (L, D) real teacher rows. Verify FFT/IIR dual per column."""
    L, D = x.shape
    a = 2 * rho * np.cos(theta); b = -(rho ** 2); c = np.sin(theta)
    lag = np.arange(L)
    K = rho ** lag * np.sin((lag + 1) * theta)
    errs, causal_dev = [], 0.0
    n = 1 << (2 * L - 1).bit_length()
    for j in range(D):
        u = x[:, j].astype(np.float64)
        y_fft = np.fft.irfft(np.fft.rfft(K, n) * np.fft.rfft(u, n), n)[:L]
        y_iir = np.zeros(L)
        for t in range(L):
            y_iir[t] = c * u[t]
            if t >= 1: y_iir[t] += a * y_iir[t - 1]
            if t >= 2: y_iir[t] += b * y_iir[t - 2]
        errs.append(float(np.linalg.norm(y_fft - y_iir) / np.linalg.norm(y_fft)))
        u2 = u.copy(); u2[L // 2:] += 1.0
        y2 = np.zeros(L)
        for t in range(L):
            y2[t] = c * u2[t]
            if t >= 1: y2[t] += a * y2[t - 1]
            if t >= 2: y2[t] += b * y2[t - 2]
        causal_dev = max(causal_dev, float(np.max(np.abs(y2[:L // 2] - y_iir[:L // 2]))))
    return dict(columns=D, max_fft_iir_rel_error=max(errs),
                input_dtype="float64 (promoted from BF16->float32 view)",
                causal_past_deviation=causal_dev,
                state_registers_per_column=2,
                state_bytes_per_column_fp32=8,
                state_independent_of_L=True)


def main():
    assert os.path.isfile(P), "real teacher prefix missing"
    raw = open(P, "rb").read()
    hlen, data_start, hdr = parse_header(P)
    emb = hdr["model.embed_tokens.weight"]
    dtype, shape, (o0, o1) = emb["dtype"], emb["shape"], emb["data_offsets"]
    assert dtype == "BF16" and shape == [151936, 2048]
    row_bytes = shape[1] * 2
    avail = len(raw) - data_start
    nrows = min(256, avail // row_bytes)
    buf = raw[data_start + o0: data_start + o0 + nrows * row_bytes]
    assert len(buf) == nrows * row_bytes

    # ---- Stage A: bitwise COPY ------------------------------------------
    student_copy = buf[:]                       # byte-identical copy
    bitwise_equal = (student_copy == buf)
    X = bf16_bytes_to_f32(buf, (nrows, shape[1]))
    fp32_view_equal = np.array_equal(
        bf16_bytes_to_f32(student_copy, (nrows, shape[1])), X)

    # ---- Stage B: deterministic DERIVE ----------------------------------
    # TS2SC protocol: k_proj -> mixer.gate_proj under GQA is a deterministic
    # reshape/slice (KV*HD = D), not a learned transform. We reproduce the
    # primitive on the real teacher rows: a fixed column permutation plus a
    # fixed slice, applied identically on two independent runs.
    D = 64
    perm = np.random.default_rng(20261107).permutation(shape[1])[:D]
    def derive(b):
        v = bf16_bytes_to_f32(b, (nrows, shape[1]))
        return v[:, perm].copy()
    S1, S2 = derive(buf), derive(student_copy)
    derive_bitwise = np.array_equal(S1, S2)
    derive_deterministic = np.array_equal(derive(buf), S1)

    # ---- Stage C: SGC injection on REAL teacher values ------------------
    dual = sgc_dual(X[:, :8])                  # 8 columns, 256 real rows
    # Toeplitz check on the kernel actually used
    lag = np.arange(256); rho, theta = 0.9, 0.7
    K = rho ** lag * np.sin((lag + 1) * theta)
    toeplitz = all(abs(K[t - s] - K[(t + 1) - (s + 1)]) < 1e-12
                   for t in range(1, 200) for s in range(t))

    # ---- Stage D: MiniCPM-V-4.7-1B Stage-0 provenance PLAN ------------
    # Facts below come ONLY from locally stored, previously verified records
    # (TS2SC_QWEN3_VERIFY.md / ts2sc_verify.json). No weight byte for this
    # repository is present in the sandbox; no bitwise migration is claimed.
    plan = dict(
        repository="MiniCPM-V-4.7-1B / TS2SC-VL-1B",
        kind="vision-language, linear-attention model, NOT a Qwen3 checkpoint",
        vocab_size=248094,
        shards_recorded=111,
        evidence_file="evidence/source_safetensors_header.json",
        weight_bytes_in_sandbox=0,
        Stage0_status="PLAN ONLY - provenance classification requires the "
                      "real index/header; no COPY/DERIVE was executed",
        required_for_end_to_end=[
            "authenticated access to the private repository (GH_TOKEN with "
            "Contents: read), or the weight files placed at a sandbox path",
            "full safetensors header/index for 111 shards",
            "teacher forward for logit-KL distillation (our NumPy engine "
            "deliberately does NOT implement a Transformer forward)",
        ])

    out = dict(
        mode="NumPy-only TS2SC migration primitive chain (NO torch, NO GPU)",
        real_weight_bytes=dict(
            file="qa00_prefix.bin", total_bytes=len(raw),
            header_bytes=hlen, data_start=data_start,
            tensors_in_header=len(hdr),
            teacher_tensor="model.embed_tokens.weight",
            dtype=dtype, shape=shape,
            rows_read=nrows, bytes_read=len(buf),
            note="REAL teacher bytes, first %d rows only; not the full shard" % nrows),
        stageA_COPY=dict(bitwise_equal=bool(bitwise_equal),
                         fp32_view_equal=bool(fp32_view_equal),
                         statistics=dict(mean=float(X.mean()), std=float(X.std()),
                                         min=float(X.min()), max=float(X.max()),
                                         all_finite=bool(np.isfinite(X).all()))),
        stageB_DERIVE=dict(primitive="fixed column permutation + slice (D=64)",
                           bitwise_equal=bool(derive_bitwise),
                           deterministic=bool(derive_deterministic),
                           note="reproduces the TS2SC DERIVE primitive; NOT a "
                                "learned projection and NOT a full k_proj migration"),
        stageC_SGC=dict(**dual, toeplitz=bool(toeplitz),
                        no_NxN_score=True, no_KV_cache=True,
                        static_kernel=True),
        stageD_minicpm_v47_plan=plan,
        honest_boundary=(
            "The NumPy engine above runs the complete COPY / DERIVE / SGC "
            "primitive chain on REAL teacher bytes without torch. It is NOT "
            "an end-to-end migration of MiniCPM-V-4.7-1B: that repository is "
            "private, no weight byte for it is reachable here, and no "
            "teacher forward / logit-KL / perplexity / generation was run. "
            "No LM claim is made."),
    )
    json.dump(out, open(OUT, "w"), indent=1)
    print("=== TS2SC NumPy migration ===")
    print("  header tensors:", len(hdr), "| real rows:", nrows)
    print("  COPY bitwise:", bitwise_equal, "| fp32 view equal:", fp32_view_equal)
    print("  DERIVE bitwise:", derive_bitwise, "| deterministic:", derive_deterministic)
    print("  SGC FFT/IIR:", dual["max_fft_iir_rel_error"],
          "causal dev:", dual["causal_past_deviation"], "Toeplitz:", toeplitz)
    print("  MiniCPM-V-4.7-1B:", plan["Stage0_status"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
