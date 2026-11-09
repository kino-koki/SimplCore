# Step 1 — supplying input-side bits closes most of the measured deficit

Architecture change pushed in this commit: `GatedEvolutionMixer` added to
`simplcore/core/arch/evolution_mixer.py` (`ARCH_STEP = 0 -> 1`), and the
training loop now passes token ids into the mixer's forward map.

## What changed, and what it costs

| | Step 0 | Step 1 |
|---|---|---|
| forward map | time-invariant (Toeplitz / FFT-IIR dual) | **input-dependent gate over long/local paths** |
| state | O(d) | O(d) |
| N×N score / KV cache | none / none | none / none |
| input-side bits | 0 | `c_t = (1/t)Σ_{s≤t} tok_s/128` |

The gate is `g_t = sigmoid(a·log1p(c_t) + b·log1p(t) + c0)` with three learnable
scalars; `c_t` depends only on token ids, never on learnable parameters, so its
stop-gradient status is structural. This **deliberately gives up strict time
invariance** — it is the P2 escape route, not a free lunch. The long path alone
still satisfies the Toeplitz/FFT-IIR checks.

## Real-corpus result (same corpus, split, optimizer, seeds, 3000 steps)

| arm | params | val bpc | long-range | short-range |
|---|---:|---:|---:|---:|
| `A_static` (Step 0) | 5,249 | 4.0833 ± 0.0476 | 4.5522 | 3.7875 |
| `B_gated` (Step 1) | 6,211 | **3.4591 ± 0.0469** | **3.9575** | **3.0393** |
| `E_linear_attn` (reference) | 7,297 | 3.7594 ± 0.0949 | 4.2906 | 3.2235 |

Paired difference (`B_gated` − `A_static`): **−0.6242 bpc**,
95% CI **[−0.6353, −0.6131]**, 5/5 same sign, sd 0.0127.
Gradcheck worst relative error across all Step-1 runs: **1.3e−07**.

This is the bridging result the theory predicts: with zero input-side bits the
fixed kernel cannot select a value by its key; supplying a deterministic
input-side statistic removes most of the gap without adding an N×N score.

## What is still wrong (the reason for Step 2)

- `B_gated` is **not** content addressing. The gate sees a running average of
  token ids, not a key; it cannot bind a specific earlier value to a specific
  query. Long-range bpc (3.9575) remains far above short-range bpc (3.0393).
- The gate's information is a hand-designed statistic. It is not learned from
  experience and does not accumulate across the training run.
- Strict time invariance is lost. Any claim that the architecture is "still
  Toeplitz" must now be scoped to the long path only, and is scoped that way
  in `invariants()`.

`E_linear_attn` remains an independent NumPy reference implementation; the
table is not an architecture ranking. All numbers come from real Adam runs on
real text, with chronological validation and no injected oracle.

## Step 2

Add the SMM×SGC fused state: a single `d×d` codebook state written by a
zero-order gate over **actual training experiences**, read out through the same
contraction as the mixer. This is the self-evolution main line. It will be
trained — not constructed analytically — on this same corpus, and pushed as a
separate commit.
