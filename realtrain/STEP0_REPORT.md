# Step 0 — real-corpus training exposes the static-kernel deficit

This is **not** the controlled synthetic binding task used elsewhere in this
project. It is a character-level language model trained with Adam on real
human text, with a chronological (never shuffled) train/validation split.

## Protocol

| item | value |
|---|---|
| corpus | tiny Shakespeare, 1,115,394 bytes, sha256 `86c4e6aa9db7c042…` |
| vocabulary | 65 characters (the exact character set of the corpus) |
| split | first 90% train / last 10% validation, **chronological** |
| objective | next-character cross-entropy, reported as bits/char (bpc) |
| optimizer | Adam, lr 3e-3, β=(0.9, 0.999), 3000 steps |
| batch / context | B=8, L=48, 5 independent seeds |
| implementation | pure NumPy fp64, hand-written analytic adjoints |
| gradient audit | finite-difference gradcheck on every parameter tensor before each run; worst relative error across all Step-0 runs `8.60e-08` |

Every run is preceded by a gradcheck. No analytic witness is injected, no
oracle answer is supplied to the model, and no number below is simulated.

## Result: the static LTI mixer is genuinely worse

| arm | params | val bpc (5 seeds) | long-range bpc | short-range bpc |
|---|---:|---:|---:|---:|
| `A_static` (fixed positive-real pole, O(d) state) | 5,249 | **4.0833 ± 0.0476** | 4.5522 | 3.7875 |
| `A_static_wide` (same architecture, d=56, ~2× params) | 10,481 | **4.0970 ± 0.0528** | 4.5611 | 3.8254 |
| `E_linear_attn` (outer-product state; **reference only**) | 7,297 | **3.7594 ± 0.0949** | 4.2906 | 3.2235 |

Paired difference (`E_linear_attn` − `A_static`): **−0.3239 bpc**,
95% CI **[−0.4280, −0.2197]**, 5/5 same sign.
Against the wide static arm: **−0.3375**, CI **[−0.4455, −0.2296]**.

Two things matter here:

1. **The deficit is real and is not a parameter-count artefact.** Doubling the
   state width makes the static mixer slightly *worse* (4.0833 → 4.0970), so
   the gap is not explained by capacity.
2. **It is concentrated in content-addressed positions.** The reference arm
   improves short-range bpc from 3.7875 to 3.2235 and long-range bpc from
   4.5522 to 4.2906, while the static kernel — whose impulse response
   `K(lag)=ρ^lag` is fixed and carries no input-dependent information — cannot
   select a value by its key. This is the empirical counterpart of the input-side
   bit-budget statement: the required bits were never supplied.

## Honest scope

- This is a **small** character LM on 1.1 MB of text, not a ≥10M-token LM run.
  It is a mechanism-level bridging experiment, not a benchmark result.
- `E_linear_attn` is an independent NumPy reference implementation, not an
  official reproduction of any published model; the table must not be read as
  an architecture ranking.
- `A_static` preserves the architecture's invariants (Toeplitz deviation 0,
  FFT↔IIR agreement, zero future-perturbation leakage, O(d) state, no N×N
  score, no KV cache). Losing those invariants is precisely what the next
  steps will do, and each step will be measured on this same real corpus.

## What Step 1 changes

Step 1 adds an **input-dependent gate** to the mixer (the L2 evidence /
L3 disagreement idea already present in `simplcore/core/arch/gates.py`). This
deliberately gives up strict time-invariance — it is the P2 escape route — and
asks whether supplying bits on the input side closes the measured gap. Step 1
is a code change to `simplcore/core/arch/evolution_mixer.py` and is pushed as
a separate commit.
