# Step 2 — the fused self-evolving state does NOT help on real text

Architecture change pushed in this commit: `FusedEvolutionMixer` added to
`simplcore/core/arch/evolution_mixer.py` (`ARCH_STEP = 1 -> 2`), and the
training loop now calls `observe()` once per real Adam step with the actual
training batch. `ARCH_STEP = 2` is the SMM×SGC main line: memory and
computation share ONE `d×d` algebraic object, written by a zero-order gate.

## What actually happened

| arm | params | val bpc | long-range | short-range |
|---|---:|---:|---:|---:|
| `B_gated` (Step 1) | 6,211 | 3.4591 ± 0.0469 | 3.9575 | 3.0393 |
| `C_fused` (Step 2) | 7,235 | **3.4769 ± 0.0306** | 3.9682 | 3.0633 |

Paired difference (`C_fused` − `B_gated`): **+0.0179 bpc** (worse),
95% CI **[+0.0018, +0.0339]**, but **only 3/5 seeds worse** — so the honest
reading is *no reliable gain, with a small average degradation*, not a clean
defeat. Gradcheck worst relative error: **5.9e−06**.

This is a **real negative result on a real corpus**, and it is the point of
running real training instead of a controlled synthetic task. The SE1–SE5
algebra is exact on orthogonal codebooks; on real text it is defeated by three
things the synthetic protocol could not reveal:

1. **Write explosion.** `commits = 412,023` over 3000 steps, `rejects = 28`.
   The evidence threshold `tau = 6` is reached by almost every (key, value)
   pair within a few batches, so the gate accepts essentially everything.
   The zero-order channel is open, but it is not selective.
2. **Time-scale mismatch.** The fused state inherits the mixer's sequence
   decay `rho = 0.9` and applies it **once per training step**. Over 3000
   steps, `0.9^age` drives every slot weight below the read-out precision,
   which in turn makes `_read` demand a re-write forever. The recurrence's
   time unit is a token; self-evolution's time unit is a training step.
3. **Alphabet collision.** The corpus has 65 characters but `VK = d/2 = 16`
   key/value buckets. Distinct symbols share a code row, so a "correct"
   registration for one symbol overwrites another. Orthogonality prevents
   cross-talk between *buckets*, not collision *within* a bucket.

## What is still conserved

`core_state_bytes = 8192` (the `d×d` state), independent of sequence length
and of the number of registered bindings; Toeplitz deviation **0**;
FFT↔IIR relative error **4.17e−16**; zero future-perturbation leakage;
no N×N score; no KV cache; zero-order (no gradient through retrieval).
The fusion did not detune the static kernel — it simply failed to improve
the language objective.

## Step 3

`CalibratedFusedMixer` repairs exactly the three diagnosed defects, with no
new theoretical claim: a separate cross-step memory decay, a cooldown per key,
and a dominance-based evidence gate that registers only the majority value of
a key. It is trained on the same corpus and pushed as a separate commit. If
it still does not beat `B_gated`, that result will be reported unchanged.
