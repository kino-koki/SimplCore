# Step 3 — calibration removes the write explosion, but yields no net gain

Architecture change pushed in this commit: `CalibratedFusedMixer` added to
`simplcore/core/arch/evolution_mixer.py` (`ARCH_STEP = 2 -> 3`). Three
corrections, each tied to a Step-2 measurement on real text:

| defect measured in Step 2 | correction |
|---|---|
| 412,023 writes from `tau=6` accepting everything | dominance gate: register only if `(k,v)` is ≥35% of that key's co-occurrence mass |
| `rho=0.9` (per token) applied once per **training step** | separate `rho_mem = 0.9995` (per step) |
| read-out decays below `eps`, forcing endless re-writes | per-key cooldown of 40 steps |

## Real-corpus result (identical protocol, 5 seeds, 3000 Adam steps)

| arm | params | val bpc | long-range | short-range | writes |
|---|---:|---:|---:|---:|---:|
| `A_static` | 5,249 | 4.0833 ± 0.0476 | 4.5522 | 3.7875 | — |
| `B_gated` | 6,211 | 3.4591 ± 0.0469 | 3.9575 | 3.0393 | — |
| `C_fused` | 7,235 | 3.4769 ± 0.0306 | 3.9682 | 3.0633 | 412,023 |
| **`D_calibrated`** | 7,235 | **3.4583 ± 0.0440** | **3.9543** | **3.0439** | **6** |

- `D_calibrated` − `B_gated`: **−0.0008 bpc**, 95% CI **[−0.0058, +0.0043]**,
  **contains zero** → no reliable difference.
- `D_calibrated` − `C_fused`: **−0.0186**, CI **[−0.0387, +0.0014]**,
  also contains zero.
- Gradcheck worst relative error: **1.3e−07**.
- Writes dropped 412,023 → **6**, rejects 5,792, slots 6. Core state stays
  **8192 bytes**, independent of L and of binding count.

## The honest reading

The calibration did what it was designed to do: it stopped the runaway write
loop and removed Step 2's small degradation. But the fused state is then
**almost never written** — only 6 registrations over 3000 steps — so it is
effectively an empty additive path, and the model reduces to `B_gated`.

The binding reason is the remaining limitation, which we do **not** silently
fix: 65 corpus symbols share 16 code buckets. On real text, most (key, value)
pairs are not dominant enough to pass a selective gate, and the ones that are
collide with other symbols in the same bucket. On the synthetic orthogonal
codebook, SE1–SE5 are exact; on a real alphabet, the fused self-evolving
state has **no measurable language-modelling benefit at this scale**.

This is the architectural weakness the real training was meant to expose. It
is reported unchanged, and appears in the paper as a measured boundary plus
an open question in the appendix. It is not a claim about language models at
scale, and no throughput or parity claim is made anywhere in this directory.

## Reproduction

```
python realtrain/train_real_lm.py --arms A_static,A_static_wide,B_gated,C_fused,D_calibrated,E_linear_attn \
    --seeds 0,1,2,3,4 --steps 3000
```

All raw curves, per-seed values, gradcheck outputs and invariant checks are in
`realtrain/results/step{0,1,2,3}_real.json`. The corpus provenance and fetch
command are in `realtrain/data/README.md`.
