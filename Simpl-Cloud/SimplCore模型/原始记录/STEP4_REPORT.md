# STEP 4 — expanded codebook fused state: a measured negative result

This round continues the architecture repair on **real text**, not on a
controlled synthetic task. It does not add a theorem, does not change the
v15 manuscript, and does not claim any improvement in language modelling.

## Bottom line

Removing the 65→16 bucket collision and re-activating the fused-state write
gate **does not produce a statistically significant gain** over the gated
mixer. The paired difference is
`+0.004849` bpc with 95% CI `[-0.004036, +0.013733]` (5 paired seeds),
bootstrap CI `[-0.002446, 0.013519]`. All four paired comparisons contain
zero. The architecture's real-training deficit is therefore **not** a
tuning artefact, **not** the bucket collision, and **not** write starvation.

| arm | mean val bpc (5 seeds) | paired diff vs `B_gated` | 95% CI | significant |
|---|---:|---:|---|---|
| `B_gated` (STEP 1) | 3.459061 | — | — | — |
| `D_calibrated` (STEP 3) | 3.458293 | -0.000768 | [-0.005814, +0.004277] | no |
| `F_expanded` (STEP 4) | 3.463910 | +0.004849 | [-0.004036, +0.013733] | no |
| `F_expanded_strict` | 3.462337 | +0.003276 | [-0.004994, +0.011546] | no |

All arms: Adam, 3000 steps, B=8, L=48, lr=0.003,
tiny_shakespeare_char_level (1,115,394 bytes), chronological 90/10, no shuffle.
Worst gradcheck relative error across all runs: `1.79e-07`.

## What was actually changed in the architecture

Two measured STEP-3 defects were attacked **together**, with the
zero-order consolidation rule otherwise inherited unchanged:

1. **Codebook collision.** STEP 2/3 folded `tok` into `tok % (d/2) = 16`
   buckets. STEP 4 gives each of the 65 real characters its own key row and
   value row from a deterministic approximately-orthogonal `(2V, d)` codebook.
   `bucket_folding = none on the real alphabet (tok < V)`;
   measured off-diagonal Gram magnitude `0.610116`
   (reported, not assumed away).
2. **Write starvation.** STEP 3 reduced commits from 412,023 to 6. STEP 4's
   pre-declared middle gate yields `18898–21156` commits
   across 5 seeds (strict-gate isolation arm: `1551–1622`).

**Conserved invariants** (all 5 seeds, both STEP-4 arms): fused core state
`8192` bytes, independent of L and of
binding count; no KV cache; no N×N score; strictly causal; every gradcheck
passed. The gate sweep is a mechanism diagnostic only — the formal
hyper-parameters were pre-declared and were **not** chosen from its results.

## The decisive mechanism: a rank/capacity lower bound, not a bug

A perfect-write probe constructs `S` by writing **every** key's majority
value from the real corpus co-occurrence counts at once, then measures top-1
read-out recovery. This is an explicit upper bound, not a training result:

| probe seed | bindings written | top-1 recovery | effective rank | min top-2 margin |
|---:|---:|---:|---:|---:|
| 0 | 65 | 0.769231 | 13.9904 | 0.005947 |
| 1 | 65 | 0.784615 | 13.7594 | 0.006956 |
| 2 | 65 | 0.800000 | 14.4445 | 0.001795 |

`d = 32`, `V = 65`, so `rank(S) <= 32 < 65`.
Even with perfect writes only `0.769–0.800` of
bindings are recoverable; effective rank is `13.76–14.44`,
and the smallest top-2 logit margin is `0.001795`. Exact
retrieval of all bindings is impossible in this state size regardless of the
write gate. This is the same input-side information-budget obstruction as the
paper's counting bound, now observed on real text rather than asserted.

### A counter-intuitive cross-check

The **strict-gate** arm writes only `1551–1622` times into
16 slots and reaches read-out top-1 recovery `0.9708–0.9917`,
while the **relaxed** arm writes `18898–21156` times into
~63 slots and reaches only `0.6775–0.7528`. Effective rank
rises from `5.52–8.42` to `11.62–14.22`,
i.e. **more writes buy more interference, not more accuracy**. Yet both arms
tied the gated mixer. So neither write frequency nor read-out precision is
the binding constraint — the state's information capacity is.

Gate sweep (seed 0, 1000 steps): writes span `128–191154`, read-out top-1 spans `0.711–0.913`. The two STEP-3 extremes
(412,023 and 6 commits) are endpoints of one continuum, not two unrelated
failure modes.

## What this means for the paper

**`simplcore.pdf` v15 is unchanged.** Its 56/56 verifier still passes, the
four legacy arms reproduce byte-identically on fresh seed 7, and no figure,
table, theorem, or conclusion was edited. The reason for not folding this in:
adding another negative ablation would re-introduce the defensive density the
review already flagged, and would not move significance. A short appendix
candidate is supplied separately as optional material, not inserted.

The result *does* strengthen the mechanistic story: the fused state's deficit
now has a quantitative cause (rank ≤ 32 < 65, measured perfect-write recovery
`0.769–0.800`) rather than a list of engineering
suspects. To make SMM×SGC fusion win, the information budget itself must grow
— larger `d`, a different state representation, or a learned input-side code —
which is a change of contribution type, not another gate sweep.

## Honest boundaries

- Negative result, not a success. No architecture claim is upgraded.
- Character-level tiny Shakespeare, 1.1 MB, 3000 Adam steps, CPU-only NumPy;
  no LM-scale, no throughput, no MQAR parity, no human review.
- The capacity probe is a synthetic perfect-write upper bound computed from
  real co-occurrence statistics; it is not a training accuracy.
- Approximately-orthogonal codebook is fixed (no gradient), with measured
  off-diagonal Gram `0.610116`; exact
  orthogonality is impossible for 130 rows in 32 dimensions and is not claimed.
- No acceptance probability, SOTA, or LM-parity claim.
