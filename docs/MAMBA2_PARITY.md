# SimplCore independent development and evaluation

## Status

SimplCore is the only candidate architecture in this project. The goal is
**not to turn SimplCore into Mamba2 or another architecture**, and not to claim
success merely by reaching parity. The goal is to develop SimplCore itself
through independently designed and implemented mechanisms, then test whether it
can surpass strong external references under reproducible matched budgets. No
such superiority has been established.

Mamba2 and Transformer in this document are evaluation-only references, not
candidate architectures, runtime dependencies, or design templates. Their
implementations must not be copied, vendored, or adapted into SimplCore.
Likewise, a newly developed SimplCore candidate must not import or transplant
the current SimplCore mixer or model implementation as its architectural core.
Existing implementations may remain intact as historical baselines while an
independently designed SimplCore implementation is developed.

“Surpass” is a measured Pareto claim, not a source-code or asymptotic claim:
on the same frozen real corpus/tokenizer, candidate and references must use
comparable parameter and token budgets across repeated seeds; quality must be
non-inferior within the declared uncertainty margin, and at least one predeclared
target metric (throughput, peak memory, or long-context quality) must improve
meaningfully without an unreported regression in the other dimensions.
Architectural properties (for example O(1) streaming state) do not substitute
for model-quality or hardware measurements.

The comparison harness in `simplcore.core.eval.equal_budget` is evaluation
infrastructure only; it does not add another architecture to the SimplCore
runtime or registry. The archived reference checks that informed validation
include fresh-seed training, parameter perturbation, full-versus-stream
equivalence, bounded-state checks, generation, and checkpoint checks. They are
test ideas, not candidate architecture source to transplant.

Specify each candidate's mechanism and falsifiable hypotheses before broader
integration; the prototype currently records them in its module documentation.
Build it without imports from existing architecture modules and add a minimal
independent test suite before connecting it to shared training infrastructure.

The first isolated prototype is `simplcore.core.arch.flux_ledger`. Its testable
hypotheses are: future tokens cannot affect earlier logits; full-sequence and
chunked execution agree exactly; recurrent memory remains fixed-size; and the
adaptive baseline/surprise readout can be trained with finite gradients. These
are correctness hypotheses only. Learning quality, throughput, memory
efficiency, and any comparative advantage remain unmeasured.

## Reproducible reference comparison

Install the optional Hugging Face reference dependency first:

```powershell
python -m pip install -e ".[mamba2]"
```

```powershell
simplcore-equal-budget --preset smoke --baseline mamba2 `
  --tokens 8000 --seq-len 64 --seed 1234 `
  --out runs/equal_budget_mamba2_smoke.json
```

This synthetic smoke run is for integration checks only. It must not be used as
evidence of parity. For a real comparison, build one frozen corpus with disjoint
`train` and `val` splits, then run:

```powershell
simplcore-equal-budget --preset small --baseline mamba2 `
  --mode equal_token --tokens 2000000 --corpus data/processed/tuan_zh `
  --seeds 1234 2345 3456 --out runs/equal_budget_mamba2.json
```

The JSON report records split
fingerprints, model configuration, parameter counts, token budgets, paired
validation results, software versions, and the execution device. The baseline
uses the Hugging Face reference implementation; without `mamba_ssm` and
`causal_conv1d`, it may fall back to slower reference kernels. Do not compare
that timing to optimized Mamba2 deployment timing.

Mamba2 comparisons currently support equal-token mode only. Its FLOPs are not
reported as a hardware-comparable equal-compute budget; measured throughput and
hardware counters are required for that claim. Parameter matching uses
non-embedding weights and refuses a configuration outside the 5% tolerance.
Loss uses the same already-shifted targets as SimplCore, avoiding the extra
internal label shift in the Hugging Face causal-LM loss helper.

Single-seed runs are debugging/evaluation records only. The aggregate command
bootstraps paired mean-loss differences across seeds and only enables the
quality gate when its upper 95% interval is within the configured PPL
non-inferiority margin. It still refuses an unqualified quality claim on
synthetic data.

Training and held-out validation are separate: corpus runs use `train` and `val`;
synthetic runs use independently seeded data. A synthetic report always has
`honesty.claim_eligible_real_corpus=false`, and cannot authorize an unqualified
quality claim. At least 16 held-out windows are required for the quality gate.
Every frozen corpus split must declare a SHA-256 for each shard; matching train
and validation content fingerprints are rejected even if the split names differ.

## Evidence required to claim that SimplCore surpasses a reference

Do not claim superiority until the following are measured and documented on
the same frozen real corpus/tokenizer, with comparable parameter budgets and
repeated seeds:

1. **Quality:** validation and downstream task metrics under equal-token and
   equal-compute budgets, including paired uncertainty intervals and failure
   cases.
2. **Throughput:** training and decode tokens/second on the same target hardware,
   with warmup and kernel paths identified.
3. **Memory:** peak training memory, optimizer/activation memory, and streaming
   state at increasing context lengths.
4. **Long context:** quality at trained and extrapolated lengths, numerical
   stability, and full-sequence versus streaming equivalence.
5. **Engineering:** deterministic data/tokenizer provenance, checkpoint resume,
   mixed-precision behavior, kernel conformance, and deployment reproducibility.

The initial CPU reference integration only establishes that a fair comparison
can be run. It does not establish optimized-kernel performance, large-scale
training quality, or superiority.
