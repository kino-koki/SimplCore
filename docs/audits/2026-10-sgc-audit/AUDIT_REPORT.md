# SGC experiment-suite audit — `G:\SimplCore-V 0.1-ALPHA\verify_sgc`

Interpreter: `G:\SimplCore-V 0.1-ALPHA\.venv311\Scripts\python.exe` (Python 3.11.15, torch 2.4.1+cpu, numpy 2.4.6).
All scratch artifacts under `.tmp_audit\`. I overwrote nothing under `verify_sgc\results\`, but the
mandated §11.5 experiment caused its own script to create three new JSON files there (see §3/§7).
Host caveat: sibling processes (`sgc_probe\dl_fineweb.py`, `scripts\probe_batch.py`) competed for CPU
during part of the audit, so the absolute wall times below are upper bounds.

---

## 1. Mixer implementation inventory

All six mixers share one LM skeleton, `SimpleLM` (`sgc_v3.py:362-377`): token embedding + learned
position embedding + N x (mixer, SwiGLU FFN), pre-norm inside each mixer, tied output head
(`build_model`, `sgc_v3.py:327-359`). Verified that `pos_emb` IS consumed (`sgc_v3.py:373`).

| key | class | file:line | params (measured) | causal | streaming state |
|---|---|---|---|---|---|
| `attn` | `AttentionMixer` | `sgc_pytorch.py:81-116` | `2D^2+2D+1` | yes | no (KV cache not implemented) |
| `sgc` | `SpectralGlobalConvV2` | `sgc_pytorch.py:24-78` | `2D^2+Lr+rD+2D+1` | yes | no |
| `sgcv3` | `ContentRoutedSGC` | `sgc_v3.py:29-69` | `3D^2+Lr+rD+4D+1` | yes | no |
| `ca` | `ContentAdaptiveMixer` | `sgc_v3.py:72-139` | `4D^2+2Dr+2D` | yes | **yes** (`forward_stream`) |
| `cagate` | `GatedAssocMixer` | `sgc_v3.py:142-205` | `5D^2+2D` | **NO — proven bug** | yes (`forward_stream`) |
| `hm` | `SGC_HM` | `sgc_v3.py:208-324` | `5D^2+3D+1` | yes | yes (`forward_stream`) |

Formulas confirmed numerically at `D=32, L=64, r=8` (all exact) and, for `attn/sgc/sgcv3/ca`, at the
report config `V=1084, D=176, L=4, H=4, seq=256, rank=8` — all four reproduce the JSON values
**exactly**: 2,221,476 / 1,987,492 / 2,236,712 / 2,142,624.

Math as actually implemented:

* **`attn`** — standard MHA: `qkv = W[3D,D]`, `scores = qk^T/sqrt(hd)`, upper-triangular mask, softmax,
  `attn@v`, `out_proj`, residual add scaled by learnable `out_scale` (init 1). Mask built once with
  `torch.triu(..., diagonal=1).bool()` (`sgc_pytorch.py:95-97`), applied via
  `masked_fill(..., -inf)` (`:111`). Causal — verified (perturbing `x[L-1]` changes only `t=L-1`).
* **`sgc`** — `u = W_in·LN(x)`, `Kern = low[:L,:] @ up` (rank-r causal kernel), FFT linear convolution
  zero-padded to `N=2L`, truncated to `[:L]`, `y = W_out·conv`, `out = x + out_scale·y`. Unit-impulse
  init `up[i,i]=1` for `i<min(r,D)`, `low[0,:]=1` (`sgc_pytorch.py:49-53`).
  **The FFT path is genuinely causal** — I checked it against an explicit `sum_{s<=t}K[t-s]u[s]`:
  max error 5.2e-6, and zeroing all future inputs moves `y[0]` by 1.4e-6. The apparent `(t-s)%L` wrap
  in `Kp` is harmless because `Xp` is zero-padded by `L`.
* **`sgcv3`** — SGC-v2 plus a content gate
  `g = sigmoid(W_out·tanh(W_k·LN_gate(x)))`, `blend = 1 + gate_scale·g`,
  `out = x + (sgc(x) - x)·blend` (`sgc_v3.py:59-69`). `gate_scale` init 0 (starts as pure SGC-v2),
  gate dropout 0.1. Causal.
* **`ca`** — causal linear attention with `phi = elu(.)+1`. Chunked: intra-chunk `tril(QK^T)·V` plus
  carried state `S = sum(k outer v)`, `z = sum k`; `out = (Q·S + tril(QK^T)V)/(Q·z + tril(QK^T)1 + eps)`
  (`sgc_v3.py:124-136`). Verified against an independent per-token einsum reference:
  `max|ref-new| = 4.8e-7` (`_verify_ca.py`). Causal.
* **`cagate`** — content-addressed slot memory, sequential per token: read
  `w = tanh(q·M^T/s)*0.5+0.5`, `out = wA/sum(w)`; write `alpha = softmax(tanh(k·M^T/s)*0.5+0.5)`,
  `M += beta(k-M)*alpha`, `A += beta(v-A)*alpha` (`sgc_v3.py:175-185`).
* **`hm`** — two-level memory. Long-term read `softmax(cos(q,M)*read_tau) @ A`; recent exact read over
  a rolling buffer plus intra-chunk causal softmax; long-term write is winner-take-all delta overwrite
  of the argmax-cosine slot with `M` re-normalised (`sgc_v3.py:285-320`).

Proven correctness bugs:

1. **`GatedAssocMixer` is not causal** (`sgc_v3.py:199-201`). The per-token loop writes the current
   token into the memory slots *before* that token reads, and the read is a softmax over all slots
   including the just-written one. Measured: perturbing `x[63]` changes the output at `t = 1..62` with
   `|dy|max` between 0.016 and 0.042. Every other mixer changes only `t = 63`. This breaks the
   autoregressive contract outright — the model can see the current token through the memory read.
2. **`GatedAssocMixer` is non-deterministic.** `forward_stream` re-initialises the key slots with
   `torch.randn(...) * 0.02` on every call (`sgc_v3.py:190`). Two identical calls on identical input
   differ by up to 9.7e-6 — RNG noise injected into every training step. `SGC_HM` avoids this with a
   seeded generator (`sgc_v3.py:261-263`); `GatedAssocMixer` does not.
3. **`ContentAdaptiveMixer` output depends on the internal chunk size** (measured `max|Δ| = 1.3`,
   `chunk=8` vs `16`; ~1.2-1.35 across all pairs). Because the numerator `Q·S` is unnormalised,
   changing where the state is split changes the result. The docstring (`sgc_v3.py:80-82`) presents
   chunking as an implementation detail; it is an architecture hyper-parameter. Training defaults hide
   this (`ca_chunk=128`), but `_train_lingap.py --chunk 64`, `stream_infer.py:67` (256) and
   `honest_max_ca512.json` (512) all differ, so those runs are not directly comparable.
4. Minor: `GatedSGC` zero-inits its gate (`sgc_gated.py:64-66`) then applies `sigmoid(0)=0.5`, so
   `sgcv3`/`GatedSGC` both start by halving the convolution output. Loss-scale effect only.
5. **`sgc`/`sgcv3` silently truncate their kernel above `max_len`.** `self.low[:L,:]`
   (`sgc_pytorch.py:60`, `sgc_v3.py:60`) zero-pads, so every offset `> max_len` contributes nothing and
   all position information beyond the construction length is identically zero. No error is raised.

---

## 2. Claim -> JSON map for `simplcore\docs\SGC_v2_诚实版报告.docx`

Read with python-docx (structural dump in `.tmp_audit\report_dump.txt`). The report has 11 tables; all
30 numeric JSON files under `verify_sgc\results\` were read.

### Backed claims (spot-checked cell by cell)

| Report claim | Supporting JSON | Result |
|---|---|---|
| 3.1 timings 0.287/0.377/0.476/0.926 vs 0.392/0.442/1.347/4.343 s | `efficiency_timing.json` | **exact** |
| 3.1 speedups +37/+17/+183/+369%, "2048 快约 4.7x" | same | arithmetic checks out (4.693x) |
| 3.1 FLOPs ratios 4.27/3.88/3.56/3.28x | same `flops` block | data matches; see note below |
| 3.2 val_acc 35.90% vs 36.27%, Δ −0.37pp | `v2_1200.json` | **exact** (0.35900 / 0.36283) |
| 3.2 loss 1.4259 vs 1.4269 | `v2_1200.json` | **exact** |
| 3.2 loss 1.617 vs 1.521 | `results_v2_causal.json` | **exact** |
| 3.2 time 119.9 s vs 178.6 s, "约快 33%" | `v2_1200.json` | means exact; 32.9% is a *reduction*, and H1 is 48.9% — the report's own text concedes the 49% is an SGC-based ratio, so it self-corrects |
| 3.2 params 103,309 vs 117,899, "省约 12%" | `v2_1200.json` | **exact** (12.4%) |
| 3.3 whole table (1.10/2.00, 4.66/3.66, 11.10/6.16, 0.61/1.27 s, 1,103,620/1,222,404) | `pytorch_compare_wd0.json` | **exact, all 6 cells** |
| 3.4 row 1 "小模型 5 步 test_ppl 28.30 / 28.52" | `gated_compare.json` | numbers exact; **not a credible protocol**, see below |
| 9.1 whole table (20 cells) | `honest_attn/sgc/sgcv3/ca.json`, `honest_compare_summary.json` | **exact, all 20 cells** |
| 9.2 complexity table (15 cells) + 48x/8.96x/9.08x | `complexity_scaling.json` | **exact, all 15 cells** |
| 10.1 max-model table (12 cells), 13.99M params, 5.3 s/step | `honest_max_ca512.json` | **exact** (13.0075/10.0170/7.0126/5.3050 -> 13.01/10.02/7.01/5.31; val 28.81/27.44/36.53/37.37; test 112.16/113.00/115.16/149.05) |
| 11.2 / 11.3 probe tables | **no JSON** — literals at `_append_hm_docx.py:72-75, 87-90` | but re-running `_probe_retention.py` (69.5 s) reproduces **everything**, including slopes −0.142 / −0.0000 |
| 11.4 "64.2 KB / 236-381 us/token" | **no JSON** — literal at `_append_hm_docx.py:100-101` | **UNVERIFIED and currently unmeasurable**, see §4c |
| 11.5 "(训练进行中)" | no JSON existed at report time | **now completed by me**, see §3 |

### Claims with NO supporting JSON at all

1. **§3.4 table row 2, §4.1, §5, §6: "独立测试集 test_acc 50 步: SGC 2.67% / Gated 1.50% /
   Attention 12.89%"** and the "训练集 acc=100% 是背答案假象" remark. The cited source
   `simplcore/docs/SGC_v2_门控消融_诚实报告.md` **does not exist anywhere in the repository** (I
   searched the whole tree). The strings `2.67%`, `12.89%`, `1.50%` occur in **no** file in the repo.
   This is the report's single most decision-relevant piece of negative evidence — the basis for
   "SGC 崩坏至随机水平" — and it is wholly unsourced. **UNVERIFIED.**
2. **§2.1 "极小步数 200 步 test_acc 97.72% vs 9.38%"** — the string `97.72` appears in no file. The
   report's handling (剔除) is correct.
3. **§11.2 / §11.3** probe panels — no JSON. Genuine on re-run, but the only copy of the numbers is
   inside the docx-generator script, and the report calls them "可复现".
4. **§11.4** memory constants — no JSON, and no functioning measurement.
5. **§6 row "门控不能修复 | gated_compare.json + 诚实报告"** — half of that citation is the missing file.

### JSON whose numbers do not match / do not support the report

* **`gated_compare.json` does not support §3.4 row 1 as a protocol.** Its own `args` are
  `dim=32, layers=1, heads=2, seq_len=64, rank=4, batch=4, steps=5, wd=0.0` — one 32-wide model
  trained for **five steps** with no weight decay. Its `train_losses` are
  `[17.79, 17.25, 29.77, 22.49, 28.56]`, i.e. noise around `ln(vocab)`. The report prints its
  `test_ppl` values next to real 800-step runs without disclosing `steps=5`. Worse, the file was
  produced with `train_gated.py --mixers sgc,gated_sgc`, so the **`attn` and `sgc_nogate` control arms
  the script exists to compare are absent** — the ablation was never run to completion.
* **§4.1 treats the two selective-copy rounds as comparable "fresh batch 验证".** Both use a *single*
  fixed batch of 200 sequences, drawn once per run (`train.py:84`,
  `val_idx, val_tgt = make_batch(200)`) and reused for every evaluation, from the same RNG and the same
  distribution as training data. There is no held-out split. The rounds also differ in steps
  (1200 vs `STEPS=400`) and in task length (`max_len=64` in `model_v2.py:36`) vs the `L=48` used by
  `train.py:25`). "两轮独立测量" overstates the independence.
* **§9.2's "SGC-CA 反超注意力" crossover is not reproducible end-to-end.** `complexity_scaling.json` is
  forward-only, single-mixer, batch 4. In the actual training protocol CA is *slower*:
  `honest_ca.json` step_ms = 991 vs `honest_attn.json` 499. My own matched measurement at
  D=176/L=4/seq=256 gives attn 1199, sgc 882, sgcv3 1210, ca 1205 ms per fwd+bwd step. The crossover
  number (131.5 ms vs 220.0 ms at L=2048) is real but is not a training-comparable measurement.
  **Flagged.**
* `pytorch_compare.json` (wd=0.1), the sibling of the §3.3 source, is never shown: SGC test_ppl 10.26 vs
  attention 6.00. Same conclusion, so this is a completeness note, not an error.

---

## 3. The unfinished experiment (§11.5) — RUN TO COMPLETION

**Script:** `verify_sgc\_train_lingap.py`. (`_train_mem_bench.py` is a different, also-incomplete
benchmark; §11.5 is `_train_lingap.py`.)

**Exact commands** (must run with cwd = `verify_sgc`; the script inserts its own dir on `sys.path`):

```
cd "G:\SimplCore-V 0.1-ALPHA\verify_sgc"
& "G:\SimplCore-V 0.1-ALPHA\.venv311\Scripts\python.exe" -u _train_lingap.py --mixer hm   --G 512
& "G:\SimplCore-V 0.1-ALPHA\.venv311\Scripts\python.exe" -u _train_lingap.py --mixer ca   --G 512
& "G:\SimplCore-V 0.1-ALPHA\.venv311\Scripts\python.exe" -u _train_lingap.py --mixer attn --G 512
```

Keep every default for comparability: `--steps 700 --dim 96 --layers 2 --bs 32 --feat 64 --seed 7
--chunk 64`. The output path is hard-coded to `results/lingap_{mixer}_G{G}.json`
(`_train_lingap.py:85-86`), one file per mixer. **Use `-u`**: without it Python block-buffers stdout
and you see nothing for the whole 15-minute run.

**Measured runtime on this CPU (700 steps, G=512, seq_len=515):**

| mixer | wall | params | loss (last) | recall_last | best | chance |
|---|---|---|---|---|---|---|
| `hm` | **905.0 s** (15.1 min) | 440,642 | 2.787 | 0.0000 | 0.0000 | 0.0625 |
| `ca` | **535.2 s** (8.9 min) | 428,352 | 2.782 | 0.03125 | 0.03125 | 0.0625 |
| `attn` | **862.3 s** (14.4 min) | 422,210 | 2.780 | 0.03125 | 0.06250 | 0.0625 |
| **total** | **2302.6 s = 38.4 min** | | | | | |

**It runs and terminates cleanly. It cannot converge — and the reason is the script, not the mixers.**

* All three final losses are **2.780-2.787, i.e. above `ln(16) = 2.773`**. All three are at or below
  chance: `hm` recall **0.000**, `ca` 0.03125, `attn` 0.03125, against chance 0.0625. `attn` — which
  has unbounded state and can retrieve any single earlier token exactly — also fails. This is a
  uniform null result.
* Root cause is `_train_lingap.py:38-40, 59`: `tgt[i] = seq[1:] + [v]` sets a target at **every**
  position, and the loss is `F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))` with **no
  masking**. At `G=512` that is 515 scored positions per sequence, of which exactly one is the
  memory-retrieval question; the other 514 are "predict the next random distractor", which is
  information-theoretically unlearnable (each distractor is uniform over 16 and independent of the
  entire history). The optimiser therefore minimises a loss the task cannot reduce, and the one
  scored position that matters is swamped.
* Independent evidence that this is the script and not the mixers: the existing
  `lingap_hm_G64.json` (written 2026-09-05 18:23, i.e. **after** the docx) already has
  `G=64, steps=300, recall_last=0.0625, best=0.125, chance=0.0625` — exactly chance. My own 60-step
  reproductions at G=64 for all three mixers plateau at loss **3.17-3.26**, above `ln(V)=2.773`, with
  recall 0.0. Nothing improves with more steps because the signal is not in the loss.
* **Minimum fix (all three needed, all in `_train_lingap.py`):**
  1. score only the final query position: `loss = F.cross_entropy(logits[:, -1, :], tv)`;
  2. log that query loss too, so "loss > ln(V)" is visible as the failure it is;
  3. at `G=512` even a fixed script is fighting a 515-long sequence at `dim=96, layers=2` on CPU —
     raise `--steps` (>=2000) and sweep `--chunk`, since `ca` is chunk-dependent (§1 bug 3). Otherwise
     a null result remains indistinguishable from under-training.

**Related already-run benchmarks — all single-seed, all at chance, none citable as evidence:**
`membench_K4_G128.json`: ca `recall_last=0.09375, best=0.09375, chance=0.03125, steps=600`.
`membench_K8_G256.json`: **only** the `attn` arm, `recall_last=0.0, best=0.0625, chance=0.03125,
steps=120` — one element, even though `_train_mem_bench.py:85` defaults to `--mixer all` and appends
three; the run was killed after the first mixer and the partial file kept.
`retention_feat64.json`: **16 of 16 rows `recall=0.0`** at chance 0.0039, across K in {4,16,48,96} and
G in {0,128,256,512,1024} at `steps=200` — a uniform null.

---

## 4. Reproducibility check (actually run)

All three exit 0, no tracebacks. Times include process start and torch import.

### 4a. `& <python> verify_sgc\sgc_pytorch.py`

```
attn params: 4487684
sgc  params: 3975684
attn logits: (2, 128, 1000) sgc logits: (2, 128, 1000)
backward OK, loss= 0.0
```
exit 0, **wall 12.23 s**. **The smoke test is worthless as a check**: it computes
`F.cross_entropy(la.reshape(-1,V), x.reshape(-1))` (`sgc_pytorch.py:196`) — the *unshifted* input as
target. With a tied embedding the easiest solution is to echo the input, so `loss = 0.0` in both
cases. It tests nothing about next-token prediction and would hide a broken mixer. `mixers.py:277-279`
documents exactly this trap and shifts correctly — the two smoke tests in the repo disagree.

### 4b. `& <python> verify_sgc\sgc_v3.py`

```
sgcv3 params: 207236
forward/backward OK, loss= 0.0
```
exit 0, **wall 12.00 s**. Same echo artifact (`sgc_v3.py:391-392`, unshifted target). Also requires
cwd = `verify_sgc`, because `from sgc_pytorch import ...` (`sgc_v3.py:26`) has no `sys.path` shim.

### 4c. `& <python> verify_sgc\stream_infer.py`

Accepts **no** command-line arguments — there is no `argparse` at all, only a bare `main()`
(`stream_infer.py:65-89`), so the "report argparse help and skip" branch does not apply and I ran it.
Needs cwd = `verify_sgc`. exit 0, **wall 229.32 s** (over the 180 s budget, so it ran as a background
job). Verbatim output:

```
模型: SGC-CA dim=256 L=4 feat=32 chunk=256
参数量: 4,579,840 (fp32≈18.3MB)
基线 RSS: 0.0 MB
         L |      峰值RSS | ΔRSS(总) | ΔRSS vs 前/2^?
     2,048 |      0.0MB |    +0.0MB |    +0.0MB | 0s
     4,096 |      0.0MB |    +0.0MB |    +0.0MB | 0s
     8,192 |      0.0MB |    +0.0MB |    +0.0MB | 1s
    16,384 |      0.0MB |    +0.0MB |    +0.0MB | 1s
    32,768 |      0.0MB |    +0.0MB |    +0.0MB | 5s
    65,536 |      0.0MB |    +0.0MB |    +0.0MB | 8s
   131,072 |      0.0MB |    +0.0MB |    +0.0MB | 14s
   262,144 |      0.0MB |    +0.0MB |    +0.0MB | 26s
   524,288 |      0.0MB |    +0.0MB |    +0.0MB | 55s
 1,048,576 |      0.0MB |    +0.0MB |    +0.0MB | 106s
```

No traceback — **but the measurement is entirely broken.** `GetProcessMemoryInfo` returns 0 with
`GetLastError() = 122` (`ERROR_INSUFFICIENT_BUFFER`) on this host for both `stream_infer.py:26-33` and
`_memcap.py:43-45`, so `rss_mb()` is always `0.0`. I reproduced the failure standalone: allocate 8 MB,
re-read, still 0. Corroborating evidence inside the repo: `memcap.json`'s `attn_delta` column is
**all zeros**, and `_stream_memproof.py:3-6` states outright that the sandbox blocks OS working-set
reads and substitutes an analytic formula. The timing column *is* usable: 1,048,576 tokens in 106 s
≈ **101 us/token**, linear. **Therefore report §11.4's "O(1) 常驻状态实测" and "L=4096 已约 268 MB"
have no working measurement behind them.** Note also that §11.4 conflates two different quantities:
4,579,840 fp32 params = 18.3 MB, whereas "64.2 KB" is the analytic buffer+slot state for
`dim=256, layers=4, slots=32` at fp16 — `(32*256 + 32)*2*4 = 65,536 B`, copied from
`_stream_memproof.py:11-14`. It is not a measurement, and the report does not say so.

Supporting run: `verify_sgc\_verify_ca.py` — arithmetic check passes
(`max|ref-new| = 4.768e-07`), and it reports CA fwd+bwd **901 ms vs attention 432 ms = 2.09x slower**
at D=176 / 4 layers / seq=256, params CA 2,142,624 vs Attn 2,221,476.

---

## 5. Corpus reality check

`verify_sgc\build_corpus.py`:

* **Where the text comes from** (`collect_documents`, lines 24-47): four globs from the project root —
  `*.txt`, `docs/**/*.md`, `java-high-performance/*.txt`, `simplcore/**/*.txt` — each file kept only if
  `len(strip()) > 500`. One file = one document. No external corpus, no download.
* **Actual size: 24 documents, 115,255 bytes on disk, 74,610 stripped characters.** Only
  **12,694 characters (17%)** are the actual Chinese prose corpora
  (`simplcore\data\corpus\{classical\daodejing,classical\lunyu,philosophy\zhuangzi,modern\prose,science\popular}`);
  the other ~62k characters (83%) are machine output and Markdown:
  `java-high-performance\final_test_out.txt` 15,820 B, `docs\PAPER_SPEC.md` 12,701 B,
  `docs\HONEST_STATUS.md` 9,513 B, `fx_trace_err.txt` 6,986 B,
  `java-high-performance\Qingxiaotuan_Training_Summary.txt` 6,026 B, `docs\REPRODUCIBILITY.md`
  5,863 B, `java-high-performance\SCARCH_FINAL_REPORT.txt` 5,339 B,
  `java-high-performance\bench_out.txt` 4,141 B, `full_test_run.txt` 4,169 B,
  `docs\PATENT_REAL.md` 4,247 B, `docs\LONG_CONTEXT_STREAMING.md` 3,471 B, and similar.
  So "高熵真实中文语料" is 83% repository logs.
* **Splits.** For the shipped `results\corpus.json` (`seq_len=256`, `step=64`):
  **vocab 1084, train 582 windows, val 490, test 165**; tokens 148,992 / 125,440 / 42,240. The report's
  §9.1 cites exactly this ("vocab=1084 train=582 / val=490 / test=165"). `results\corpus512.json` was
  built later from a **different 21-document tree**: vocab **956**, train 345 / val 65 / test 89
  (`honest_max_ca512.json`'s args confirm `vocab_size=956`). So "同一语料" holds inside §9.1 and inside
  §10, but **not between them**.
* **Is the split leak-free?** The document-level mechanism is real and I verified it end to end: each
  file is assigned whole to exactly one split (`build_corpus.py:98-107`), the vocabulary is built from
  train documents only (`:109-111`), and **0 of 77 val and 0 of 101 test 128-character slices occur
  verbatim anywhere in a train document.** Greedy assignment lands on 4 val docs / 4 test docs /
  16 train docs. So *document-level* leakage: none. Two honest caveats:
  1. Windows **overlap 4x** (`step = seq_len//4 = 64` at `seq_len=256`), so 582 train windows contain
     only ~149k tokens at ~4x redundancy. The effective training set is far smaller than 582 implies,
     and each val window is 75% redundant with its neighbours. Document disjointness is preserved;
     the sample count is inflated ~4x.
  2. The split is not a balanced or topic-random split: `test` is dominated by one document
     (`docs\PAPER_SPEC.md` = 125 of 192 test windows, 65%), which is why test `<unk>` rate is
     **12.86%** in `corpus.json` vs 1.53% train / 0.31% val. The report does disclose the OOD rate, but
     it means §9.1's test column is largely "how well does the model do on PAPER_SPEC.md".
* **On-disk size of the raw source text: 115,255 bytes (112.6 KiB)** across those 24 files, 74,610
  characters after stripping. The whole corpus is ~75k characters; `corpus.json`'s train split is
  labelled 582 x 256 = 149k tokens. This is a micro-corpus, roughly 5 orders of magnitude below what a
  scaling claim would require.

---

## 6. Blunt gap list, ordered by impact

1. **The headline negative evidence is unsourced.** §3.4/§4.1/§5/§6 rest on "SGC test_acc 2.67% /
   Attention 12.89%" from `SGC_v2_门控消融_诚实报告.md`, which **does not exist**; the strings appear
   in no file. *Minimum fix:* regenerate that evaluation with a committed script + JSON, or delete the
   claim and rebuild §4.1/§5 on evidence you actually have. Everything downstream of it is currently
   indefensible.
2. **`GatedAssocMixer` is non-causal and non-deterministic.** `sgc_v3.py:199-201` writes before reading
   (measured: perturbing `x[63]` moves `y` at `t=1..62`, `|dy| ~ 0.02`); `sgc_v3.py:190` re-draws `M`
   with `torch.randn` every forward (measured: identical input, two calls differ by 9.7e-6).
   *Minimum fix:* read before write in `_step_pos`, and initialise `M` deterministically using the
   seeded-generator pattern already present in `SGC_HM._init_state` (`sgc_v3.py:261-263`).
3. **The §11.5 experiment cannot produce a result as written.** `_train_lingap.py:59` scores all `G+3`
   positions, so 514 of 515 targets are unknowable noise; I ran all three mixers to completion and all
   three end at or below chance with loss above `ln(V)`. *Minimum fix:* score the final position only
   (`F.cross_entropy(logits[:, -1, :], tv)`), then re-run with `--steps 2000`.
4. **The memory claims have no instrumentation.** `stream_infer.py:26-33` and `_memcap.py:43-45`
   silently return `0.0` because `GetProcessMemoryInfo` fails with `ERROR_INSUFFICIENT_BUFFER` here;
   `memcap.json`'s attn column is all zeros; `_stream_memproof.py` substitutes an analytic formula.
   *Minimum fix:* use `torch.cuda.max_memory_allocated` on a GPU, or `psutil.Process().memory_info()`,
   or fix the struct sizing **and assert the call returned non-zero**. Never publish a measurement
   whose API returned an error code.
5. **No real corpus.** 74,610 characters, 83% machine logs and Markdown, from a 24-file glob of the
   repository itself (`build_corpus.py:26-31`). Every "real-language" number — train_ppl 1.10, test_ppl
   11.10 vs 6.16, val_ppl 12,542, all of §9 and §10 — is measured on this. *Minimum fix:* point
   `collect_documents` at the real corpus trees already present under `simplcore\data\corpus\`, reach
   >=10^7 characters, keep the document-level split.
6. **"O(n log n) beats O(n^2) in practice" does not survive the training protocol.**
   `complexity_scaling.json` is forward-only and single-mixer; end-to-end, `honest_ca.json` is
   991 ms/step vs `honest_attn.json` 499 ms (2.0x *slower*), and my matched measurement gives
   attn 1199 / sgc 882 / sgcv3 1210 / ca 1205 ms per fwd+bwd step at D=176 / L=4 / seq=256. The clean
   speedup in `efficiency_timing.json` belongs to fixed-kernel `sgc` only. *Minimum fix:* claim
   efficiency for `sgc` and separately qualify or drop §9.2's `ca` crossover.
7. **`ContentAdaptiveMixer` results depend on an undeclared hyper-parameter.** Output changes by up to
   1.3 when `chunk` changes (measured across 8/16/32/64). `ca_chunk` is 128 (`train_honest.py` default),
   256 (`stream_infer.py:67`), 512 (`honest_max_ca512.json`), 16/32 in probes. *Minimum fix:* make the
   chunked form invariant (normalise `den` consistently) or promote `ca_chunk` to a documented
   architecture parameter held fixed across every comparison.
8. **The "training" runs do not train.** Selective-copy plateaus near 36% against a 12.5% random
   baseline and the report itself says "远未收敛" — yet it is the headline §3.2 result.
   `gated_compare.json` is literally 5 steps with 1 layer, dim 32, wd 0, its train losses wandering
   around `ln(vocab)`, and those numbers appear in report table 3. `retention_feat64.json` is 16/16 rows
   at recall 0.0. `membench_K8_G256.json` holds only the `attn` arm because the run was killed mid-way
   and the partial file kept. *Minimum fix:* pick one task that actually has headroom, train it to a
   real plateau, and only then compare mixers.
9. **`sgc`/`sgcv3` have a silent context ceiling and no long-context path.** `self.low[:L,:]`
   (`sgc_pytorch.py:60`, `sgc_v3.py:60`) zero-pads beyond `max_len`, so usable context is exactly the
   `max_len` passed to `build_model` (769 in `_train_lingap.py:50`) and nothing errors. There is no
   `forward_stream` for `sgc`/`sgcv3` at all; `stream_infer.py` demonstrates streaming only for `ca`;
   and `SGC_HM`'s recent read materialises a `(B, n, buffer+n)` score matrix per chunk
   (`sgc_v3.py:292-301`), so its peak is `O(buffer·chunk)`, not `O(1)`. *Minimum fix:* assert on
   `max_len`, and implement an overlap-add FFT streaming path for `sgc` — the rank-r causal kernel
   makes that natural, and it is the one genuinely differentiating capability in this suite.
10. **Result files are not traceable to the invocation that produced them.** `complexity_scaling.json`
    has no `args`; the lingap and membench files record no host or wall time; several scripts
    (`_verify_ca.py`, `_probe_retention.py`, `_stream_memproof.py`, `_ar_quick.py`, `_probe_scale.py`,
    `_probe_variance.py`) never write JSON at all, so §11.2/§11.3/§11.4 and the missing `ar_quick.json`
    are unreproducible by construction; `verify_sgc\results\` has no manifest. *Minimum fix:* a single
    runner that writes `{args, source hash, host, wall_s, metric}` for every script and refuses to
    write a JSON whose measurement API returned an error.

---

## 7. §11.5 completion log (background job `pwsh-94`)

Raw log: `.tmp_audit\lingap_G512.txt`. Started 18:26:32, finished 19:07:25.

```
  hm   G= 512  loss=2.787  recall=0.000 best=0.000 params=440,642
saved ...\verify_sgc\results\lingap_hm_G512.json
  [hm] exit=0 run=905s
  ca   G= 512  loss=2.782  recall=0.031 best=0.031 params=428,352
saved ...\verify_sgc\results\lingap_ca_G512.json
  [ca] exit=0 run=535.2s
  attn G= 512  loss=2.780  recall=0.031 best=0.062 params=422,210
saved ...\verify_sgc\results\lingap_attn_G512.json
  [attn] exit=0 run=862.3s
TOTAL elapsed=2302.6s
```

Written JSON (exact):

```json
lingap_hm_G512.json   {"V":16,"task":"interval-memory","mixer":"hm","G":512,"recall_last":0.0,
                       "best":0.0,"chance":0.0625,"params":440642,"steps":700}
lingap_ca_G512.json   {"V":16,"task":"interval-memory","mixer":"ca","G":512,"recall_last":0.03125,
                       "best":0.03125,"chance":0.0625,"params":428352,"steps":700}
lingap_attn_G512.json {"V":16,"task":"interval-memory","mixer":"attn","G":512,"recall_last":0.03125,
                       "best":0.0625,"chance":0.0625,"params":422210,"steps":700}
```

Report §11.5's three "(训练进行中)" rows can now be filled — but the filled table says **no mixer learns
the task**, which is a statement about `_train_lingap.py`, not about HM vs CA vs attention.

**Side effect, flagged:** running the mandated experiment caused its own script to **create three new
files in `verify_sgc\results\`** (`lingap_hm_G512.json`, `lingap_ca_G512.json`,
`lingap_attn_G512.json`, 174-182 B each). Nothing there was overwritten. If the `results\` tree must
stay byte-frozen, delete them; otherwise they are the first legitimate §11.5 data.
