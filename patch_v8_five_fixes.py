#!/usr/bin/env python3
"""Apply the five reviewer-identified fixes to build_simplcore_pdf_v8.py.

Every replacement is asserted to occur exactly the expected number of times,
so a silent no-op cannot pass.  All injected numbers come from result JSONs.
"""
import json, os, re, sys

P = "/data/workspace/build_simplcore_pdf_v8.py"
s = open(P, encoding="utf-8").read()
ci = json.load(open("/data/workspace/ts2sc_vl47/se2_ci.json", encoding="utf-8"))

R = []
_WS = re.compile(r"\s+")


def _locate(src, needle):
    """Whitespace-insensitive substring search; return (start,end) spans in src."""
    pat = _WS.sub(" ", needle).strip()
    chars, starts, ends = [], [], []
    i = 0
    while i < len(src):
        if src[i].isspace():
            j = i
            while j < len(src) and src[j].isspace():
                j += 1
            chars.append(" ")
            starts.append(i)
            ends.append(j)
            i = j
        else:
            chars.append(src[i])
            starts.append(i)
            ends.append(i + 1)
            i += 1
    comp = "".join(chars)
    out = []
    for m in re.finditer(re.escape(pat), comp):
        a, b = m.start(), m.end()
        out.append((starts[a], ends[b - 1]))
    return out


def rep(tag, old, new, n=1):
    spans = _locate(s, old)
    assert len(spans) == n, "%s: expected %d occurrence, found %d\n  %r" % (
        tag, n, len(spans), old[:90])
    R.append((tag, old, new, n))


# ---------------------------------------------------------------- FIX 1 ----
# Theorem A(a): <=1/2 requires B <= log2|A|-1; general bound is 2^B/|A|.
rep("F1/A(a) main statement",
    '"acc(F | v) <= 2^B / |A(v)|. If B < log2 |A(v)| then "\n'
    '       "acc(F | v) <= 1/2. This is a conditional bound on a single fibre, not "\n'
    '       "a marginal claim."',
    '"acc(F | v) <= 2^B / |A(v)|. If B < log2 |A(v)| the bound is < 1 but "\n'
    '       "need not be <= 1/2 (for |A(v)| = 3, B = 1 it is 2/3); the familiar "\n'
    '       "<= 1/2 form requires the stronger B <= log2|A(v)| - 1. This is a "\n'
    '       "conditional bound on a single fibre, not a marginal claim."')

rep("F1/A.1 proof corollary",
    '"hence acc(F | v) <= 2^B / |A(v)|; this proves (a). Exactness on every "',
    '"hence acc(F | v) <= 2^B / |A(v)|; this proves (a). The <= 1/2 "\n'
    '       "corollary follows only when 2^B <= |A(v)|/2, i.e. "\n'
    '       "B <= log2|A(v)| - 1. Exactness on every "')

# ---------------------------------------------------------------- FIX 2 ----
# Result 2(ii) "exactly H layers" conditioned on its premises everywhere.
PREM = ("under the H6 interface, last-layer read-out (A4), distinct keys, "
        "at most one content match resolved per layer, and per-hop successor "
        "sets of rank c as in P5")

rep("F2/abstract",
    '"with a sharp depth bound of exactly H static-LTI layers for H hops "\n'
    '       "(Result 2, the central claim of this paper). The difficulty is that "',
    '"with a sharp depth bound of H static-LTI layers for H hops "\n'
    '       "under H6, last-layer read-out, distinct keys and one content "\n'
    '       "match per layer (Result 2, the central claim of this paper). "\n'
    '       "The difficulty is that "')

rep("F2/abstract design rule",
    '"choose exactly H layers and an order-2 state; we make no language-"',
    '"choose H layers under Result 2(ii)\'s stated premises and an order-2 "\n'
    '       "state; we make no language-"')

rep("F2/contribution Result 2",
    '"order for one hop is exactly 2, and exact H-hop retrieval under last-"\n'
    '       "layer read-out requires exactly H static-LTI layers, with O(1) state "\n'
    '       "in L. Every other result in this paper serves this bound.")',
    '"order for one hop is exactly 2, and exact H-hop retrieval "\n'
    '       "requires exactly H static-LTI layers, with O(1) state in L, "\n'
    '       "%s. The depth claim is conditional on these premises, "\n'
    '       "not an architecture-independent law; with a binary branching "\n'
    '       "factor P5\'s state-order statement reduces accordingly. "\n'
    '       "Every other result in this paper serves this bound.")' % PREM)

rep("F2/intro",
    '"static time-invariant kernel and last-layer read-out, H content hops "\n'
    '       "need exactly H layers, and the state stays O(1) in sequence length.",',
    '"static time-invariant kernel and last-layer read-out, H content hops "\n'
    '       "need H layers under H6, A4, distinct keys and one content match "\n'
    '       "per layer (Result 2(ii)), and the state stays O(1) in L.",')

rep("F2/A(e) frontier",
    '"B >= log2 |A(v)| the minimum depth is still exactly H (Result 2); for "',
    '"B >= log2 |A(v)| the minimum depth is still H (Result 2(ii), "\n'
    '       "under its stated premises); for "')

rep("F2/linear-attn boundary",
    'and the layer count is exactly H by Result 2(ii).',
    'and the layer count is H by Result 2(ii) under its stated premises.')

rep("F2/section 4.12",
    'B >= log2|A| the minimum depth is exactly H (Result 2(ii)) and the',
    'B >= log2|A| the minimum depth is H (Result 2(ii), under its "\n'
    '      "premises) and the')

rep("F2/conclusion stack",
    '"layer stack needs exactly H layers for H hops with O(1) state in L, "',
    '"layer stack needs H layers for H hops under Result 2(ii)\'s premises "\n'
    '       "and with O(1) state in L, "')

rep("F2/conclusion rule",
    '"budget log2|A(v)| bits at the input, then choose exactly H layers and "',
    '"budget log2|A(v)| bits at the input, then choose H layers under "\n'
    '       "Result 2(ii)\'s premises and "')

rep("F2/A.3 proof",
    '"Result 2(ii) gives minimum depth exactly H. The frontier is a right "',
    '"Result 2(ii), under its stated premises, gives minimum depth H. "\n'
    '       "The frontier is a right "')

# ---------------------------------------------------------------- FIX 3 ----
# SE2: the 0.10% gap is a coincidence of the mean; give the 95% CI.
rep("F3/Table16 caption",
    '"%.4f. SE2\'s coupon-collector bound predicts E[T_cover] = %.2f "\n'
    '       "rounds against a measured %.2f (relative gap %.2f percent); the "\n'
    '       "premise was fixed to i.i.d. arrival BEFORE measurement, not fitted "\n'
    '       "after. SE4 caps the information entering the state at T w log2(V_K) "',
    '"%.4f. SE2\'s coupon-collector bound predicts E[T_cover] = %.2f "\n'
    '       "rounds against a five-seed mean of %.2f (relative gap %.2f "\n'
    '       "percent, 95%% t-CI [%.2f, %.2f] rounds, n = 5). The mean agrees "\n'
    '       "with the prediction, but the seed-to-seed spread is large: the "\n'
    '       "small relative gap is a coincidence of the MEAN and is NOT a "\n'
    '       "precision claim. The premise was fixed to i.i.d. arrival BEFORE "\n'
    '       "measurement, not fitted after. SE4 caps the information entering "\n'
    '       "the state at T w log2(V_K) "')

rep("F3/Table16 tuple",
    '        AGREE["SE2_measured_mean_first_cover"],\n'
    '        100 * AGREE["SE2_relative_gap"]),\n'
    '       size=SMALL, lead=SLEAD, after=2.5)',
    '        AGREE["SE2_measured_mean_first_cover"],\n'
    '        100 * AGREE["SE2_relative_gap"],\n'
    '        SE2_CI["ci95"][0], SE2_CI["ci95"][1]),\n'
    '       size=SMALL, lead=SLEAD, after=2.5)')

rep("F3/A.8 proof",
    '"%.2f (relative gap %.2f percent), with per-seed values %s. The "\n'
    '       "arrival model was set to i.i.d. BEFORE measuring. BOUNDARY. A "',
    '"%.2f (relative gap %.2f percent), with per-seed values %s. With "\n'
    '       "n = 5 the sample SD is %.2f and the 95%% t-CI is [%.2f, %.2f] "\n'
    '       "rounds: the prediction lies inside the interval, but the "\n'
    '       "interval spans %.0f%% of the mean, so this is weak evidence and "\n'
    '       "the small gap is NOT a precision claim. The arrival model was set "\n'
    '       "to i.i.d. BEFORE measuring. BOUNDARY. A "')

rep("F3/A.8 tuple",
    '        100 * AGREE["SE2_relative_gap"],\n'
    '        ", ".join(str(x) for x in AGREE["SE2_per_seed_first_cover"])),\n'
    '       size=SMALL, lead=SLEAD, after=2.5, span=False)',
    '        100 * AGREE["SE2_relative_gap"],\n'
    '        ", ".join(str(x) for x in AGREE["SE2_per_seed_first_cover"]),\n'
    '        SE2_CI["sample_sd"], SE2_CI["ci95"][0], SE2_CI["ci95"][1],\n'
    '        100 * (SE2_CI["ci95"][1] - SE2_CI["ci95"][0]) / SE2_CI["mean"]),\n'
    '       size=SMALL, lead=SLEAD, after=2.5, span=False)')

rep("F3/Table28 rows",
    '         ["SE2 relative gap", "%.4f%%" % (100 * AGREE["SE2_relative_gap"]),\n'
    '          "derived", "-", "prediction vs measurement"],',
    '         ["SE2 relative gap", "%.4f%%" % (100 * AGREE["SE2_relative_gap"]),\n'
    '          "derived (mean coincidence, NOT precision)", "-",\n'
    '          "prediction vs measurement"],\n'
    '         ["SE2 cover 95% t-CI", "[%.2f, %.2f]" %\n'
    '          (SE2_CI["ci95"][0], SE2_CI["ci95"][1]),\n'
    '          "t(4), sample SD %.2f" % SE2_CI["sample_sd"], "5",\n'
    '          "per-seed first full cover"],')

# ---------------------------------------------------------------- FIX 4 ----
# Notation table: B vs b vs state bytes; unify the "state B" header.
rep("F4/Notation pointer in Section 2",
    '"Given v, a decoder can observe at most 2^B distinguishable "\n'
    '       "refinements of v.", size=SMALL, lead=SLEAD, after=2.0)',
    '"Given v, a decoder can observe at most 2^B distinguishable "\n'
    '       "refinements of v. Uppercase B (total input-side bits) and "\n'
    '       "lowercase b (bits per insight per consolidation round) are "\n'
    '       "different quantities; state bytes is a size in bytes, never a "\n'
    '       "bit budget. The full notation table is Appendix A.0.",\n'
    '       size=SMALL, lead=SLEAD, after=2.0)')

rep("F4/Appendix A.0 notation table",
    '    Hh(d, "A.1  Theorem A(a) and A(b)", size=9.8, before=4.0, after=2.0)\n',
    '    Hh(d, "A.0  Notation and unit ledger", size=9.8, before=4.0,\n'
    '        after=2.0, span=False)\n'
    '    d.table([\n'
    '        ["symbol", "meaning", "unit / where used"],\n'
    '        ["B", "total input-side bit budget, B = sum_t b_t",\n'
    '         "bits; Theorem A"],\n'
    '        ["b_t", "per-position alphabet excess log2|pi^(-1)(v_t)|",\n'
    '         "bits; Section 2"],\n'
    '        ["b", "bits per insight per consolidation round",\n'
    '         "bits; CT7(b), SE4; here b = 5 = log2 V_K"],\n'
    '        ["w, T", "insights per round, number of rounds",\n'
    '         "count; CT7(b), SE4"],\n'
    '        ["state bytes", "structural size of the recurrent state",\n'
    '         "BYTES, not bits; Table 8, Appendix D"],\n'
    '        ["32768", "d x d fp64 core state, d = 64 (fusion config)",\n'
    '         "bytes; Tables 16, 18, 34; DIFFERENT from Table 8"],\n'
    '        ["c, o", "orthogonal successor rank per hop, tensor order",\n'
    '         "count; P5, Result 2"],\n'
    '    ], [36, 100, 82], size=6.4, lead=7.6, after=2.0, span=False)\n'
    '    P(d, "<b>Table A0.</b> Uppercase B and lowercase b are different "\n'
    '       "quantities; state bytes is a size in bytes and never a bit "\n'
    '       "budget. Table 8 (throughput configuration) and the fusion "\n'
    '       "tables (d = 64) use DIFFERENT state configurations and must "\n'
    '       "not be compared numerically.",\n'
    '       size=SMALL, lead=SLEAD, after=2.5, span=False)\n'
    '    Hh(d, "A.1  Theorem A(a) and A(b)", size=9.8, before=4.0, after=2.0)\n')

rep("F4/state B header", '"H_L", "L", "us/token", "state B", "state const."',
    '"H_L", "L", "us/token", "state bytes", "state const."')

# ---------------------------------------------------------------- FIX 5 ----
# Superseded single-seed values: mark NOT USED, bold warning.
rep("F5/Table10 header",
    '         ["quantity", "earlier values", "value here", "statistic / source"],',
    '         ["quantity", "superseded (NOT USED)", "value here", "statistic / source"],')

rep("F5/Table10 warning",
    '     P(d, "Earlier drafts quoted several values for the same configuration. "\n'
    '       "Table 10 records each drift, the single value used in this submission, "\n'
    '       "its statistic, and its source. No number in this paper appears "\n'
    '       "without one of these rows.", size=SMALL, lead=SLEAD, after=2.0)',
    '     P(d, "Earlier drafts quoted several values for the same configuration. "\n'
    '       "Table 10 records each drift, the single value used in this submission, "\n'
    '       "its statistic, and its source. No number in this paper appears "\n'
    '       "without one of these rows. <b>Only the five-seed column enters any "\n'
    '       "claim; all superseded single-seed values are quoted for provenance "\n'
    '       "only and are NEVER used as evidence.</b>",\n'
    '       size=SMALL, lead=SLEAD, after=2.0)')

# Appendix D: point at a machine-readable long-form CSV.
rep("F5/Appendix D CSV pointer",
    '       "1/32 = 0.0313. The core state is 32768 bytes in every row.",',
    '       "1/32 = 0.0313. The core state is 32768 bytes in every row "\n'
    '       "(d = 64, fp64; notation table in Section 2). A machine-readable "\n'
    '       "long-form CSV of every row below is supplied in the artifact as "\n'
    '       "appendix_D_fusion_long.csv; no value is OCR-transcribed.",')

# Transpose direction: unify on C_val^T C_key (value-to-key swap).
rep("F5/T* direction (both occurrences)",
    'T* = C_key^T C_val', 'T* = C_val^T C_key', n=2)

rep("F5/H6 duplicate sentence",
    'The read-out is last-layer only (A4). The read-out is last-layer only (P4).',
    'The read-out is last-layer only (A4 / P4); the 1/R codebook scaling is "\n'
    '      "absorbed by the unit normalisation in H6.')

# ---------------------------------------------------------- SE2_CI load ---
rep("inject SE2_CI constant",
    'ROOT = "/data/workspace"\n',
    'ROOT = "/data/workspace"\n'
    'SE2_CI = json.load(open(ROOT + "/ts2sc_vl47/se2_ci.json",\n'
    '                        encoding="utf-8"))  # 5-seed t-CI from raw seeds\n')

# ---------------------------------------------------------------- apply ----
for tag, old, new, n in R:
    spans = _locate(s, old)
    assert len(spans) == n, "apply %s: %d spans" % (tag, len(spans))
    a, b = spans[0]
    # s[:a] already carries the original line indentation; strip only the
    # leading whitespace of the replacement so we never double-indent.
    s = s[:a] + new.lstrip() + s[b:]

open(P, "w", encoding="utf-8").write(s)
print("APPLIED %d replacements:" % len(R))
for tag, _, _, _ in R:
    print("  -", tag)
print("\nSE2_CI source:", ci["ci95"], "| n_seeds:", ci["n_seeds"])
