"""expand4_v7.py — final non-repetitive hard content: escape boundaries and
reproduction protocol. Fills the last main-paper page without padding."""
G = "/data/workspace/build_simplcore_pdf_v7.py"
s = open(G, encoding="utf-8").read()


def ins(anchor, block, label):
    global s
    assert s.count(anchor) == 1, (label, s.count(anchor))
    s = s.replace(anchor, block + anchor)
    print("  inserted", label)


ins('    Hh(d, "4  Experiments: One Table, One Numeric Convention",\n',
    r'''
    Hh(d, "3.6  Why the three escapes do not beat the bound",
       size=9.5, before=4.0, after=2.0)
    P(d, "P2's conclusion depends on its premise: a single mixer whose "
      "history enters the final computation only through a fixed linear "
      "summary A. Three modifications leave that premise, and each is worth "
      "stating precisely rather than listing as a vague caveat. "
      "<b>(i) Deep stacks with per-position nonlinear FFNs.</b> A and q then "
      "pass through the SAME nonlinear map, producing cross-terms that are "
      "not captured by A = sum_p L_p v_p + c; the zero-space construction of "
      "Section 3.4 no longer applies. "
      "<b>(ii) Input-dependent gates.</b> The recurrence coefficients become "
      "functions of x_t, so the coefficients are no longer time-invariant; "
      "by P6 the impulse response is non-Toeplitz and the single-FFT path "
      "fails. This is a real architectural difference, not a defect. "
      "<b>(iii) Outer-product states.</b> Keeping S = sum_p k_p v_p^T "
      "retains the binding explicitly instead of compressing it into one "
      "vector, so dim ker L >= nd - d no longer bounds what the state can "
      "distinguish. Construction C is an instance of (iii).",
      size=SMALL, lead=SLEAD, after=2.0)
    P(d, "None of the three escapes changes Theorem A(b). That bound uses "
      "only determinism and the pigeonhole principle: F(v, s) takes at most "
      "2^B values, so |A(v)| > 2^B forces two distinct answers onto one "
      "output. It makes NO assumption about P3. Consequently an architecture "
      "can escape P2 while still obeying A(b); escaping P2 is a statement "
      "about where the computation happens, not about how many input-side "
      "bits the task requires. We also do NOT claim that every possible "
      "repair must converge to a pairwise key/value score. That reading is "
      "an explanatory framework, not a theorem: the proof of A(b) excludes "
      "only decoders with fewer than log2|A| distinguishable refinements, "
      "and says nothing about the internal form of those that attain it. "
      "Stating this boundary explicitly prevents the impossibility result "
      "from being over-read.", size=SMALL, lead=SLEAD, after=2.5)

''', "3.6 escapes")

ins('    Hh(d, "5  What This Does NOT Establish", size=10.2, before=5.0,\n',
    r'''
    Hh(d, "4.10  Reproduction protocol", size=9.5, before=4.0, after=2.0)
    P(d, "The anonymised artifact contains one reference implementation of "
      "the layer, seven minimal mechanism references used ONLY for the "
      "Section 4.6 audit, the reachability enumeration of Section 4.3, and "
      "all raw result files. All probes are CPU-only NumPy with no external "
      "model weights and complete in minutes on a single core. Seeds are "
      "listed per table: 1000+s for training, 7000+s for held-out evaluation, "
      "s in {0,..,4} for the five-seed rows; analytic witnesses use fixed "
      "seeds recorded with each table. Every number in Tables 2-12 is "
      "injected from the raw files at build time by string formatting, so a "
      "changed source file changes the PDF and a stale transcription is "
      "impossible. A single command regenerates the paper and a second "
      "command re-runs all synthetic probes. The verifier checks statement "
      "formation, witness agreement and protocol consistency "
      "(%d/%d gates pass); it is NOT a mathematical proof, and all proofs in "
      "this paper are analytic." %
      (next(iter(ADJ["checks"].values())) is not None and ADJ["n_pass"],
       ADJ["n_checks"]),
      size=SMALL, lead=SLEAD, after=2.0)
    P(d, "Compute statement: total CPU time for every reported synthetic "
      "probe is under one hour; no GPU, no pretrained model, no private "
      "weight and no external dataset is used. Token counts, perplexity "
      "and generation quality are NOT measured. The 2027 call for papers "
      "and submission deadline had not been published when this version was "
      "prepared; page count, the author-side LLM-use disclosure and "
      "reciprocal-reviewer obligations must be rechecked against the "
      "official template when it appears.", size=SMALL, lead=SLEAD,
      after=2.5)

''', "4.10 reproduction")

open(G, "w", encoding="utf-8").write(s)
print("EXPANSION 4 WRITTEN")
