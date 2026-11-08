Appendix A.36  The final TSC-H layer (architecture specification)

Input.  V = 2 V_K, C = Sylvester Hadamard(V), C C^T = I.  Three symbol
classes, distinguished ONLY by id (no run-time tag, no position rule):
    KEY symbol k        -> id k        -> C[k]            (first  V_K rows)
    VALUE symbol k      -> id V_K + k  -> C[V_K + k]      (last   V_K rows)
    FILLER              -> zero one-hot -> 0
This is the single role bit of R3.  C is applied by FWHT, O(V log V) / O(V).

Per layer h = 1..H (K_h(t) = sum_{p=1..P} w_hp rho_hp^(L-1-t), 0<rho<1):
    A_t = C[x_(t-1)];  b_t = C[x_t]                    (kernel-external)
    S_h = sum_t K_h(t) A_t (x) b_t                     (order-2, O(1) state)
    m_h[t] = <q_h, A_t>                                (content match)
    z_h    = sum_t K_h(t) m_h[t] b_t = S_h^T q_h
    q_(h+1) = normalize(T_h z_h)                        (learned, FIXED in t)
    T_h = U_h W_h^T / sqrt(r),  r >= m = n_pairs+1      (R4)
Read-out:  logits_j = tau <z_H, C[V_K + j]>,  tau > 0 learned.

Invariants (all machine-checked on the FINAL learned-T construction):
    Toeplitz deviation 0.0; FFT/IIR relative error 2.89e-16;
    last-time deviation 1.45e-16; strict causality;
    state = 2 P V floats per layer, independent of L;
    no N x N score matrix, no softmax over tokens, no KV cache,
    no input-dependent kernel, no external role tag, no position gate.

Cost:  parameters = (H-1) * 2 V r + 1;  codebook = 0 stored bytes (FWHT);
       at V = 2^18 (Qwen3 scale) with r = 8: 4,194,305 parameters,
       one FWHT 0.0119 s, versus 549.8 GB for a dense codebook.
