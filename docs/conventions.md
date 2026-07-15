# Scientific conventions

Authoritative reference for the polarization, RIME, and χ² conventions used
across NeuralDMD. Grounded in **Thompson, Moran & Swenson (TMS),
*Interferometry and Synthesis in Radio Astronomy*, 3rd ed. (Springer, 2017),
Chapter 4**, and cross-checked numerically against
`ehtim/observing/pol_conventions.py` (Chael 2026, which itself cites "TMS Ch. 4").
Every physics PR updates this file.

## 1. Polarization basis and Stokes ↔ correlation products

We use the **IAU / IEEE engineering convention** (time dependence
`exp(+j ω t)`, as in TMS Ch. 4), with the circular feeds defined from the
linear (X = north, Y = east on sky) feeds by

```
R = (X + iY) / √2 ,   L = (X − iY) / √2 .
```

### Coherency products in terms of Stokes (I, Q, U, V)

| circular (R, L) | linear (X, Y) |
|-----------------|---------------|
| RR = I + V      | XX = I + Q    |
| LL = I − V      | YY = I − Q    |
| RL = Q + iU     | XY = U + iV   |
| LR = Q − iU     | YX = U − iV   |

Implemented in `src/neuraldmd/physics/stokes.py` as the `_PRODUCT_COEFFS`
table / `stokes_to_products_matrix`.

### Derivation (why these, from TMS)

* **Linear feeds — TMS Eq. (4.28)** gives the coherencies directly:

  ```
  ⟨Ex Ex*⟩ = ½(I + Q)    ⟨Ey Ey*⟩ = ½(I − Q)
  ⟨Ex Ey*⟩ = ½(U + jV)   ⟨Ey Ex*⟩ = ½(U − jV)
  ```

  (the overall ½ is absorbed into the gain, per TMS's note after Eq. 4.29).

* **Circular feeds — TMS Eq. (4.29)** (the general Morris/Weiler formula) with
  the ideal circular-feed ellipticities `χ_R = −π/4`, `χ_L = +π/4` and feed
  position angle `ψ`:

  ```
  RR = ½ G [ I + V ]
  LL = ½ G [ I − V ]
  RL = ½ G e^{−j(ψ_m+ψ_n)} [ Q + jU ]
  LR = ½ G e^{+j(ψ_m+ψ_n)} [ Q − jU ]
  ```

  After field-rotation (parallactic) derotation, `ψ → 0` and
  `RL = Q + iU`, `LR = Q − iU`. The `e^{∓j·2ψ}` phase is exactly the feed
  rotation carried by the RIME (§3).

* **Cross-check** — `tests/test_stokes_ehtim.py` asserts our
  `stokes_to_products_matrix` equals ehtim's `stokes_to_circ` / `stokes_to_lin`
  bit-for-bit over random Stokes vectors.

### Analytic sanity checks (in `tests/test_stokes.py`)

* Pure I → RR = LL = I, RL = LR = 0.
* V = 0 → RR = LL (equal parallel hands).
* The 4×4 circular matrix is unitary up to scale (round-trip identity).

## 2. Coherency (brightness) matrix

The sky brightness matrix in the **linear (X, Y) basis** is (TMS Eq. 4.28):

```
B = Σ_s S_s σ_s = [[I + Q,  U + iV],
                   [U − iV,  I − Q]] = [[XX, XY],
                                        [YX, YY]]
```

with `stokes_pauli_matrices()` returning
`σ_I = [[1,0],[0,1]]`, `σ_Q = [[1,0],[0,−1]]`, `σ_U = [[0,1],[1,0]]`,
`σ_V = [[0,i],[−i,0]]`. This is identical to ehtim's `_coherency_matrix` in the
XY basis (asserted in `test_stokes_ehtim.py`).

## 3. RIME (measurement equation)

TMS Eqs. (4.44)–(4.52) (Hamaker–Bregman–Sault). The observed 2×2 coherency on
baseline (m, n) is

```
V'_mn = J_m · B · J_n^H          (TMS Eq. 4.52, 2×2 form of the outer product)
```

where `J^H` is the conjugate transpose (note the **conjugate on the second
antenna** — TMS writes `(J_m ⊗ J_n*)`). The per-station Jones matrix factors as

```
J = G · (I + D) · R_ψ
```

| factor | form | TMS |
|--------|------|-----|
| gain      | `G = diag(g_R, g_L)`            | Eq. (4.47) |
| leakage/D | `(I + D) = [[1, d_R], [d_L, 1]]` | Eq. (4.46) |
| feed rot. | `R_ψ = diag(e^{−jψ}, e^{+jψ})`  | Eq. (4.45) |

This is the same factoring as `ehtim.pol_conventions.jones_matrix`
(`J = G @ (I + D)`), pinned in `test_stokes_ehtim.py`.

**We apply the RIME to the MODEL** visibilities (corrupt the prediction, then
χ² against the data). This differs from KINE, which corrects the *data*; the two
are equivalent for pure diagonal gains but **not** once D-terms enter, so
model-side application is the correct general choice.

Fast paths: when `D = 0`, `g_R = g_L = g`, and Stokes-I only, the chain reduces
to the scalar `V'_mn = g_m g_n* V_mn`.

## 4. χ² normalization

χ² is computed per **real** degree of freedom: each complex-visibility term is
divided by `2·Σ(mask)` (real + imaginary parts), consistent across all Stokes
and products, with equal weights (KINE pattern). Padded visibility entries
(σ = 1e6, mask = 0) contribute zero. Closure phases stay gain-invariant and
serve as a calibration canary.

## 5. Derived quantities

* **EVPA** (electric-vector position angle): `χ = ½·atan2(U, Q)` (TMS/IAU;
  equals ehtim's `0.5·angle(Q + iU)`).
* **Linear polarized intensity**: `P = √(Q² + U²)`. We **never divide by I**
  anywhere in a loss (no fractional-polarization parameterization).

## 6. Degeneracies & sign register

| item | note / mitigation |
|------|-------------------|
| **V sign ↔ basis** | Tied to the circular basis. We use IAU/IEEE `R = (X+iY)/√2` (positive V = RCP → RR = I + V). The "physics" basis `R = (X−iY)/√2` flips V's sign. |
| flux ↔ gain-amp | hard gain bounds; warmup freeze; optional mean-\|g\|≈1 penalty. |
| global phase ↔ image shift | phase gains OFF by default; reference-station φ = 0 when on. |
| W ↔ b scale | existing unit-RMS gauge fix (stop-gradient), kept. |
| EVPA sign | tested vs ehtim (`test_evpa_matches_ehtim_definition`). |
| P ≤ I | optional soft penalty only, default weight 0.0. Negativity penalty applies to **I only** (Q, U, V are signed). |
