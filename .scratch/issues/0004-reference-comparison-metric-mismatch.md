# Issue #4: Square-vs-Reference "mismatch" is a metric artifact, not a kernel regression

**GitHub:** [#10](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/10) — the full
report is published there. Extends GitHub [#2](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/2)
(cross-linked in [this comment](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/2#issuecomment-5730785802)).
Note the local `.scratch/issues/` numbering diverged from GitHub after `0002` — local `0003` is
GitHub #7, local `0004` is GitHub #10.

**Role / Label:** `open — diagnosis complete, fixes not yet applied`
**Status:** 🔍 Diagnosed 2026-09-18. Root cause identified and quantified with code evidence.
No code fix applied yet: the report chart still draws the bad comparison, and the fresh
reference baseline has been built but **not yet run on hardware**.
**Target:** `jbpi2` (BananaPi M2 Pro, Spacemit X60, VLEN=256)
**Shapes:** Square `S1-S5` (1000³…5000³, `MR32_NR7`) vs BERT `B1-B5` (`MR32_NR6`/`MR16_NR15`/`MR32_NR7`)
**Primary Sources:**
- Report under investigation: `tests/pure_xdsl_blis_rvv/logs/report_2026-09-18/`
  (`report.html`, `build_report_data.py`, `HANDOFF.md`)
- Run data: `tests/pure_xdsl_blis_rvv/logs/run_2026-09-18_{004053,014128,024151,034155,044110}/`
- Archived reference tables: `ext/square_comparison.txt`, `ext/bert_large_comparison.txt`,
  `ext/output/retro_europar_plot.txt`, `ext/plot_square.py`, `ext/plot_bert_large.py`
- Real reference design: `ext/RVV_ukernels_benchmark/gemm_blis_family/`
- New tooling: `tests/pure_xdsl_blis_rvv/ref_baseline/`

---

## 0. Presenting symptom

On the 2026-09-18 report chart, the BERT family sits at or above the reference-design
line while the Square family sits below it. Square was suspected of a performance
regression.

**It is not a regression.** Square is within ±3% of the archived reference on the
metric the chart uses. Three independent artifacts stack up to produce the illusion.

---

## 1. Root cause A (dominant): the chart compares two different metrics

| | formula | packing |
|---|---|---|
| ours | `2mnk / total_kernel_time` — `templates/driver_benchmark.cpp:695` | **excluded** |
| reference | `2mnk / wall(gemm_blis_B3A2C0)` — `ext/.../src/test_gemm.c:399-400` | **included** |

The generated nest brackets A-pack and B-pack into `pt`, and only the `jr`/`ir`
microkernel calls into `kt` (`pj_gen/.../gemm_dispatch_xdsl.cpp:71-144`). The report
plots `KerGOPs_runs` against `REF[...].xdsl_ref` (`report.html:665-668`).

Excluding packing inflates our number by `1/(1-pack_fraction)`, and that fraction is
strongly shape-dependent:

```
pack fraction -> "free" inflation of KerGOPs vs a pack-inclusive reference

Square S1-S5   ██░               6.6 - 8.6%   ->  +7 to +9%
BERT B1/B4/B5  ████░            15.0 - 16.5%  ->  +18 to +20%
BERT B3        ███████████░           40.6%   ->  +68%
```

BERT receives roughly **double** the free boost Square receives. This alone accounts
for the observed split.

Verified: loop nests are the same B3A2C0 order, A repacked once per `(jc,pc,ic)` in
both; the only difference is instrumentation.

## 2. Root cause B: the report's reference line is not the archived data

`logs/report_2026-09-18/build_report_data.py:36-47` hand-codes the Square reference as
**8.5 / 8.7 / 8.9 / 9.0 / 9.0** — monotonically rising. The archived reference is
**8.32 / 8.53 / 8.51 / 8.31 / 8.36** — flat, peaking at S2.

Three independent sources agree on the flat curve:
- `ext/square_comparison.txt`
- `ext/output/retro_europar_plot.txt`
- the `xdsl_gflops` array in `ext/plot_square.py` that generated the published PDF
  (`tile_labels = ["32x7"]*5`, same pinning we use)

Every hand-typed value exceeds its archived counterpart (+0.18 to +1.04 GFLOPS). BERT
had 6–18% of real headroom so it still cleared the raised bar; Square had ~0% headroom
and dipped below.

The code comment says the figures came "directly from the user" and that the `ext/`
files are "stale". **That provenance is unverified and is the one open question in
this issue** (see §7).

## 3. Root cause C: the reference under-samples the Square family specifically

The reference does no warm-up: `while (time <= tmin) { nreps++; ... }` then divides by
`nreps` (`ext/.../src/test_gemm.c:340-360`), with `TIMIN=3.0` in `run_gemm.sh`.
At ~8 GFLOPS that yields:

```
B2, B3   1272 reps
B1         30 reps
B4, B5      8 reps
S1         13 reps
S2          2 reps      <- thin
S3          1 rep       <- single cold sample
S4          1 rep       <- single cold sample
S5          1 rep       <- single cold sample
```

Every BERT shape is well averaged. **S3/S4/S5 are single, un-averaged, cold runs.**
Our side reports a median of 5 timed reps taken after 5 discarded warm-up reps
(`driver_benchmark.cpp:564-569`, `:707`).

## 4. The numbers

```
                      KerGOPs   pack    e2e |  vs ARCHIVED ref (pack-inclusive)
case  shape           (chart)   frac   GOPs |   ref     chart     like4like
S1    1000^3             8.56   8.6%   7.83 |  8.32     +2.9%        -5.9%
S2    2000^3             8.27   8.0%   7.61 |  8.53     -3.1%       -10.8%
S3    3000^3             8.49   7.2%   7.86 |  8.51     -0.3%        -7.6%
S4    4000^3             8.29   7.0%   7.69 |  8.31     -0.2%        -7.5%
S5    5000^3             8.48   6.6%   7.91 |  8.36     +1.4%        -5.3%
B1    1024x384x1024      8.16  15.8%   6.87 |  7.38    +10.6%        -7.0%
B2    384x384x64        12.90   5.5%  12.10 | 12.20     +5.7%        -0.8%
B3    64x384x384         9.19  40.6%   5.79 |  8.53     +7.7%       -32.2%
B4    4096x384x1024      8.22  15.0%   6.99 |  7.46    +10.2%        -6.3%
B5    1024x384x4096      8.01  16.5%   6.68 |  6.80    +17.8%        -1.8%
```

Means vs archived: on `KerGOPs`, Square **+0.1%**, BERT **+10.4%**. On `e2e`, Square
**−7.4%**, BERT **−9.6%** (−4.0% excluding B3).

The dramatic "BERT beats reference by 10–18%" exists **only** on the pack-excluded
metric.

## 5. Secondary finding: B3 is the real performance problem, and the chart hides it

B3 (`64×384×384`, `MR16_NR15`) spends **40.6%** of its time packing. Pack-inclusive it
is **−32.2%** against the archived reference, while the current chart shows it at
**+4.4% above** reference. This is the shape that actually warrants attention, and the
metric is masking it. Related to the still-open residual in Issue #1 (§8 there).

## 6. Provenance finding: the archived tables are not from the reference design

`ext/square_comparison.txt` / `ext/bert_large_comparison.txt` are labelled with kernels
`32x7` and `16x15`. The reference design at `ext/RVV_ukernels_benchmark` ships **only
even-NR** Exo kernels (NR ∈ 2,4,…,32; MR ∈ 8,16,24,32,40,48) — it has no 32x7 or 16x15
kernel and cannot produce one without re-running Exo.

Those numbers came from `xdsl_sweep_driver/xdsl_test_gemm.c:357-417`, our own older
sweep driver, and are reported as **best-of-sweep**, not a single pinned config.

So the long-standing "reference design" comparison has been against an older xDSL
generation, not against the C/Exo reference at all.

## 7. Open question (blocks closing §2)

Were the Square figures **8.5 / 8.7 / 8.9 / 9.0 / 9.0** produced by a genuine re-run of
a reference, or transcribed from a chart? If transcribed, §2 resolves immediately and
the Square "gap" collapses to noise. No source for them exists in the repo.

## 8. Fairness caveats found while validating (for whoever quotes these numbers)

Biasing **against** us:
- **Fringe tiles.** Our dispatcher always runs the full MR×NR kernel on partial tiles —
  zero-padded pack plus `memset` and gather/scatter through `tmp_C`
  (`gemm_dispatch_xdsl.cpp:127-139`; `cur_mr`/`cur_nr` are accepted but unused). The
  reference has 384 specialized kernels covering every `(mr,nr)` from 1×1 to MR×NR and
  calls the right-sized one. ≥3.7% of tile executions at 1000³ (structural floor; the
  true figure depends on `mc`/`nc` alignment to `mr`/`nr`, which is not rounded).
- **Per-call timers.** ~36.6k `chrono::now()` pairs per rep at 1000³ ≈ 1–3% of measured
  `Compute_Time_ms`, which *deflates* our reported KerGOPs.

Biasing **for** us if the wrong reference variant is used:
- The reference's `loadAB/bcast` kernel materializes B into 6 vector registers, needing
  24 C + 4 A + 6 B = **34 live vector registers on a 32-register machine**, and provably
  spills. Compiled at `-O3 -march=rv64gcv -mabi=lp64d`, body of
  `gemm_RVV_32x6_b0_col_fp32`:
  ```
  bcast : 199 instructions,  6x vs1r.v + 6x vl1re32.v   (spills)
  macc  : 147 instructions,  0 vector spills
  ```
  The sibling `loadAB/macc` variant uses `vfmacc.vf` with B as a scalar operand —
  structurally identical to our AXPY kernel. **Always benchmark the reference at
  `macc`**, not `bcast`.

Confirmed equivalent (no bias): packed layout (`A_packed[k*MR+i]`, `B_packed[k*NR+j]`),
beta schedule (b0/b1, 24 C stores each), B3A2C0 loop order, and the blocking model —
`cache_config.py` is a straight port of the reference's `modelLevel/model_level.c`,
identical constants and clamp order. Neither side multiplies by `alpha`; both assume
alpha=1.

## 9. Tooling shipped: `tests/pure_xdsl_blis_rvv/ref_baseline/`

Reproduces our exact case set (shapes + pinned MR/NR) inside the real reference design
and emits a fresh baseline.

| file | role |
|---|---|
| `cases.tsv` | case set: layer, M/N/K, pinned MR/NR — single source of truth |
| `gen_ref_datasets.py` | splits into one `.dat` per built (MR,NR); reports kernel availability, predicted MC/NC/KC, and under-sampling warnings |
| `run_ref_baseline.sh` | pre-flight cross-build → stage → build+run per (MR,NR) → retrieve → compare |
| `compare_ref_baseline.py` | joins reference CSVs against `logs/run_*`, printing both `Ker` and `e2e` |
| `repair_ref_kernels.py` | repairs 6 truncated vendored kernel files (see below) |
| `blis_shim/blis.h` | lets the reference build without a real BLIS install (`FAMILY_EXO` only) |

Validated locally: generated `.dat` tab layout is byte-identical to the reference's own
`cnn_models/square.dat`; all 4 shapes × 2 variants cross-compile with
`/opt/riscv/bin/riscv64-unknown-linux-gnu-gcc`; the comparison join works.
**Not yet run on hardware.**

Notes:
- Only `32x6` (B1, B5) can be reproduced **exactly**. `32x7` and `16x15` are bracketed
  by nearest even-NR neighbours (32x6+32x8, 16x14+16x16). Never quote a `bracket` row
  as "the reference number".
- The reference unconditionally `#include`s `blis.h` and calls `bli_init()` /
  `bli_gks_query_cntx()` under `#if !defined(BLIS)`, but under `FAMILY_EXO` the BLIS
  ukernel is never invoked (`gemm_blis.c:117-133` dispatches via `ukrmatrix`; the
  `gemm_kernel(...)` call at `:139` is compiled out). Hence the shim.
- **6 of 4032 vendored kernel files are truncated upstream** — all `16x14/*/macc`,
  each missing the 16-byte prefix `#include "kernel` from line 1. `repair_ref_kernels.py`
  restores it in the *staged copy only* and refuses to run on any path containing
  `ext/`, so the vendored tree stays pristine.

## 10. Recommended next actions

**Report**
1. Plot `2mnk/(pack+compute)` alongside `KerGOPs`, or relabel the chart
   "compute only — packing excluded". As drawn the two bars are not comparable.
2. Restore the Square ref line to archived values, or drop it pending §11.
3. Surface B3 instead of letting the metric mask it.

**Measurement**
4. Run the baseline:
   ```bash
   REMOTE_PASS="…" REMOTE_HOST="fej" REMOTE_PORT="3322" \
     ./tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh \
       --jump --riscv-ip jbpi2 --bracket --variants 2,0 --tmin 156
   ```
   `--tmin 156` so S3–S5 are not single cold samples; `--variants 2,0` so the reference
   is judged at its best. ~15–25 min board time.

**Engineering, ranked by likely payoff**
5. Specialize fringe-tile kernels in the generator (the reference's approach) — a real
   win on any M/N not a clean multiple of MR/NR.
6. Bracket the `jr`/`ir` nest once per `ic` instead of per microkernel call — recovers
   1–3% at no fidelity cost.
7. Investigate B3's 40.6% pack fraction.

## 11. Closure criteria

- [ ] Provenance of the 8.5/8.7/8.9/9.0/9.0 Square figures resolved (§7)
- [ ] Fresh reference baseline run on `jbpi2` and committed
- [ ] Report chart redrawn on a like-for-like metric
- [ ] B3 pack-fraction finding split into its own issue
