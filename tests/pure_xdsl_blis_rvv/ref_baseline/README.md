# Reference-design baseline

Reproduces **our exact benchmark problem** — same layers, same M/N/K, same pinned
micro-kernel MR×NR — inside the real reference design at
`ext/RVV_ukernels_benchmark/gemm_blis_family`, and emits a fresh baseline to
compare against.

This replaces `ext/bert_large_comparison.txt` / `ext/square_comparison.txt`. Those
archived tables were **not** produced by this reference design: they came from an
older xDSL sweep driver (`xdsl_sweep_driver/xdsl_test_gemm.c`), and they report
*best-of-sweep* numbers. Their provenance is exactly what this directory removes
from the loop.

## Run it

```bash
# from the project root (RVV_code_gen_via_MLIR_xDSL/)
REMOTE_PASS="<pw>" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh \
    --jump --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 --bracket
```

`--dry-run` stops after generating the datasets and run plan, without touching the
board. Results land in `tests/pure_xdsl_blis_rvv/logs/refbase_<stamp>/`.

## Files

| file | role |
|---|---|
| `cases.tsv` | the case set: layer, M/N/K, pinned MR/NR. Single source of truth. |
| `gen_ref_datasets.py` | splits `cases.tsv` into one `.dat` per built (MR,NR); reports kernel availability and predicted MC/NC/KC |
| `run_ref_baseline.sh` | stages the reference tree on the board, builds one binary per (MR,NR), runs, retrieves, compares |
| `compare_ref_baseline.py` | joins the reference CSVs against our `logs/run_*` data |
| `blis_shim/blis.h` | lets the reference build without a real BLIS install (valid for `FAMILY_EXO` only — see the header comment) |

## Three things to know before trusting the output

### 1. The two harnesses do not report the same metric

| | formula | packing |
|---|---|---|
| reference | `2mnk / wall(gemm_blis_B3A2C0)` (`src/test_gemm.c:399-400`) | **included** |
| ours | `2mnk / total_kernel_time` (`templates/driver_benchmark.cpp:695`) | **excluded** |

The loop nests are otherwise the same B3A2C0 order with the same blocking model, so
the *only* difference is instrumentation. `compare_ref_baseline.py` therefore prints
both our `Ker` (pack-excluded, what the dashboards plot) and our `e2e`
(`2mnk/(vector_pack+compute)`). **Only `e2e` is comparable to the reference column.**

Because pack fraction varies a lot by shape (≈7% for Square, ≈15–16% for B1/B4/B5,
≈41% for B3), comparing `Ker` against the reference silently flatters the
pack-heavy shapes by 2–3× more than the pack-light ones.

### 2. The reference cannot build two of our three pinnings

Its Exo kernels are pre-generated and checked in at
`kernels/RVV_256_<OPT>/fp32/<MR>x<NR>/`, and only exist for **even NR** (2,4,…,32)
with MR ∈ {8,16,24,32,40,48}.

| pinning | cases | reference kernel |
|---|---|---|
| 32x6 | B1, B5 | **exists** — exact match |
| 32x7 | B4, S1–S5 | missing (odd NR) |
| 16x15 | B2, B3 | missing (odd NR) |

`--bracket` handles this by also building the nearest even-NR neighbours (32x6 and
32x8 for 32x7; 16x14 and 16x16 for 16x15), so the true reference value is bracketed
rather than skipped. Rows are labelled `exact` or `bracket` in the output — do not
quote a `bracket` row as "the reference number".

Generating true 32x7 / 16x15 reference kernels would require re-running Exo, which
is not vendored here.

### 3. Blocking is derived, not forced

The reference computes MC/NC/KC itself from MR/NR via
`modelLevel/model_level.c:get_optim_mc_nc_kc`, clamped to the shape. Our
`cache_config.py` is a straight port of that function — identical cache constants
(NL1=128, CL1=64, WL1=4, …) and identical clamp order — so both sides pick the same
blocking for the same MR/NR automatically. `gen_ref_datasets.py` prints the
predicted MC/NC/KC so any divergence from what the reference actually prints at
runtime is immediately visible.

This was not always true: `modelLevel/archs/bananapi.h`'s `WL2` briefly drifted to
16 (commit `40d1ea8`, meant to reflect the real board's L2 associativity), which
silently changed MC/NC on 8 of 10 comparison shapes while `cache_config.py` kept
predicting the reference's blocking (`WL2=8`). Settled in issue #12 by reverting
`WL2` back to 8 — parity over hardware-fidelity, since the whole point of this
comparison is that the two sides differ only in packing and microkernel
generation. `bananapi.h` and `cache_config.py` must be kept in sync; see the
comment on `cache_config.py:WL2`.

## Variant selection — use `macc`, not `bcast`

The reference ships three kernel families per shape under `loadAB/`. Two of them
are *not* equivalent to our `AXPY_FP32_Pack` kernel, and picking the wrong one
silently handicaps the reference:

| variant | `--gather` | inner loop (32x6, per k) |
|---|---|---|
| `macc` | `2` (**default**) | 4 `vle32` + 24 `vfmacc.vf` — B stays a scalar operand |
| `bcast` | `0` | 4 `vle32` + 6 `vfmv.v.f` + 24 `vfmacc.vv` — B materialized into 6 vector regs |
| `gather` | `1` | gather-load variant, not comparable to AXPY |

`bcast` needs 24 C accumulators + 4 A + 6 B = **34 live vector registers on a
32-register machine**, so it spills. Verified by compiling
`kernels/RVV_256_OPT/fp32/32x6/loadAB/{bcast,macc}/kernels_RVV_32x6_fp32.c` with
`riscv64-unknown-linux-gnu-gcc -O3 -march=rv64gcv -mabi=lp64d` and counting the
body of `gemm_RVV_32x6_b0_col_fp32`:

```
bcast : 199 instructions,  6x vs1r.v + 6x vl1re32.v   (spills)
macc  : 147 instructions,  0 vector spills
```

`macc` is the variant that structurally matches our design (`vfmacc.vf`, B as a
scalar), so it is the default. **Benchmarking the reference at `bcast` would
credit our design with a ~26% instruction-count advantage that is an artifact of
variant choice, not of our codegen.**

`--variants 2,0` runs several and keeps the best per case (the `kern` column in
the comparison says which won), which is the fairest reading of "the reference
design's performance". `--opt BASE|LDX|ALL` and `--swap 1` select the other
families.

## Known bias caveats to quote alongside any number

1. **Fringe tiles penalize us.** Our dispatcher always runs the full MR×NR kernel
   on partial tiles — zero-padded pack, plus a `memset` and gather/scatter through
   a `tmp_C` buffer (`gemm_dispatch_xdsl.cpp:127-139`; `cur_mr`/`cur_nr` are
   accepted but unused). The reference has a specialized kernel for every (mr,nr)
   from 1x1 up to MRxNR and calls the right-sized one
   (`ukrmatrix[mr][nr][be]`, `gemm_blis.c:119`). So we do strictly more work
   on every fringe tile.

   The extra multiply-adds, as a share of the real work
   (`ref_baseline/fringe.py`; the `fringe` column of `compare_ref_baseline.py`),
   each block of M and N padding its own last tile at the blocking our binary runs
   (section 3, plus the `nc_overrides` in `config.json`):

   | case | B1 | B2 | B3 | B4 | B5 | S1 | S2 | S3 | S4 | S5 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | extra MACs | 0.00% | 1.56% | 5.47% | 0.26% | 0.00% | 2.50% | 1.25% | 0.60% | 0.45% | 1.00% |

   B3 is the high one because `config.json` pins its `nc` to 128 (issue #1): three
   128-wide blocks each pad to 135, so 405 columns of work for 384. At the
   heuristic's own `nc=384` it would be 1.56%, which is what B2 gets, and what the
   reference's blocking gives B3. A run made with `--no-nc-overrides` (issue #16) is
   that configuration: give `compare_ref_baseline.py` its `*_no_nc_override.txt` log
   with `--raw-name` and the `fringe` column drops B3 to 1.6%. `RVV_FORCE_NC` / `--force-nc` change these figures and are
   not modelled.

   That counts multiply-adds only. The `memset` and the copy in and out of `tmp_C`
   cost time on top and are not in these figures. Dropping the `tmp_C` round trip
   (writing the masked tile straight into C) would be the cheap partial fix; it is
   deferred with the right-sized kernels (issue #10), so quote these percentages
   next to any comparison instead.
2. **Our per-call timers cost us — far more than this file used to claim.** We
   bracket every microkernel invocation with a `chrono::now()` pair, ~36.6k
   calls/rep at 1000^3. The 1-3% figure previously quoted here was inferred from
   `RVV_Time_ms − (Vector_Pack_Time_ms + Compute_Time_ms)`, which only sees the
   sliver of the pair that falls *between* tiles. Measured directly by building
   both ways (issue #14, `make INSTRUMENTED=0`), the real cost on jbpi is:

   | case | instrumented `RVV_Time_ms` | uninstrumented | tax |
   |---|---|---|---|
   | B1 (1024x384x1024) | 118.8-119.5 | 98.6-101.4 | 14.6-17.0% |
   | B5 (1024x384x4096) | 484.0-492.8 | 411.6-423.0 | 13.6-14.9% |
   | B2 (384x384x64)    | 1.686 (mean of 5) | 1.500 (mean of 5) | 11.0% |

   Ranges are across two A/B sessions on `jbpi`; the spread is board noise, the
   sign and magnitude are stable.

   A `chrono::now()` pair costs ~160 ns on this board (measured), which for B1's
   24576 tiles/rep is only ~3.9 ms of the ~17-20 ms removed. The rest is the cost
   of having an opaque external call sitting in the innermost loop; it cannot be
   estimated from the timers themselves, only compiled out. B3 is excluded from
   the table: it is the shape with the packing instability of issue #1 and its
   uninstrumented numbers swing run-to-run (2.07-3.94 ms) rather than settling.

   So any pack-inclusive number we quote from an instrumented build is **deflated
   by roughly 11-17%**, not 1-3%. Use `--no-instr` for a headline figure.
3. **The reference does no warm-up** and averages over however many reps fit in
   `TIMIN`. See the under-sampling warning `gen_ref_datasets.py` prints. Since
   issue #27 our uninstrumented build does the same **by default** (the reference
   protocol; `--tmin` matches `TIMIN`), and warns the same way when fewer than 3 reps
   fit. `--no-ref-protocol` gives the older warm-up protocol. Measured effect of the
   warm-up on B1/B5/B2: under 1% (`docs/metrics.md`).
4. **Neither side multiplies by alpha** — both kernels ignore the `alpha` argument
   and assume alpha=1. Fine for this benchmark (alpha=1, beta=0 on both sides),
   but neither is a general GEMM.
