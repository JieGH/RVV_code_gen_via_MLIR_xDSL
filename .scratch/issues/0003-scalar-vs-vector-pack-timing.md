# Issue #7: Isolate & Compare Scalar-Pack vs RVV Vector-Pack Timing

**Role / Label:** `ready-for-agent` → results landed, see summary below
**Status:** 🟢 Correctness proven (both host-native and on-board), full 10-case matrix
re-run live on hardware (2026-09-16) with the new `Vector_Pack_Time_ms`/
`Scalar_Pack_Time_ms` columns.
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256), reached via jump host `fej`.
**Primary Sources:**
- Correctness test: `tests/pure_xdsl_blis_rvv/test_pack_correctness.py`
- Log run: `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_MR{16NR15,32NR6,32NR7}_2026-09-16_151844/`
- Related: [[0002-fixed-dim-pack-compute-breakdown]] (this issue is its Track 1 sub-issue,
  investigating finding #2 and next-step #1 from that doc)

---

## 1. Correctness: pack functions vs. the canonical BLIS reference

Before trusting any pack-timing comparison, proved `pack_RB_scalar`/`pack_CB_scalar`
and the generated RVV `pack_RB_v_if`/`pack_CB_v_if`/`_else` functions match the
canonical BLIS layout (`pack_RB`/`pack_CB` in
`ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c`), via two TDD'd tests
in `test_pack_correctness.py`:

1. **`test_scalar_pack_matches_canonical_blis_reference`** (host-native, no RVV
   hardware needed): compiles `pack_RB_scalar`/`pack_CB_scalar` — via a new
   `render_scalar_pack_functions()` extracted from `generate_pure_blis_project.py` so
   the test exercises the exact same source the benchmark pipeline generates, not a
   hand-copied duplicate — against a reference oracle transcribed independently from
   `gemm_blis.c`'s `pack_RB`/`pack_CB` (both the `orderM=='C'` and `else` branch), run
   locally with a native compiler. **PASS.**
2. **`test_rvv_vector_pack_matches_scalar_reference_on_board`**: builds the real RVV
   `pack_CB_v_if`/`pack_CB_v_else`/`pack_RB_v_if`/`pack_RB_v_else` through the same
   xDSL → EmitC → `mlir-translate` pipeline `generate_pure_blis_project.py` uses, with
   `RR=15` (not a multiple of `vlen=8` for VLEN=256 fp32 — exercises the vector tail
   chunk), cross-builds+runs the comparison on the real board via the `fej` jump host,
   and skips gracefully if the board isn't reachable. **PASS.**

**Finding:** `pack_RB_v_if`/`pack_CB_v_if` match `pack_RB_scalar`/`pack_CB_scalar`
exactly (both implement the reference's `orderM=='C'`/`transM=='N'` branch — the only
branch the benchmark dispatcher ever calls). `pack_RB_v_else`/`pack_CB_v_else` also
match the canonical reference correctly, but implement the reference's *other*
branch (row-major source access) — **they are generated and lowered but never wired
into `gemm_dispatch_xdsl.cpp`'s dispatch tables** (only `_if` is referenced there, for
both pinned and search-mode dispatch). This is dead-but-correct code, not a bug: this
project always packs column-major, non-transposed `A`/`B`, so the `_if` branch is the
only one ever exercised in production. Worth knowing if a future issue considers
pruning it or ever needs the transposed/row-major packing path.

Both tests genuinely detect mismatches — verified by deliberately injecting an
off-by-one into the oracle and confirming each test fails with a `[MISMATCH]` report,
then reverting.

---

## 2. Instrumentation: explicit scalar-pack timing column

- Renamed the issue #2 `Pack_Time_ms` raw-log column to `Vector_Pack_Time_ms`
  (breaking rename — historical logs under `tests/pure_xdsl_blis_rvv/logs/` predate
  this and should be read with the old name in mind).
- Added `Scalar_Pack_Time_ms`, reusing the scalar-pack-only timing pass that already
  ran on every benchmark row to produce `Scl_Pack_GBs` (no new timing pass): now also
  exposes `(total_scalar_pack_time / N_REPS) * 1000.0` in ms, matching the existing
  `Vector_Pack_Time_ms`/`Compute_Time_ms` per-rep-averaged convention.

---

## 3. Test Matrix Re-Run

Re-ran the exact same 10-case Square (S1-S5) / BERT (B1-B5) matrix from issue #2
(same `config.json`, same pinned MR/NR, same `N_REPS`), now with the new columns.

| Case | Shape (m×n×k) | MRxNR | KerGOPs (median) | Scalar Pack ms | Vector Pack ms | Compute ms | Vector Pack Gain % |
|---|---|---|---|---|---|---|---|
| S1 | 1000×1000×1000 | 32×7 | 8.82 | 23.01 | 21.35 | 226.6 | **+7.2%** |
| S2 | 2000×2000×2000 | 32×7 | 8.53 | 179.40 | 161.10 | 1876.7 | **+10.2%** |
| S3 | 3000×3000×3000 | 32×7 | 8.73 | 448.14 | 389.69 | 6186.6 | **+13.0%** |
| S4 | 4000×4000×4000 | 32×7 | 8.52 | 1253.56 | 1124.27 | 15015.4 | **+10.3%** |
| S5 | 5000×5000×5000 | 32×7 | 8.39 | 1966.69 | 1705.45 | 29806.9 | **+13.3%** |
| B1 | 1024×384×1024 | 32×6 | 8.60 | 19.48 | 17.12 | 94.0 | **+12.1%** |
| B2 | 384×384×64 | 16×15 | 13.12 | 0.14 | 0.08 | 1.4 | **+44.7%** |
| B3 | 64×384×384 | 16×15 | 11.02 | 1.05 | 1.18 | 1.8 | **-12.7%** |
| B4 | 4096×384×1024 | 32×7 | 8.25 | 75.06 | 67.79 | 390.7 | **+9.7%** |
| B5 | 1024×384×4096 | 32×6 | 8.22 | 80.05 | 73.17 | 391.8 | **+8.6%** |

Gain % = `(Scalar_Pack_Time_ms − Vector_Pack_Time_ms) / Scalar_Pack_Time_ms × 100`.
Positive = vector packing faster; negative = vector packing slower.

Full raw logs: `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_MR{16NR15,32NR6,32NR7}_2026-09-16_151844/raw_fp32_AXPY_FP32_Pack.txt`.

### Findings

1. **The Square-shape regression from issue #2 is *not* a packing-time effect.**
   Issue #2 found RVV vector packing trailing a scalar-pack *reference design's*
   total-GOPs by 4-8% on every Square shape (S2-S5). This run measures scalar vs.
   vector pack time directly, head-to-head, in the same binary/board conditions —
   and vector packing is **faster than scalar packing on every single Square shape**
   (+7.2% to +13.3%), actually *more* consistently than most BERT shapes. Since pack
   time is only 5-9% of Square's total wall time (see table above), and vector
   packing isn't the slower path, packing cannot explain the Square total-GOPs gap
   issue #2 found. That gap's cause is elsewhere — most likely the external
   reference numbers (issue #2's "ref" column was a different, user-supplied design
   run under different, uncontrolled conditions, not measured in the same run) or
   `N_REPS=5` sampling noise on the Square group, not the packing implementation.
   This answers issue #2's open next-step #1 and closes the "is it packing?"
   question: **no, it isn't.**
2. **`B3` is the one shape where vector packing is slower than scalar (-12.7%).** B3
   is the smallest/highest-pack-overhead shape in the matrix (`mc=64, kc=128`,
   39.4% of its total time in packing — see [[0001-b3-m7-packing-instability]] and
   issue #2 finding #1 on the same shape). At this block size the RVV path's fixed
   per-call overhead (vsetvl, the two-chunk loop needed since `RR=15` isn't a
   multiple of `vlen=8`) is large relative to the small amount of data actually
   moved, letting the scalar loop's simpler control flow win. This is the opposite
   direction from every other case and is worth a closer look if B3-shaped
   (small-M, high-reuse) workloads are a real target, not just a stress point.
3. **`B2` shows the largest vector-packing win (+44.7%)** but on an absolute time of
   under 0.15ms — a real result, but too small in absolute terms to matter for
   total wall-clock time on that shape (pack is only 5.2% of B2's total).
4. Every row still has `skip_ref_run=true` (unchanged from issue #2), so
   correctness of the macro-kernel/microkernel compute path at these exact shapes
   remains unverified by this run — this issue only proved *pack-function*
   correctness (§1), independent of the macro-kernel loop.

---

## 4. Future-work note: trimmed the Square dataset

`S3`-`S5`'s `KerGOPs`/pack-vs-compute split all land in the same narrow band as
`S1`/`S2` regardless of problem size (8.4-8.8 GOPs, 5-13% pack-of-total across all
five) — the Square family's behavior is already fully captured by `S1`+`S2`. Per
user direction, commented out `S3`-`S5` (not deleted) in
`tests/pure_xdsl_blis_rvv/datasets/issue2_square_bert_32x7.dat` to cut board time on
future re-runs of this matrix; re-enable if a future issue needs the larger-size
data points specifically.

---

## 5. Files changed this issue

- `tests/pure_xdsl_blis_rvv/test_pack_correctness.py`: new, 2 tests (host-native
  scalar-vs-reference, on-board RVV-vs-reference), TDD'd against injected bugs.
- `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`: extracted
  `render_scalar_pack_functions()` (pure refactor, verified byte-identical
  regenerated output) so the correctness test and the production generator share one
  source of truth for the scalar pack functions.
- `tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp`: renamed
  `Pack_Time_ms` → `Vector_Pack_Time_ms`; added `Scalar_Pack_Time_ms` from the
  already-running scalar-pack timing pass.
- `tests/pure_xdsl_blis_rvv/pj_gen/*/driver_benchmark.cpp`: regenerated to match.
- `tests/pure_xdsl_blis_rvv/datasets/issue2_square_bert_32x7.dat`: commented out
  S3-S5 for future re-runs (see §4).
- `docs/ROADMAP.md`, GitHub issue #7: updated with this summary.

## 6. Next steps

1. If B3-shaped (small-M, high pack-reuse) workloads are a real target, investigate
   why vector packing loses to scalar there specifically (finding #2) — candidate:
   amortize/skip the vector path below some block-size threshold.
2. Issue #8 (non-wall-clock, more accurate performance metric) remains open and
   unaffected by this issue's findings.
3. Correctness of the macro-kernel/microkernel compute path at large shapes
   (`skip_ref_run=true` throughout) is still unverified — carried over from issue #2,
   still open.
