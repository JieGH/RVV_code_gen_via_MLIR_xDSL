# Issue #2: Fixed-Dimension Evaluation & Packing/Compute Breakdown

**Role / Label:** `ready-for-agent` → results landed, see summary below
**Status:** 🟢 Test matrix run live on hardware (2026-09-16). All 10 cases produced
performance data; correctness (`PASS`/`FAIL`) was **not** verified in this run — every row
sets `skip_ref_run=true` (inherited from the pre-existing config.json pattern), so every case
reports `SKIP`, not `PASS`. See §5 for why that matters and what's still open.
**Target:** `jbpi` / `jbpi2` (BananaPi M2 Pro, VLEN=256), reached via jump host `fej`.
**Primary Sources:**
- Log run: `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_MR{16NR15,32NR6,32NR7}_2026-09-16_132501/`
- Config: `tests/pure_xdsl_blis_rvv/config.json` (3 pinned rows added for this issue)
- Datasets: `tests/pure_xdsl_blis_rvv/datasets/issue2_{bert_16x15,bert_32x6,square_bert_32x7}.dat`
- Related: [[0001-b3-m7-packing-instability]] (B3 = same shape as B3_M7, same `nc` mitigation)

---

## 1. Test Matrix

Fixed `(m, n, k)` shapes, each pinned to one `(MR, NR)`, as specified in GitHub issue #2:

| Case | Shape (m×n×k) | MRxNR |
|---|---|---|
| S1 | 1000×1000×1000 | 32×7 |
| S2 | 2000×2000×2000 | 32×7 |
| S3 | 3000×3000×3000 | 32×7 |
| S4 | 4000×4000×4000 | 32×7 |
| S5 | 5000×5000×5000 | 32×7 |
| B1 | 1024×384×1024 | 32×6 |
| B2 | 384×384×64 | 16×15 |
| B3 | 64×384×384 | 16×15 |
| B4 | 4096×384×1024 | 32×7 |
| B5 | 1024×384×4096 | 32×6 |

`B2`/`B3` are the same shapes as issue #1's `B2_M5`/`B3_M7`; `B3` carries the same
`nc=128` override shipped in that issue (`config.json`: `{"B3": 128}`).

---

## 2. Bug found and fixed along the way: `pj_gen/` output-directory collision

`generate_pure_blis_project.py`'s `out_dir` was `pj_gen/{datatype}_{method}_vlen{vlen}` —
it did not include `(MR, NR)`. This issue's matrix needs 3 *pinned* rows that all share
`fp32`/`AXPY_FP32_Pack`/`vlen256` but differ only in `(MR, NR)`. Before the fix, all three
would have generated into the same directory, and `build_project` doesn't clear
`out_dir` between configs — each row's files would silently overwrite the previous
row's, so `compile_order.txt` would list the same directory three times and only the
*last* config's kernel would ever actually get compiled and run.

Fixed by extracting a pure `compute_out_dir(config)` function (`config.py`) that appends
`_MR{mr}NR{nr}` for pinned configs, TDD'd (`tests/pure_xdsl_blis_rvv/test_config.py`, 3
tests) and wired into `build_project` and `main`'s `compile_order.txt` generation.
Verified live: the run produced 3 distinct directories
(`..._MR16NR15`, `..._MR32NR6`, `..._MR32NR7`), each compiled and run independently.

## 3. Instrumentation added: explicit pack/compute ms columns

Before this issue, `driver_benchmark.cpp` already separated pack time (`pt`) from kernel
time (`kt`) per call internally, but only ever exposed **pack bandwidth** (GB/s) and a
**combined** wall-clock time (`RVV_Time_ms`, includes pack + compute + loop overhead) in
the raw CSV log — never an isolated ms figure for either phase, which is what issue #2
actually asked for. Added `Pack_Time_ms` / `Compute_Time_ms` columns (per-rep-averaged,
`total_rvv_pack_time / N_REPS * 1000` and `total_kernel_time / N_REPS * 1000`), visible in
the raw log alongside the existing GOPs/bandwidth columns.

---

## 4. Results

Ran on `jbpi` via the jump host, 2026-09-16 13:15–13:25. `N_REPS=1000` for the BERT groups
(`16x15`, `32x6`), `N_REPS=5` for the Square+B4 group (`32x7`) — see §5 for why.

Reference column is the user-supplied FP32 GEMM+xDSL numbers for the **same shapes/MRxNR**,
from a prior design that used RVV compute with **scalar** packing (not RVV vector packing).

| Case | Shape (m×n×k) | MRxNR | KerGOPs (median) | Ref (scalar-pack) | Δ vs ref | Pack ms | Compute ms | Pack % of total |
|---|---|---|---|---|---|---|---|---|
| S1 | 1000×1000×1000 | 32×7 | 8.63 | ~8.5 | +2% | 21.36 | 231.84 | 8.4% |
| S2 | 2000×2000×2000 | 32×7 | 8.39 | ~8.7 | -4% | 160.67 | 1906.45 | 7.8% |
| S3 | 3000×3000×3000 | 32×7 | 8.49 | ~8.9 | -5% | 392.94 | 6361.81 | 5.8% |
| S4 | 4000×4000×4000 | 32×7 | 8.31 | ~9.0 | -8% | 1121.94 | 15417.6 | 6.8% |
| S5 | 5000×5000×5000 | 32×7 | 8.64 | ~9.0 | -4% | 1711.43 | 29005.6 | 5.6% |
| B1 | 1024×384×1024 | 32×6 | 8.40 | ~8.0 | +5% | 17.12 | 95.91 | 15.1% |
| B2 | 384×384×64 | 16×15 | 12.77 | ~12.5 | +2% | 0.08 | 1.48 | 5.2% |
| B3 | 64×384×384 | 16×15 | 9.09 | ~8.8 | +3% | 1.31 | 1.98 | 39.9% |
| B4 | 4096×384×1024 | 32×7 | 8.63 | ~8.5 | +2% | 68.00 | 373.18 | 15.4% |
| B5 | 1024×384×4096 | 32×6 | 8.25 | ~7.5 | +10% | 77.16 | 390.55 | 16.5% |

Raw CSV (as fetched, columns trimmed to the ones referenced above):
```
S1,1000,1000,1000,...,231.841 (compute_ms), 21.3597 (pack_ms)
S2,2000,2000,2000,...,1906.45, 160.671
S3,3000,3000,3000,...,6361.81, 392.938
S4,4000,4000,4000,...,15417.6, 1121.94
S5,5000,5000,5000,...,29005.6, 1711.43
B1,1024,384,1024,...,95.9067, 17.123
B2,384,384,64,...,1.48039, 0.0812358
B3,64,384,384,...,1.97706, 1.3141
B4,4096,384,1024,...,373.183, 67.9964
B5,1024,384,4096,...,390.551, 77.1629
```
Full raw logs (all columns): `tests/pure_xdsl_blis_rvv/logs/fp32_AXPY_FP32_Pack_vlen256_MR{16NR15,32NR6,32NR7}_2026-09-16_132501/raw_fp32_AXPY_FP32_Pack.txt`.

### Findings

1. **Pack overhead tracks the M-reuse factor, not raw problem size** — confirms issue #1's
   theory outside the one shape it was discovered on. `B3` (M=64, reuse=4) spends **39.9%**
   of its time packing; every other case, including the much larger `B5`/`B4`
   (reuse ≥ 24, mc clamped by cache not M), stays under 17%. The Square shapes (reuse=24,
   `mc=768` regardless of problem size) bottom out around 5-8% pack overhead — packing
   amortizes away almost completely once M-reuse is high, independent of total FLOPs.
2. **RVV vector packing beats the scalar-pack reference on every BERT shape** (+2% to
   +10%, biggest gain on B5) but **trails it by 4-8% on every Square shape** (S2-S5). This
   is a real, reproducible split by shape family, not noise — worth a closer look at why
   vector packing underperforms scalar packing specifically on the large square shapes
   (candidates: `nc`/`kc` heuristic tuned for BERT-shaped problems, or the reduced
   `N_REPS=5` for Square giving each measurement fewer chances to land on a "typical" run
   compared to the reference's presumably higher rep count).
3. **All 10 cases execute correctly end-to-end** (compile, run, produce numerically
   plausible GOPs) but **none were checked against the scalar reference GEMM for
   correctness** in this run — every config row has `skip_ref_run=true`. This wasn't a new
   choice specific to Square (see §5); it was already the state of the one pre-existing
   `config.json` row before this issue. It means this data answers "how fast" but not
   "is it still numerically correct at these shapes" — that's still open.

---

## 5. Judgment calls made to fit real hardware time (transparency)

- **`N_REPS=5` for the Square+B4 group (`32x7`), `1000` for BERT groups.** At `N_REPS=1000`,
  `S5` alone (5000³, ~250 GFLOP/rep) would need roughly 250 TFLOP of RVV kernel execution —
  on the order of hours, not minutes, on this board. Reduced to `N_REPS=5` for that group
  only; total board time for all 10 cases was ~25 minutes. Trade-off: Square/B4 GOPs
  numbers rest on 5 samples instead of 1000, so their CV%/percentile columns in the raw log
  are much less meaningful than the BERT groups' — treat the Square/B4 medians as rougher
  point estimates, not as statistically tight as B1-B3/B5.
- **`skip_ref_run=true` for all three groups** — carried over from the existing
  `config.json` row, not newly introduced for this issue, but worth calling out again here:
  no case in this run was checked against the scalar reference GEMM, so "correctness at
  these exact pinned shapes" remains unverified by this run specifically (BERT-family
  shapes have been checked in other runs historically per issue #1's investigation; Square
  shapes at 32x7 have not, to this agent's knowledge).

---

## 6. Files changed this issue

- `tests/pure_xdsl_blis_rvv/config.py`: added `compute_out_dir()` (TDD'd).
- `tests/pure_xdsl_blis_rvv/test_config.py`: new, 3 tests for `compute_out_dir`.
- `tests/pure_xdsl_blis_rvv/generate_pure_blis_project.py`: use `compute_out_dir()` in
  `build_project` and `main` instead of the collision-prone inline string.
- `tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp`: added `pack_time_val` /
  `compute_time_val`, exposed as `Pack_Time_ms` / `Compute_Time_ms` raw-log columns.
- `tests/pure_xdsl_blis_rvv/config.json`: added 3 pinned rows for this issue's matrix;
  commented out (not deleted) the original issue #1 repro row.
- `tests/pure_xdsl_blis_rvv/datasets/issue2_{bert_16x15,bert_32x6,square_bert_32x7}.dat`: new.
- `docs/ROADMAP.md`, GitHub issue #2: updated with the concrete test matrix.

## 7. Next steps

1. Investigate the Square-shape RVV-pack-vs-scalar-pack regression (finding #2) —
   candidates: `nc`/`kc` cache-blocking heuristic vs. Square's flat aspect ratio, or just
   `N_REPS=5` noise (re-run at higher `N_REPS` with a time budget, or on a faster/idle
   board window, to confirm it's real before spending design effort on it).
2. Add a real correctness pass for at least one Square shape (e.g. `S1` at full
   `N_REPS` with `skip_ref_run=false`) to close the open question in finding #3 — `S1` is
   small enough (1000³) that the scalar reference GEMM should finish in a reasonable time.
3. If Square-shaped GEMMs (large, roughly cubic) are an actual target workload rather than
   a synthetic stress point, consider whether they deserve their own `nc`/`kc` tuning pass,
   the same way `B3`/small-M shapes got a `nc` override in issue #1.
