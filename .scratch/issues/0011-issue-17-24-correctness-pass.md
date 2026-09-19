# Issues #17 + #24 (local): the correctness check, run on the board, for all three pinnings

**Role / Label:** `done` — the check ran on the board and passed; GitHub #17 and #24 are **not** closed or commented on (left to the owner).
**Status:** 🟢 9 of 9 rows `PASS`, twice, and the vector packers match the oracle at all five shipped `RR`. The shipped shapes (B1–B5, S1–S5) were **not** run through the oracle; see §4.
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256), reached via `fej`.
**Primary Sources:**
- GitHub #17 (finding 6) and #24 (finding 13), children of #11. Ranked "Tier 1 … whether the result is real" in [[0009-issue-11-midpoint-the-result-flipped]] §5.
- `tests/pure_xdsl_blis_rvv/config_correctness.json`, `datasets/correctness_small.dat`, `test_correctness_pass.py`; edits to `test_pack_correctness.py`, `config.json` (comment), `docs/agents/execution.md` ("Correctness pass").
- Commit `7b94543` on `feat/combined`.
- Board runs `logs/run_2026-09-19_195354/` (core 4) and `logs/run_2026-09-19_195900/` (`--no-nc-overrides --cpu 0`).

---

## 1. What the two issues asked for, and what was done

| the issue said | done |
|---|---|
| #17.1 fix the brace structure in `gemm_ref` | **Nothing to fix.** The braces were already right (commit `d6e0a4d`, 2026-07-11, two months before #17 was filed). Diffed against `test_gemm.c:442-610`: the only differences are the element type, `tmp = 0` for `0.0`, the quick-return block written inline instead of `#include "quick_gemm.h"`, and the reference's `printf`+`exit` on an invalid `transA/transB`. `beta` is guarded by `if (beta==zero)` in all four branches. #17's reason is also off: `alpha*tmp + beta*C` is the correct formula for any `beta`, so unguarded code is wrong only at `beta = 0` when `C` holds NaN/Inf (BLAS says `C` is not read then), and the driver zeroes `Cg` before every oracle call. A test now pins the guard (§2). |
| #17.2 turn `SKIP_REF_RUN` off, one small shape per pinning | A separate config, not an edit of `config.json` (whose datasets include 5000³, where the scalar oracle is hours). Same three pinnings, `skip_ref_run` false, three shapes (§2). |
| #17.3 / #24 parameterise the pack test over `[6, 7, 15, 16, 32]` | Done: `SHIPPED_RR`, and the harness now runs four `(mc, kc)` shapes per `RR` so a partial block ends in each vector chunk (§3). |
| #24 also: test the buffer "as the benchmark produces it (memsets + pack)" | The memsets no longer exist (#18 removed them). What the benchmark produces is now checked end to end by the driver run: packers, kernel and fringe path together. |

## 2. The correctness pass

`config_correctness.json` has the three shipped pinnings (16x15, 32x6, 32x7), `skip_ref_run` false, one dataset. Never merged into `config.json`, so a shipped benchmark run is unchanged.

`datasets/correctness_small.dat` (all B2-sized or smaller; every pinning gets all three):

| label | M×N×K | what it reaches |
|---|---|---|
| `C_full` | 96×210×128 | every tile full (96 = 3·32 = 6·16, 210 = 14·15 = 35·6 = 30·7), one K block (k = kc) |
| `C_edges` | 123×57×300 | row tail 27 at MR=32 (inside the **4th** vector chunk), 11 at MR=16; column tail 12 at NR=15 (2nd chunk), 3 at NR=6, 1 at NR=7; k = 128+128+44, so the beta=1 kernel runs on the 2nd and 3rd K block |
| `C_tiny` | 12×5×17 | the whole matrix inside one partial tile, k < kc |

Host tests, `test_correctness_pass.py` (10, none need the board):
- `gemm_ref`, taken from the real `driver_benchmark.cpp` (its `main` renamed), against a naive triple loop at eight `(alpha, beta)` pairs, including `beta ≠ 0`, `alpha = 0`, and a `beta = 0` call whose `C` starts as NaN. **It passed at once** (the bug was already gone), so it was checked for sensitivity: with the guard replaced by `if (0)` in a scratch copy of the template it fails (`got nan want 5.125`). Only the NaN-`C` case can fail it, since for finite `C` the unguarded formula gives the same answer. The tracked template was not touched.
- the config covers every pinning in `config.json`, all pinned, all with the check on;
- no config row may turn the check on for a shape bigger than B2 (checked for `config.json` too);
- per pinning: a partial panel whose tail reaches the last vector chunk (both the M and N axis), a full tile, a shape with k > kc, a shape inside one partial tile. `kc` comes from `cache_config`, the Python copy of the driver's model.

**Board result, both runs: 9/9 `PASS`.** `--no-nc-overrides --cpu 0` is the run quoted in `docs/agents/execution.md`; `config_correctness.json` has no nc overrides, so the flag changes nothing but the log name. The pinned core was confirmed from `provenance.txt` (affinity list `0`).

## 3. The pack test

`test_rvv_vector_pack_matches_scalar_reference_on_board` is now `[RR6, RR7, RR15, RR16, RR32]`, all green on the board (6.1–6.4 s each, i.e. compiled and run on `jbpi`; a skip would show as `s` and take no time). Per `RR`, four shapes: `mc` = 37, 59, 51, 96 (kc 22, 22, 9, 7): tails that end in the 1st chunk, in the 4th (59 → 27 at RR=32), in the 3rd (51 → 19), and none at RR 6, 16 and 32 (96 is a multiple of each; at RR=15 and 7 it still leaves a tail). RR=32 is the only one with 3rd and 4th chunks; they now match the oracle both as whole chunks and as partial ones. The test file now also puts its own directory on `sys.path`, so it runs on its own (it only imported when another test module had already done that). `pack_CB_v_if` and `pack_RB_v_if` (the ones the dispatcher calls) matched at every shape.

**A latent bug in the existing test, found by the new shapes.** The first run failed everywhere, on `pack_RB_v_else` only (the variant no dispatcher calls). Cause: the `_else` packers read their source row-major, `M[(i+ii)*ldM + j]`, and the harness sized the source `ldM*cols` (column-major). At `mc > kc` the read goes past the buffer — including at the old 37×22 shape (worst index 1461 in an 880-float buffer). It passed only because what sat past the buffer read the same on both calls; my extra shapes changed the heap layout and the two reads disagreed. Sources are now `ldM * max(kc, mc)` and zeroed. The failure output also now names the `RR` and shape.

## 4. What this does not establish

- **`PASS` is a tolerance**, `‖C_ours − C_ref‖₂ / ‖C_ref‖₂ ≤ max(1e-5, k·1e-6)` on random [0,1] data, `alpha = 1`, `beta = 0`. At `C_edges` (bound 3e-4) one wrong `k`-step in one entry is about 5e-5 and would pass; a wrong 8-row chunk is at least about 4e-2. The pack test is the bitwise check, and it covers the packed layout only.
- **The shipped shapes were not run through the oracle.** B1–B5 and S1–S5 are checked by implication (same pinnings, same code, small shapes). What the small shapes do not reach: `ic`/`jc` blocking (mc and nc are ≥ m and n at every shape here, so those loops run once), and B3's shipped `nc = 128` override. [[0008-ref-protocol-warmup-effect-measured]] §2b recorded `PASS` for B1, B2, B3, B5 on ad hoc copies of the projects; this is the first `PASS` on the shipped-generator MR32NR7.
- **#24's "never verified" was too strong for MR=32 in the compute path**: B1 and B5 (m = 1024) already ran full 4-chunk packs under `PASS` in 0008. What was missing was a *partial* 3rd/4th chunk and any check on 32×7. Both are covered now.
- **`beta ≠ 0` from the caller** is not exercised: the driver never passes it. The beta=1 microkernel runs only as the later K blocks' accumulate. The oracle handles `beta ≠ 0` (host test) but nothing feeds it.
- `pack_*_v_else` is generated and never called; it is compared here, not used.
- The DOT and int8 methods are untouched.
- Console `MISMATCH` lines use an absolute 1e-4 and are for debugging only; `Status` uses the norm above.

## 5. Side effects

- Running any `--config` through `compile_macro_bench.sh` overwrites the tracked `pj_gen/`. It was regenerated from `config.json` afterwards; `git status` shows no `pj_gen/` change. This also cleared the nine `.mlir` files that had lost their trailing newline before this session.
- The board workspace `xdsl_rvv_microkernel/` was overwritten by the two runs (its `pj_gen/` is replaced on every run), and `xdsl_rvv_microkernel/pack_correctness_test/` on `fej` by the pack tests.
- New logs under `tests/pure_xdsl_blis_rvv/logs/` (untracked, like the others).
- The shipped `config.json` still skips the check on every row, so a plain benchmark run still logs `SKIP`; the check is the separate command in §2. Chosen because its datasets include 5000³.

## 6. Review

Two review agents (standards, spec). Fixed: a claim that the test "refuses Square or BERT shapes" (it enforces a size bound, B2 is a BERT shape); #17's reason for the `beta` guard (§1); vocabulary (`Fringe Tile`, scalar oracle) against `CONTEXT.md`; a duplicated decorator; a hardcoded arch; `SHIPPED_RR` now tied to `config.json` by a test; the missing `E501` entry for the new test file. Left as is: the stub header and `model_level.o` build are copied from `test_ref_protocol.py` rather than shared. One slip of mine on the way: a scripted re-indent of the pack harness duplicated a block of the file; the tests still passed (later definitions shadowed the earlier), so it was found by reading the diff, and fixed. `pj_gen/`'s `.mlir` files showed as modified once during the review (trailing newline gone; cause not found) and were restored.

## 7. Next

1. Comment on and close #17 and #24 (not done here).
2. If the tolerance is too loose for the write-up of #11, tighten it per pinning, or add a bitwise check of `C` for the integer-valued case (A, B small integers make every partial sum exact in fp32, so a match can be exact).
3. A fourth shape with m > mc (800×40×20 gives two `ic` blocks at 768 + 32) would cover the block loops cheaply, if that matters.
