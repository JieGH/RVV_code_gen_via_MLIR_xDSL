# Issue #29: the reference design alone (LDX, macc) over the #26 sweep: same layers, boards, schedule

**Role / Label:** `done` — tooling written, reviewed and tested; the 16-run sweep finished 16 of 16 (22:58 on 20 Sep to 06:28 on 21 Sep, none failed); results in §5, and the report page shows them (§7).
**Status:** ✅ Against the reference's nearer kernel, our vector pack is within noise on 4 of 10 layers, faster on 5 and slower on 1 (B3). B3's gap to the reference is the packer, as far as one comparison at a different tile and a different time of day can say (§5).
**Target:** `jbpi` and `jbpi2` (BananaPi M2 Pro, VLEN=256), through `fej`, core 0, governor `performance`, as the #26 sweeps.
**Primary Sources:**
- [[0013-issue-26-pack-swap-end-to-end]] — the sweeps this repeats for the reference design (§7 there: vector pack overnight, scalar pack in the afternoon, 16 runs each).
- [[0009-issue-11-midpoint-the-result-flipped]], [[0004-reference-comparison-metric-mismatch]] — the metric (both sides are packing-inclusive) and the published table.
- `tests/pure_xdsl_blis_rvv/logs/refbase_2026-09-20_223620/` (the validation run), `logs/sweep_ref_<stamp>/` (the sweep), `logs/sweep_ref.out`.

---

## 1. What was asked, and what "the same experiment" means here

Run the complete reference design from `ext/`, on the same dimensions and input sizes as #26, with reference code only, using the **LDX** kernels of the 256-bit set and the **macc** ("acc") variant.

- **Same experiment** = the #26 sweep: the ten layers of `config.json`, `jbpi` and `jbpi2` taking turns, one run every 30 min, 8 runs per board, core 0, `tmin` 10 s, uninstrumented, reference protocol. Only the design under test changes.
- **Reference code only** = `ext/RVV_ukernels_benchmark/gemm_blis_family`: its scalar packers (`pack_RB`, `pack_CB`), its Exo microkernels, its blocking model. Nothing of ours runs.
- **LDX** = `kernels/RVV_256_LDX`. The one difference from the `OPT` set (diffed on 32x6 macc): the LDX kernels take `lda` and `ldb` as run-time arguments (`B[k*ldb]`), the OPT ones have the strides as constants (`B[k*6]`, `A[k*32]`). The reference calls them with `lda = MR`, `ldb = NR` (`gemm_blis.c:126-130`), so both compute the same thing; LDX pays a run-time multiply. The earlier reference baseline (`refbase_2026-09-18_164740`, B1 and B5 only) was already LDX/macc.
- **macc** = `loadAB/macc`: `vfmacc.vf`, B stays a scalar operand, C accumulates in registers. The variant that matches our kernel (README of `ref_baseline`).

The tooling to run the reference already existed (`run_ref_baseline.sh --opt LDX --variants 2 --bracket`). What was missing was the 16-run repeat with an error bar, and the comparison against our two sets.

## 2. What was built

- `run_ref_sweep.sh`: the counterpart of `run_scalar_pack_sweep.sh`. One `run_ref_baseline.sh` run per slot, under the shared lock. A run that exits non-zero, leaves a plan row out, or writes a GFLOPS that is not a number is named as failed and left out of `runs.txt`; the sweep goes on. `run_ref_baseline.sh` alone does not do this: it only warns when a build fails, and exits 0.
- `ref_baseline/compare_ref_sweep.py`: per layer and per built kernel, the reference's mean, range and run count beside our vector-pack set (the overnight runs) and scalar-pack set (the newest `sweep_scalar_*`), `vector/ref` and `scalar/ref` with a faster / slower / within-noise call against the larger spread between runs, the published figure (exact rows only), pin warnings, and the limits.
- `test_compare_ref_sweep.py` (18 tests) and `test_run_ref_sweep.py` (14 tests, 7 of them bad-flag cases, against a stub reference runner, no board).
- `run_ref_baseline.sh` now passes `--tmin` to `gen_ref_datasets.py`; before, the under-sampling warning in every run log assumed 3 s whatever `--tmin` was.
- `compare_ref_baseline.py`: `_stamp`, `_pin`, `_pin_words` made public (`stamp_of`, `pin_of`, `pin_words`) so the new script does not import private names.
- Docs: `docs/agents/execution.md`, `ref_baseline/README.md`.

The code review (two sub-agents, standards and spec) found, and these were fixed: the private imports; a stale line in `execution.md`; the header saying "the exit code says how many"; a row check that accepted `nan`; a `published` column that divided a neighbour kernel's number by a 32x7 figure; no pin warning; the comparison's limits missing from its output; thin test coverage. Left as they were: about 150 lines of shell copied from `run_scalar_pack_sweep.sh` (the house habit; a shared library is a separate change), and the overnight window written in five places.

## 3. Checks made before the sweep

- **Correctness of the LDX macc builds** (the reference's own `TEST=T` mode, on `jbpi`, core 0): 16x14, 16x16, 32x6 and 32x8, each on 100x90x70, 64x64x300, 37x29x130 and 129x131x257 (fringes, three K blocks). All 16 checks `OK`, error 0.00e+00 as printed. Nothing in the timed runs checks the reference's output, so this is the only check. The one-off script was in the scratchpad, not the repo.
- **One real run** (`refbase_2026-09-20_223620`, `jbpi`): all 18 rows, 11 min wall (22:36-22:47), so it fits the 30 min slot.
- **Tests**: 76 pass across the new files and `test_compare_ref_baseline.py`, `test_board_pin.py`. Pyright strict is not clean on the new files (`load_plan` returns a bare `list[dict]`, as in the old script); the pre-commit hook is bypassed as agreed.

## 4. The one run so far (not a result)

`refbase_2026-09-20_223620`, `jbpi`, GFLOPS, packing included:

| built kernel | S1 | S2 | S3 | S4 | S5 | B1 | B4 | B5 | B2 | B3 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32x6 (exact for B1, B5; low bracket for 32x7) | 8.49 | 8.30 | 8.94 | 8.65 | 8.81 | 7.30 | 7.01 | 7.46 | | |
| 32x8 (high bracket) | 4.61 | 4.36 | 4.49 | 4.57 | 4.45 | | 4.30 | | | |
| 16x14 (low bracket for 16x15) | | | | | | | | | 10.10 | 7.25 |
| 16x16 (high bracket) | | | | | | | | | 6.68 | 3.55 |

- The high brackets (32x8, 16x16) run at about half the speed of the low ones. Both hold 32 C accumulators plus the A registers, so they need more than the 32 vector registers and spill (reasoned from the tile shape; the assembly was not read). The low bracket is the nearer one to a true 32x7 / 16x15.
- At 32x6 the reference is level with our two sets on S1 (8.49 against 8.49 and 8.45) and lands within a few percent on S3-S5. On B1 it is below ours (7.30 against 8.04 and 7.91). This is one run against 16; do not read the sign of any gap from it.
- B3 at 16x14 (7.25) is above our vector-pack B3 (5.30) and near our scalar-pack B3 (7.88). That fits the idea in #26 that the reference's scalar packing is part of the B3 story, but one run at a bimodal shape proves nothing.

## 5. Result

`logs/sweep_ref_2026-09-20_225831/comparison.txt`. 16 of 16 runs (8 per board, all pinned to core 0), every row present in every run. GFLOPS by mean time, packing included, on all three sides; the reference column is its nearer kernel (exact on B1, B5; 32x6 for the 32x7 layers, 16x14 for the 16x15 layers). Range in brackets is lowest to highest run.

| Layer | reference (kernel) | vector pack | scalar pack | vector vs ref | scalar vs ref |
|---|---|---|---|---|---|
| S1 | 8.41 (32x6) | 8.49 | 8.45 | +1.0%, within noise | +0.5%, within noise |
| S2 | 8.44 (32x6) | 8.75 | 8.62 | +3.7%, within noise | +2.2%, within noise |
| S3 | 8.80 (32x6) | 9.05 | 8.94 | +2.8%, within noise | +1.5%, within noise |
| S4 | 8.62 (32x6) | 8.83 | 8.77 | +2.5%, within noise | +1.7%, within noise |
| S5 | 8.78 (32x6) | 9.16 | 9.09 | +4.4%, faster (noise 3.3%) | +3.6%, faster (noise 3.3%) |
| B1 | 7.26 (32x6 exact) | 8.04 | 7.91 | +10.8%, faster | +9.0%, faster |
| B2 | 10.07 (16x14) | 12.72 | 12.15 | +26.3%, faster | +20.7%, faster |
| **B3** | **7.35 (16x14)** | **5.30** | **7.88** | **−27.9%, slower** | **+7.2%, within noise (22.6%)** |
| B4 | 6.91 (32x6) | 7.80 | 7.58 | +13.0%, faster | +9.8%, faster |
| B5 | 7.35 (32x6 exact) | 7.82 | 7.78 | +6.4%, faster (noise 5.2%) | +5.8%, faster (noise 5.2%) |

The other neighbour ran at means of 4.30-4.64 (32x8) and 3.73-6.73 (16x16): 51-67% of the near kernel on every layer (§4's reading, now over 16 runs; that it spills is still reasoned from the tile shape, not read from the assembly).

What the table says, and what it does not:

- **The square layers (S1-S4) are level with the reference** within the spread between runs (1.0-3.7%); S5 is 4.4% ahead, barely outside its noise. The reference's spread across runs is 2.3-4.8% here, the same size as ours.
- **B3 is the one layer where we are clearly behind, and only with the vector pack.** Vector pack 5.30 against the reference's 7.35 (−27.9%, outside the 12.5% noise). With the scalar pack (the reference packs with scalar loops) B3 is 7.88, 7.2% above the reference and within the 22.6% noise of the scalar set. That fits #26: the B3 gap to the reference is the packer. It is one comparison, at a different tile (16x14 against our 16x15) and at a different time of day (the scalar runs are from the afternoon, the reference from the night), so "explains it" is the reading, not a proof.
- **B1, B4, B5, B2: we are ahead of the reference's near kernel** by 6-26% (B5 barely outside noise). B1 and B5 are exact matches (same 32x6 tile), so those two are the clean comparison: +10.8% and +6.4% with the vector pack. B2 and B4 are against 16x14 and 32x6, not our 16x15 and 32x7, so part of the gap may be the tile, not the code generation; there is no way to tell them apart without the true kernels.
- **The reference does not reproduce the published table.** Exact rows: B1 7.26 against published 8.0 (−9.3%), B5 7.35 against 7.5 (−2.0%). B4's near kernel is 6.91 against 8.5 (−18.7%, computed here; the page leaves bracket rows without a published comparison). The earlier single run of 18 Sep (B1 7.02, B5 7.06, jbpi2, tmin 3, not pinned) is below these two by 3-4%. So the published B4 figure, which our runs also miss by 8-11% (#26), is not reached by the reference either at its close kernel. The published table's origin is still unknown.
- **Time of day.** The reference sweep and our vector-pack sweep are both overnight, so they are the closer pair; the scalar set is from the afternoon.

## 6. What this cannot show (written down before the result)

- **No true 32x7 or 16x15 reference.** S1-S5, B4, B2, B3 are bracketed, and the brackets are lopsided (§4). "The reference number" for those layers is at best the low bracket, and that is an inference.
- **Not the same hours.** Ours: vector 23:06-06:36 (19-20 Sep), scalar 13:32-21:02 (20 Sep). The reference: 22:58 on 20 Sep to about 06:40 on 21 Sep. The reference sweep is at night like the vector one, so vector against reference is the cleaner pair; the scalar set is from the afternoon. Time of day and board temperature are not controlled.
- **S3-S5 are about one rep per run** at `tmin` 10 on both sides (single cold rep, no warm-up): their noise is the spread between runs only.
- **GFLOPS is printed with 3 significant digits** by the reference, so about 0.1% rounding.
- **The LDX set, not OPT.** Whether OPT (constant strides) is faster was not measured here; `run_ref_sweep.sh --opt OPT` would.
- **Nothing was committed.** The new files and edits are in the working tree.

## 7. The report page

`logs/report_2026-09-20/index.html` (built by `build_report.py`, template `template.html`) now carries the reference sweep: a cyan cross in *Layer by layer* (mean and lowest-highest run of the near kernel, right of the boards), its tooltip (both neighbours, runs, the change against ours), a *Reference sweep* series in the legend, in the *Versus reference* chooser and in the *Table* tab, a tile, the table of §5 under the chart, and a source note. `build_report.py` reads the newest `logs/sweep_ref_*` (or `--ref-sweep DIR`). Checked by building the page, a headless-Firefox screenshot (cross, tile and table drawn), and the tooltip and table code run under node against the real data (S1, B1, B3 values equal `comparison.txt`). Not checked: the light/dark colour of the new cyan against the page in a live browser, the *Versus reference* chart with the new option selected, and touch devices.
