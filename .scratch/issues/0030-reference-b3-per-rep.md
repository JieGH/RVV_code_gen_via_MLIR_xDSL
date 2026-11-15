# Issue #46: Does the reference design's B3 also have two speeds? Per-rep times of the reference (follow-up of #36 and #41)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/46

**Comes from:** the engineering report, Limitations (first bullet) and Recommendation 4:
[issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf)
([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)).

**Role / Label:** `needs-triage` — follow-up of #36 and #41. Measure only: no fix is sought here.
**Status:** ✅ Measured 2026-09-29 on both boards, first under another user's load, then **repeated in quiet windows on both boards** (no other user's process at any reading; §7.8, §7.9). **Answer: B3 with K=256 is the same slow state in both designs; on plain B3 the reference is far steadier than ours** (quiet window: 87-89% of our reps slow at 4.4-4.5 GOPs; the patched reference 22% on `jbpi` and 0% on `jbpi2`, at 7.1-7.6 GOPs). The control run does not pass case by case, and the patch's effect on B3 has opposite sign on the two boards (§7.3, §7.9), which points at buffer placement. The first series (§7.1-§7.6) carries a 🔴 RED NOTE for other users' load.
**Target:** `jbpi` first, then `jbpi2`; core 4; house rules of #39 apply.
**Shape(s):** B3 (64×384×384) at the reference's `nc`=384. Controls: B2 (one speed in our runs), B3 with K=256, Reuse24.
**Primary Sources:**
- [[0021-b3-vector-packer-slow]] (#36) — the slow state, found with per-rep records of our side.
- [[0025-b3-slow-state-root-cause]] (#41) §8 — what the slow state is on our side (L1D prefetcher does about half its work).
- [[0015-direct-untouched-reference-check]] (#30) — the earlier "run the reference directly" check, and why the staging pipeline was distrusted.
- `tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh`, `repair_ref_kernels.py` — how the reference is staged on the board today.
- `ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c` (read-only) — the reference's timing loop.
- `tests/pure_xdsl_blis_rvv/reps_report.py` — reads the per-rep file we would produce.

---

## 1. Question

Our B3 runs at two speeds (#36, #41). Nobody has measured whether the reference design's B3 does too. The
report says it probably does, because the cause (the X60's L1D prefetcher) and the memory walk are shared
by both designs, but that is a guess. **Does the reference's B3 fall into a slow state, and if so, how
often and after how long?** This decides whether B3's instability is a difference between the designs at
all.

## 2. Why it has not been seen yet

The reference driver (`test_gemm.c`, around lines 340-360) calls the GEMM in a loop `while (time <= tmin)`,
reads the clock after each call, but keeps only the total. It prints one mean time per call
(`time = time/nreps`). There is no per-rep record, so two speeds inside a run are invisible: they average
into one number. Our side has had per-rep records since #35.

`ext/` is read-only for this project. The staging step already copies the reference to the board and
repairs six kernel files there (`repair_ref_kernels.py`), so a small change to the **staged copy** has a
precedent; the files in `ext/` stay untouched.

## 3. Plan

1. **Per-rep times on the staged copy.** In the staged `test_gemm.c` only, store the clock read after each
   call in a pre-sized array (no allocation, no output inside the loop) and write it out after the timed
   loop, in the same CSV layout as our `reps_*.txt` (`Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,...`) so
   `reps_report.py` reads it unchanged. The array is filled inside the loop, the file is written after it.
2. **Control for the change itself.** Run the unchanged staged reference and the changed one inside the
   same launches. Their mean GOPs must agree within noise (under about 3%, and same sign in every launch);
   if they do not, stop and report that before reading any per-rep result.
3. **The measurement.** Reference B3 at its own `nc`=384, `--tmin 10`, core 4, next to our B3 with
   `--no-nc-overrides` (also `nc`=384) in the same launches. Order A-B-B-A (ours, ref, ref, ours) and at
   least 3 launches; then the same on `jbpi2`. Controls in the same launches: B2, B3 with K=256, Reuse24.
4. **Read it** with `reps_report.py`: slow-rep share, fast start (reps and seconds), clumping, CV.

Board time: not yet estimated; each case is `--tmin 10` s plus build and staging, so a few minutes per
launch is the expectation, not a measured figure.

## 4. How to read it

- **The reference has two speeds too** (slow-rep share well above 0 for B3, small for B2): B3's instability
  is not a difference between the designs. It is the board and the shared access pattern. Our slow-state
  chain (#41, #43, #44) then applies to both designs.
- **The reference is steady on B3** (one speed): there is a real difference. Candidate differences to list,
  not yet to test: the reference's own packers, its kernels (LDX kernels take `lda`/`ldb` at run time),
  buffer alignment and allocation, the call order inside a rep.
- **Reference slow reps come late or rarely:** compare fast start against our scalar packer (0.5-8 s, #36)
  and our vector packer (0.1-0.5 s); the reference may sit between them.

## 5. Out of scope

Fixing anything; changing our generated code; editing `ext/`; adding the reference to the default
head-to-head command; a new comparison table against the reference (this issue reads slow-rep shares, not
GOPs against GOPs).

## 6. Closure

- [x] Staged reference writes a per-rep file that `reps_report.py` reads; the change is described in the issue. (§7.1: `ref_baseline/ref_per_rep.py`, third version.)
- [~] Control: changed and unchanged reference agree on mean GOPs, inside the same launches, on both boards. Run and reported; **not met case by case** (7 of 16 board/tile/case rows agree, 8 disagree, 1 is a same-sign gap; §7.3). The average gap over 60 launch pairs is +0.6%.
- [x] B3 (and B2, B3_K256, Reuse24 as controls) measured, 3 or more launches per board, logs committed. (`jbpi` 4 launches; `jbpi2` 4, but 3 for B3_K256 and Reuse24, §7.6.)
- [x] Written answer: "the reference B3 has / does not have two speeds", with the slow-rep share, fast start
  and the run folders cited. (§7.4.)

---

## 7. Results (2026-09-29)

### 7.1 What was built

- `ref_baseline/ref_per_rep.py` (test `test_ref_per_rep.py`, 19 tests): `patch` edits a **staged copy** of `test_gemm.c`
  (refuses any path under `ext/`; the file in `ext/` is untouched) and `convert` turns the driver's file into
  `reps_ref_<tile>_<variant>.txt.gz` in the layout of our `reps_*.txt` (label for the driver's numeric layer id,
  `Pack_*`/`Compute_*` = `nan`), checking each case's total against the GFLOPS the driver printed.
- `run_ref_baseline.sh --per-rep` and `--cases FILE`; the case list is `ref_baseline/cases_issue46.tsv` (B3, and B2,
  B3_K256, Reuse24 as controls, all at 16x15, which the reference builds as the two even neighbours 16x14 and 16x16).
- `run_issue46.sh` (test `test_run_issue46.py`, 16 tests): launches of three runs on one board, core 4, `--tmin 10`:
  **O** ours (`config_issue46_ref_b3.json`, `--no-nc-overrides`, `nc` 384), **P** the reference unchanged, **R** the reference with the
  per-rep record. Orders O P R, R P O, O R P, P R O; boards take turns; other users' processes read before and after every launch.
- `issue46_report.py` (test `test_issue46_report.py`): the control table and the per-rep table of a series.
- `reps_report.py` names the new files "reference <tile>" and prints a 🔴 RED NOTE for any run folder whose `provenance.txt`
  records another user's job (`board_pin.sh` now records it; `docs/agents/execution.md` rule 5).

**Deviations from §3, on purpose:** the order is a balanced three-way rotation instead of A-B-B-A (there are three runs,
not two); the boards alternate launch by launch instead of `jbpi` then `jbpi2`, so a change over the hours falls on both;
4 launches per board, not 3.

### 7.2 The patch had to be changed twice (found by the control)

The C change is one array and one store after the clock read the driver already does. The control run showed it was
not neutral:

| version | what it did | B3_K256 at 16x16, plain vs patched (GFLOPS) |
|---|---|---|
| 1 | array from `malloc`, output through `fopen` | 4.64 vs 3.32 (-28%), 4.65 vs 3.22 (-31%); `jbpi2`: 4.77 vs 3.28 (-31%) |
| 2 | array static, output still through `fopen` | 4.72 vs 3.22 (-32%) |
| 3 | array static, output through `open`/`write` from a static buffer: nothing from the heap | 4.60 vs 4.52, 4.73 vs 4.58, 4.72 vs 4.65, 4.72 vs 4.63 (-1.5 to -3.2%) |

Reading (consistent with the numbers, not proven): the driver aligns its A, B, C and pack buffers to 32 bytes only, so
where they land depends on what was allocated before them; a heap allocation (the array, or the second `FILE`) moved
them, and B3-like cases are sensitive to that. Version 3 takes nothing from the heap. The failed runs are kept as
evidence: `run_2026-09-29_111635` to `_115420`, `refbase_2026-09-29_111636` to `_120434` (version 1: series
`issue46_2026-09-29_111401`, version 2: `issue46_2026-09-29_115638`); a one-pair check of version 3 is
`refbase_2026-09-29_121133` (plain) and `refbase_2026-09-29_121706` (patched). **This also says that the slow state
depends on where the buffers sit in memory**, which is a result for #41 in its own right.

### 7.3 Control, version 3 (series `issue46_2026-09-29_122340`, both boards)

Gap = (with per-rep record - unchanged) / unchanged, by the mean GFLOPS the driver printed, per launch:

| rows | count |
|---|---|
| agree (every launch under 3%) | 3 |
| agree only inside the case's own launch-to-launch spread (the unchanged reference alone moves 8-26%) | 4 |
| disagree (a launch over 3% and over the spread) | 8 |
| same sign in every launch, mean over 1% and over the spread | 1 (`jbpi` 16x16 B3, +5.0%) |

B2, the steady case, has gaps from -6.2% to +5.2% (`jbpi` 16x14: +5.2, -5.6; `jbpi2` 16x16: -6.2). Averaged
over the 60 launch pairs the gap is +0.6% (mean absolute gap 4.3%), so there is no overall bias, but single launches differ by
several percent, more than the 3% the plan asked for. **The control does not pass case by case.** Full table:
`python tests/pure_xdsl_blis_rvv/issue46_report.py logs/issue46_2026-09-29_122340`. The per-rep numbers below are read with
this in mind: they show whether the reference has two speeds, not their exact size.

### 7.4 Answer: does the reference's B3 have two speeds?

Slow-rep share (average over 4 launches; `jbpi` / `jbpi2`), uninstrumented, reference protocol, `--tmin 10`, core 4, `nc` 384;
GOPs by mean time (by median rep):

| case | ours 16x15 | reference 16x14 | reference 16x16 (spills, 34 registers) |
|---|---|---|---|
| B3 (64x384x384) | 89% / 88%; 4.30 (4.12) / 3.89 (3.71) | **20% / 14%; 6.92 (7.46) / 6.96 (7.37)** | 0% / 11%; 4.41 / 3.98 |
| B3_K256 (64x384x256) | 88% / 90%; 4.75 (4.53) / 4.20 (4.00) | **73% / 76%; 4.79 (4.25) / 4.20 (3.78)** | 0% / 0%; 4.59 / 4.33 |
| Reuse24 (384x384x384) | 7% / 7%; 8.35 / 8.25 | 3% / 6%; 8.41 / 7.80 | 0% / 0%; 5.09 / 4.92 |
| B2 (384x384x64) | 0% / 0%; 12.45 / 11.92 | 0% / 0%; 9.67 / 9.49 | 0% / 0%; 6.54 / 6.50 |

- **The reference's B3 at 16x14 has two speeds, in 6 of 8 launches** (slow share per launch: `jbpi` 0, 13, 28, 39%; `jbpi2` 23, 0, 17,
  17%). It is milder than ours (ours 88-89% in every launch) and later: the first slow rep comes after 3.0 s on average
  (`jbpi`) and 3.4 s (`jbpi2`), against 0.07-0.08 s for ours.
- **B3 with K=256 is the same slow state in both designs**: 64-84% of the reference's reps are slow, 88-90% of ours; both
  switch within a fraction of a second (reference 0.00-0.28 s, ours 0.02-0.04 s).
- **Reuse24 and B2 look alike** in both designs (few or no slow reps). Ours had 7% here against 40% in the survey of #47:
  same shape, same `nc`, but 4 shapes in the launch here and 5 there (§7.6).
- **The reference at 16x16** (a kernel that needs 34 registers and spills) is steady on B3 and K256 and slower: no slow state to
  see because it is already slow (4.4 GOPs, our slow speed).

Reading against §4: **not "steady" and not "the same"**: the state exists in the reference (so it is the board and the shared
walk, and the chain of #41/#43/#44 applies to both designs), but B3 is much less affected there. What differs could be: the
tile (16x14 against 16x15; not tested), the reference's own packers, its LDX kernels, buffer alignment (§7.2 shows it matters),
the call order inside a rep. Listed, not tested.
Speeds are not compared here (different tile, other user's load); the numbers are side by side for context only.

### 7.5 Run folders

Series `issue46_2026-09-29_122340` (`runs.tsv` in that folder lists them in run order):

| board | launch | ours (O) | reference unchanged (P) | reference with per-rep record (R) |
|---|---|---|---|---|
| jbpi | 1 | `run_2026-09-29_122606` | `refbase_2026-09-29_122606` | `refbase_2026-09-29_123151` |
| jbpi2 | 1 | `run_2026-09-29_123944` | `refbase_2026-09-29_123945` | `refbase_2026-09-29_124513` |
| jbpi | 2 | `run_2026-09-29_130533` | `refbase_2026-09-29_125615` | `refbase_2026-09-29_125041` |
| jbpi2 | 2 | `run_2026-09-29_132103` | `refbase_2026-09-29_131256` | `refbase_2026-09-29_130537` |
| jbpi | 3 | `run_2026-09-29_132334` | `refbase_2026-09-29_132912` | `refbase_2026-09-29_132335` |
| jbpi2 | 3 | `run_2026-09-29_133736` | `refbase_2026-09-29_134404` | `refbase_2026-09-29_133736` |
| jbpi | 4 | `run_2026-09-29_140349` | `refbase_2026-09-29_135022` | `refbase_2026-09-29_135554` |
| jbpi2 | 4 | `run_2026-09-29_141839` | `refbase_2026-09-29_140351` | `refbase_2026-09-29_140954` |

Per-rep files: `refbase_*/reps_ref_16x14_macc.txt.gz`, `reps_ref_16x16_macc.txt.gz`; ours: `run_*/…/reps_*_noinstr_refproto_no_nc_override.txt.gz`. Load records:
`load_<board>_L<n>_{before,after}.txt` and `load_warnings.txt` in the series folder.

### 7.6 Limits

- **🔴 RED NOTE, other users' load.** Every run of this series was made with `psiwins` jobs on the boards. `jbpi`: `spmv`
  at 92-99% of core 0 for the whole series. `jbpi2`: `cc1plus`, `ld` and `mixed-precision` at 48-100% of a core, moving
  between cores 0-6 (unpinned, affinity 0-7), once found on core 4 itself (launch 4, before the reference run
  `refbase_2026-09-29_140954`). Even on other cores this can move the chip's thermal state; the 8 launches are **not clean**. Repeat:
  series `issue46_2026-09-29_122353`, queued for 00:00, is not yet analysed.
- **Own error:** at 13:12 a test run of mine (`test_board_pin.py`, which starts the scripts with `--dry-run`) rewrote
  `ref_baseline/generated/` while the series was running. The plain reference run `refbase_2026-09-29_131256` (`jbpi2`, launch 2)
  staged that folder and ran only B2 and B3; its B3_K256 and Reuse24 rows are missing, so those two cases have 3 launches
  on `jbpi2` (still the minimum). The rows it did produce are real measurements. `execution.md` now says to run no tests while
  a series is going. (The other 23 runs staged before or after; nothing else was affected.)
- The control does not pass (§7.3); B3's two-speed behaviour makes single launches move by 10-25%.
- The reference was run at 16x14 and 16x16, never at our 16x15 (it ships no odd NR).
- One kernel set (LDX, `macc`), one `nc` (384); `--tmin 10` gives 700-5000 reps per case.

### 7.8 Quiet-window repeat on `jbpi` (series `issue46_2026-09-29_215925`, 21:59-22:52, 4 launches)

No other user's process at any of the 8 load readings (`load_jbpi_L*_{before,after}.txt` in the series folder show only root's `sshd`
and kernel threads): **no red note**. Same script, patch version 3, core 4, `--tmin 10`. Run folders: `runs.tsv` in the series folder
(ours `run_2026-09-29_220147`, `_222544`, `_222808`, `_225219`; unchanged reference `refbase_2026-09-29_220148`, `_221802`, `_223337`,
`_223907`; patched reference `refbase_2026-09-29_220713`, `_221239`, `_222808`, `_224431`).

Slow-rep share, average of 4 launches; GOPs by mean time (by median rep):

| case | ours 16x15 | reference 16x14 (patched) | reference 16x16 (patched) | unchanged reference 16x14 (mean GOPs per launch) |
|---|---|---|---|---|
| B3 | 87%; 4.53 (4.29) | **22%** (12-29% in every launch); 7.12 (7.69) | 0%; 4.47 | 7.56 / 7.52 / 7.61 / 7.53 |
| B3_K256 | 94%; 4.71 (4.56) | **80%** (77-82%); 4.89 (4.43) | 0%; 4.68 | 5.06 / 5.29 / 6.50 / 5.24 |
| Reuse24 | 10%; 8.48 (8.85) | 3% (0-9%); 8.46 (8.62) | 0%; 5.17 | 7.94 / 8.36 / 7.87 / 8.15 |
| B2 | 0%; 12.56 | 0%; 9.89 | 0%; 6.64 | 9.81 / 9.93 / 9.96 / 9.80 |

Control (patched against unchanged, 8 rows): 3 agree, 1 agrees inside its own spread (K256 at 16x14: the unchanged reference alone
moves 5.06-6.50), 1 disagrees (Reuse24 at 16x14, gaps +7.4, -1.2, +10.0, +2.9%), **3 have the same sign in every launch**: B3 at 16x14
(**-5.7%**: unchanged 7.52-7.61, patched 6.81-7.49), B2 at 16x16 (-1.1%), Reuse24 at 16x16 (+3.0%).

What this changes:
- **Confirmed:** ours falls into the slow state on B3 (87%) and K=256 (94%) as before; K=256 is the same slow state in the reference
  (80% of reps slow; the unchanged reference's mean moves 5.06-6.50 between launches, which is what two speeds look like from outside).
  The reference at 16x16 has none. Reuse24 and B2 are steady in both.
- **Sharper:** for B3 at 16x14 the unchanged reference is **steady across all four quiet launches (7.52-7.61, spread 1%)** and 6%
  above the patched one (same sign in every launch), while the patched driver shows 12-29% slow reps. So either the patch (even with
  no heap use) still tips B3 into a mild slow state, or the unchanged driver has slow reps too that its total hides. The per-rep record cannot
  tell which, because the unchanged driver has none: that is the limit of the method. What holds in both readings: **our B3 (87% slow, 4.5
  GOPs) is far more affected than the reference's (at most 22%, 7.1-7.6 GOPs)**, and K=256 is the same in both.
- The first series' unclean B3 numbers for the unchanged reference (6.05-7.37 on the two boards) were more scattered than these quiet ones
  (7.52-7.61), which fits other users' load adding dips on top.
- `jbpi2` is in §7.9.

### 7.9 Quiet-window repeat on `jbpi2` (series `issue46_2026-09-29_225532`, 22:55-23:48, 4 launches)

Board checked idle before it started (no other user's process on either board); no other user at any reading in the series (no
`load_warnings.txt`): **no red note**. Run folders (`runs.tsv` in the series folder): ours `run_2026-09-29_225755`, `run_2026-09-29_232157`, `run_2026-09-29_232420`, `run_2026-09-29_234845`; unchanged reference `refbase_2026-09-29_225755`, `refbase_2026-09-29_231411`, `refbase_2026-09-29_232945`, `refbase_2026-09-29_233514`; patched reference `refbase_2026-09-29_230318`, `refbase_2026-09-29_230848`, `refbase_2026-09-29_232421`, `refbase_2026-09-29_234046`.

Slow-rep share, average of 4 launches; GOPs by mean time (by median rep):

| case | ours 16x15 | reference 16x14 (patched) | reference 16x16 (patched) | unchanged reference 16x14 (mean GOPs per launch) |
|---|---|---|---|---|
| B3 | 89%; 4.40 (4.25) | **0% in every launch**; 7.44 (7.74) | 2%; 4.45 | 7.06 / 7.33 / 7.23 / 6.93 |
| B3_K256 | 88%; 4.90 (4.59) | **60%** (37-82%); 5.50 (6.13) | 0%; 4.68 | 5.28 / 5.77 / 5.32 / 5.07 |
| Reuse24 | 12%; 8.62 (9.16) | 4% (0-12%); 8.33 (8.61) | 0%; 5.19 | 8.35 / 8.61 / 7.81 / 8.10 |
| B2 | 0%; 12.65 | 0%; 9.96 | 0%; 6.85 | 9.95 / 9.95 / 9.96 / 9.98 |

Control (8 rows): 3 agree, 1 agrees inside its spread (Reuse24 at 16x14), 3 disagree (B3 at 16x14: +4.8, +3.0, +0.8, **+8.7%**; K256 at 16x14, whose
two speeds make single launches move -10% to +21%; B3 at 16x16: +11.4%), 1 same sign (Reuse24 at 16x16, +3.3%).

What this adds:
- **On `jbpi2` the reference's B3 shows no slow reps at all** (0% in all four launches, 7.3-7.55 GOPs), while ours is 89% slow at 4.4. So on plain B3 the
  designs differ clearly; the reference's mild slow state on `jbpi` (22%, §7.8) is board-dependent (and in the first, loaded series `jbpi2` had 0-23%).
- **K=256 is a slow state in both designs on both boards**: reference 80% (`jbpi`) and 60% (`jbpi2`); ours 94% and 88%.
- **The patch's effect on B3 at 16x14 has opposite sign on the two boards**: patched 5.7% *slower* than unchanged on `jbpi`, 4.3% on average *faster*
  on `jbpi2`. A fixed cost of the patch would have the same sign on both; this looks like the patch moving buffers, and B3 reacting differently on each
  board. So single-case gaps of the control are mostly this layout effect, and the mild `jbpi` slow share of the patched B3 may be part of it.

### 7.7 Next

1. (Done) `jbpi2` repeated in a quiet window, §7.9; the 00:00 series was cancelled as redundant.
2. Ours at 16x14 (our generator can emit it) against the reference at 16x14, to take the tile out of the comparison.
2b. Give the reference's B3 a way to be watched without the patch changing it (for example a hardware counter read from outside the process), so that its B3 can be settled.
3. Test the buffer alignment: give the reference's `Ac`/`Bc` and ours the same offsets modulo 4096 and see whether B3 changes.
