# Issue #47: The slow state across every shape, from the per-rep logs we already have (follow-up of #41)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/47

**Comes from:** the engineering report, Discussion (cause 3, "only one still open") and Recommendation 5:
[issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf)
([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)).

**Role / Label:** `needs-triage` — follow-up of #41 (its §5, "what switches it"). Analysis only: no board time, no change to the generator or the driver.
**Status:** ✅ Closed 2026-11-18 (housekeeping: answered, results merged). Was: ✅ Done 2026-09-29 (analysis only, no board time). Results in §7: the slow state is not only B3; it is in every shape with M up to 384 and K of 256 or more that ran on the 16x15 tile, and in none of the shapes with M of 1000 or more, which all ran on other tiles (32x6, 32x7), so the two cannot be told apart yet. It does not switch on at a fixed time, rep count or byte count.
**Target:** the committed logs in `tests/pure_xdsl_blis_rvv/logs/`. No board.
**Shape(s):** all: B1-B5, S1-S5, Reuse8/12/24, and the B3 variants (K256, N192-N320, `nc` sweeps).
**Primary Sources:**
- [[0021-b3-vector-packer-slow]] (#36), [[0020-b3-smaller-nc-exploration]] (#35), [[0025-b3-slow-state-root-cause]] (#41) — what is known about the slow state on B3.
- `tests/pure_xdsl_blis_rvv/reps_report.py` — already gives slow-rep share, fast start, clumping and CV per case and launch.
- `tests/pure_xdsl_blis_rvv/logs/run_*/*/reps_*.txt.gz` and each run's `provenance.txt`.
- `docs/metrics.md`, "Per-rep log".

---

## 1. Question

The slow state is documented for B3 and, in part, for Reuse24 (7% slow reps in #42). Nobody has looked at the
other shapes. **Which shapes ever enter the slow state, how much of a run they spend in it, and does the
switch depend on time, on the number of reps, or on the amount of data packed?** The answers narrow down
what the prefetcher reacts to, and they say how far B3's problem reaches.

## 2. What data exists (rough count, to be confirmed by step 1)

A first count of the per-rep files, by run folder (not yet by launch), found:

- **Many folders:** B2, B3, B3_K256, B3_nc128 and Reuse24 (84-96 folders each, uninstrumented reference
  protocol and instrumented builds). These include the overnight series, so they mix packers (vector,
  scalar), `--zero-pack-lines`, `nc` overrides and boards.
- **Few folders:** B1, B4, B5 and S1-S5: 3-4 folders each, from #37 and the B1/B5 follow-up (two tile
  variants each). At the #39 minimum of 3 launches, so they are usable but thin.
- **Six folders each:** the `nc` sweeps of #35 (B3, Reuse8/12/24 at `nc` 32-384) and the B3 N and K variants.
- The count found no core-0 run and no run with an unknown core; step 1 checks this properly.

So the survey is rich for five shapes and thin for the rest. If the thin ones show one speed in all their
launches, "no slow state seen in 3 launches" is the honest statement, not "no slow state".

## 3. Plan

1. **Inventory.** One table, one row per run folder: board, build (instrumented or not), protocol, core,
   packer, `--zero-pack-lines`, `nc` override, layers present, launches. Read from `provenance.txt` and the
   file suffixes. Drop what the #39 rules drop (core 0, cases seen in fewer than 3 launches go under "no
   result").
2. **Slow-state table.** Per shape and setting (packer, `nc`, zeroline): slow-rep share, fast start,
   clumping and CV, across launches with the lowest and highest launch (the `reps_report.py` outputs).
   Uninstrumented reference-protocol logs for the shares; instrumented logs only for counts, not for speed.
3. **Time, reps or bytes?** For every case with two speeds, take the first slow rep: its index, the seconds
   since the start of the case, and the bytes packed up to then. The packed bytes per rep have to be worked
   out from the loop nest (A and B block sizes and how often each is packed); write the formula down in the
   issue before using it. Then see which of the three is closest to constant across shapes.
4. **Which shape features go with the slow state.** Put slow-rep share beside M, N, K, `nc`, the size of the
   packed B and A blocks, and whether the buffers are powers of two. Look for a threshold or a pattern; do
   not fit a model.

A small script for steps 2-3 (`slow_state_survey.py`, reusing `reps_report.py`'s reader) with a test on a
made-up per-rep file is allowed. Nothing else in the repository changes.

## 4. How to read it

- **Only B3 (and Reuse24 a little) slow:** the state needs a small M and something in B3's blocking; that
  supports the #44 idea (fewer streams) and the padding idea (#38), and puts B4/B1/B5 outside the problem.
- **Many shapes slow, some briefly:** the state is a general property of the prefetcher on this board, and
  B3 is only where it is largest.
- **The switch time is roughly constant in seconds:** something time-based (a prefetcher training or
  throttle window). **Constant in reps or in bytes packed:** something that depends on the access history.
  These need different next experiments (a time-based cause cannot be fixed by changing the packing).

## 5. Out of scope

New board runs; any fix; comparing with the reference (that is [[0030-reference-b3-per-rep]]); quoting
speeds from instrumented logs; using a shape seen in fewer than 3 launches as a result.

## 6. Closure

- [x] Inventory table of the per-rep logs, with each run's build, protocol, core and packer. (§7.1, and one row per file in the survey log.)
- [x] Slow-state table for every shape that has 3 or more clean launches; the others marked "no result". (§7.2; the full table is in the survey log. No case fell under 3 launches, so there is no "no result" row.)
- [x] Time, reps or bytes: a written answer, with the packed-bytes formula and the numbers. (§7.3.)
- [x] Shape-feature table and a short list of what the slow shapes have in common. (§7.4.)
- [x] The script and its test committed, and the run folders cited. (`slow_state_survey.py`, `test_slow_state_survey.py`; run folders in §7.)

---

## 7. Results (2026-09-29)

**What was added:** `tests/pure_xdsl_blis_rvv/slow_state_survey.py` (reads every `logs/run_*/*/reps_*.txt.gz` with
`reps_report.py`'s reader and prints the four tables of §3) and `test_slow_state_survey.py` (made-up run folders, 14
tests). `reps_report.py`'s `read_reps` now also keeps each case's M, N, K, MC, KC and tile. The full output is
committed as `tests/pure_xdsl_blis_rvv/logs/issue47_slow_state_survey.md` (four tables, one row per case and per
file). Run it again with `python tests/pure_xdsl_blis_rvv/slow_state_survey.py --until 2026-09-28_235959 --out <file>` (`--until` leaves out the runs made after the logs this issue read, e.g. those of #46).

A launch is in a **slow state** when at least 5% of its reps are slow (`reps_report.py`'s two-means split, rep 0 left
out). One or two slow reps in a hundred are an interrupt, not a state; a case can show a small slow share without
having a state (for example B1 with one slow rep in 100).

### 7.1 Inventory

204 per-rep files in 193 run folders. All were made on core 2 or core 4 (no core-0 run, no run without provenance, no
unpinned run). Only the uninstrumented, reference-protocol files (104) are used for speed; the 100 instrumented
(warm-up protocol) files are counted but not used, as §5 says.

| board | build / protocol | packer | files | shapes |
|---|---|---|---|---|
| jbpi | uninstrumented, reference, core 4 | vector | 24 | B2, B3, B3_K256, the `nc` sweeps and N variants of #35, Reuse8/12/24 |
| jbpi | uninstrumented, reference, core 4 | scalar | 3 | the same sweep shapes |
| jbpi | uninstrumented, reference, core 4 | vector + zero-line (#43) | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | uninstrumented, reference, core 4 | vector | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| jbpi2 | uninstrumented, reference, core 4 | vector, `nc` overrides ignored | 14 (7 folders x 2 tiles) | B1, B4, B5, S1-S5 |
| jbpi2 | uninstrumented, reference, core 4 | vector + zero-line (#43) | 21 | B2, B3, B3_K256, B3_nc128, R24 |
| both | instrumented, warm-up (not used for speed) | vector, scalar, zero-line | 100 | B2, B3, B3_K256, R24 and the sweep shapes |

Every case in the speed tables has at least 3 launches (the thinnest are B4 and S1-S5 with 3 each), so "no result" is
empty. Older logs carry no record of other users' load on the board (the provenance probe for it was added on
2026-09-29, see the red note in §7.6).

### 7.2 Which shapes enter the slow state (vector packer, uninstrumented, reference protocol, core 4)

Slow reps as a share of a launch, average (lowest-highest launch); GOPs by mean time, then by median rep. Examples
of run folders: `run_2026-09-27_153743` (B3, B3_K256, R24 launch on `jbpi`), `run_2026-09-27_104338` (the `nc` sweep
on `jbpi`), `run_2026-09-28_122304` (B1/B4/B5/S1-S5 on `jbpi2`).

| shape (M x N x K), tile | launches | slow reps % | GOPs by mean time / by median rep (`jbpi` / `jbpi2`) | in a slow state? |
|---|---|---|---|---|
| B3 64x384x384, 16x15 (nc 384) | 21 `jbpi` + 21 `jbpi2` | 87-89 (73-95) | 4.49 / 4.28, 4.64 / 4.31 | yes, every launch |
| B3_K256 64x384x256, 16x15 | 24 + 21 | 95-97 (79-98) | 4.72 / 4.60, 4.68 / 4.59 | yes, every launch (the worst) |
| Reuse8 128x384x384 (nc 384), 16x15 | 3 | 66 (32-84) | 6.44 / 6.71 | yes, every launch |
| Reuse12 192x384x384 (nc 384), 16x15 | 3 | 64 (58-68) | 7.29 / 7.58 | yes, every launch |
| Reuse24 384x384x384, 16x15 | 21 + 21 | 40 (25-64) | 8.02 / 8.23, 7.96 / 8.14 | yes, every launch |
| B2 384x384x64, 16x15 | 21 + 21 | 0 | 12.40 / 12.48, 12.41 / 12.48 | no |
| B1 1024x384x1024, 32x6 / 32x7 | 4 + 4 (`jbpi2`) | 0 | 7.97 / 7.97 (32x6), 8.12 / 8.13 (32x7) | no |
| B4 4096x384x1024, 32x6 / 32x7 | 3 + 3 | 0 | 7.80 / 7.80 (32x6), 8.09 / 8.09 (32x7) | no |
| B5 1024x384x4096, 32x6 / 32x7 | 4 + 4 | 0 | 7.69 / 7.69 (32x6), 7.75 / 7.75 (32x7) | no |
| S1-S5 (1000-5000 cube), 32x6 / 32x7 | 3 + 3 each | 0 | 8.32-9.53 (by median rep the same to 0.01) | no |

The `nc` sweeps (#35, `jbpi`, 3 launches each) show the same picture for the smaller shapes: B3 at `nc` 64-384 and the N
variants (N192-N320) are slow in every launch (22-86%); Reuse8 and Reuse12 too (30-76%); Reuse24 only sometimes (0-13%
at `nc` 120-384, see §7.6). The scalar packer has it later and less (B3 at `nc` 384: 46%, at `nc` 128: 6%, B3_K256: 93%).
Both boards agree (B3 87-89% on both, B3_K256 95-97% on both, Reuse24 40% on both).

### 7.3 Time, reps or bytes?

Packed bytes per call, from the loop nest of `generate_pure_blis_project.py` (jc over N by `nc`, pc over K by `kc`,
pack the B block `kc x nc`, ic over M by `mc`, pack the A block `mc x kc`; the same count the driver keeps in
`pack_bytes`): every element of B goes through the packer once, and every element of A once per `nc` block of N,

`bytes_per_rep = 4 * (K*N + M*K*ceil(N/nc))` (fp32).

For B3 at `nc` 384 that is 0.66 MiB per rep; for Reuse24 at 384 it is 1.12 MiB.

Where the first slow rep sits (uninstrumented, vector packer, `jbpi`; slow-state launches only):

| case | launches in a slow state | first slow rep: reps (min / median / max) | seconds (min / median / max) | MiB packed before (mean) |
|---|---|---|---|---|
| B3, 16x15 | 21 | 3 / 14 / 80 | 0.007 / 0.036 / 0.244 | 14 |
| B3_K256 | 24 | 5 / 13 / 39 | 0.008 / 0.018 / 0.055 | 7 |
| Reuse24 | 21 | 20 / 78 / 411 | 0.25 / 0.99 / 4.9 | 103 |
| B3, scalar packer, `nc` 384 | 3 | 188 / 209 / 247 | 0.44 / 0.47 / 0.57 | 141 |

The spread of each case's switch point over its own launches is mostly 20-100% of the mean (0-140% at the ends), and the spread across cases is
66-92% for reps, 75-139% for seconds and 73-107% for bytes packed (per board and packer, table 3 of the survey log).
**None of the three is constant.** Reps has the smallest spread on every board and packer, but 66-92% is not "constant".
Within a case the first slow rep varies 8-27 times between launches (B3: 3 to 80 reps), and its spread is close
to 100% of its mean, which is what a chance event with a fixed rate looks like, not a threshold. That is a reading of
the numbers, not something tested here.

### 7.4 What the slow shapes have in common

Features are in table 4 of the survey log, slowest first (M, N, K, `nc`, `mc`, `kc`, loop blocks, block sizes, packed
bytes per FLOP, which leading dimensions are powers of two). What can be read from it, without a model:

- **All slow cases have M of 384 or less, K of 256 or more, N of 192 or more, and the 16x15 tile.** The slow share
  falls as M grows at K=384 and `nc`=384: B3 (M=64) 86-89%, Reuse8 (128) 66%, Reuse12 (192) 64%, Reuse24 (384) 40%.
- **B2 (M=384, K=64) has none.** Same tile and N as Reuse24 and B3, but one K block (`kc` = K = 64) instead of 2-3.
  B3_K256 (two K blocks) is the slowest of all, so it is not simply "more K blocks is worse".
- **Every shape with M of 1000 or more has none, but each of them also ran on 32x6 or 32x7, never on 16x15.** In this data,
  shape and tile move together, so the survey cannot say which of the two matters (§7.6).
- **Not the block sizes.** B and A blocks are far under the 512 KiB L2 in every case (B block plus A block at most
  384 KiB). The slow cases have B blocks of 48-192 KiB and A blocks of 32-192 KiB; the steady B2 has 96 and 96, inside both
  ranges. Not powers of two either: B3 (ldb=384) and B3_K256 (ldb=256) are both slow.
- **Not packed bytes per FLOP alone.** B2 packs 10.4 bytes per kFLOP, the same as Reuse24 (10.4), but has no slow state; B3 packs 36.5.
  B1 packs 7.2 and is steady, which fits, but B2 does not.

### 7.5 Which reading of §4 holds

- "Only B3 (and Reuse24 a little)": **no.** Reuse8, Reuse12 and every B3 variant have it, Reuse24 in every launch at
  25-64%. B3 is where it is largest, not where it is alone.
- "Many shapes, some briefly": **yes for M up to 384**, none of the big shapes. It is a property of the board for this
  kind of run (small M, several K blocks), with B3 at the top end. It does **not** put B4, B1, B5 and the squares "outside
  the problem" with certainty (§7.6).
- "Time, reps or bytes": **none is constant**; the switch looks like a chance event per rep, so it is not a time-based
  window (which would be constant in seconds) and not an access-history count (constant in bytes). This does not rule out
  a cause that depends on the history in some other way.

### 7.6 Limits

- **🔴 RED NOTE, other users' load.** These logs were made before the board's other users were recorded (added
  2026-09-29). On 2026-09-29 morning another user's job (`psiwins`, `spmv`, about 92% of a core, on core 0 of both boards
  for over an hour) was running; nothing shows whether one ran during the logs of 26-28 Sep. Even on another core it can
  move the chip's thermal state, so shares of 25-64% for Reuse24 and the launch-to-launch spread may include that.
- **Tile and M are not separated.** All big shapes ran on 32x6/32x7 (with `mc` 768), all small ones on 16x15. No 16x15 run of
  a big shape, and no 32x6 run of B3, exists.
- **Same case, different launch, different share.** Reuse24 at `nc` 384 was 25-64% in the 5-shape launches (21 per board)
  but 18%, 0% and 0% in the 3 sweep launches of #35 (`run_2026-09-27_104338`, `run_2026-09-27_111536`,
  `run_2026-09-27_112614`), which ran about 55 shape labels in one go. Reuse8 at `nc` 384 was 84%, 32% and 83% in those same three.
  The state may depend on what ran before in the launch, or on the time of day; nothing here tells which.
- **The fast start is coarse.** "First slow rep" is decided by the split of `reps_report.py`; a state that starts and stops
  again is only read at its first entry. The 5% cut is a choice, not a measurement.
- **Thin data.** B1, B4, B5 and S1-S5 are 3-4 launches on `jbpi2` only: "no slow state seen in 3-4 launches", not "no slow
  state".

### 7.7 Next steps (not run; each needs board time)

1. B3 (M=64) on the 32x6 tile, and B1 or Reuse24 on the 16x15 tile: separates tile from M.
2. An M sweep at K=384, N=384 on one tile (M = 64, 128, 192, 384, 768, 1024): finds where the slow state stops.
3. Reuse24 alone in a short launch against the same case last in a long launch: tests whether launch history matters.
4. All of these in a quiet window (midnight, no other user's job), on both boards, in the order A-B-B-A.
