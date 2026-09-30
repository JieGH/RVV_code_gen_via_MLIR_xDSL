# Issue #48: B3 and the packing: sweep the microkernel dimensions (MR x NR) and read packing and compute apart (follow-up of #41, #46, #47)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/48

**Comes from:** the engineering report, Key findings 6, 8 and 10 and Recommendation 4:
[issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf)
([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)).

**Role / Label:** `needs-triage` — new direction for B3, opened 2026-09-30 at the user's request before the session closed. Step 1 and the tooling are done (§7); no board run yet.
**Status:** 🔧 Step 1 done 2026-09-30: the packer table is in §7.1, from the packer code, checked by a test against the builders. The scripts for the sweep are written and tested (§7.3); **no board run has been made yet** (board time waits for the user's go, §7.3).
**Target:** `jbpi` and `jbpi2`, core 4, quiet window (no other user's job on the board; see `docs/agents/execution.md`, rule 5), house rules of #39.
**Shape(s):** B3 (64×384×384) first; B3 with K=256 and Reuse24 as controls; B2 as the steady control.
**Primary Sources:**
- [[0017-b3-b4-nr-sweep-eval-plan]] (#32) — the earlier NR sweep for B3 (MR=16, NR 16 down to 8), run on core 0 and read only as total speed.
- [[0025-b3-slow-state-root-cause]] (#41) and [[0030-reference-b3-per-rep]] (#46) — the slow state and the reference's steadier B3.
- [[0031-slow-state-survey-all-shapes]] (#47) — the slow state appears on the 16x15 tile with few rows and not on the 32-row tiles; tile and rows are not separated.
- `tests/pure_xdsl_blis_rvv/pack_builder.py`, `generate_pure_blis_project.py` — how the vector packers are built from MR and NR.
- `tests/pure_xdsl_blis_rvv/config_dim_sweep_b3.json` — the #32 sweep config, a starting point.
- New for this issue: `pack_dims_table.py` (the table of §7.1), `config_issue48_sweep.json` and `config_issue48_sweep_scalar.json` (the tiles),
  `run_issue48.sh` (the series), `issue48_report.py` (reads a series); tests `test_pack_dims_table.py`, `test_run_issue48.py`, `test_issue48_report.py`.

---

## 1. Direction (the user's idea)

The vector packer is not independent of the microkernel dimension. A vector register on this board holds 8 fp32 values
(VLEN 256), so how many vector loads and stores one packed panel takes, and how the last, partial vector of a panel is handled,
depend on MR (the height of an A panel) and NR (the width of a B panel). B3 has the smallest M of the shapes we run and the
largest share of its time in packing, and it is the shape that falls into the slow state. The suggestion is to **sweep the
microkernel dimensions for B3 and read packing and compute separately**, to see whether the packing side of B3 changes with
MR and NR in a way the total speed hides.

## 2. Why this direction

- The slow state hits both packing (1.6-3.1 times longer) and compute (16-66% longer) in slow reps (#36), but the report so far
  cannot say which side starts it.
- #46: on plain B3 the reference (16x14) is steady or mildly slow (0-22% of reps) and ours (16x15) is 87-89% slow. The tiles differ
  (14 against 15 columns) and nobody has tested that difference alone.
- #47: every slow shape ran on 16x15 and every steady big shape on 32x6/32x7. MR and NR are the one thing that moves with the
  split, so a sweep over MR and NR on one shape is the direct test of whether the tile matters.
- The earlier NR sweep (#32) ran on core 0, read total speed only, and did not look at reps or at the pack/compute split.
- (Added 2026-09-30, §7.2: #36 had run our 16x14 and 16x16 once, instrumented, so "nobody has tested it" holds only for the uninstrumented speed over several launches.)

## 3. Plan (the draft; §7.3 says what was built, and where it differs: controls on every tile, a scalar arm on 6 tiles)

1. **Read the packers.** Write down, from `pack_builder.py`, what the vector A packer and B packer do per panel for each MR and NR:
   vectors per column of an A panel (MR/8), the partial vector for NR not a multiple of 8, strides, bytes per panel row, whether
   panel rows straddle 64-byte cache lines (NR=15 is 60 bytes, NR=14 is 56, NR=16 is 64). Put the table in this issue before any run.
2. **Choose the sweep.** MR in {8, 16, 24, 32} and NR from 4 up to the register limit for each MR
   ((MR/8)×(1+NR) ≤ 32 as in #32's plan; include the over-budget 16x16 once as a known-bad reference). Keep B3 at `nc`=384 and `--no-nc-overrides`.
3. **Two builds per point.** Uninstrumented (reference protocol, `--tmin 10`) for speed and the per-rep slow-rep share (`reps_report.py`),
   and instrumented (`--raw-events x60-l2` where it helps) for the split of every rep into pack time and compute time
   (`Pack_Seconds`, `Compute_Seconds`). Instrumented numbers explain, they are not quoted as speed.
4. **Both packers.** Vector (shipped) and scalar (`--scalar-pack`), so a packer effect and a tile effect can be told apart.
5. **Repeats.** At least 3 launches per board, order alternated (A-B-B-A style rotation as in `run_issue46.sh`), core 4, quiet window, both boards.
   Controls in the same launches: B3 with K=256, Reuse24 and B2 (built as all four shapes on every tile, §7.3).
6. **Read it** as a table per MR×NR: GOPs by mean time and by median rep, slow-rep share, fast start, pack share of the time, pack seconds
   and compute seconds in fast and in slow reps.

## 4. How to read it

- **Packing time and the slow state move with NR or MR** (for example the slow share drops at NR=14 or at a multiple of 8): the tile decides
  it, and the reference's 16x14 result is explained by the tile. Next: choose the tile for B3 from this table, not from the register budget alone.
- **Only compute moves with the tile, packing does not:** the packer is not where B3's problem starts; look at the microkernel's access pattern.
- **Neither moves:** the tile is not the cause; go back to buffer placement (#46 §7.2) and the loop nest (#44, #38).
- Watch for a **register cliff** first: points above the 32-register limit will be slow for a reason that is already known.

## 5. Out of scope

Changing the packers; changing the shipped tiles (any change is a separate decision, on the evidence of this table); the `cbo.zero` switch (#43);
the reference design (#46) except as a same-launch control if a 16x14 build of ours is added.

## 6. Closure

- [x] Table of what each vector packer does per MR and NR, written into this issue before any run (§7.1, 2026-09-30).
- [ ] Sweep run on both boards in quiet windows, at least 3 launches per point, logs committed, no red note (or the note stated).
- [ ] Packing and compute read apart, for fast and slow reps, per tile.
- [ ] Written answer: does the packing side of B3 depend on MR and NR, and does the slow state?

---

## 7. Step 1 and the tooling (2026-09-30, no board time)

### 7.1 What the vector packers do per MR and NR

Both packers are the `_if` functions of `pack_builder.py` (the driver never calls the `_else` ones). For a block of `mc` x `nc` x `kc`:

- **A packer, `pack_RB_v_if`.** For each panel of MR rows, for each k, for each chunk of 8 rows: one `vle32` from column k of A (contiguous) and one
  `vse32` into `Ac` (row k of the panel, MR floats = MR*4 bytes apart). MR/8 vectors per k. A chunk's length is `max(min(rows, chunk end), chunk start) - chunk start`,
  so a panel narrower than a chunk still issues the chunk's load and store with `vl = 0` (MR=24 on 64 rows: the last panel has 16 rows, chunks 8+8+0).
- **B packer, `pack_CB_v_if`.** For each panel of NR columns, for each k, for each chunk of 8 columns: one **strided** `vlse32` (stride `ldb`*4 bytes, one element
  from each of up to 8 columns of B) and one `vse32` into `Bc` (row k of the panel, NR*4 bytes apart). ceil(NR/8) vectors per k. **The panel reads NR columns of B at once**,
  so it walks NR read streams in parallel, each moving 4 bytes per k: the thing #44 suspects the prefetcher cannot follow. NR is also the row width of `Bc`.
- **What moves with MR:** the number of A vectors per k (MR/8), the number of A panels and whether the last A panel is partial (only MR=24 on 64 rows). All stores of `Ac` are whole 32-byte chunks on 32-byte boundaries: no store crosses a cache line for any MR that is a multiple of 8.
- **What moves with NR:** the number of read streams (NR), the number of B vectors per k (1 up to NR=8, 2 up to 16, then 3, 4), the partial vector (NR not a multiple of 8:
  15 = 8+7, 14 = 8+6, 12 = 8+4), the bytes between rows of `Bc` (60 for NR=15, 56 for 14, 64 for 16), and so how many of its vector stores straddle two lines
  (41% for NR=15, 31% for 14, 12% for 12, **0% for 8, 16 and 24**), and whether the last column panel is partial (384 is divisible by 6, 8, 12, 16 and 24, not by 7, 14, 15, 31).
- **The tile also changes the blocking.** (kc, "K blocks" and "on a line" in the table are for the first, full K block; the last block is smaller, e.g. 170, 170, 44 for kc=170 and 132, 132, 120 for kc=132, and its panels can differ in alignment.) `get_optim_mc_nc_kc` takes MR and NR: for B3 it gives kc = 128 at 16x12..16x16 and at MR=32, **256** at 16x6..16x8 and 8x8..8x16,
  **170** at MR=24 and 8x24, **132** at 8x31 (mc = 64 and nc = 384 everywhere; for MR=16 only NR=8 and below gets 256, NR=9..16 all get 128). So a tile change is also a kc change, and with kc = 170
  or 132 some `Bc` panels are not a whole number of lines (8x31, 24x6 and 24x9, marked NO below). The reps logs carry the real mc/kc of every run; the report prints kc next to each tile, and kc
  has to be read as a second thing that moved.
- **Fringe tiles go the slow way.** The shipped build is pinned to one tile; every tile with `cur_mr < MR` or `cur_nr < NR` runs the kernel into a temp buffer and copies C in and out
  with scalar loops (`generate_pure_blis_project.py`, `kernel_call_block`; C is copied in only when beta is not 0, i.e. from the second K block on). How many tiles that is per block is the last column of the table: 4 of 104 at 16x15 and 4 of 112 at 16x14,
  none at 16x16, 16x12 and 16x8, 64 of 192 at 24x6. It is compute time, not pack time, and it moves with the tile.
- **The register limit** is (MR/8) x (1 + NR) live vectors (the #32 rule): 16x16 needs 34 and is the only over-budget tile in the sweep (the known-bad reference). 16x15, 8x31 and 32x7 use
  all 32; 16x14 and 24x9 use 30.

The numbers (from `python tests/pure_xdsl_blis_rvv/pack_dims_table.py`, B3 = 64x384x384; "K blocks" = ceil(384/kc); stores over a line assume a panel that starts on a line, which the
"on a line" column says is true for every tile except the ones marked NO):

| tile | mc | nc | kc | K blocks | A: vectors per k (vl) | A panels (fringe rows, vl) | Ac row B | A stores over a line | B: strided loads per k (vl) | B panels (fringe cols, vl) | Bc row B | B stores over a line | Bc panel B, on a line | live vregs | fringe tiles / tiles |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16x6 | 64 | 384 | 256 | 2 | 2 (8+8) | 4 (none) | 64 | 0% | 1 (6) | 64 (none) | 24 | 25% | 6144 yes | 14 | 0 / 256 |
| 16x7 | 64 | 384 | 256 | 2 | 2 (8+8) | 4 (none) | 64 | 0% | 1 (7) | 55 (6, 6) | 28 | 38% | 7168 yes | 16 | 4 / 220 |
| 16x8 | 64 | 384 | 256 | 2 | 2 (8+8) | 4 (none) | 64 | 0% | 1 (8) | 48 (none) | 32 | 0% | 8192 yes | 18 | 0 / 192 |
| 16x12 | 64 | 384 | 128 | 3 | 2 (8+8) | 4 (none) | 64 | 0% | 2 (8+4) | 32 (none) | 48 | 12% | 6144 yes | 26 | 0 / 128 |
| 16x14 | 64 | 384 | 128 | 3 | 2 (8+8) | 4 (none) | 64 | 0% | 2 (8+6) | 28 (6, 6+0) | 56 | 31% | 7168 yes | 30 | 4 / 112 |
| 16x15 | 64 | 384 | 128 | 3 | 2 (8+8) | 4 (none) | 64 | 0% | 2 (8+7) | 26 (9, 8+1) | 60 | 41% | 7680 yes | 32 | 4 / 104 |
| 16x16 | 64 | 384 | 128 | 3 | 2 (8+8) | 4 (none) | 64 | 0% | 2 (8+8) | 24 (none) | 64 | 0% | 8192 yes | 34 (over) | 0 / 96 |
| 8x8 | 64 | 384 | 256 | 2 | 1 (8) | 8 (none) | 32 | 0% | 1 (8) | 48 (none) | 32 | 0% | 8192 yes | 9 | 0 / 384 |
| 8x14 | 64 | 384 | 256 | 2 | 1 (8) | 8 (none) | 32 | 0% | 2 (8+6) | 28 (6, 6+0) | 56 | 31% | 14336 yes | 15 | 8 / 224 |
| 8x15 | 64 | 384 | 256 | 2 | 1 (8) | 8 (none) | 32 | 0% | 2 (8+7) | 26 (9, 8+1) | 60 | 41% | 15360 yes | 16 | 8 / 208 |
| 8x16 | 64 | 384 | 256 | 2 | 1 (8) | 8 (none) | 32 | 0% | 2 (8+8) | 24 (none) | 64 | 0% | 16384 yes | 17 | 0 / 192 |
| 8x24 | 64 | 384 | 170 | 3 | 1 (8) | 8 (none) | 32 | 0% | 3 (8+8+8) | 16 (none) | 96 | 0% | 16320 yes | 25 | 0 / 128 |
| 8x31 | 64 | 384 | 132 | 3 | 1 (8) | 8 (none) | 32 | 0% | 4 (8+8+8+7) | 13 (12, 8+4+0+0) | 124 | 42% | 16368 NO | 32 | 8 / 104 |
| 24x6 | 64 | 384 | 170 | 3 | 3 (8+8+8) | 3 (16, 8+8+0) | 96 | 0% | 1 (6) | 64 (none) | 24 | 25% | 4080 NO | 21 | 64 / 192 |
| 24x8 | 64 | 384 | 170 | 3 | 3 (8+8+8) | 3 (16, 8+8+0) | 96 | 0% | 1 (8) | 48 (none) | 32 | 0% | 5440 yes | 27 | 48 / 144 |
| 24x9 | 64 | 384 | 170 | 3 | 3 (8+8+8) | 3 (16, 8+8+0) | 96 | 0% | 2 (8+1) | 43 (6, 6+0) | 36 | 22% | 6120 NO | 30 | 45 / 129 |
| 32x4 | 64 | 384 | 128 | 3 | 4 (8+8+8+8) | 2 (none) | 128 | 0% | 1 (4) | 96 (none) | 16 | 0% | 2048 yes | 20 | 0 / 192 |
| 32x6 | 64 | 384 | 128 | 3 | 4 (8+8+8+8) | 2 (none) | 128 | 0% | 1 (6) | 64 (none) | 24 | 25% | 3072 yes | 28 | 0 / 128 |
| 32x7 | 64 | 384 | 128 | 3 | 4 (8+8+8+8) | 2 (none) | 128 | 0% | 1 (7) | 55 (6, 6) | 28 | 38% | 3584 yes | 32 | 2 / 110 |

`test_pack_dims_table.py` checks the vector counts of the table against the real builders (the loads, stores and `setvl`s in the innermost loop of `build_pack_CB_v_if` and
`build_pack_RB_v_if` for 11 NR and 7 MR values), so the table cannot drift from `pack_builder.py`.

### 7.2 Two things found while reading that change §2

- **#36 already ran our 16x14 and 16x16, once.** With the vector packer B3 was two-speed at 16x14, 16x15 and 16x16 alike (instrumented build, `jbpi`, one launch: 412, 842 and 569 slow reps of
  1000); with the scalar packer 16x14 and 16x15 were two-speed and 16x16 was not ([[0021-b3-vector-packer-slow]] §2). The reference's 16x14 is steady in #46. So if the 14th column were the
  cause, our own 16x14 would have been steady too, and that one instrumented launch says it was not. The claim in §2 ("nobody has tested that tile difference alone") is therefore only true for
  the uninstrumented speed, three launches per board and a quiet window; this sweep is that test, and it starts from the expectation that 16x14 is two-speed too (412 against 842 slow reps of 1000: not steady, about half as often).
- **#32 already moved kc, and read only speed.** Its MR=16 sweep crossed from kc = 256 (16x8) to kc = 128 (16x9 and up). It found the packer gap (scalar ahead of vector by 21-78% on B3 at every NR,
  both boards) and the register cliff, not a slow-rep share per NR. The new tiles bring kc = 170 and 132 too. If the slow state follows kc instead of NR, it shows up as a step at the tiles where kc
  changes, not as a slope in NR.

### 7.3 The sweep, as built

- **Tiles** (`config_issue48_sweep.json`, 19 pinned rows, all four shapes of `datasets/issue46_ref_b3.dat` on each: B3, B3_K256, B2, Reuse24, so every tile carries its own controls):
  MR=16 x NR 6, 7, 8, 12, 14, 15, 16; MR=8 x NR 8, 14, 15, 16, 24, 31; MR=24 x NR 6, 8, 9; MR=32 x NR 4, 6, 7. NR stops at the register limit for each MR (31, 15, 9, 7), except 16x16, kept once as the
  known-bad reference. The plan's "NR from 4" is not followed for every MR and some NR between are left out (16x9..11 and 16x13, 8x17..23 and 25..30): 19 tiles already take about 50 min a launch, and
  the points kept are the ones that tell the causes above apart. Any gap in the curve is a later, cheaper follow-up. Why these: 16x14 against 16x15 is #46's open pair; 16x6/8/12/16 have no fringe column panel and 16x7/14/15 have one, so a fringe panel and a partial vector come apart
  (12 = 8+4 has a partial vector and no fringe panel); MR=8 has one A vector per k-step; MR=32 is the row count of the steady big shapes (#47).
- **Scalar arm** (`config_issue48_sweep_scalar.json`, 6 tiles: 16x8, 16x14, 16x15, 16x16, 8x15, 32x6): the scalar packer reads the same addresses in the same order, so a packer effect and a tile effect come apart
  on the tiles that matter, without doubling the board time.
- **Four arms** in `run_issue48.sh`: `vec` and `scalar` (uninstrumented, reference protocol, `--tmin 10`: speed, slow-rep share), `vec-instr` and `scalar-instr` (`--instr`, warm-up protocol, 1000 reps:
  `Pack_Seconds` and `Compute_Seconds` of every rep; `--raw-events x60-l2` optional). Always `--no-nc-overrides --cpu 4`. The arms rotate from launch to launch (vec, scalar / scalar, vec / ...), the two boards take
  turns, one run at a time under the shared lock, other users' processes are read before and after every run (RED NOTE, never a stop).
- **Board time** (about, from the 2.4-minute run of four shapes on one tile on 2026-09-29): 50 min for one `vec` or `vec-instr` launch on one board, 20 min for one scalar arm. 3 launches on each of the two boards:
  `vec` + `scalar` about 7 h, `vec-instr` + `scalar-instr` about 7 h. So two quiet windows (two nights), or one if the instrumented arms are cut to one board.
- **Reading** (`issue48_report.py <series folder>`): per board and shape, one row per tile and packer: GOPs by mean time (average, lowest-highest launch), by median rep, slow-rep share, fast start, CV, kc;
  and for the instrumented arms: pack share of pack+compute, and pack and compute milliseconds per rep in fast reps and in slow reps. Fewer than 3 launches and any RED NOTE are flagged.
- **Tests:** `test_pack_dims_table.py` (34), `test_run_issue48.py` (13, stubbed bench, no board), `test_issue48_report.py` (10), plus one new test in `test_reps_report.py` (`reps_report.read_reps` now keeps
  each rep's `Pack_Seconds` and `Compute_Seconds`). `pyright` clean on the new files.

### 7.4 Still to do

- Board runs (§7.3): waiting for the user's go. Start with `run_issue48.sh --start-at 00:00 --arms vec,scalar`, then `--arms vec-instr,scalar-instr`.
- Closure items 2-4 of §6 (sweep run, packing and compute read apart, written answer).
