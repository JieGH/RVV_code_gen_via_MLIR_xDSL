# Issue #48: B3 and the packing: sweep the microkernel dimensions (MR x NR) and read packing and compute apart (follow-up of #41, #46, #47)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/48

**Comes from:** the engineering report, Key findings 6, 8 and 10 and Recommendation 4:
[issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf)
([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)).

**Role / Label:** `needs-triage` — new direction for B3, opened 2026-09-30 at the user's request before the session closed. Step 1 and the tooling are done (§7); no board run yet.
**Report:** [issue-48-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-48-brief.tex)), 1 October 2026.
**Status:** ✅ Measured 2026-09-30 on both boards (25 tiles, vector and scalar packer, uninstrumented and instrumented, 3-4 launches each, core 4, no other user's job at any reading). **Answer (§8): the slow state of B3 follows NR, not MR or kc. With the vector packer every tile with NR up to 6 is steady and every tile with NR of 7 or more is two-speed, at every MR from 4 to 32; the scalar packer is far steadier at NR of 8 and up. Our 16x14 is as slow as our 16x15, so the tile does not explain why the reference's 16x14 is steady.** Packer table in §7.1; the sweep, the tooling and the results in §7.3 and §8.
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

## 3. Plan (the draft; §7.3 says what was built, and where it differs: controls on every tile, a scalar arm on 7 tiles)

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
- [x] Sweep run on both boards, at least 3 launches per point (4 for the vector and scalar arms on `jbpi2`), logs committed, no red note: no other user's job at any reading (§8.1, 2026-09-30). Not in a midnight window; none was needed, the boards were quiet.
- [x] Packing and compute read apart, for fast and slow reps, per tile (§8.4).
- [x] Written answer: does the packing side of B3 depend on MR and NR, and does the slow state? (§8.5)

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
- **The tile also changes the blocking.** (kc, "K blocks" and "on a line" in the table are for the first, full K block; the last block is smaller, e.g. 170, 170, 44 for kc=170 and 132, 132, 120 for kc=132, and its panels can differ in alignment.) `get_optim_mc_nc_kc` takes MR and NR: for B3 it gives kc = 128 at 16x12..16x16 and at MR=32, **256** at 16x4..16x8, 4x16 and 8x8..8x16, **384** (the whole K, one K block) at 4x4, 4x8 and 8x4,
  **170** at MR=24, 12x8 and 8x24, **132** at 8x31 (mc = 64 and nc = 384 everywhere; for MR=16 only NR=8 and below gets 256, NR=9..16 all get 128). So a tile change is also a kc change, and with kc = 170
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
| 4x4 | 64 | 384 | 384 | 1 | 1 (4) | 16 (none) | 16 | 0% | 1 (4) | 96 (none) | 16 | 0% | 6144 yes | 5 | 0 / 1536 |
| 4x8 | 64 | 384 | 384 | 1 | 1 (4) | 16 (none) | 16 | 0% | 1 (8) | 48 (none) | 32 | 0% | 12288 yes | 9 | 0 / 768 |
| 4x16 | 64 | 384 | 256 | 2 | 1 (4) | 16 (none) | 16 | 0% | 2 (8+8) | 24 (none) | 64 | 0% | 16384 yes | 17 | 0 / 384 |
| 8x4 | 64 | 384 | 384 | 1 | 1 (8) | 8 (none) | 32 | 0% | 1 (4) | 96 (none) | 16 | 0% | 6144 yes | 5 | 0 / 768 |
| 16x4 | 64 | 384 | 256 | 2 | 2 (8+8) | 4 (none) | 64 | 0% | 1 (4) | 96 (none) | 16 | 0% | 4096 yes | 10 | 0 / 384 |
| 12x8 | 64 | 384 | 170 | 3 | 2 (8+4) | 6 (4, 4+0) | 48 | 13% | 1 (8) | 48 (none) | 32 | 0% | 5440 yes | 18 | 48 / 288 |

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

- **Tiles** (`config_issue48_sweep.json`, 25 pinned rows, all four shapes of `datasets/issue46_ref_b3.dat` on each: B3, B3_K256, B2, Reuse24, so every tile carries its own controls):
  MR=16 x NR 6, 7, 8, 12, 14, 15, 16; MR=8 x NR 8, 14, 15, 16, 24, 31; MR=24 x NR 6, 8, 9; MR=32 x NR 4, 6, 7; and, added at the user's request on 2026-09-30, the small and odd tiles 4x4, 4x8, 4x16, 8x4, 16x4 and 12x8 (a side of 4 is one partial vector with vl = 4; 12 is an MR that is not a multiple of 8, so its A chunks are 8+4; 4x4 has 1536 tiles per block, so per-tile cost should dominate there). NR stops at the register limit for each MR (31, 15, 9, 7), except 16x16, kept once as the
  known-bad reference. The plan's "NR from 4" is not followed for every MR and some NR between are left out (16x9..11 and 16x13, 8x17..23 and 25..30): 25 tiles already take about 60 min a launch, and
  the points kept are the ones that tell the causes above apart. Any gap in the curve is a later, cheaper follow-up. Why these: 16x14 against 16x15 is #46's open pair; 16x6/8/12/16 have no fringe column panel and 16x7/14/15 have one, so a fringe panel and a partial vector come apart
  (12 = 8+4 has a partial vector and no fringe panel); MR=8 has one A vector per k-step; MR=32 is the row count of the steady big shapes (#47).
- **Scalar arm** (`config_issue48_sweep_scalar.json`, 7 tiles: 16x8, 16x14, 16x15, 16x16, 8x15, 32x6, 4x4): the scalar packer reads the same addresses in the same order, so a packer effect and a tile effect come apart
  on the tiles that matter, without doubling the board time.
- **Four arms** in `run_issue48.sh`: `vec` and `scalar` (uninstrumented, reference protocol, `--tmin 10`: speed, slow-rep share), `vec-instr` and `scalar-instr` (`--instr`, warm-up protocol, 1000 reps:
  `Pack_Seconds` and `Compute_Seconds` of every rep; `--raw-events x60-l2` optional). Always `--no-nc-overrides --cpu 4`. The arms rotate from launch to launch (vec, scalar / scalar, vec / ...), the two boards take
  turns, one run at a time under the shared lock, other users' processes are read before and after every run (RED NOTE, never a stop).
- **Board time** (about, from the 2.4-minute run of four shapes on one tile on 2026-09-29): 60 min for one `vec` or `vec-instr` launch on one board, 20 min for one scalar arm. 3 launches on each of the two boards:
  `vec` + `scalar` about 8 h, `vec-instr` + `scalar-instr` about 8 h. So two quiet windows (two nights), or one if the instrumented arms are cut to one board.
- **Reading** (`issue48_report.py <series folder>`): per board and shape, one row per tile and packer: GOPs by mean time (average, lowest-highest launch), by median rep, slow-rep share, fast start, CV, kc;
  and for the instrumented arms: pack share of pack+compute, and pack and compute milliseconds per rep in fast reps and in slow reps. Fewer than 3 launches and any RED NOTE are flagged.
- **Tests:** `test_pack_dims_table.py` (34), `test_run_issue48.py` (13, stubbed bench, no board), `test_issue48_report.py` (10), plus one new test in `test_reps_report.py` (`reps_report.read_reps` now keeps
  each rep's `Pack_Seconds` and `Compute_Seconds`). `pyright` clean on the new files.

### 7.4 Still to do

Nothing for the closure of this issue; the follow-ups are in §8.6.

---

## 8. Results (2026-09-30)

### 8.1 What was run, and under what conditions

- **Runs.** The 25-tile sweep on both boards, `--no-nc-overrides` (B3 at nc=384), core 4, `--tmin 10` for the uninstrumented arms (reference protocol), `--instr` (warm-up protocol, 1000 reps) for the instrumented ones. Each tile ran B3, B3_K256, B2 and Reuse24.
  Vector arms and scalar arms, uninstrumented and instrumented. Launches: `jbpi2` 4 for `vec` and `scalar` (one from the series that was stopped and restarted, see below) and 3 for the instrumented arms; `jbpi` 3 for each.
- **The boards ran at the same time**, one series each (separate board workspaces and host checkouts: the boards share one NFS home). The two timings should be independent (separate hardware; only the NFS server and the jump host are shared), but the runs were not one after the other.
- **Other users: none.** All 26 run folders record `other users : none` at the start of their run, and no series wrote a `load_warnings.txt` (the processes of each board are read before and after every run). The other cores of both boards were at 0-1% busy in every per-tile timing window but two (`busy share` in each `provenance.txt`), both instrumented: 14% on core 7 during 8x14 on `jbpi` (`run_2026-09-30_163007`) and 2-4% on cores 3 and 5 during 8x31 on `jbpi2` (`run_2026-09-30_192902`). The process snapshots before and after those runs list no other user's job. Neither touches a speed number. `issue48_report.py` prints no RED NOTE. (Earlier in the day, 12:19, a `psiwins` job of 1.6-1.8 cores was on both boards; it was gone by 14:04 and did not come back. The short smoke test at 12:26-12:40, made under that job, was deleted and is not used.)
- **Series folders** (all committed): `logs/issue48_2026-09-30_124654` (stopped after 2 runs, when the work was split across the boards; its 2 runs are kept: `run_2026-09-30_134329`, `run_2026-09-30_135936`), `logs/issue48_2026-09-30_140902` (`jbpi2`, 12 runs) and `logs/issue48_2026-09-30_140902_jbpi` (`jbpi`, 12 runs; renamed on copy, both series started in the same second).
  12 of 12 runs ok in each of the last two. Full report, all shapes and tiles: `logs/issue48_2026-09-30_report.md` (`python tests/pure_xdsl_blis_rvv/issue48_report.py <the three series folders>`). Each run's own console log is `<series folder>/<board>_L<n>_<position>_<arm>.log`.
- **Numbers below** are GOPs by mean time (the reference's formula) averaged over the launches, with the share of slow reps in brackets, from the uninstrumented builds. The instrumented builds (§8.4) explain where time goes and are not quoted as speed.

### 8.2 B3 (64x384x384) per tile: speed and slow-rep share

| tile | kc | vector, jbpi2 (4 launches) | vector, jbpi (3) | scalar, jbpi2 (4) | scalar, jbpi (3) |
|---|---|---|---|---|---|
| 4x4 | 384 | 3.55 (0%) | 3.55 (0%) | 3.39 (0%) | 3.38 (0%) |
| 4x8 | 384 | 2.75 (78%) | 2.62 (82%) | - | - |
| 4x16 | 256 | 2.62 (81%) | 2.73 (76%) | - | - |
| 8x4 | 384 | 4.91 (0%) | 4.91 (0%) | - | - |
| 8x8 | 256 | 3.46 (88%) | 3.84 (73%) | - | - |
| 8x14 | 256 | 3.61 (89%) | 3.48 (93%) | - | - |
| 8x15 | 256 | 3.37 (92%) | 3.54 (90%) | 4.73 (49%) | 4.83 (46%) |
| 8x16 | 256 | 3.57 (88%) | 3.75 (81%) | - | - |
| 8x24 | 170 | 4.92 (54%) | 5.20 (53%) | - | - |
| 8x31 | 132 | 4.32 (75%) | 4.11 (84%) | - | - |
| 12x8 | 170 | 2.44 (72%) | 2.36 (82%) | - | - |
| 16x4 | 256 | 5.32 (0%) | 5.30 (0%) | - | - |
| 16x6 | 256 | 6.71 (0%) | 6.77 (0%) | - | - |
| 16x7 | 256 | 4.76 (68%) | 4.44 (79%) | - | - |
| 16x8 | 256 | 3.56 (94%) | 3.72 (89%) | 7.28 (2%) | 7.03 (8%) |
| 16x12 | 128 | 5.10 (78%) | 4.71 (88%) | - | - |
| 16x14 | 128 | 4.72 (86%) | 4.51 (81%) | 6.98 (33%) | 7.19 (32%) |
| 16x15 | 128 | 4.43 (92%) | 4.53 (89%) | 6.77 (39%) | 6.13 (52%) |
| 16x16 | 128 | 3.53 (67%) | 3.29 (25%) | 3.83 (60%) | 3.79 (55%) |
| 24x6 | 170 | 5.35 (0%) | 5.38 (0%) | - | - |
| 24x8 | 170 | 3.30 (89%) | 3.33 (87%) | - | - |
| 24x9 | 170 | 3.69 (83%) | 3.55 (89%) | - | - |
| 32x4 | 128 | 5.78 (0%) | 5.80 (0%) | - | - |
| 32x6 | 128 | 6.33 (0%) | 6.32 (0%) | 5.89 (0%) | 5.92 (0%) |
| 32x7 | 128 | 5.39 (83%) | 5.30 (87%) | - | - |

- **Steady** (0% slow reps, both boards, every launch): 4x4, 8x4, 16x4, 16x6, 24x6, 32x4, 32x6. **Two-speed** (53-94% slow): every tile with NR of 7 or more, at every MR from 4 to 32: 4x8, 4x16, 8x8, 8x14..8x31, 12x8, 16x7, 16x8, 16x12..16x15, 24x8, 24x9, 32x7. 16x16 (over the register budget) is two-speed on both boards too (67% and 25% of reps), slowly.
- **kc does not explain it.** The steady tiles have kc = 384 (4x4, 8x4), 256 (16x4, 16x6), 170 (24x6) and 128 (32x4, 32x6); the two-speed ones have kc = 384 (4x8), 256, 170 and 128.
- **8x24** (kc = 170, 3 B vectors per k) is the least slow two-speed tile (53-54% of reps). **16x16** and 8x31 (the ends of the register range) are slow too.
- **The best B3 tiles are NR=6**: 16x6 at 6.71 / 6.77 GOPs and 32x6 at 6.33 / 6.32, against the shipped 16x15 at 4.43 / 4.53 (`jbpi2` / `jbpi`). But see §8.3.

### 8.3 The controls on the same tiles (vector packer; `jbpi2` / `jbpi`; GOPs, slow reps in brackets)

| tile | B3 | B3_K256 | B2 | Reuse24 |
|---|---|---|---|---|
| 16x6 | 6.71 (0%) / 6.77 (0%) | 4.49 (58%) / 4.96 (48%) | 10.27 (0%) / 10.25 (0%) | 9.14 (0%) / 9.11 (0%) |
| 16x8 | 3.56 (94%) / 3.72 (89%) | 6.74 (0%) / 6.76 (0%) | 10.24 (0%) / 10.33 (0%) | 8.94 (0%) / 8.64 (0%) |
| 16x14 | 4.72 (86%) / 4.51 (81%) | 4.44 (93%) / 4.52 (93%) | 11.87 (0%) / 11.84 (0%) | 8.29 (0%) / 8.31 (0%) |
| 16x15 | 4.43 (92%) / 4.53 (89%) | 4.93 (88%) / 4.76 (94%) | 12.59 (0%) / 12.56 (0%) | 8.45 (17%) / 8.79 (4%) |
| 16x16 | 3.53 (67%) / 3.29 (25%) | 3.45 (88%) / 3.73 (77%) | 7.95 (0%) / 7.95 (0%) | 4.93 (0%) / 4.89 (0%) |
| 32x6 | 6.33 (0%) / 6.32 (0%) | 5.54 (30%) / 5.54 (19%) | 10.65 (0%) / 10.62 (0%) | 9.37 (0%) / 9.35 (0%) |
| 32x7 | 5.39 (83%) / 5.30 (87%) | 5.93 (62%) / 6.01 (59%) | 11.80 (0%) / 11.70 (0%) | 9.81 (0%) / 9.81 (0%) |
| 24x6 | 5.35 (0%) / 5.38 (0%) | 3.56 (85%) / 3.57 (81%) | 9.33 (0%) / 9.30 (0%) | 6.36 (14%) / 6.29 (69%) |
| 8x8 | 3.46 (88%) / 3.84 (73%) | 3.54 (89%) / 3.39 (95%) | 8.14 (0%) / 8.14 (0%) | 7.55 (0%) / 7.52 (0%) |
| 4x4 | 3.55 (0%) / 3.55 (0%) | 3.93 (0%) / 3.91 (0%) | 4.19 (0%) / 4.20 (0%) | 4.21 (0%) / 4.22 (0%) |

- **B2 and Reuse24 are steady on almost every tile**, so the slow state is not the board or the day: on the same launches where B3 is 90% slow, B2 is at 0%. The exceptions are Reuse24 at 16x15 (17% / 4% slow reps), 24x6 (14% / 69%) and 8x24 (11% / 4%).
- **B3_K256 is not the same rule as B3.** At 16x8 it is steady (6.74 / 6.76 GOPs, 0% slow) though B3 at 16x8 is 89-94% slow; at 16x6 it is two-speed (58% / 48%) though B3 at 16x6 is steady; at 32x6 it is partly slow (30% / 19%). So "NR of 6 or less is steady" holds for B3 (K=384) and not for K=256: the threshold moves with K.
- **No tile wins every shape.** 16x6 is the best for B3 (6.7) and Reuse24 (9.1) and loses B2 (10.3 against 12.6 at 16x15). 16x15 is the best for B2 and loses B3 (4.5) badly. The shipped tile is a choice between the shapes, not a setting.

### 8.4 Packing and compute apart (instrumented builds, B3; debug numbers, not speed)

Mean time per rep in milliseconds, pooled over the launches: fast reps, then slow reps, and pack's share of pack + compute.

| tile, packer | jbpi2: fast reps pack / compute ms; slow reps; pack share | jbpi: same |
|---|---|---|
| 4x4 vector | 1.09 / 4.48; slow no slow reps; pack 20% | 1.08 / 4.47; slow no slow reps; pack 19% |
| 16x6 vector | 0.97 / 1.85; slow no slow reps; pack 34% | 0.99 / 1.86; slow no slow reps; pack 35% |
| 32x6 vector | 1.32 / 1.73; slow no slow reps; pack 43% | 1.32 / 1.74; slow no slow reps; pack 43% |
| 32x7 vector | 1.25 / 1.90; slow no slow reps; pack 40% | 1.29 / 1.94; slow 1.47 / 2.14; pack 40% |
| 8x8 vector | 0.79 / 2.27; slow 1.70 / 3.81; pack 30% | 0.83 / 2.29; slow 1.69 / 3.76; pack 31% |
| 16x8 vector | 0.76 / 1.86; slow 1.69 / 3.32; pack 33% | 0.76 / 1.88; slow 1.69 / 3.29; pack 33% |
| 16x14 vector | 0.72 / 1.89; slow 1.52 / 2.74; pack 33% | 0.66 / 1.80; slow 1.43 / 2.66; pack 32% |
| 16x15 vector | 0.71 / 1.80; slow 1.48 / 2.68; pack 34% | 0.73 / 1.79; slow 1.46 / 2.67; pack 34% |
| 16x8 scalar | 0.77 / 1.79; slow (2 slow reps); pack 30% | 0.80 / 1.79; slow 1.50 / 2.68; pack 31% |
| 16x14 scalar | 0.69 / 1.67; slow 1.15 / 1.97; pack 30% | 0.70 / 1.68; slow 1.20 / 1.97; pack 30% |
| 16x15 scalar | 0.71 / 1.51; slow 1.13 / 1.82; pack 33% | 0.72 / 1.51; slow 1.39 / 2.00; pack 35% |
| 32x6 scalar | 1.50 / 1.73; slow no slow reps; pack 46% | 1.50 / 1.74; slow no slow reps; pack 46% |

- **In a slow rep both halves slow down**, on every tile: pack takes 1.6-2.2 times as long (vector, 16x8: 0.76 to 1.69 ms; 16x15: 0.71 to 1.48; scalar 16x15: 0.71 to 1.13) and compute 1.2-1.8 times (vector, 16x8: 1.86 to 3.32 ms; scalar 16x15: 1.51 to 1.82). This is the #36 finding, now on 25 tiles: the slow state is not a packer-only effect.
- **Tiles that never go slow have no slow reps in either half** (4x4, 16x6, 32x6 vector, 32x6 scalar): pack and compute are each one speed.
- **In fast reps the pack time depends on the tile** (vector packer): 16x6 needs 0.97-0.99 ms against 0.66-0.76 ms for 16x8..16x15, and 32x6 needs 1.32 ms against 0.76 at 16x8. The reads and writes are the same number of bytes for B3 at every tile, so this is cost per vector op and per panel, not data; the sweep does not say which. Pack's share of a fast rep (vector): 19-20% at 4x4, 30-35% at MR 8 and 16, 40-43% at MR=32.
- **The instrumented build is not the same experiment.** 32x7 has no slow reps on `jbpi2` and 578 of 2997 on `jbpi` in the instrumented build, against 83-87% slow reps uninstrumented on both boards. The instrumented timers and the warm-up protocol change when the state sets in, so only the fast-against-slow split is read from it.

### 8.5 Answer

**Does the packing side of B3 depend on MR and NR, and does the slow state?**

1. **The slow state depends on NR, and not on MR or kc.** With the vector packer, B3 is steady on every tile with NR of 4 or 6 (MR 4, 8, 16, 24 and 32, each with the NR values tried) and two-speed on every tile with NR of 7 or more (MR 4, 8, 12, 16, 24 and 32), on both boards, in every launch. This is the "packing and the slow state move with NR" outcome of §4: the tile decides it. MR changes the speed level but not whether the state appears.
2. **The threshold sits between 6 and 7 columns of the B panel.** NR is the number of columns of B the packer reads at once, each one a separate read stream (§7.1). Six streams are steady, seven are not (16x7: 68-79% slow; 32x7: 83-87%). #44's suspicion that the prefetcher cannot follow more streams fits this, and so does the scalar packer below. It is still a suspicion: nothing here counts prefetcher events.
3. **The scalar packer is much steadier from NR=8 up** (same addresses, same order, one element at a time): 16x8 is 2% / 8% slow at 7.28 / 7.03 GOPs, against 94% / 89% slow and 3.56 / 3.72 for the vector packer, which makes the vector packer about half as fast there. At 16x14 it is 33% / 32% slow (6.98 / 7.19 GOPs), at 16x15 39% / 52% and at 8x15 49% / 46%, so it is not free of the state, only later into it. At NR of 6 or less the vector packer is equal or ahead (32x6 6.33 against 5.89 GOPs, 4x4 3.55 against 3.39).
4. **#46: the tile does not explain the reference.** Our 16x14 is 86% / 81% slow (4 and 3 launches), as much as our 16x15 (92% / 89%). The reference's 16x14 is steady (#46). So the 14th column is not the cause; what differs between the two designs at 16x14 is not the tile (the packers, the buffer placement: #46 §7.2, or the kernels).
5. **#47: rows matter too.** The steady big shapes ran at 32x7 and were steady (#47); B3 at 32x7 is 83-87% slow. So NR of 7 is enough to go slow on B3 and not on a shape with a large M; the state needs a small M (few A panels) as well.
6. **K matters** (§8.3): B3_K256 goes slow at 16x6 and stays steady at 16x8, the opposite of B3. The clean rule above is for K = 384.

### 8.6 What to do with it (not done here: changing the shipped tile or the packer is out of scope)

- **A tile for small-M shapes is a decision, on this table:** 16x6 gives B3 6.7 GOPs against 4.5 at the shipped 16x15 and Reuse24 9.1 against 8.5-8.8, and costs B2 (10.3 against 12.6). B3_K256 at 16x6 is slow. The shapes of the real models (issue #2's matrix) need the same sweep before a tile is chosen; B1, B4, B5 and S1-S5 have M of 1000 and more, where 32x6 and 32x7 are steady (#47).
- **Test the stream count directly** (#44 is the test: pack B one column at a time). This table is the prediction: with one stream per pack, NR=15 should behave like NR=6.
- **Count the prefetcher events** at 16x6 against 16x7 (`--raw-events x60-l2`, #42), the smallest pair that flips the state, to see whether the prefetch hit and refill counts halve between them as in #41.
- **Repeat the decisive points at midnight** only if another user's job appears: all readings here were quiet.

## 9. Sub-issues (opened 2026-10-14, in priority order)

The B3_K256 rows of §8.3 do not follow the "NR up to 6 is steady" rule (16x6 goes slow at K=256, 16x8 goes steady), so the follow-ups start from that.

1. **#49** [[0033-b-stream-layout-k256-flip]]: where the B read streams land in memory. The step between streams is `ldb`*4 = 1536 B at K=384 and 1024 B at K=256. A desk table first, then a padded-`ldb` board test. It tests the premise of #44.
2. **#50** [[0034-prefetcher-counters-four-corners]]: prefetcher counters at the four tiles that flip (16x6/16x7 at K=384, 16x6/16x8 at K=256).
3. **#51** [[0035-slow-state-m-sweep]]: sweep M at fixed tiles, so the rows and the tile come apart (#47's open point).
4. **#52** [[0036-reference-16x14-vs-ours-code-read]]: read the reference and ours side by side at 16x14 (desk only).
5. **#53** [[0037-scalar-vs-vector-packer-on-controls]]: scalar against vector packer on all four shapes, from the committed logs (desk only).
