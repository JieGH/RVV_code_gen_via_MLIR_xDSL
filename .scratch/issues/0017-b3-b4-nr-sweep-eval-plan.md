# Issue #32: B3 and B4 kernel-dimension (NR) sweep — packing and compute perf across the register-pressure boundary, evaluation plan

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/32

**Closed 2026-09-27 after audit** (§7). **Audit report:** [issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.tex)) — brief and audit of #31-#34, 2026-09-27.
**Role / Label:** `plan` — §2-§5 are the original plan, kept as written.
**Status:** 🟢 Run 2026-09-24/25 (§4.2, §4.3, §4.5 done; §4.1 and §4.4 not done, §4.4 moved to #36).
Results were first posted as a GitHub comment on 2026-09-25 and are copied into §6 here. Bottom line:
the register rule holds (only the two tiles over 32 registers, 16x16 and 32x8, fall, by 38-45%); the
shipped tiles are at or near the best; on B3 the scalar packer beats the vector packer by 21-78%.
**Target:** `jbpi` and `jbpi2` (BananaPi M2 Pro, VLEN=256), reached through jump host `fej`, core 0,
governor `performance` (audit note: all runs did use core 0, which #31 §6.6 found runs 5-10% slower;
see §7) — same target as every other benchmark issue. "Both machines" in the ask means
these two boards.
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/config.json:56-60` — B3 ships at `MR16_NR15` (dataset
  `issue2_bert_16x15.dat`, `nc` override `{"B3": 128}`); B4 ships at `MR32_NR7`
  (`issue2_square_bert_32x7.dat`, shared with S1-S5).
- `tests/pure_xdsl_blis_rvv/datasets/issue2_bert_16x15.dat` — B3's own shape: M=64, N=384, K=384.
- `tests/pure_xdsl_blis_rvv/datasets/issue2_square_bert_32x7.dat` — B4's own shape: M=4096, N=384,
  K=1024.
- `tests/pure_xdsl_blis_rvv/pj_gen/fp32_AXPY_FP32_Pack_vlen256_MR16NR15/gemm_b1_xdsl.mlir` — the
  generated beta=1 kernel for B3's shipped tile, read for §2 below.
- [[0016-b4-32x7-register-pressure-eval-plan]] — the open sibling plan for B4's `32x7` tile (issue #31).
  §2 there is the same kind of register count, done for B4; §4.3 there already proposed two extra B4
  rows (`MR32_NR6`, `MR32_NR8`); this plan supersedes that narrower bracket with the full `32x8→32x4`
  sweep the user asked for, and adds the parallel B3 sweep #31 never covered.
- [[0006-match-reference-design-everything-but-the-two-intended-differences]] — 32x6 confirmed
  spill-free (28/32 registers live, 4 spare), the clean baseline both sweeps are measured against.
- [[0014-issue-29-reference-alone-sweep]] — 32x8 (reference's nearest built neighbour above 32x6) runs
  at 51-67% of 32x6's speed, reasoned to be spilling (36 registers needed, over by 4).
- [[0003-scalar-vs-vector-pack-timing]] — the existing vector-vs-scalar packing split methodology and
  numbers (2026-09-16), the baseline any fresh split in this sweep should be compared against.
- `docs/agents/execution.md` — `run_pack_swap.sh` (vector→scalar→scalar→vector back-to-back per config
  row, issue #26) and `compile_macro_bench.sh`'s `--scalar-pack`/`--no-nc-overrides`/`--cpu`/`--tmin`
  flags, which this plan reuses rather than inventing new tooling.
- `tests/pure_xdsl_blis_rvv/config.json:29-32` (commented-out rows: `[16,16]`, `[16,11]`, `[16,17]`,
  `[16,6]`) — an informal MR16 dimension sweep was tried once before, pre-dating the current reference
  protocol (issue #27) and `nc`-override work (issue #1/#16). Those commented rows are not reusable data
  — different protocol, no provenance, no board pairing — but they show a sweep like this has been of
  interest before.
- `docs/metrics.md` — `KerGOPs` includes packing since issue #13; always say which build
  (`INSTRUMENTED=0`/`=1`) a number came from, per house rule.

---

## 1. What's being asked

Investigate how the kernel's second dimension (`NR`, holding `MR` fixed) affects performance, for both
B3 and B4, using this design's own packing (both the vector packer and the scalar packer) and its RVV
compute kernel. Two sweeps, both stepping `NR` down by 1 each time:

- **B3**: `MR=16` fixed, `NR` from 16 down to 8 (`16x16, 16x15, ..., 16x8` — 9 shapes). B3 ships today at
  `16x15`.
- **B4**: `MR=32` fixed, `NR` from 8 down to 4 (`32x8, 32x7, ..., 32x4` — 5 shapes). B4 ships today at
  `32x7`. Measure both packing methods and RVV compute performance across this range.

Run on both boards (`jbpi`, `jbpi2`). This document is the plan only — nothing below has been run.

## 2. Why this is worth running, not just re-deriving 0016 (read-only check, done now)

[[0016-b4-32x7-register-pressure-eval-plan]] already raised a register-pressure doubt about B4's
shipped `32x7` tile: at `MR=32`, live vector registers at the compute nest's peak are `MR/8` (A-tile,
kept live across all NR columns) `+ (MR/8)×NR` (C-tile, one accumulator per output column-group),
against a 32-register file. That plan proposed bracketing B4 with just two extra rows (`NR=6`, `NR=8`).

Reading B3's shipped kernel source directly (`gemm_b1_xdsl.mlir` for `MR16_NR15`) shows the identical
story on the *other* pinned tile:

```
$ grep -c vle32 gemm_b1_xdsl.mlir   # A-tile + C-tile loads
32
$ grep -c vfmacc gemm_b1_xdsl.mlir  # one per C accumulator
30
$ grep -c vse32 gemm_b1_xdsl.mlir   # C-tile stores
30
```

`MR/8 = 2` A-tile registers, `2×15 = 30` C-tile registers, peak live = **32 of 32**, zero spare — the
same knife-edge as B4's `32x7`. This wasn't checked when #31 was written because #31 only looked at
B4. The general rule, `(MR/8)×(1+NR) ≤ 32`, gives the exact boundary for any `MR`:

| tile | `MR/8` | live = `(MR/8)×(1+NR)` | spare | known outcome |
|---|---|---|---|---|
| MR16, NR=16 | 2 | 34 | −2 (over) | not checked |
| **MR16, NR=15 (shipped B3)** | 2 | **32** | **0** | **not checked — same edge as B4** |
| MR16, NR=14 | 2 | 30 | 2 | not checked |
| MR16, NR=13 | 2 | 28 | 4 | not checked |
| MR16, NR=12..8 | 2 | 26..18 | 6..14 | not checked |
| MR32, NR=8 | 4 | 36 | −4 (over) | confirmed spills, 51-67% of NR=6's speed [0014] |
| **MR32, NR=7 (shipped B4)** | 4 | **32** | **0** | **open doubt [0016/#31]** |
| MR32, NR=6 | 4 | 28 | 4 | confirmed clean, byte-identical to reference [0006] |
| MR32, NR=5 | 4 | 24 | 8 | not checked |
| MR32, NR=4 | 4 | 20 | 12 | not checked |

Both of this project's shipped, hand-picked-by-search tiles (`config.json`, issue #2) landed exactly on
the zero-spare boundary, not with any margin. That's either a coincidence of the search objective, or a
sign the search is quietly picking the fastest *raw FLOP-to-load ratio* tile without a register-slack
penalty, landing right at the edge every time. This sweep is the direct way to tell which: if perf falls
off sharply exactly at `NR=15`/`NR=7` and recovers moving either direction... no — moving to fewer
registers doesn't recover it, but if perf is flat or still climbing right up to the boundary and only
craters just past it (`NR=16`/`NR=8`), that says the search's boundary tiles are fine and the shipped
values are close to optimal already. If instead perf was already degrading a few steps before hitting
the boundary (scheduling-slack cost, not an outright spill — the same risk [0016] §2 flagged and couldn't
resolve without a real run), the shipped tiles are leaving performance on the table and a smaller `NR`
might beat them.

## 3. What we already know, so the plan doesn't re-derive it

- **B3's shape** (`issue2_bert_16x15.dat`): M=64, N=384, K=384 — this is B3's *only* logged shape; no
  B3 number exists today at any tile but `16x15`.
- **B4's shape** (`issue2_square_bert_32x7.dat`): M=4096, N=384, K=1024 — B4 is also the only BERT case
  pinned to `32x7`; S1-S5 share the tile but not the shape.
- **Existing vector-vs-scalar pack split** [0003 §3.1, 2026-09-16, nine days stale as of [0016]]: B3 at
  16x15 and B4 at 32x7 both already have a pack-fraction number on record; this sweep's `NR=15` and
  `NR=7` rows are a fresh rerun of exactly those two points, which also serves as a sanity check that
  the new sweep's numbers agree with the old ones before trusting the new NR values.
- **0016 §4.3** already proposed `MR32_NR6`/`MR32_NR8` as a narrow B4 bracket. This plan's `32x8→32x4`
  sweep is a superset of that — if this runs, §4.3 of [0016] doesn't need its own separate execution.
- The commented-out `config.json` rows (`[16,16]`, `[16,11]`, `[16,17]`, `[16,6]`) show someone tried an
  MR16 sweep before, but predate the current reference protocol (issue #27) and the `nc`-override
  mitigation (issue #1), so none of that data is reusable here.

## 4. The evaluation plan

### 4.1 Static register check for every shape (no board time)

Before generating anything: for each of the 9 B3 shapes and 5 B4 shapes, the peak live-register count is
known in advance from §2's formula — no need to guess afterward which shapes should spill. Confirm two
or three of the un-checked ones (e.g. `MR16_NR16`, `MR16_NR8`, `MR32_NR4`) against actual generated
`.mlir`/`.cpp` source once each project exists (§4.2), the same `grep vle32/vfmacc/vse32` count used in
§2, as a source-level cross-check that the formula holds outside the two tiles already verified.

### 4.2 Generate the new kernel projects

Neither sweep can reuse `config.json` (it would overwrite the shipped B3/B4 pinning and their `pj_gen/`
output — see [[project_pjgen_regenerate_before_commit]] on `pure-xdsl` gitignoring generated output).
Two new, dedicated config files, modeled on `config_pack_swap_b2b3b4.json`:

- `tests/pure_xdsl_blis_rvv/config_dim_sweep_b3.json` — 9 rows, `["fp32", "AXPY_FP32_Pack", [16, NR],
  256, "jbpi2", ["issue2_bert_16x15.dat"], true, 1000]` for `NR` in `16..8`, each landing in its own
  `pj_gen/fp32_AXPY_FP32_Pack_vlen256_MR16NR<NR>/` directory (`config.compute_out_dir` keys on MR/NR, so
  no collision with the shipped `MR16NR15` dir or each other).
- `tests/pure_xdsl_blis_rvv/config_dim_sweep_b4.json` — 5 rows, same shape but `[32, NR]` for `NR` in
  `8..4`, dataset `issue2_square_bert_32x7.dat` (B4's row only — S1-S5 are a separate question per
  [[0016]] §5 and would multiply board time for no gain here).
- The B3 sweep drops the shipped `{"B3": 128}` nc override and runs with `--no-nc-overrides`, per the
  house default run command ([[feedback_default_run_command]]) for numbers not being quoted against the
  reference. This means B3's `nc=384` here will show more run-to-run variance than the shipped `nc=128`
  configuration (issue #1's whole point) — call that out explicitly when reading results, so a `NR`-driven
  slowdown isn't confused with the already-known `nc`-driven instability.

12 new `pj_gen/` projects total (8 new B3 tiles + 4 new B4 tiles; `NR=15` and `NR=7` already exist and
don't need regenerating, but are included in the sweep configs anyway so the whole curve comes from one
consistent run rather than splicing in nine-day-old numbers).

### 4.3 Run: both packing methods, both boards

Reuse `run_pack_swap.sh --config <file>` (issue #26's existing tool, already built for exactly
"vector-live vs. scalar-live, back to back") rather than inventing a new script:

```bash
./tests/pure_xdsl_blis_rvv/run_pack_swap.sh --config config_dim_sweep_b3.json --cpu 0 --tmin 10 ...
./tests/pure_xdsl_blis_rvv/run_pack_swap.sh --config config_dim_sweep_b4.json --cpu 0 --tmin 10 ...
```

Each invocation cycles vector→scalar→scalar→vector over *every row in the pointed config in one pass*
(not "4 runs" flat — that count is per invocation, and board time scales with the number of rows inside
it, since each row is its own compile). Run each config once on `jbpi`, once on `jbpi2` — the "both
machines" half of the ask — so every shape gets an independent same-day repeat on different hardware
before trusting a per-NR trend line, the same pairing discipline [0014] §6 flags as otherwise missing.

Total board work: 2 configs × 4 (vector/scalar/scalar/vector passes) × 2 boards = 16 compile-and-run
passes, each internally looping over 9 (B3) or 5 (B4) rows. This is the uninstrumented, default-protocol
build (`INSTRUMENTED=0`) — the GOPs numbers from this pass are the headline ones, reported per
[[feedback_report_both_gops_metrics]] as both by-mean-time and by-median-rep.

### 4.4 Optional deeper split: instrumented compute-vs-pack breakdown

If §4.3's vector/scalar delta alone doesn't cleanly separate "packing changed" from "compute kernel
changed" at the boundary tiles, rerun just the boundary-adjacent shapes (`NR=16/15/14` for B3,
`NR=8/7/6` for B4) on the `INSTRUMENTED=1` build, which reports `Vector_Pack_Time_ms`,
`Scalar_Pack_Time_ms`, and `Compute_Time_ms` directly ([0016] §4.2's method). Never blend this build's
numbers into the `INSTRUMENTED=0` headline GOPs — report it as its own line, per house rule
(`docs/metrics.md`).

### 4.5 Read the results against §2's prediction

Plot GOPs (both metrics) against `NR` for each of B3 and B4, both boards overlaid. §2's formula predicts
where each curve should fall off a cliff (`NR=16` for B3, `NR=8` for B4 — both already known to spill)
and where it should be flat and near-optimal (`NR≤14` for B3, `NR≤6` for B4). The interesting region is
the one or two steps just below the shipped boundary tile (`NR=15→14` for B3, `NR=7→6` for B4): flat
there says the shipped tile is fine; a rise there says the shipped tile was already paying a
scheduling-slack cost §2 couldn't detect statically.

## 5. What this plan cannot show

- §4.1/§2's register counts are static (source-level, pre-backend-allocation) — they show what the
  generator emitted, not what the RISC-V backend compiler chooses to do with it. A tile with spare
  registers on paper could still spill after backend scheduling; this plan doesn't disassemble every
  shape the way [0016] §4.1 proposed doing for just B4's `32x7`.
- B3's sweep runs with `--no-nc-overrides`, so it is not directly comparable to the shipped `nc=128`
  numbers anywhere else in the repo — only to other rows inside this same sweep.
- Neither sweep touches S1-S5, which share B4's `32x7` tile but not its shape; a good `NR` for B4's
  BERT shape is not guaranteed to be good for the Square family's very different reuse ([0016] §5 makes
  the same point).
- Board time and time-of-day are not controlled between the `jbpi`/`jbpi2` runs beyond running them the
  same day — same caveat as every other cross-run comparison in this project ([0014] §6).
- This plan does not decide whether to change the shipped tiles. §4.5 says how to read the result; acting
  on it (redesigning B3's or B4's kernel dimension) is explicitly out of scope, same as [0016] §4.4.

---

## 6. Results (2026-09-24 day run and 2026-09-25 night run)

All `INSTRUMENTED=0`, reference protocol, `--no-nc-overrides --tmin 10`, core 0. `run_pack_swap.sh`,
one session per config per board, each session = 4 launches (vector, scalar, scalar, vector). Night
sessions: `logs/pack_swap_2026-09-25_{002831,013410}.txt` (B2/B3 on `jbpi`, `jbpi2`) and
`..._{023319,030711}.txt` (B4); day sessions `logs/pack_swap_2026-09-24_{155849,184417,194206,201556}.txt`.
Cells are GOPs by mean time / by median rep, averaged over the two launches of each packer. Night run
shown; the day run agrees within a few percent.

`issue2_bert_16x15.dat` carries both B2 and B3, so the MR=16 sweep measured both.

### 6.1 B3 (64×384×384), MR=16

| NR | live regs | jbpi vector | jbpi scalar | jbpi2 vector | jbpi2 scalar |
|---|---|---|---|---|---|
| 16 | 34 | 3.10 / 3.05 | 4.86 / 5.00 | 3.23 / 3.08 | 4.84 / 4.98 |
| **15 (shipped)** | 32 | 5.29 / 4.44 | 8.24 / 8.88 | 5.72 / 5.79 | 8.46 / 8.83 |
| 14 | 30 | 4.57 / 4.32 | 7.30 / 8.14 | 4.70 / 4.33 | 6.68 / 7.50 |
| 13 | 28 | 5.55 / 5.80 | 7.02 / 7.58 | 5.19 / 4.44 | 8.02 / 8.31 |
| 12 | 26 | 4.98 / 4.50 | 6.03 / 5.70 | 4.96 / 4.50 | 6.88 / 7.65 |
| 11 | 24 | 4.29 / 4.01 | 6.30 / 6.64 | 4.22 / 4.02 | 6.17 / 6.56 |
| 10 | 22 | 4.09 / 3.96 | 7.07 / 7.53 | 4.13 / 4.00 | 6.76 / 7.29 |
| 9 | 20 | 4.33 / 4.05 | 7.49 / 7.51 | 4.36 / 4.06 | 7.46 / 7.54 |
| 8 | 18 | 3.92 / 3.88 | 5.52 / 5.76 | 3.88 / 3.90 | 5.72 / 6.11 |

B3's reps switch between two speeds, so its medians jump around (see #26 §3); read B3 by mean.

### 6.2 B2 (384×384×64), MR=16 — measured but not in the 2026-09-25 comment

| NR | live regs | jbpi vector | jbpi scalar | jbpi2 vector | jbpi2 scalar |
|---|---|---|---|---|---|
| 16 | 34 | 8.00 / 8.01 | 7.72 / 7.74 | 7.88 / 7.92 | 7.73 / 7.75 |
| **15 (shipped)** | 32 | **12.81 / 12.88** | 12.12 / 12.29 | **12.56 / 12.69** | 12.17 / 12.37 |
| 14 | 30 | 12.15 / 12.41 | 11.95 / 11.98 | 11.82 / 12.14 | 11.77 / 11.98 |
| 13 | 28 | 12.25 / 12.29 | 11.71 / 11.75 | 12.12 / 12.16 | 11.77 / 11.82 |
| 12 | 26 | 12.20 / 12.35 | 11.79 / 11.84 | 12.32 / 12.46 | 11.65 / 11.82 |
| 11 | 24 | 11.29 / 11.36 | 10.96 / 11.00 | 11.38 / 11.41 | 10.95 / 10.98 |
| 10 | 22 | 9.82 / 10.61 | 10.86 / 10.89 | 10.46 / 11.03 | 10.83 / 10.87 |
| 9 | 20 | 10.91 / 10.94 | 10.61 / 10.64 | 10.82 / 10.85 | 10.59 / 10.61 |
| 8 | 18 | 10.25 / 10.37 | 9.87 / 9.88 | 10.34 / 10.37 | 9.85 / 9.89 |

### 6.3 B4 (4096×384×1024), MR=32

| NR | live regs | jbpi vector | jbpi scalar | jbpi2 vector | jbpi2 scalar |
|---|---|---|---|---|---|
| 8 | 36 | 4.43 / 4.44 | 4.37 / 4.40 | 4.38 / 4.43 | 4.37 / 4.37 |
| **7 (shipped)** | 32 | 7.84 / 7.85 | 7.66 / 7.67 | 7.92 / 7.93 | 7.64 / 7.68 |
| 6 | 28 | 8.04 / 8.05 | 7.83 / 7.85 | 7.99 / 8.00 | 7.82 / 7.83 |
| 5 | 24 | 7.59 / 7.60 | 7.45 / 7.46 | 7.58 / 7.60 | 7.48 / 7.49 |
| 4 | 20 | 6.92 / 6.92 | 6.83 / 6.84 | 6.97 / 6.99 | 6.92 / 6.92 |

### 6.4 Reading (§4.5)

1. **The register rule holds.** The only tiles over 32 live registers (16x16: 34, 32x8: 36) are the only
   ones that fall: −38% (B2), −41% (B3), −44% (B4). At or under 32, nothing falls off; speed rises
   with NR toward the limit.
2. **Shipped tiles.** 16x15 is best or tied for B2 and B3 (scalar). For B4, 32x6 was ahead of 32x7
   in 8 of 8 pairs by 0.5-3.3%, all on core 0 — later contradicted on core 4 (§7).
3. **Packer.** B3: scalar beats vector by 21-78% (by mean) at every NR, both boards, both runs. B2: vector
   0-6% ahead; B4: vector 0-3.5% ahead.

## 7. Audit (2026-09-27)

Audited against the logs by a later model version; full brief in the audit report linked at the top.

- **Holds.** The comment's tables match the pack-swap logs (e.g. B4 32x7 7.84 / 7.92, B3 16x15
  vector 5.28 / 5.73). The register rule held on all three shapes.
- **"32x6 beats 32x7" is not established.** All 8 pairs were on core 0. #34 on core 4: 32x7 ahead by
  6%; on core 2: 32x6 ahead by 1%. The two tiles are tied within launch noise. Follow-up: #37.
- **"Biggest finding: scalar beats vector on B3 by 35-90%"** was already known from #26 (+48.8% over 16
  runs). The logs give +21% to +78% by mean.
- **Not reported at the time:** the B2 sweep (§6.2).
- **Protocol.** Every launch used core 0, although #31 §6.6 had found the core-0 penalty earlier the same
  day. Comparisons inside one launch still stand; B3, the noisiest case, is the most affected. Follow-up:
  #39.
- **Records.** Until 2026-09-27 this file said "plan only" and was untracked, as were the two sweep
  configs; the results lived only in a GitHub comment. Now committed with the logs.
- **Not done:** §4.1 (register count read from the new kernel files) and §4.4 (instrumented split; moved
  to #36).
- **Side effect still open:** the uncommitted `config.json` change drops B3's `nc`=128 override (5 tests
  fail). Decision: #35.
