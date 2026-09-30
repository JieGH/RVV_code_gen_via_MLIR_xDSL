# Issue #48: B3 and the packing: sweep the microkernel dimensions (MR x NR) and read packing and compute apart (follow-up of #41, #46, #47)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/48

**Comes from:** the engineering report, Key findings 6, 8 and 10 and Recommendation 4:
[issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf)
([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)).

**Role / Label:** `needs-triage` — new direction for B3, opened 2026-09-30 at the user's request before the session closed. Not started; nothing run.
**Status:** 📝 Open, direction only. This file records the idea; the plan below is a first draft to be checked against the packer code.
**Target:** `jbpi` and `jbpi2`, core 4, quiet window (no other user's job on the board; see `docs/agents/execution.md`, rule 5), house rules of #39.
**Shape(s):** B3 (64×384×384) first; B3 with K=256 and Reuse24 as controls; B2 as the steady control.
**Primary Sources:**
- [[0017-b3-b4-nr-sweep-eval-plan]] (#32) — the earlier NR sweep for B3 (MR=16, NR 16 down to 8), run on core 0 and read only as total speed.
- [[0025-b3-slow-state-root-cause]] (#41) and [[0030-reference-b3-per-rep]] (#46) — the slow state and the reference's steadier B3.
- [[0031-slow-state-survey-all-shapes]] (#47) — the slow state appears on the 16x15 tile with few rows and not on the 32-row tiles; tile and rows are not separated.
- `tests/pure_xdsl_blis_rvv/pack_builder.py`, `generate_pure_blis_project.py` — how the vector packers are built from MR and NR.
- `tests/pure_xdsl_blis_rvv/config_dim_sweep_b3.json` — the #32 sweep config, a starting point.

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

## 3. Plan (draft)

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
   Controls in the same launches: B3 with K=256, Reuse24 and B2 at the shipped tile.
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

- [ ] Table of what each vector packer does per MR and NR, written into this issue before any run.
- [ ] Sweep run on both boards in quiet windows, at least 3 launches per point, logs committed, no red note (or the note stated).
- [ ] Packing and compute read apart, for fast and slow reps, per tile.
- [ ] Written answer: does the packing side of B3 depend on MR and NR, and does the slow state?
