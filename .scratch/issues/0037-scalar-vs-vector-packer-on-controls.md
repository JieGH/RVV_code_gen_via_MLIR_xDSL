# Issue #53: Scalar against vector packer on all four shapes of the #48 sweep, from the committed logs (sub-issue of #48)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/53

**Comes from:** #48 §8.5 (answer 3: on B3 the scalar packer is far steadier and faster from NR=8 up; 16x8 gives 7.28 against 3.56 GOPs)
and the B3_K256 rows of the #48 report, where the scalar packer is slow at 16x14 and 16x15 (85-90% slow reps on `jbpi`).

**Role / Label:** `needs-triage` — sub-issue of #48, priority 5 of the #48 follow-ups (opened 2026-10-14). Analysis only: no board time, no code change.
**Status:** 🟡 Answered (§6-§7, below).
**Target:** the committed #48 logs (`tests/pure_xdsl_blis_rvv/logs/issue48_2026-09-30_*`). No board.
**Shape(s):** B3, B3_K256, B2, Reuse24 on the 7 tiles of the scalar arm: 16x8, 16x14, 16x15, 16x16, 8x15, 32x6, 4x4.
**Primary Sources:**
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) §7.3 (the scalar arm), §8.2, §8.4.
- `tests/pure_xdsl_blis_rvv/logs/issue48_2026-09-30_report.md` — every shape, tile and packer, both boards.
- `tests/pure_xdsl_blis_rvv/issue48_report.py`, `reps_report.py` — the readers.
- [[0021-b3-vector-packer-slow]] (#36), [[0003-scalar-vs-vector-pack-timing]] — earlier scalar against vector results.

---

## 1. Question

#48 compared the two packers on B3 only. "Use the scalar packer at NR of 7 or more" would be an option for the shipped build only if it holds
on the other shapes too. **On each of the 7 scalar-arm tiles, which packer is faster and steadier on each of B3, B3_K256, B2 and Reuse24, on both boards?**

## 2. Why it matters (why priority 5)

- It is free: the data is committed.
- It decides whether a packer choice per tile is worth a spec, or whether the scalar packer only helps B3 at K=384. The K=256 rows already hint at the second.
- It adds a second packer to the layout question ([[0033-b-stream-layout-k256-flip]]): the scalar packer reads the same addresses in the same order, one element at a time.

## 3. Plan

1. With `issue48_report.py` (or a small read-only addition to it), print one table per shape: rows are the 7 tiles; columns are vector and scalar on `jbpi2` and on `jbpi`.
   Each cell has GOPs by mean time, GOPs by median rep and the slow-rep share.
2. A summary line per tile: which packer wins on how many of the four shapes, per board.
3. From the instrumented arms, the fast-rep pack time of the two packers per tile and shape (debug numbers, not speed), to see whether the
   scalar packer wins by being steadier or by being cheaper per rep.

## 4. How to read it

- **Scalar wins on all four shapes from NR=8 up:** a per-tile packer choice is worth a spec (separate decision).
- **Scalar wins on B3 only:** its B3 advantage is a K=384 layout effect, read together with [[0033-b-stream-layout-k256-flip]], not a better packer.
- **Scalar loses on B2 or Reuse24:** the packer choice is per shape, not per tile, so it is out of reach of a build-time switch.

## 5. Out of scope

Changing the shipped packer; new board runs (if a gap needs one, it is proposed here, not run).

## 6. Tables (2026-10-14)

`issue53_packer_compare.py` reads `logs/issue48_2026-09-30_report.md` directly (no hand-copied
numbers) and prints the full per-shape tables, the per-tile summary and the instrumented
fast-rep pack times: `python3 issue53_packer_compare.py`. The per-tile summary (shapes won of
4, by mean GOPs):

| tile | jbpi2: vector / scalar | jbpi: vector / scalar |
|---|---|---|
| 16x8 | 3 / 1 | 2 / 2 |
| 16x14 | 1 / 3 | 1 / 3 |
| 16x15 | 2 / 2 | 1 / 3 |
| 16x16 | 2 / 2 | 2 / 2 |
| 8x15 | 1 / 3 | 2 / 2 |
| 32x6 | 4 / 0 | 4 / 0 |
| 4x4 | 4 / 0 | 4 / 0 |

On B2, vector wins **every** scalar-arm tile on **both** boards (e.g. 16x8: 10.24/10.33 against
9.89/9.92 GOPs) — the one unconditional pattern in the whole table.

## 7. Written answer

**Scalar wins on B3 only; on B2 it loses on every tile, so the choice is per shape, not per
tile** (§4's third reading). On B3, scalar is clearly ahead at NR ≥ 8 (16x8, 16x14, 16x15,
8x15: 6.1-7.3 against 3.3-4.7 GOPs) and roughly level at NR ≤ 6 (32x6, 4x4: vector slightly
ahead, matching #48 finding 3). On B3_K256 and Reuse24 the pattern is mixed, not a repeat of
B3: on B3_K256, vector wins 16x8 outright (6.74-6.76 against 5.65-5.71 GOPs, scalar still 3-5%
slow there) and the two packers are close everywhere else; on Reuse24, scalar wins the same
NR≥8 tiles it wins on B3 (16x14, 16x15, 8x15) but loses 32x6 and 4x4, same as B3. **On B2 the
rule reverses completely**: vector is ahead on every one of the 7 tiles, on both boards, with
no slow reps on either side — B2 is steady regardless of packer, and the vector packer is
simply faster there.

The instrumented fast-rep pack times (debug numbers, not speed) answer plan step 3: **scalar's
B3 win is not a cheaper pack, it is a steadier one.** At the NR≥8 tiles the two packers' fast-rep
pack times are close (16x8: vector 0.76-0.76 ms against scalar 0.77-0.80 ms — scalar is not
faster when both are fast); scalar's advantage on B3 comes entirely from having far fewer slow
reps (e.g. 16x8: 2-8% slow for scalar against 89-94% for vector), not from packing faster per
rep. On B2, where neither packer ever goes slow, scalar's pack is uniformly *more* expensive
than vector's (e.g. 16x8: 0.083-0.090 ms vector against 0.156-0.166 ms scalar) — consistent
with #53 §2's point 3: the scalar packer reads the same addresses in the same order, one
element at a time, so when there is no slow state to avoid it has nothing to win back.

**What this means for #44 and [[0033-b-stream-layout-k256-flip]]:** the scalar packer is not a
free global win and a build-time packer switch keyed on NR alone would cost B2 on every tile.
It is further evidence that the slow state (not the packer's raw cost) is what the layout
question is about: the scalar packer reads the same strided addresses in the same order as the
vector one, one element at a time, and that alone is enough to avoid most of the slow reps on
B3 (though not on B3_K256's 16x14/16x15, which stay 85-94% slow under either packer) — so
whatever the vector packer's wide reads trigger is not purely about doing fewer, wider loads.

## 8. Closure

- [x] Per-shape tables (7 tiles x 2 packers x 2 boards) and the fast-rep pack times, written here, every number from a committed log. See §6 and `issue53_packer_compare.py`.
- [x] Written answer: does the scalar packer's B3 advantage hold on B3_K256, B2 and Reuse24? See §7: no — it reverses on B2, and is mixed on B3_K256/Reuse24.
