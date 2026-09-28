# Spec: pack B one column at a time, so the prefetcher follows one stream instead of 15 (B3 slow state, fix 2 of 2)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/44

**Engineering report (#31-#44, all issues together):** [issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)), 2026-09-28.

**Status:** 📝 Specified 2026-09-27. **Implementation on hold until the user says go.**
**Parent:** #41 (§8: the slow state is the L1D prefetcher doing about half its work). Sibling: #43.
**Baseline:** `config.json` pins B3 to `nc`=128 again (user's decision, 2026-09-27); this work is measured
against that baseline and against `nc`=384.

## Problem Statement

B3 (64×384×384) has two speeds. In its slow reps the L2 is accessed exactly as often as in fast reps, but L1D
prefetch hits and refills about halve and demand L2 misses are 2-4× higher (#41 §8). The prefetcher backs off.
The vector packer spends far more reps in the slow state than the scalar one (B3: 51% against 0% in #42's
launches; 66-86% against 0-46% in #35).

The B packer (`pack_CB`, 16×15 tile) walks B across a whole 15-column panel at once: for each row `i` of the
`kc` rows it loads one element from each of the 15 columns (the vector packer with one strided load, `vlse32`,
stride `ldb`; the scalar one with 15 loads), then stores the 15 values contiguously into `Bc`. That is **15
read streams in flight at once**, each advancing 4 bytes per step. A hardware prefetcher tracks a limited
number of streams; with more streams than it can hold, it keeps losing and re-learning them, and many designs
then throttle themselves. The vector packer sends these reads faster than the scalar one, which fits it
falling into the slow state more often (#36 found both packers read the same addresses in the same order).

This is a hypothesis about the prefetcher's design: SpacemiT does not document the X60 prefetcher's stream count
or its throttling rule. This change tests it.

## Solution

A second read order for the B packer: **one column at a time**. For each column of the panel, read its `kc`
elements in one unit-stride pass (one stream, the simplest pattern a prefetcher follows) and write them into
`Bc` with a strided store (stride = the panel width, 15 elements). Stores are not prefetched, so the reads stay
on one stream. The packed layout of `Bc` is exactly the same as today, so nothing downstream changes.

Behind a run-time switch, off by default, for both the vector and the scalar packer, so it can be compared
launch against launch with the house rules and the per-rep counters of #42.

## User Stories

1. As the researcher, I want a B packer that reads one column of B at a time, so that the prefetcher follows
   one stream instead of 15.
2. As the researcher, I want it to produce exactly the same `Bc` as today's packer, byte for byte, so that the
   microkernel and the result are unchanged.
3. As the researcher, I want it for the vector packer (unit-stride load along the column, strided store into
   `Bc`), so that the packer that falls into the slow state most is the one tested.
4. As the researcher, I want it for the scalar packer too, so that the two read orders can be compared with the
   vector unit taken out of the picture.
5. As the researcher, I want a switch to choose the read order per launch, off (today's order) by default, so
   that the baseline and all earlier numbers stay as they are.
6. As the researcher, I want runs in the new order to write to their own log names, so that they never mix with
   baseline logs.
7. As the researcher, I want the run overview and `provenance.txt` to name the read order, so that every number
   names its build (house rule 4).
8. As the researcher, I want the tail of a partial panel (fewer than 15 columns) to pack correctly, so that
   every `nc` works, including `nc`=128 (128 = 8×15 + 8).
9. As the researcher, I want the tail of a column (`kc` not a multiple of the vector length) to pack correctly,
   so that B3_K256 and any `kc` work.
10. As the researcher, I want the GEMM result in the new order to pass the same correctness check, so that I
    know it is safe.
11. As the researcher, I want the per-rep counts (`--raw-events x60-l2`) in both orders, so that I can see
    whether the prefetch hits stay up and the demand L2 misses stay down in the new order.
12. As the researcher, I want the share of slow reps in both orders, over at least three launches each, so that
    I can tell whether the slow state is gone, rarer, or unchanged.
13. As the researcher, I want the same comparison at `nc`=128 (the baseline) and `nc`=384, so that I know whether
    the new order and the smaller `nc` add up.
14. As the researcher, I want controls (B2, Reuse24) in the same launches, so that a change in B3 is not a change
    in the board.
15. As the researcher, I want the uninstrumented, reference-protocol speed of B3 in both orders, so that a
    decision to ship the new order rests on a quotable number.
16. As the researcher, I want the result written into #41 with committed logs, so that the slow state is either
    named ("too many streams") or this cause is ruled out.

## Implementation Decisions

- **Same `Bc`, new walk.** For panel `j` (width `nr` ≤ 15) and each column `jj` in it: load `kc` elements of
  column `j+jj` of B with unit stride, and store them into `Bc` at the panel's base plus `jj`, stride 15 elements
  (the panel width `RR`, not `nr`, so the partial panel keeps today's layout). Pad columns of a partial panel stay
  unwritten, as today (#18).
- **Vector packer.** Built in `pack_builder.py` next to `build_pack_CB_v_if`, as a new builder for the column
  order, strip-mined over `kc` with `vsetvl`. It needs an fp32 strided store; the project's RVV dialect has
  `vlse32` and `vse32` for fp32 and `vsse8` for int8, but **no `vsse32` for fp32**: add it to the dialect
  (`src/xdsltemplate/dialects/rvv.py`) following `vsse8_v_i8m1Op`, with its lowering.
- **Scalar packer.** A column-order twin of `pack_CB_scalar` in the generator's scalar pack functions (same
  single source used by the tests and the benchmark).
- **Scope.** The B packer (`pack_CB`, the `_v_if` case the AXPY kernels use) for `AXPY_FP32_Pack`. The A packer is
  untouched (A's panel is read along its columns already), as are the DOT and int8 packers.
- **Switch.** A driver flag `--pack-b-by-column` and the same flag on `compile_macro_bench.sh`, passed through
  like `--scalar-pack`, with a `_packbcol` log suffix, a run-overview line and a `provenance.txt` line. Works with
  and without `--scalar-pack`. Off by default. How the choice reaches the generated dispatch (a parameter on the
  generated GEMM entry point, like `use_scalar_pack`, choosing between two packer function tables) is the
  implementer's choice, as long as the default path is unchanged.
- **Measurement** (not code): `config_issue41_l2_counters.json` rows plus B3 at `nc`=128, instrumented,
  `--raw-events x60-l2`, core 4; four settings (vector/scalar × row/column order), each over three launches in an
  alternating order; then uninstrumented, reference protocol, `--tmin 10` for B3 in the better order.

## Testing Decisions

- A good test checks the packed layout, the GEMM result, the flags and the logs, not the instructions emitted.
- **Scalar pack seam** (host, prior art: `test_scalar_pack_matches_canonical_blis_reference` in
  `test_pack_correctness.py`): the column-order scalar packer gives exactly the canonical reference's `Bc` for
  full and partial panels, and for `kc` both a multiple and not a multiple of 8.
- **Vector pack seam** (on board, prior art: `test_rvv_vector_pack_matches_scalar_reference_on_board`): the
  column-order RVV packer, built through the real xDSL/MLIR pipeline with RR=15, vlen=8, matches the scalar
  reference byte for byte, including the partial-panel and `kc` tails; skips when `jbpi` is not reachable.
- **Dialect seam** (prior art: the dialect's existing tests for `vsse8`/`vlse32`, if any; otherwise a lowering
  test beside them): `vsse32` builds, verifies and lowers to the strided-store intrinsic.
- **Dispatch seam** (prior art: `test_pad_zeroing.py`): the generated dispatch with the switch on gives the right
  GEMM result on the host with stand-in packers; off, it is unchanged.
- **Driver and script seams** (prior art: `test_ref_protocol.py`, `test_board_pin.py`): the flag, its log suffix,
  overview and provenance lines, and pass-through by `compile_macro_bench.sh`, alone and with `--scalar-pack`.

## Out of Scope

- `cbo.zero` on the pack buffers (that is #43; the two are measured apart first, together only after).
- Changing `Bc`'s layout or the microkernel.
- Changing the A packer, the DOT kernels or int8.
- Blocking two or three columns together as a middle ground (worth a follow-up only if one column helps and
  costs too much elsewhere).
- Shipping the new order by default: a decision for after the measurement.

## Further Notes

- The strided store writes 4 bytes every 60 bytes of `Bc` (15 floats); consecutive columns fill the same lines,
  so `Bc` lines are written 15 times each, spread over the panel. If store traffic turns out to cost more than
  the reads save, #43's `cbo.zero` on `Bc` removes the reads of those lines; that is why the two are separate
  issues with a joint run after.
- If the slow state stays in the new order, the "too many streams" idea is ruled out and #41 moves on (per-100-rep
  windows around the switch, `ldb` padding off 384).
- Counter numbers come from the instrumented build and are not to be quoted as speed (house rule 4).
