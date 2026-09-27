# Spec: clear each pack-buffer line with `cbo.zero` before packing into it (B3 slow state, fix 1 of 2)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/43

**Status:** 📝 Specified 2026-09-27. **Implementation on hold until the user says go.**
**Parent:** #41 (§8: the slow state is the L1D prefetcher doing about half its work). Sibling: #44.
**Baseline:** `config.json` pins B3 to `nc`=128 again (user's decision, 2026-09-27); this work is measured
against that baseline and against `nc`=384.

## Problem Statement

B3 (64×384×384) has two speeds: most reps fall into a slow state where both packing and the microkernel take
~1.5-2× longer (#36). The per-rep counters of #42 show that in the slow state the L2 is accessed exactly as
often, but the L1D prefetcher does about half its work and demand L2 misses are 2-4× higher (#41 §8).

Every line the packers write to Ac or Bc is first **read** from the L2 before it is overwritten: on the board a
pure write stream reads each line once (`l2_load_access` ≈ 1.0 per 64 B line written, #42's probe). The packers
overwrite whole lines, so those reads are wasted. They add L2 and memory traffic, and they are extra streams for
the prefetcher to track, next to the B (and A) streams the packer is reading. That is load on exactly the part
that backs off in the slow state.

## Solution

The X60 has `cbo.zero` (Zicboz): it sets a whole 64-byte cache line to zero without reading it first. Checked
on `jbpi` (2026-09-27): it runs in user mode (no SIGILL) and clears exactly one 64-byte line; the device tree
gives `riscv,cboz-block-size = 64`.

Before a packer writes a block of Ac or Bc, clear with `cbo.zero` every 64-byte line that the block will
overwrite in full. The packer's writes then land on lines already in the cache, with no read. Behind a run-time
switch, off by default, so it can be compared launch against launch with the house rules and turned on only if
it helps.

## User Stories

1. As the researcher, I want the packers to stop reading Ac/Bc lines they are about to overwrite, so that the
   L2 and the prefetcher carry less traffic.
2. As the researcher, I want a switch to turn this on per launch, so that I can compare it against the baseline
   A-B-B-A-A-B.
3. As the researcher, I want the switch off by default, so that the baseline and every earlier number stay as
   they are.
4. As the researcher, I want runs with the switch on to write to their own log names, so that they never mix
   with baseline logs (as `--scalar-pack` runs do).
5. As the researcher, I want the run overview and `provenance.txt` to say whether line clearing was on, so that
   every number names its build (house rule 4).
6. As the researcher, I want only lines the pack block overwrites in full to be cleared, so that no packed data,
   and no pad slot the kernel relies on, is lost.
7. As the researcher, I want the result of the GEMM with the switch on to pass the same correctness check as
   without, so that I know the clearing is safe.
8. As the researcher, I want the clearing to cover both the vector and the scalar packers, so that the two can
   be compared under the same conditions.
9. As the researcher, I want it to cover the Bc pack and the Ac pack, so that all pack writes lose their reads.
10. As the researcher, I want the time the clearing takes to fall inside the pack timer, so that it is counted,
    not hidden (as #18 did for the pad zeroing).
11. As the researcher, I want the build to fall back to plain stores on a machine without Zicboz (the host), so
    that the host tests still build and run the same code.
12. As the researcher, I want the `--raw-events x60-l2` counts per rep with the switch on and off, so that I can
    see whether the pack's L2 loads drop and whether the prefetch hits stay up.
13. As the researcher, I want a config for the comparison (B3 at `nc`=128 and 384, B3_K256, B2 and Reuse24 as
    controls), so that it is one command per launch.
14. As the researcher, I want the result written into #41 with committed logs, naming build, protocol and core,
    so that it can be quoted under the house rules.
15. As the researcher, I want the uninstrumented, reference-protocol speed of B3 with and without the switch, so
    that the decision to ship it rests on a quotable number, not on counters.

## Implementation Decisions

- **One helper, one place.** A small helper clears the whole 64-byte lines inside a byte range. On RISC-V with
  Zicboz it issues `cbo.zero` per line; elsewhere (the host) it does nothing, so host tests exercise the same
  call sites and the same packed result. The line size is 64 (the board's `riscv,cboz-block-size`); a build for
  another size is out of scope.
- **Which lines.** Only lines lying wholly inside the range the current pack call writes: from the first
  64-byte boundary at or after the block's start to the last boundary at or before its end. Ac and Bc are
  allocated 64-byte aligned by the driver, so for full blocks this is every line. A partly written line at either
  end is left alone.
- **Pad slots.** For the AXPY kernels the pad slots of the last partial panel need not be 0 (#18); clearing them
  to 0 is harmless. The DOT kernels' K pad (`kc_pad`) is zeroed on purpose today, inside the pack timer; this
  work is scoped to `AXPY_FP32_Pack` and must not change the DOT path.
- **Where it runs.** In the generated dispatch, just before each timed pack of Bc and of Ac, inside the pack
  timer and the pack counter window.
- **Switch.** A driver flag `--zero-pack-lines` and the same flag on `compile_macro_bench.sh`, passed through to
  the driver like `--scalar-pack`, with a `_zeroline` log suffix, a run-overview line and a `provenance.txt`
  line. Off by default. How the flag reaches the generated dispatch (an extra parameter on the generated GEMM
  entry point, like `use_scalar_pack`) is the implementer's choice, as long as the default path is unchanged.
- **Measurement** (not code): with `config_issue41_l2_counters.json` rows plus B3 at `nc`=128, instrumented
  build, `--raw-events x60-l2`, core 4, switch off/on alternating A-B-B-A-A-B, then the same pair uninstrumented,
  reference protocol, `--tmin 10`, for the speed number.

## Testing Decisions

- A good test checks what a user of the code sees: the packed result, the GEMM result, the flags, the log names
  and overview lines. Not which instructions were emitted.
- **Dispatch seam** (prior art: `test_pad_zeroing.py`, the real generated dispatch built on the host with
  stand-in kernels and packers, buffers pre-filled with NaN): with the switch on, the GEMM result is still right
  and no pad slot leaks into C; with it off, the result is unchanged from today.
- **Line-range seam** (host, prior art: `test_perf_counters.py`'s header harness): the helper, given ranges that
  start and end on and off 64-byte boundaries, asks to clear exactly the whole lines inside the range and none
  outside. On the host the helper is a no-op, so the test observes which lines it would clear through a test hook
  or a pure "which lines" function, not the instruction itself.
- **Driver and script seams** (prior art: `test_ref_protocol.py`, `test_board_pin.py`): the flag, its log
  suffix, the overview and provenance lines, and that it passes through `compile_macro_bench.sh`.
- **On-board check** (prior art: `test_rvv_vector_pack_matches_scalar_reference_on_board`): the RVV packers with
  the clearing on produce the same packed layout as the scalar reference; skips when `jbpi` is not reachable.

## Out of Scope

- Changing the order in which the packers read B (that is #44).
- `cbo.zero` on C, or in the microkernel.
- The DOT kernels (`kc_pad` zeroing stays as it is).
- Shipping the switch on by default: that is a decision for after the measurement.

## Further Notes

- Checked on `jbpi` 2026-09-27: `cbo.zero` works in user mode; `riscv,cbom-block-size` and
  `riscv,cboz-block-size` are 64; there is **no** `riscv,cbop-block-size` and no `zicbop` in the ISA string, so
  the X60 has no prefetch hint instructions (`prefetch.r` would be a no-op).
- Expected signal in the counts: the pack window's `l2_load_access` should fall by about one per 64-byte line
  written. Whether the slow-rep share falls is the real question.
- All counter numbers come from the instrumented build and are not to be quoted as speed (house rule 4).
