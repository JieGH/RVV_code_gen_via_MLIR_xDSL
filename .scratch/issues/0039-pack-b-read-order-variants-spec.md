# Spec: B packer read-order variants in C intrinsics, behind a `--pack-b` switch, plus a pack-only loop (for #54)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/55

**Report:** [issue-54-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-54-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issue-54-brief.tex)), 23 October 2026 (results of this spec's variants, issue #54).

**Status:** 📝 Specified 2026-10-22. Not built. Workflow (phases, gates, who does what) and progress: [[0039-progress-log]].
**Parent:** #54 [[0038-b3-which-read-property-of-pack-b-sets-off-the-slow-state]]. Related: #44 [[0028-pack-b-fewer-streams-spec]] (this
spec builds #44's one-column order as one of its variants, in C instead of xDSL; #44 itself stays on hold).
**Baseline:** the shipped vector packer (`row`), measured **in the same build and the same launches** as every variant. Numbers from this
build are never compared with logs from earlier builds (the switch adds one branch inside the pack timer, as #43's did).

## Problem Statement

On B3 the shipped vector B packer drives the board into a slow state (#41: L1D prefetch hits halve, demand L2 misses rise 2-6x, both pack
and compute slow down) on every tile with NR ≥ 7 (#48). The scalar packer reads the same addresses in the same order and mostly avoids it
(#53). Three properties of the reads could be the trigger: how a read is issued (one instruction over 8 lines, or one load per element),
how many columns are read at once, and how many bytes are taken from one column per touch (#54 §1). There is no build in which these can be
changed one at a time.

## Solution

A small family of B packers written as RVV C intrinsics in one new header, chosen at run time by a driver flag `--pack-b <name>`. They all
produce **exactly the same `Bc`** as today, byte for byte, so the microkernel and the result do not change. The shipped xDSL vector packer
stays the default, untouched. A stand-alone pack-only loop runs the same packers without any compute, to see whether a packer alone goes two-speed.

## Definitions (shared by every variant)

- B is column-major: column `j` of the current block starts at `b_ptr + j*ldb`, and its `kc` values are contiguous (`generate_pure_blis_project.py:556`).
- `Bc` layout (`pack_CB_scalar`, `generate_pure_blis_project.py:213-226`): panel `p` covers columns `p*RR .. p*RR+nr-1` (`nr = min(cur_nc - p*RR, RR)`),
  and element `(i, jj)` of the panel sits at `Bc[p*RR*kc + i*RR + jj]`. Pad columns `jj ≥ nr` of the last panel are **not written** (the driver
  allocates `Bc` zeroed, #18). Every variant keeps this exactly.
- Signature of every variant: `void pack_CB_<name>(const float* B, float* Bc, int32_t kc, int32_t nc, int32_t ldb, int32_t RR)`.

## The variants

| `--pack-b` | loop order | loads | stores |
|---|---|---|---|
| `row` (default) | today's xDSL `pack_CB_v_if_…_NR<RR>` | unchanged | unchanged |
| `row-intr` | per panel, per k, per chunk of 8 columns | `__riscv_vlse32_v_f32m1(B + (j+c)*ldb + i, ldb*4, vl)` | `__riscv_vse32_v_f32m1(Bc_panel + i*RR + c, …, vl)` |
| `col-gG`, G in {1, 2, 4, all} | per panel, per group of G columns, per strip of k (`vsetvl e32m1`, up to 8) | for each column of the group: `__riscv_vle32_v_f32m1(B + (j+jj)*ldb + i, vl)` | for each column: `__riscv_vsse32_v_f32m1(Bc_panel + i*RR + jj, RR*4, v, vl)` |
| `col-g1-m4` | one column at a time, strips of up to 32 k (`e32m4`) | `__riscv_vle32_v_f32m4` | `__riscv_vsse32_v_f32m4` stride `RR*4` |

- `all` means G = nr of the panel (all columns of the panel side by side; at RR=15 that is 15 live vector registers, inside the 32).
  Within a group, all G loads of a k-strip are issued before the G stores, so G columns are in flight at once: that is what makes G the
  stream count.
- `scalar` stays the existing `--scalar-pack`; it is not a `--pack-b` value. `--pack-b` with `--scalar-pack` is an error (exit, message).
- Tails: a k-strip shorter than the vector (`kc` not a multiple of 8 or 32) is handled by `vsetvl`; the last, partial panel by `nr`.

## User Stories

1. As the researcher, I want each variant to give exactly today's `Bc`, so that any change in speed is the read order and nothing else.
2. As the researcher, I want `row-intr` (today's order in C intrinsics), so that I can tell a C-path effect from a read-order effect.
3. As the researcher, I want `col-g1`, `col-g2`, `col-g4` and `col-gall`, so that the number of columns read at once moves while each read stays 32 B down one column.
4. As the researcher, I want `col-g1-m4`, so that I can see whether reading two whole lines per load changes anything over 32 B.
5. As the researcher, I want to pick the variant per launch with `--pack-b <name>`, default `row`, so that the shipped path and all earlier numbers stay as they are.
6. As the researcher, I want the variant in the log names (`_packb-<name>`), in the run overview and in `provenance.txt`, so that every number names its build (house rule 4).
7. As the researcher, I want `--pack-b` passed through by `compile_macro_bench.sh`, so that the usual entry point and the series scripts work unchanged.
8. As the researcher, I want the GEMM result to pass the existing correctness check under every variant, so that I know it is safe.
9. As the researcher, I want a pack-only loop that runs one variant over the B blocks of one shape and tile, rep after rep, with one clock read per rep, so that I can see whether the packer alone goes two-speed.
10. As the researcher, I want the pack-only loop's per-rep times in a format `reps_report.py` (or a thin reader of it) splits into fast and slow reps, so that it is read the same way as the full runs.
11. As the researcher, I want a series script that runs the variants on 16x15, 16x8 and 16x6 over the four shapes of `issue46_ref_b3.dat`, 3 launches, the variant order rotating between launches, so that the house rules hold.
12. As the researcher, I want a report script that prints, per board, tile and shape, one row per variant: GOPs by mean time, GOPs by median rep, slow-rep share, fast start, CV, launches, RED NOTE, so that the answer of #54 is read from one table.
13. As the researcher, I want `--pack-b` refused for DOT methods and int8, so that a variant is never run where it was not built.

## Implementation Decisions

- **New header** `tests/pure_xdsl_blis_rvv/templates/pack_b_variants.h` (prior art: `templates/pack_lines.h`): the variants, `#include <riscv_vector.h>`
  under `__riscv`, and a plain C fallback with the same loop order on the host so the host tests can check the index arithmetic. One
  entry `pack_CB_variant(int variant, …)` with a `switch`.
- **Generator** (`generate_pure_blis_project.py`): add `int pack_b_variant` to `gemm_blis_generic` and to the `gemm_blis_fn` typedef
  (lines 162, 165, 513), after `zero_pack_lines`. In the B pack block only (`render_timed_pack_block`, line 584), the vector branch becomes
  `if (pack_b_variant != 0) pack_CB_variant(pack_b_variant, b_ptr, Bc, cur_kc, cur_nc, ldb, nr); else <today's call>`. The A pack block is unchanged.
  Line clearing (#43) stays before the pack, as it is.
- **Driver** (`templates/driver_benchmark.cpp`): parse `--pack-b <name>` beside `--pad-ldb` (lines 547-575), map the name to the variant number,
  reject unknown names, `--scalar-pack`, and DOT/int8 methods. Pass it at every `variant.func(…)` call (lines 904, 1001 and the timed loop).
  **The warm-up scalar-pack pass (line 1001) passes 0**, so its scalar timing stays what it measures today. Log suffix `_packb-<name>` after
  `_zeroline`; an overview line "Pack B"; a `provenance.txt` line.
- **`compile_macro_bench.sh`**: `--pack-b <name>` pass-through (beside `--pad-ldb`, line 117), its own `BUILD_LOGS` suffix and overview line.
- **Pack-only loop** `tests/pure_xdsl_blis_rvv/pack_only/pack_b_only.cpp` + a make target: arguments M N K, MR NR, variant, reps, cpu.
  It takes mc, nc, kc from the same `get_optim_mc_nc_kc` the driver uses (print them), allocates B and Bc as the driver does (64-byte aligned, Bc zeroed),
  and per rep packs every `(jc, pc)` block of B exactly as the GEMM loop nest does. It runs `row-intr` and the `col-*` variants (the xDSL `row` is
  not linked here; `row-intr` stands in for it, which is why the control of user story 2 matters), and a scalar twin copied from
  `render_scalar_pack_functions` by including the generated text, not a hand copy. Output: `Rep,Pack_Seconds` per rep, plus a header naming board, cpu, variant, tile, mc/nc/kc.
- **Series script** `run_issue55.sh`, modelled on `run_issue48.sh` (lock, `--boards`, `--launches`, `--start-at`, `--workspace`, `--direct`,
  other-user check before and after each run, `load_warnings.txt`): arms are variants; `--variants a,b,c` picks a series; the variant order rotates per launch
  (A-B-C / C-B-A / B-A-C …); config `config_issue55_tiles.json` (16x15, 16x8, 16x6 on `issue46_ref_b3.dat`); always `--no-nc-overrides --cpu 4 --tmin 10`.
  An optional `--pack-only` arm runs the loop for each variant on B3 and B3_K256 at 16x8 and 16x15.
- **Report script** `issue55_report.py` (prior art: `issue48_report.py`), reading per-tile `reps_*.txt.gz` files one tile at a time (the #49 lesson:
  running `reps_report.py` on a whole run folder merges tiles that share a dataset row).

## Testing Decisions

- A good test checks the packed bytes, the GEMM result, the flags and the logs, not the instructions emitted.
- **Variant layout, host** (prior art: `test_scalar_pack_matches_canonical_blis_reference`, `test_pack_correctness.py:190`): every variant's host
  fallback gives exactly the canonical `Bc` for RR in {6, 8, 15}, nc a multiple and not a multiple of RR, kc in {128, 130, 256}; pad slots stay zero.
- **Variant layout, on board** (prior art: `test_rvv_vector_pack_matches_scalar_reference_on_board`, `test_pack_correctness.py:394`): the same cases
  with the real intrinsics on `jbpi`; skips when the board is not reachable. **Never run while a board series is running** (it rewrites `generated/`).
- **Dispatch** (prior art: `test_zero_pack_lines.py:211-235`, `test_pad_zeroing.py`): the generated dispatch passes `pack_b_variant` into the B pack
  block only, inside the pack timer, and the A pack block is unchanged; the GEMM result is right with each variant on the host.
- **Driver and script** (prior art: `test_ref_protocol.py`, `test_pad_ldb.py`): the flag, the refusals, the log suffix, overview and provenance lines, pass-through by `compile_macro_bench.sh`.
- **Pack-only loop** (host): it packs the same blocks the GEMM does (block count and per-block kc, nc printed match `get_optim_mc_nc_kc`), its `Bc` matches the canonical one after a rep, and its output parses.
- **Series and report scripts** (prior art: `test_run_issue48.py`, `test_issue48_report.py`): stubbed bench, no board; rotation order, lock, RED NOTE; the report on a fixture.

## Out of Scope

Porting any variant to xDSL; a `vsse32` dialect op; segment stores (`vssseg`); changing `Bc`'s layout, the A packer, the kernel, the tiles or the blocking;
`cbo.zero` combinations; `ldb` padding combinations; instrumented counter runs (`--instr --raw-events x60-l2` already works with the switch, and can be
added for the winner only); shipping a variant by default.

## Further Notes

- The `col-*` variants write `Bc` with stride `RR*4` (60 B at RR=15): each `Bc` line is written by several columns. `Bc` is 7.5 KB per panel at kc=128,
  so it stays in L1; if the stores turn out to cost more than the reads save, that shows in the fast-rep times and is a finding, not a bug.
- The X60 has no Zicbop (#43 spec): software prefetch is not an option, which is why the read order is the lever tried here.

## Amendments after phase 1 (2026-10-22; read-only check of every code claim, decision-changing points re-checked by hand)

These override the sections above where they differ. G = `generate_pure_blis_project.py`, D = `templates/driver_benchmark.cpp`, C = `compile_macro_bench.sh`.

1. **Only fp32 AXPY gets the branch.** The dispatch is generated for every config; for int8 `c_type_b` is `int8_t` (G:296) and DOT packs with `kc_pad`.
   The generator emits the variant branch only when `c_type_b == "float"` and not `is_dot`; elsewhere `pack_b_variant` is accepted and ignored, so every project still compiles.
   The driver refusal (DOT, non-fp32) stays as a second guard.
2. **Every site of the signature** (add `int pack_b_variant` after `zero_pack_lines`): G:162, G:165, G:513; D:904, D:1001, D:1061; and the test stubs
   `test_pad_ldb.py:41,64`, `test_pad_zeroing.py:124,145`, `test_zero_pack_lines.py:101,125`, `test_ref_protocol.py:51,53,87`, `test_correctness_pass.py:93,94,107`.
   Also D's positional `RunSettings` initializer (D:674-677), the usage string (D:607), and parse `--pack-b` before the unknown-argument fallthrough (D:~601), which would otherwise take a stray word as the dataset.
3. **Default call first.** The B block becomes `if (pack_b_variant == 0) { <today's call>; } else { pack_CB_variant(...); }`, so `pack_CB_direct(` stays the first `_direct(` after the timer line (`test_zero_pack_lines.py:235-241` cuts its window there).
4. **Header plumbing.** `templates/pack_b_variants.h` is copied beside `pack_lines.h` (G:745) and included in the `gemm_headers.h` string (G:479). Every function `static inline`. Intrinsics and `<riscv_vector.h>` use under `#if defined(__riscv)`; host tests compile with an empty stub `riscv_vector.h` and no `-march`, so the host build uses the plain-C fallback.
5. **Intrinsic spelling:** `__riscv_` prefixed (v0.12), as the shipped generated code uses with the board's GCC 13.2 (Bianbu). Host clang 14 does not know these names, so **the intrinsic path is first compiled and checked on the board (phase 5 gate)**; host tests check the fallback, which must keep the same loop order, group order and stores.
6. **Log names.** Driver suffix order: `_refproto _scalarpack _zeroline _padldb<N> _packb-<name> _no_nc_override` (D:654-659). C's `BUILD_LOGS` gets `_packb-<name>` in the same place. (C never adds `_padldb` today, a known gap left alone here.) `results_*.csv` (templates/Makefile:52-62) not changed.
7. **`provenance.txt` is written by C** (C:~343-352): add a `pack B` line there. Add a "Pack B" overview line in D and in C. C refuses `--pack-b` with `--scalar-pack` **or** `USE_SCALAR_PACK` set, before any board time; D refuses the same (D:542 env var included).
8. **Any RR.** Search-mode projects call the packer with any RR in 1..max_dim, so every variant works for any RR. `col-gall` = all columns of the panel up to 24 at once (24 live f32m1 registers); a wider panel goes in groups of 24. Layout tests cover RR in {1, 4, 6, 7, 8, 9, 15, 16, 24, 31}, `ldb > kc` (as with `--pad-ldb`: stride is `ldb*4`, not `kc*4`), nc a multiple and not a multiple of RR, kc in {1, 7, 128, 130, 256}.
9. **Pack-only loop:** uses the C `get_optim_mc_nc_kc(sizeof(float), m, n, k, mr, nr, &mc, &nc, &kc)` (`modelLevel/model_level.c:59`, included inside `extern "C"`, linked as the Makefile does), not the Python one; no nc overrides (the series runs `--no-nc-overrides`); Bc allocated as D does, `max((1+kc)*(nr+nc), padded_nc*kc_pad)` floats, 64-byte aligned, zeroed.
10. **Rotation:** left rotation as `run_issue48.sh`'s `order_of` (`a b c / b c a / c a b`, already tested), not the order written in the Implementation Decisions. `complete()` counts reps files per tile row of the one config, per variant arm.
11. **On-board test path:** `test_pack_correctness.py` reaches `BOARD_HOST = "jbpi"` through `JUMP_HOST = "fej"` (:54-55) and compiles with `g++ -O2 -std=c++17 -march=rv64gcv` on the board.
12. Line cites corrected: `render_timed_pack_block` is G:247 (the B block G:584); driver flag parsing D:562-576; `b_ptr` G:557.
