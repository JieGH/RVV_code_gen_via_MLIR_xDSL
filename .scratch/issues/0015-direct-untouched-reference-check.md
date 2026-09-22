# Issue #30: run the untouched reference directly, no staging pipeline, one shape, both boards, both designs

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/30

**Role / Label:** `ready-for-agent` — done, see §7
**Status:** ✅ Done (2026-09-22). Not a repeat of #29 ([[0014-issue-29-reference-alone-sweep]]) — that measured through
`run_ref_baseline.sh`, a pipeline the user no longer trusts (2026-09-22, after being shown the #29 numbers).
This issue is a from-scratch, minimally-scripted, hand-auditable check: run `ext/RVV_ukernels_benchmark`'s
own commands, as close to byte-identical as it will compile, and put the raw driver output next to ours —
no dataset generator, no comparison script, no derived percentages.
**Target:** `jbpi` and `jbpi2` (BananaPi M2 Pro, VLEN=256), reached through jump host `fej`.
**Primary Sources:**
- [[0014-issue-29-reference-alone-sweep]] — the pipeline being distrusted; §2 there lists everything it does
  besides "build and run."
- `tests/pure_xdsl_blis_rvv/ref_baseline/run_ref_baseline.sh` — what today's reference run actually does
  (§1 below is a line-by-line account of it).
- `tests/pure_xdsl_blis_rvv/ref_baseline/repair_ref_kernels.py`, `ref_baseline/blis_shim/blis.h` — the two
  accommodations the pipeline makes to get the reference to compile at all (§2).
- `ext/RVV_ukernels_benchmark/gemm_blis_family/{Makefile,run_gemm.sh,execute_all.sh}` — the reference's own
  build/run commands, unwrapped.
- `tests/pure_xdsl_blis_rvv/compile_macro_bench.sh` — our own design's normal build/run command
  (`--no-nc-overrides --cpu 0 --tmin 10` for head-to-head numbers).

---

## 1. What the user asked for, and what they're reacting to

> "I don't trust this result... run the reference command directly in the original directory... For the
> rest of the benchmark and codebase, you are using the reference without change... run two designs each on
> two different boards, four times in total, and return the performance."

Today's reference number (§5 of #29, and the table shown just before this request) comes from
`run_ref_baseline.sh`, which does five things before a number appears:

1. rsyncs `ext/RVV_ukernels_benchmark/gemm_blis_family/` to a remote staging directory named
   `xdsl_rvv_refbase/` — **not** the original directory;
2. copies the kernel tree through `repair_ref_kernels.py` first, which rewrites 6 of 4032 vendored kernel
   files (all `16x14/*/macc`) that are missing the first ~16 bytes of their `#include` line upstream (§2);
3. ships a stand-in `blis.h` (`blis_shim/`) because the reference `Makefile` hardcodes
   `BLIS_HOME = /home/adcastel/opt/blis`, a path that exists on nobody's machine here (§2);
4. `sed -i`s `TIMIN=` in-place in the staged copy of `run_gemm.sh`;
5. generates its own `xdsl_cases/*.dat` input files (`gen_ref_datasets.py`) instead of using the reference's
   own `cnn_models/*.dat`, and runs 10 layers × 2 brackets per board, folded through `compare_ref_sweep.py`
   into means, ranges and "faster/slower/within noise" calls.

None of the five is hidden or undocumented (each has a comment explaining why), but it is a lot of machinery
between "the reference's own make + run_gemm.sh" and the number the user was shown. The ask is to strip that
back to (1) and (3) only — the two that are unavoidable to get anything to compile — run one shape, both
designs, both boards, and hand back the driver's own printed line untouched.

## 2. What "the reference, without change" can and cannot literally mean

Two things stand between a pristine `ext/RVV_ukernels_benchmark` checkout and a running binary. Both need a
decision before this can run; neither is "changing the codebase" in the sense of touching computed logic.

- **The BLIS header.** `src/gemm_blis.h:43` and `src/test_gemm.c:44` unconditionally `#include "blis.h"`,
  and the `Makefile`'s default `INCLUDE`/`LIBS` point at a real BLIS install that isn't on `jbpi`/`jbpi2`
  (and isn't in this repo). Two ways to satisfy "without change":
  - install real BLIS on both boards and build with `RUN_MODE=FAMILY_BLIS` or the real `-I.../blis/` —
    closest to "the reference untouched," but RISC-V BLIS availability/build time is unknown and untested
    here, and it changes which code path runs (`FAMILY_BLIS` vs `FAMILY_EXO`) unless BLIS is only ever
    linked, never taken;
  - reuse `blis_shim/blis.h` as-is — already independently reasoned through (its header comment) and cited
    in #29: under `RUN_MODE=FAMILY_EXO` the Exo kernel table (`gemm_blis.c:117-139`) is dispatched to
    directly, the BLIS calls the shim stubs out are compiled out or never dereferenced. It changes zero
    lines of `gemm_blis_family` or `kernels/`; it supplies the one header the vendored source assumes will
    exist. **Proposed default: reuse the shim**, named plainly (not tucked into a directory called
    `blis_shim` deep in `ref_baseline/`) so it's visible in the run's own directory listing.
- **The truncated kernel files.** Only `16x14/*/macc` (6 files) are damaged upstream. **This is exactly why
  §3 proposes picking a 32x6 shape** — it sidesteps the question entirely: 32x6 kernel files are not among
  the damaged six, so a 32x6 run needs zero source repair, zero judgment call, fully byte-identical
  `kernels/RVV_256_LDX/fp32/32x6/` to what's checked into `ext/`.

Everything else — the Makefile, `run_gemm.sh`, the kernel `.c`/`.h` files actually exercised, `test_gemm.c`,
`gemm_blis.c` — would run untouched, straight from a copy of `ext/RVV_ukernels_benchmark`.

## 3. Candidate approach

1. **Shape:** `M=1024 N=384 K=1024`, `MR=32 NR=6` (this project's own B1, `OPT=LDX`, `GATHER=2` i.e. `macc`,
   `SWAP=0`) — an **exact** match to a kernel the reference actually ships, not a bracket. Picked because it
   needs no kernel repair (§2) and is not an inference from a neighbouring NR the reference doesn't build;
   it is literally the dimension "we had" that the reference also has. Open question: confirm this is the
   dimension meant, since #29's pinned dims for most other layers (`32x7`, `16x15`) don't exist in the
   reference at all and would require generating a new kernel file — out of scope for "without change."
2. **Connection scheme (new, minimal — not the sweep/lock machinery of `run_ref_sweep.sh` /
   `cron_bench_run.sh`, since this is a one-off, not a scheduled run sharing `/nfs` with other jobs):**
   reuse only the jump-host `ssh`/`rsync` pattern already in `run_ref_baseline.sh` (`REMOTE_PASS`,
   `REMOTE_HOST=fej`, `REMOTE_PORT`, `--riscv-user --riscv-ip --riscv-port`) — a plain `ssh_board()` /
   `rsync_board()` pair, no lock file, no plan.txt, no result-dir naming scheme.
3. **Staging, per board, per design:**
   - *Reference*: `rsync -a --exclude build/ --exclude out_models/ --exclude output/
     ext/RVV_ukernels_benchmark/gemm_blis_family/` plus `kernels/RVV_256_LDX/fp32/32x6/` (the one MR×NR
     needed for both B1 and B5 — same kernel dir, see §6 — the full kernel tree is thousands of files and
     irrelevant here) into a plainly-named remote dir (e.g. `ref_direct_check/`), plus the one shim header.
     Hand-write a two-row dataset file (B1, B5) in the format `run_gemm.sh` already reads
     (`#layer m n k Flops`, confirmed against `ext/.../cnn_models/square.dat`) instead of running
     `gen_ref_datasets.py`.
   - *Ours*: `compile_macro_bench.sh` at the same two shapes, `--no-nc-overrides --cpu 0 --tmin 10`,
     unmodified — this is already the head-to-head command, no new wrapper needed on our side.
4. **Build + run, on each board, each design** — the literal commands, not a generated script wrapping
   them:
   ```
   # reference — one build serves both B1 and B5, same MR/NR
   make clean && make MR=32 NR=6 OPT=LDX GATHER=2 SWAP=0 \
       SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO INCLUDE='-I../blis_shim' LIBS='-lm'
   taskset -c 0 ./run_gemm.sh <two-row B1/B5 dataset> result.csv
   ```
   and `compile_macro_bench.sh`'s own build+run for ours, same `--cpu 0`, same two shapes.
5. **Eight runs total:** {reference, ours} × {jbpi, jbpi2} × {B1, B5}, one rep each (no sweep, no repeats) —
   report each design's own printed driver line (the reference's `DRIVER FOR THE EVALUATION OF GEMM` table
   row; ours, whatever `compile_macro_bench.sh` prints) side by side, unprocessed, per shape. No mean, no
   spread, no verdict — that was the point of the complaint. B2/B3/B4 are out of scope per §6, not silently
   skipped.

## 4. Open questions — user response (2026-09-22)

> "I just want you to run the external repository as it is designed to be, but with the dimension of the
> matrix. I want you to do this using their work but the only things that are different are: the matrix
> input, the set microkernel dimension, the jump server setup that I'm using in this design. Just use as
> much as you can from the reference external repository to run them without doing major or smart
> modification."

This resolves question 1: exactly three things are allowed to differ from a pristine `ext/RVV_ukernels_benchmark`
run — the matrix shape fed in, the MR×NR picked, and the jump-host transport (this repo's boards aren't reachable
any other way). Everything else — Makefile, `run_gemm.sh`, `test_gemm.c`, `gemm_blis.c`, the kernel source
actually exercised, how the driver prints its result — stays exactly as the reference ships it. That is what §3
already proposed, and it's also *why* §3 picked B1 (1024×384×1024, 32×6): it's a shape and MR×NR pair the
reference already has a real, undamaged, checked-in kernel for. Picking a shape the reference doesn't ship
(e.g. 32×7, 16×15) would force writing a new kernel file to make it build — that's a fourth kind of difference,
and exactly the "smart modification" the user is ruling out. So B1/32×6 stands as the shape, not open anymore.

Remaining, narrower than before:
- The BLIS header stand-in (§2) isn't one of the three named differences, but it's not a fourth one either —
  it's the one thing required just to get the vendored source to compile at all (it supplies a header the code
  `#include`s, never touches computed logic, and doesn't touch matrix input, MR×NR, or the jump-host setup).
  Treating it as in-scope by default; flag if the user wants real BLIS installed instead.
- One rep each, no averaging — still assumed, since "return the performance" / "four times in total" reads as
  four raw numbers, not a sweep with error bars. Will surface the driver's own printed line as-is per run.

## 6. Scope widened to the full BERT-large series, then narrowed back (2026-09-22)

User asked for the whole BERT-large series to be checked, not just B1 — "all of the BERT model needed to be
tested," confirmed as "B1 2 3 4 5." BERT-large's five encoder/FFN shapes, as pinned in this project (see
`tests/pure_xdsl_blis_rvv/datasets/issue2_bert_16x15.dat`, `issue2_bert_32x6.dat`,
`issue2_square_bert_32x7.dat`, `bert_large.dat`):

| Shape | M×N×K | MR×NR | In the reference's kernel tree? |
|---|---|---|---|
| B1 | 1024×384×1024 | 32×6 | yes — undamaged, already the plan |
| B5 | 1024×384×4096 | 32×6 | yes — same kernel dir as B1, only the matrix input differs |
| B2 | 384×384×64 | 16×15 | **no** |
| B3 | 64×384×384 | 16×15 | **no** |
| B4 | 4096×384×1024 | 32×7 | **no** |

Checked every kernel variant the reference ships (`kernels/RVV_256_{LDX,OPT,ALL,BASE}/fp32/`): all four only
carry even NR (…6, 8… / …14, 16…). 16×15 and 32×7 are dimensions this project generated for itself; they
don't exist anywhere in the reference's vendored source. Building B2/B3/B4 "as the reference is designed"
would need writing new kernel files — the "major/smart modification" ruled out earlier this same issue.

**Decision: run B1 and B5 only.** They're the only two BERT shapes with real, undamaged reference kernels, and
the B1/B5 pair happens to be the cleanest possible demonstration of the three-axis rule (§4): same MR×NR, same
kernel source, only the matrix input (M/N/K) differs. B2/B3/B4 are recorded here as untestable without
modification, not silently dropped — reopen if real BLIS (`FAMILY_BLIS`) gets installed on the boards later,
since that path isn't limited to pre-generated kernel files.

## 7. Done (2026-09-22)

Ran. `tests/pure_xdsl_blis_rvv/ref_baseline/run_direct_check.sh` (new, ~100 lines, no dataset generator, no
comparison script) does exactly §3/§6: stages a copy of `ext/RVV_ukernels_benchmark/gemm_blis_family` plus
only the `kernels/RVV_256_LDX/fp32/32x6/` kernel dir plus the BLIS shim header onto each board over the `fej`
jump host (a plain `~/.ssh/config` alias with its own key — no `REMOTE_PASS`/`sshpass` needed, since `fej`'s
home is the same NFS mount the boards see), builds with the reference's own `Makefile` unmodified
(`MR=32 NR=6 OPT=LDX GATHER=2 SWAP=0 SIMD_MODE=RVV_EXO RUN_MODE=FAMILY_EXO`), and runs the reference's own
`run_gemm.sh` against a hand-written two-row dataset (B1, B5) in its own `#layer m n k Flops` format. For
"ours," the existing `compile_macro_bench.sh --config config.json --no-nc-overrides --cpu 0 --tmin 10` ran
unmodified — no new wrapper, per §3.

Raw driver output, both sides, unprocessed:

| Board | Reference B1 (GFLOPS) | Reference B5 (GFLOPS) | Ours B1 (GOPs) | Ours B5 (GOPs) |
|---|---|---|---|---|
| jbpi  | 7.15 | 7.05 | 8.04 | 7.83 |
| jbpi2 | 7.18 | 7.16 | 8.05 | 7.82 |

No mean, no spread, no verdict across the two designs — that was the point. The reference's own `PASS` column
read `-` (not checked) on every row: `run_gemm.sh`'s own default is `TEST=F`, unmodified from the shipped
script, so this reflects the reference's own default behavior, not something this check turned off.

Full transcripts: `tests/pure_xdsl_blis_rvv/logs/issue30_direct_check_2026-09-22_180052.txt` (reference, both
boards) and `tests/pure_xdsl_blis_rvv/logs/run_2026-09-22_181113/` (ours, jbpi) /
`run_2026-09-22_181822/` (ours, jbpi2).
