# Issue #30: run the untouched reference directly, no staging pipeline, one shape, both boards, both designs

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/30

**Role / Label:** `ready-for-agent` — done. Original scope (§7) and the follow-on investigation (§9) both
resolved (§10)
**Status:** ✅ Done (2026-09-22). B1/B5-direct-check (§7) done; the publication-vs-fresh "discrepancy" (§9)
turned out not to be a reference-design question at all — resolved in §10. Not a repeat of #29 ([[0014-issue-29-reference-alone-sweep]]) — that measured through
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

## 8. Full BERT B1–B5, our design only (2026-09-22)

User asked to add B2/B3/B4 for "the other design" — ours doesn't have the reference's kernel-availability
constraint (§6), so all five BERT-large shapes are valid runs on our side, just not comparable against the
reference (B2/B3/B4 have no reference-side number, per §6). No new board time: `compile_macro_bench.sh
--config config.json` already swept every row in `config.json` (B1–B5 across three MR×NR pinnings) in the
same two runs used for §7 — this just surfaces the B2/B3/B4 rows that were in those logs and not reported
before.

| Shape | M×N×K | MR×NR | jbpi (GOPs) | jbpi2 (GOPs) |
|---|---|---|---|---|
| B1 | 1024×384×1024 | 32×6  | 8.04  | 8.05  |
| B2 | 384×384×64    | 16×15 | 12.59 | 12.72 |
| B3 | 64×384×384    | 16×15 | 5.19  | 5.62  |
| B4 | 4096×384×1024 | 32×7  | 7.67  | 7.80  |
| B5 | 1024×384×4096 | 32×6  | 7.83  | 7.82  |

"By mean time" GOPs, whole GEMM call including packing, same protocol as §7 (reference protocol, tmin=10s,
`--no-nc-overrides --cpu 0`). Correctness not checked (`skip_ref_run` in `config.json`, same as B1/B5). Same
source logs as §7: `tests/pure_xdsl_blis_rvv/logs/run_2026-09-22_181113/fp32_AXPY_FP32_Pack_vlen256_{MR16NR15,MR32NR7}/`
(jbpi) and `run_2026-09-22_181822/fp32_AXPY_FP32_Pack_vlen256_{MR16NR15,MR32NR7}/` (jbpi2) — already committed
in `53f16a9`, nothing new to add there.

## 9. What this is really for: publication-vs-fresh discrepancy (2026-09-22, ongoing)

**The actual question behind all of §§1–8**: the user has a "past publication ref design" table — GFLOPS
they published earlier for this same reference, S1–S5 and B1–B5, stated from memory, not from a run either of
us has seen (see memory `project-past-publication-ref-design`):

| case | pinned config | published GFLOPS |
|---|---|---|
| B1 | 32×6 | ~8.0 |
| B2 | 16×15 | ~12.5 |
| B3 | 16×15 | ~8.8 |
| B4 | 32×7 | ~8.5 |
| B5 | 32×6 | ~7.5 |

Today's fresh, byte-faithful runs of the untouched reference (§7) come in lower: B1 7.15–7.18, B5 7.05–7.16.
The user is trying to find out whether that gap is a real regression (hardware, compiler, thermal, something
changed since publication) or an artifact of how the check was run — and the session so far has surfaced a
real candidate for the second explanation.

**What's been found, working on the board directly (2026-09-22, `jbpi2`, user driving manually, this agent
guiding step by step — no board access from this session):**

1. A **real, already-built BLIS install** exists on the board's shared filesystem at
   `/nfs/gap/jlei/ws_bpi/blis` (`include/generic/blis.h`, `lib/generic/libblis.a`) — built but never
   `make install`ed into the flat layout the reference's `Makefile` expects by default. This is better than
   the `blis_shim` workaround (§2): it's the real library, not a stand-in, so `bli_init()` etc. now run for
   real instead of being stubbed out. Confirmed working: `make MR=32 NR=6 ... INCLUDE=-I../../blis/include/generic
   LIBS="../../blis/lib/generic/libblis.a -lm"` builds and runs cleanly.
2. **First result under real BLIS** (B1+B5, `bert_b1_b5.dat`, `jbpi2`, same 32×6 build as §7 but linked
   against real BLIS instead of the shim): B1 = 7.30, B2 = 8.65, B3 = 5.71, B4 = 7.36, B5 = 6.82 GFLOPS. Only
   B1/B5 are meaningful here — the same `bert_b1_b5.dat` file was run against the **32×6 binary for all five
   rows**, so the B2/B3/B4 numbers above are those shapes computed through the *wrong* tile size (32×6
   instead of their pinned 16×15/32×7), not a real B2/B3/B4 result. That this happened at all is itself the
   likely source of at least part of the publication gap: **`MR`/`NR` is a compile-time constant per binary in
   this reference — one build serves one tile size, so getting a real per-layer number needs a separate build
   per pinned `MR×NR`, and there is no way to get all five layers correctly from one run.**
3. Confirmed again (independently of §6, now via `make -n` dry run and by inspecting the actual kernel
   source): `kernels/RVV_256_{BASE,LDX,OPT}/fp32/{16x15,32x7}/` do not exist, in any `OPT` variant. This is
   **not a BLIS problem** — real BLIS only supplies a handful of harmless setup-call stubs; the actual tile
   compute always comes from a vendored, `MR×NR`-specific Exo-generated kernel file (21,534 lines for 32×6,
   confirmed autogenerated — function-header style like `gemm_RVV_10x1_b0_col_fp32(...)` is Exo's own
   annotation format, and no Exo generator/schedule source is vendored in this repo, only its output). So
   `RUN_MODE=FAMILY_BLIS` does not unblock B2/B3/B4 either — `SIMD_MODE=RVV_EXO` unconditionally requires the
   kernel file to exist and compile, regardless of `RUN_MODE`. Writing new 16×15/32×7 kernels by hand isn't
   realistic at that scale.
4. Mapped exactly what **is** buildable: `MR ∈ {8,16,24,32,40,48} × NR ∈ {2,4,...,32}` (even only) = 96 combos
   per `OPT` mode, for `BASE`/`LDX`/`OPT` (`UNROLL` has no kernel directory on this board at all). The
   reference's own sweep script, `execute_all.sh`, assumes a wider range (`NR` up to 48) than exists here, and
   has no error handling — failed builds under it just leave a missing/empty result file and the loop
   continues silently.

**Where this is headed, not yet run:** a scoped sweep — `OPT=LDX`, `GATHER=2` (macc), `SWAP=0` fixed (matching
everything else in this issue), `MR×NR` varied across the 96 buildable combos, `bert_b1_b5.dat` as the
dataset — to find each B-layer's *best available* tile size under the untouched reference, and only then
compare that against the publication table. The user is driving this manually on `jbpi2` with step-by-step
guidance from this agent (no direct board access from this session); a script for it was handed over but not
yet executed. Even a full sweep can only report each layer's best *available* result — B2/B3/B4's true
published pins (16×15, 32×7) still can't be reproduced, so the eventual comparison will be "best we can
actually build" vs. "what was published at a pin we can't build," not a clean apples-to-apples match for
those three. B1/B5 (32×6, both real, both buildable) are the only shapes where a same-pin comparison is
possible at all.

Nothing here is committed as a numeric result yet — this section is a record of the investigation in
progress, not a finding.

## 10. Resolved: the "publication" table is `prev_xdsl`, not the reference (2026-09-22)

Searched the user's Notion workspace (~12 pages, Feb–Sept 2026) for where the §9 table (B1 ~8.0, B2 ~12.5,
B3 ~8.8, B4 ~8.5, B5 ~7.5) actually came from. Found it, quoted directly from Notion page "debug perfromance
differnces" (2026-07-29, https://app.notion.com/p/3a968a923c2a8043b3c4fdf2c741fe55):

> "xdsl **privious ref design** is 8.8" ... "Comparison: Current xDSL AXPY vs **Previous xDSL Design
> (prev_xdsl)**"

**The table is `prev_xdsl` — a previous generation of this project's own xDSL codegen, kept as an internal
baseline. It was never the third-party reference (`ext/RVV_ukernels_benchmark`), at any point.** The user's
own notes use "ref design" ambiguously: sometimes the third-party C/Exo reference, sometimes `prev_xdsl`. §9's
entire framing — "publication-vs-fresh discrepancy" against the reference — was built on that ambiguity. There
was never a real discrepancy between the reference and a publication, because the publication table was never
about the reference.

**The discrepancy that *is* real (prev_xdsl vs current design) is also resolved, not just relabeled.** The
same Notion page ran the controlled experiment issue #30 kept re-deriving from scratch: forcing the *current*
xDSL design to the same tile size as `prev_xdsl` (16×15) recovered near-identical numbers — B3 8.72 vs
`prev_xdsl`'s 8.8, B2 12.59–12.64 vs `prev_xdsl`'s 12.5. So the apparent gap is mostly tile-size mismatch —
the exact failure mode hit again today when B2/B3/B4 got forced through a 32×6 reference build (§9 point 2).
The residual instability beyond that was traced, in the same Notion page, to a real bug:
`get_optim_mc_nc_kc()` in the xDSL driver hardcodes `*kc = k`, bypassing K-dimension cache blocking and
destroying stability specifically on B3-shaped problems (small M, large K). **Not yet checked whether that bug
is still present in the current driver** — worth a follow-up, separate from this issue.

**Corrected memory**: `project_reference_baseline_perf_user_stated.md` rewritten to reflect this; it had
already been rewritten once this session (first correction: "not the reference, it's our own design" — itself
incomplete, since it's specifically a *previous generation* of our own design, not the current one).

**What this means for §9's open investigation**: it's closed. §7's B1/B5 reference numbers (7.15–7.18,
7.05–7.16 GFLOPS) were never meant to be compared against the §9 table at all — they're the third-party
reference, correctly isolated. The comparison worth having is `prev_xdsl` vs the *current* design's own
numbers (§8: B1 8.04–8.05, B2 12.59–12.72, B3 5.19–5.62, B4 7.67–7.80, B5 7.82–7.83 GOPs) at matched tile
sizes — B1/B2 already land close to `prev_xdsl`; B3 is the one still off, consistent with the `kc=k` bug above
if it's still present.

**Double-checked (2026-09-22): user initially pushed back, "it's the reference design, not prev_xdsl."**
Verified directly against the live upstream repos on GitHub — `adcastel/RVV_ukernels_benchmark` (vendored in
`ext/`) and the related `adcastel/RVV_ukernels_Exo` (broader kernel set, more `MR` values). Neither has ever
shipped a `16×15` or `32×7` kernel, at any `VLEN`, in any `OPT` variant — both are even-`NR`-only in the
author's own published kernel history. With that shown, the user confirmed: it is `prev_xdsl`. Closed with
confirmation, not just inference from the Notion quote.

## 11. Final synthesis: what to conclude (2026-09-22)

**Reference material for this section — the `prev_xdsl` performance table and its source, documented here
directly so it doesn't have to be re-found:**

| case | config | `prev_xdsl` GFLOPS |
|---|---|---|
| S1 | 32x7 | ~8.5 |
| S2 | 32x7 | ~8.7 |
| S3 | 32x7 | ~8.9 |
| S4 | 32x7 | ~9.0 |
| S5 | 32x7 | ~9.0 |
| B1 | 32x6 | ~8.0 |
| B2 | 16x15 | ~12.5 |
| B3 | 16x15 | ~8.8 |
| B4 | 32x7 | ~8.5 |
| B5 | 32x6 | ~7.5 |

**Source, exact location:** Notion page **"debug perfromance differnces"**, dated **2026-07-29**, at
**https://app.notion.com/p/3a968a923c2a8043b3c4fdf2c741fe55**. This is where `prev_xdsl` is named directly
("Comparison: Current xDSL AXPY vs Previous xDSL Design (prev_xdsl)") and where "xdsl privious ref design is
8.8" appears — the direct evidence behind §10's conclusion. The same B1–B5 rows (not S1–S5) are also quoted,
citing this same prior value, in two more recent pages: **"verify fixed kernel dimentions and impact of
packing"** (2026-09-18, https://app.notion.com/p/3dd68a923c2a80debcf4e7ffbf856c86) and **"debug istability
with CC"** (2026-09-21, https://app.notion.com/p/3dd68a923c2a80c6b938dced75ab53f6), both under a column
labeled `FP32 GEMM+xDSL`.

---

The user's own read of all the numbers gathered in this issue: `prev_xdsl` (the publication table, §10) tracks
closely with the current design's own numbers (§8), and both sit apart from the untouched-reference numbers
(§7). Initial instinct was to disregard the reference sweep entirely as unreliable. Checked that against the
actual sizes of the gaps before accepting it — the two are not the same kind of gap.

**The only valid, tile-matched comparison against the reference is B1 and B5** — the sole two shapes the
reference can build at all (§6):

| Layer | `prev_xdsl` | Current design (§8) | Reference, untouched (§7) | Design vs. reference |
|---|---|---|---|---|
| B1 | ~8.0 | 8.04–8.05 | 7.15–7.18 | **+12%** |
| B5 | ~7.5 | 7.82–7.83 | 7.05–7.16 | **+9–10%** |

That is not a large gap. It is a modest, plausible difference between two independent microkernel
implementations, consistent in direction and magnitude across both shapes, and it agrees with `prev_xdsl`
sitting above the reference too — three independent-ish signals (prev_xdsl, current design, both boards)
all landing the same side of a ~10% line. That is what a real, believable result looks like, not a broken
measurement.

The numbers that *did* look drastically different — B2/B3/B4 forced through the reference's only buildable
tile (32×6) instead of their real 16×15/32×7 pins, 30–40% off — were never a valid comparison in the first
place (§6/§9): the reference has no kernel at those dimensions, full stop. That gap being large says nothing
about whether the B1/B5 gap is trustworthy; it was already known to be an invalid comparison before it ran.

**Conclusion, and the user's decision on how to treat it (2026-09-22):**
- The reference sweep (§7) stands as valid, real evidence — not disregarded. It is the only apples-to-apples
  head-to-head this issue has, and it shows the current design outperforming the untouched reference by
  roughly 9–12% on B1/B5.
- B2/B3/B4 have no reference counterpart to compare against at all — a scope limit (the reference's own
  kernel set doesn't cover those dimensions), not a measurement to disregard because it's inconvenient.
- The right thing to report, for either the paper or internal tracking: "design vs. reference" only for
  B1/B5 (+9–12%), and "design vs. `prev_xdsl`" (this project's own prior generation) for the full B1–B5 set,
  clearly labeled as two different comparisons, not blended into one table.

**Meeting-readiness note (2026-09-22), separate from validity:** the reference numbers (§7) stay documented
here as valid — kept, not deleted or hidden — but the user does not want them presented in an upcoming
meeting yet, since they cannot yet self-explain *why* the reference runs slower (only that it plausibly does,
by ~9–12%, not what in either implementation causes that specific margin). This is a presentation-readiness
call, not a data-validity one: nothing here says the numbers are wrong, only that the "why" isn't understood
well enough yet to defend in front of an audience. Revisit before presenting if that explanation gets found.

## 12. See also: interactive website report (2026-09-22)

`tests/pure_xdsl_blis_rvv/logs/report_2026-09-20/` holds a standalone interactive HTML report
(`index.html`, built by `build_report.py` from `template.html`) — not generated or touched as part of this
issue, but covering the same project and adjacent time window, so it's worth cross-checking against the
numbers here rather than treating this issue's findings in isolation. Dated 2026-09-20 — after the `prev_xdsl`
source found in §11 (2026-07-29), just before this issue's own board runs (§7/§8, 2026-09-22); not verified
here whether it uses `prev_xdsl`, the reference, or some other baseline as its comparison point — check before
citing a number from it alongside anything in this issue.

## 13. The scoped MR×NR sweep from §9, run (2026-09-23/24)

The sweep §9 described as "handed over, not yet executed" was run. Log pasted into the session directly (not
saved under `tests/pure_xdsl_blis_rvv/logs/` — no path recorded for where the user ran it beyond `fej
jlei@jbpi2:~/ws_bpi/RVV_ukernels_benchmark/gemm_blis_family`). Params match §9's plan: `OPT=LDX`, `GATHER=2`
(`macc`), `loadAB`, `FAMILY_EXO`/`RVV_EXO`, FP32, dataset `bert_b1_b5.dat` — which turns out to carry **5**
shapes, not 2: `1024×384×1024` (B1), `384×384×64` (B2), `64×384×384` (B3), `4096×384×1024` (B4),
`1024×384×4096` (B5). `MR ∈ {8,16,24,32,40,48}`, `NR ∈ {2,4,...,32}` even, tmin 3.0 s, one rep per shape (no
jitter stats — raw driver GFLOPS only, `PASS` column is `-` throughout, i.e. no correctness check requested
this run).

**Buildable: 54 of 96 combos.** The rest fail one of two ways, both at compile time (no board time wasted on
either):
- **Missing 1×1 fringe kernel** (`gemm_RVV_1x1_b0_col_fp32` undeclared) — the vendored kernel set stops
  existing past a rough `MR×NR` ceiling: NR≥30 for MR=16, NR≥20 for MR=24, NR≥16 for MR=32, NR≥12 for MR=40,
  NR≥10 for MR=48. Matches §9 point 3/4's finding that the reference only ships kernels it happened to
  generate, not a full grid.
- **One corrupted vendored file**: `16×14`'s header include line is truncated
  (`kernels_RVV_16x14_fp32.c:1` reads `s_RVV_16x14_fp32.h"` — missing the `#include "kernel`/`fp32_kernel`
  prefix). Not a missing-kernel gap like the others; this one file is bad. Not investigated further (out of
  scope for this issue — flag for whoever next needs 16×14).

**Best available tile per B-shape (GFLOPS, single rep, `--tmin 3`, jbpi2, `OPT=LDX`):**

| Shape | Best MR×NR | GFLOPS | Runner-up |
|---|---|---|---|
| B1 (1024×384×1024) | **32×6** | 7.19 | 48×4 (7.00), 24×8 (6.93) |
| B2 (384×384×64)    | **16×10** | 10.4 | 24×8 (9.53), 16×8 (9.49) |
| B3 (64×384×384)    | **8×28** | 6.25 | 8×20 / 24×8 (6.12) |
| B4 (4096×384×1024) | **32×6** | 7.20 | 24×8 (7.12), 48×4 (6.87) |
| B5 (1024×384×4096) | **32×6** | 7.91 | 48×4 (7.37), 24×8 (6.87) |

**What this adds to §11's conclusion:** B1/B4/B5 all peak at the same tile, **32×6** — which is also the
*only* real, published pin the reference can build for those shapes (§6/§9). So §11's B1/B5 comparison
wasn't just the only buildable option, it was also the best buildable option; nothing else in the 54-combo
sweep beats it for those three shapes. B1's 7.19 here is close to but not identical to §7's 7.15–7.18 (same
tile, different day/rep count — expected noise, not a discrepancy).

B2/B3 have no real pin to compare against at all (§6: the reference never shipped 16×15). Their "best
available" numbers (16×10 → 10.4, 8×28 → 6.25) are informative for tuning intuition but still not a
same-pin comparison to anything — consistent with §11's existing caveat, not a new finding for those two.

**Not done:** other `OPT` modes (`BASE`, plain `RVV_EXO` without `LDX`), other `GATHER` variants, `MC`/`NC`
overrides, and a correctness pass (`PASS` column unused this run) — all out of scope for what was asked.
