# Issue #52: Why the reference's 16x14 is steady and ours is not: read the two designs side by side (sub-issue of #48)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/52

**Comes from:** #48 §8.5 (answer 4: our 16x14 is 81-86% slow on B3, as slow as our 16x15, while the reference's 16x14 is steady in #46,
so the tile does not explain the gap) and #46 §7.2 (moving the buffers changes the reference's speed by about 30%).

**Role / Label:** `needs-triage` — sub-issue of #48, priority 4 of the #48 follow-ups (opened 2026-10-14). Desk work: no board time, no code change. `ext/` stays read-only.
**Status:** 🟡 Answered (§8-§10, below).
**Target:** none (code read). A board test only as a later, separate step.
**Shape(s):** B3 (64x384x384) and B3_K256, as built by each design.
**Primary Sources:**
- [[0030-reference-b3-per-rep]] (#46) — the reference's per-rep times; §7.2 on buffer placement.
- [[0032-b3-microkernel-dimension-sweep-pack-and-compute]] (#48) §8.2, §8.5 — our 16x14 on both boards.
- `../ext/RVV_ukernels_benchmark/gemm_blis_family/src/gemm_blis.c`, `test_gemm.c`, `inutils.c` (read-only) — the reference's GEMM, packers, timing loop and allocation.
- `tests/pure_xdsl_blis_rvv/templates/driver_benchmark.cpp`, `generate_pure_blis_project.py`, `pack_builder.py` — ours.
- [[0033-b-stream-layout-k256-flip]] — the layout question; the code read should answer its questions for the reference too.

---

## 1. Question

At the same tile (16x14) and the same shape, the reference is steady and ours is two-speed. **What is different between the two at 16x14,
and which differences could change where the B read streams and the pack buffers land, or how the prefetcher sees them?**

## 2. Why it matters (why priority 4)

The reference is the only steady 16x14 we know of. A difference between the two designs that makes it steady is the cheapest way to find
the trigger, and #46 has been open on it since 2026-09-29.

## 3. What to compare (a checklist, one row each, with `file:line` on both sides)

- **Layout of B and `ldb`:** row- or column-major, `ldb` at run time, any padding. Ours: column-major, `ldb = K` (`driver_benchmark.cpp:752`).
- **Allocation and alignment** of A, B, C, Ac, Bc: `malloc` / `aligned_alloc` / static, the alignment (ours: 32 bytes, #46 §7.2), the order they are allocated in, page alignment, whether the pack buffers are reused across reps.
- **The B packer:** read order (row by row or column by column), vector or scalar, strided loads or not, the width of one read, the Bc layout (row width, padding of the fringe panel).
- **The A packer:** the same questions.
- **Blocking:** mc, nc, kc chosen for B3 at 16x14 in each design, the loop order (jc, pc, ic, jr, ir), where packing sits in the nest.
- **The kernel:** how it reads Ac and Bc (unit-stride or not, how far ahead), how it handles the fringe tiles (ours: a temp buffer and scalar copy, #48 §7.1), and whether it takes `ldc`/`ldb` at run time (the reference's LDX kernels do, #46).
- **The timing loop:** what runs between reps (flushes, allocations, clock reads); #46 §7.2 shows that a heap allocation between reps matters.

## 4. Plan

1. Two independent read-only passes (subagents, `model: "sonnet"`), each filling the checklist with `file:line` for both designs, without seeing the other's answer.
2. Compare the two passes. Any row where they disagree, or any row that would change a decision, is checked by hand in the source before it goes in.
3. Rank the differences by how well they fit what is known: steady at NR=14 in the reference; in ours NR up to 6 is steady at K=384 and the K=256 flips ([[0033-b-stream-layout-k256-flip]]).
4. Write down, for the top two or three, the smallest change to *our* build that would copy the reference on that point, as a proposed board test. Do not build it here.

## 5. How to read it

- **A difference in B layout, `ldb` or buffer alignment:** it lines up with the layout question. Run it as an arm of [[0033-b-stream-layout-k256-flip]] step 2 or 3.
- **A difference in the packer read order:** it lines up with #44 (one column at a time). The reference is then evidence for #44.
- **No difference that touches memory layout or read order:** look at the kernel and the timing loop rows next.

## 6. Out of scope

Editing `ext/`; changing our generator or driver; any board run (the proposed tests go to a later step or to [[0033-b-stream-layout-k256-flip]]).

## 8. Two independent passes, compared (2026-10-14)

Two blind subagent passes (model `sonnet`), neither seeing the other or any prior write-up of
this question, filled the §3 checklist independently. They agreed on six of seven rows:

| row | agreement |
|---|---|
| 1. `ldb` | Both: ours has `--pad-ldb` (default 0, so `ldb=k` today, identical to the reference); reference has no such option and always `ldb=k`. Not an active difference at today's settings. |
| 2. Allocation | Both: ours `aligned_alloc(64,...)`, `Ac`/`Bc` allocated fresh **per shape and per variant** (`driver_benchmark.cpp:784-787, 891-892`), freed after that variant's reps; reference `my_aligned_alloc(32,...)` (calloc'd), `A/B/C/Ac/Bc/Cc` allocated **once before the whole shape loop** (`test_gemm.c:267-277`), reused for every shape and rep until the program exits. Both flag this as plausibly layout-relevant. |
| 3-4. Packers | Both: same read order and strides on both designs; ours issues RVV vector loads (`vlse32`/`vle32`, up to 8 lanes/instruction), the reference reads one scalar element at a time, same addresses and order either way. Both rank this as the least likely of the real differences to change *where* anything lands, since the addresses visited are identical. |
| 5. Blocking/loop order | Both: `mc=64, nc=384, kc=128` on both sides (ours is a direct port of the reference's own `model_level.c`), loop order `jc→pc→ic→jr→ir` identical, packing sits in the same place in the nest on both sides. No real difference. |
| 7. Timing loop | Both: one clock read per rep, nothing else inside the loop, on both designs under the comparable (uninstrumented / `--ref-protocol`) build. No real difference. |

**Row 6 (the microkernel/fringe path) disagreed and was checked by hand, per §4's plan step
2.** One pass found the reference calls a dedicated, specialized microkernel compiled per
exact `(mr, nr)` pair (reading `kernels/RVV_256_LDX/fp32/16x14/...`, and writing straight into
`C`, no temp buffer, even on a fringe tile). The other pass found the reference calling into a
BLIS-library microkernel (`bli_cntx_get_l3_sup_ker_dt`, `test_gemm.c:91`) whose internals are
compiled, not visible in this source tree -- a quite different picture.

**Resolution, read directly from the source and the build scripts:** `test_gemm.c:91`'s BLIS
kernel pointer is only used when `FAMILY_BLIS` or plain `BLIS` is the active `RUN_MODE`.
`run_ref_baseline.sh` (the script this project's own benchmarks actually use) always passes
`RUN_MODE=FAMILY_EXO` (`ref_baseline/run_ref_baseline.sh:183,274`). Under `FAMILY_EXO`,
`gemm_blis.c:119` does `ukr = *ukrmatrix[mr][nr][be]`, where `mr`/`nr` are **already the
fringe-adjusted sizes** (`mr = min(mc-ir, MR)`, line 108; `nr = min(nc-jr, NR)`, line 106), and
calls that kernel **directly on `Cptr`, with `ldC`**, no temp buffer at all (`gemm_blis.c:119-131`).
The matching per-size kernel files are real and present at `kernels/RVV_256_LDX/fp32/16x14/...`
(and `RVV_256_BASE`, `RVV_256_OPT`) -- confirmed by listing the directory, not just citing it.
**The first pass was right about the path this project actually runs; the second pass described
an inactive mode.**

So row 6 is a real, confirmed difference: ours (pinned mode) always calls the one fixed-size
16x14 kernel and routes any fringe tile through a stack temp buffer plus a scalar copy-in/copy-out
(`generate_pure_blis_project.py:606-623`); the reference generates and calls a separate exact-size
kernel for every `(mr, nr)` pair that actually occurs, writing straight into `C` every time, fringe
or not. B3's `N=384` is not a multiple of `NR=14` (384 = 27×14 + 6), so this fringe path runs on
every one of B3's three K-blocks (`kc=128`, `K=384` needs `ceil(384/128)=3` blocks).

## 9. Ranked differences

1. **Microkernel/fringe handling (row 6).** Confirmed real (§8). Not previously considered by
   either #44 (stream count) or [[0033-b-stream-layout-k256-flip]] (ldb/layout) -- it is a third
   mechanism. Ours adds an extra read-from-temp-buffer and copy-to-C pass on every fringe tile
   that the reference's design never does at all.
2. **Allocation alignment and lifetime (row 2).** Confirmed real, and it already has empirical
   support: #46 §7.2 found that moving the reference's own buffers by one heap allocation changed
   its B3_K256 speed by about 30%. Ours reallocates Ac/Bc fresh for every shape and variant; the
   reference allocates them once and never moves them again for the whole run.
3. **Packer instruction width (rows 3-4).** Real but, by both passes' agreement, the least likely
   to move where anything lands, since the addresses visited are identical either way. This is
   exactly what #44 already proposes to test (pack B one column/element at a time) -- no new test
   needed here, defer to #44.
4. **`ldb` padding (row 1).** Not an active difference at today's settings (both designs run with
   `ldb=k`); ours merely has an unused capability the reference lacks. Not board-test-worthy on
   its own.

## 10. Proposed board tests (not run here)

- **For rank 1 (fringe path):** before touching the generator, isolate whether the fringe path is
  even implicated: run B3 at 16x14 against a shape with the same M, K and tile but N an exact
  multiple of 14 (e.g. N=378 or N=392), so there is no N-fringe at all. If the slow-rep share drops
  sharply with no fringe, the temp-buffer-and-copy path is a real contributor; if it does not move,
  this row drops in the ranking. (A generator change to give fringe tiles their own exact-size
  kernels, matching the reference, is a separate, bigger spec -- not proposed here.)
- **For rank 2 (allocation):** the smallest change that copies the reference's pattern: allocate
  Ac/Bc once per run (outside the per-shape, per-variant loop) in `driver_benchmark.cpp`, zeroed
  once, reused across every shape and rep, instead of the current alloc-per-variant/free-after.
  Run B3 at 16x14 before and after, same tile, same board, same launches; if the slow-rep share
  drops, allocation lifetime is a real contributor, independent of (and possibly compounding with)
  [[0033-b-stream-layout-k256-flip]]'s `ldb`-padding test.
- **For rank 3 (packer):** no new test -- this is #44's own planned spec.

## 11. Closure

- [x] Checklist filled for both designs, with `file:line`, two independent passes compared, disagreements checked by hand. See §8.
- [x] Ranked list of differences with a one-line reason each. See §9.
- [x] Proposed board test(s) for the top differences, written here, not run. See §10.
