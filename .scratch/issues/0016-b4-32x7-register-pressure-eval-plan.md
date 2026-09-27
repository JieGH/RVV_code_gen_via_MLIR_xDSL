# Issue #31: B4's 32x7 kernel — register-pressure doubt, and a plan to check it

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/31

**Closed 2026-09-27 after audit** (§8). **Audit report:** [issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.tex)) — brief and audit of #31-#34, 2026-09-27.
**Role / Label:** `resolved` — the register-pressure doubt this issue was opened to check (§1-§2) is
**answered: no.** §7 below closes it out. The separate, still-open question this work surfaced — why
B4 specifically underperforms the square family despite sharing an identical, proven-fine tile — is
**not** this issue's question and moves to a new sub-issue (§7).
**Status:** 🟢 Closed on its original question. §4.1's question got an unplanned shortcut: the
reference's own true 32x7 kernel turned out to exist (§6.1) and was extracted, verified correct, and
run in place of ours (§6.2-6.4). Across both cores tried (core 0: ours 7.1% faster; core 4: roughly
tied, §6.7), our kernel was never meaningfully *worse* than the reference's own equally
register-constrained kernel at this tile — the opposite of what §2's doubt predicted. Separately, the
square family (S1-S5) — the same tile, the same zero-spare-register profile, swept across a 25x range
of problem size — holds tight, low-jitter (CV 0%), size-ordered performance the whole way (§7.1). Both
lines of evidence say the same thing: **the tile is not the problem.** What *is* the problem — B4 sits
below every one of S1-S5 despite the identical tile and a FLOP count that lands between S1 and S2 — is
now [[0019-b4-bert-shape-underperformance|a separate issue]], because it's a shape question (BERT's
M=4096/N=384/K=1024 proportions), not a tile/register question. See §7.
**Target:** `jbpi2` (BananaPi M2 Pro, VLEN=256), reached through jump host `fej`. Originally core 0;
switched to **core 4** partway through §6 after finding the board is shared (§6.6) — core 0 itself
turned out to run 5-10% slower than core 4 for the identical binary (§6.6), so core 0 numbers and
core 4 numbers must not be mixed. Every result below says which core it's from.
**Primary Sources:**
- `tests/pure_xdsl_blis_rvv/config.json:60`, `datasets/issue2_square_bert_32x7.dat`,
  `datasets/issue26_b4.dat` — B4 is pinned to `MR32_NR7`: M=4096, N=384, K=1024. It shares that tile
  with S1-S5 (the whole Square family); B1/B5 use `MR32_NR6`, B2/B3 use `MR16_NR15`.
- [[0006-match-reference-design-everything-but-the-two-intended-differences]] §"Confirmed matching" —
  at 32x6, both our design and the reference use 28 of 32 vector registers, zero spills, byte-identical
  instruction mix. This is the baseline the doubt below is measured against.
- [[0014-issue-29-reference-alone-sweep]] §4-5 — the reference has no true 32x7 or 32x8 Exo kernel;
  32x8 (the nearest built neighbour above 32x7) runs at 51-67% of 32x6's speed on every shape, reasoned
  (not read from assembly) to be spilling because it needs more than 32 registers.
- [[0003-scalar-vs-vector-pack-timing]] §3 — the existing with/without-packing split for B4 (and every
  other case), from 2026-09-16: B4 packing was 14.8% (vector) / 16.1% (scalar) of total time. This is
  the baseline the "redo with and without packing" half of the ask should be compared against, not a
  fresh discovery.
- `tests/pure_xdsl_blis_rvv/pj_gen/fp32_AXPY_FP32_Pack_vlen256_MR32NR7/gemm_b1_xdsl.cpp` — the generated
  microkernel source read for §2 below.
- `docs/metrics.md:32-78` — our `KerGOPs` has included
  packing since issue #13; never compare a pre-#13 number to a post-#13 one, and always say which build
  (`INSTRUMENTED=0`/`=1`) a number came from.

---

## 1. What's being asked

The user has doubts about the B4 layer's `32x7` kernel dimension (`MR=32`, `NR=7`) — both how fast it
actually runs, and whether that shape should be redesigned. Two things were asked for, in order:

1. **First, an evaluation plan** for this layer (this document) — not the evaluation itself.
2. Later: measure B4 with and without packing, and consider redesigning the kernel dimension.

This plan covers both halves: how to get a trustworthy with/without-packing number for B4, and how to
check whether `32x7` itself is the problem before spending time on a redesign.

## 2. Why there's something real to doubt here (read-only check, done now)

`32x7` was chosen by the search behind `config.json`'s pinned tiles (issue #2), not hand-picked for
register-file fit. Whether it fits is a separate question nobody checked directly. Reading the actual
generated kernel answers the register-count part of that without needing a board:

`gemm_b1_xdsl.cpp:7` — `gemm_microkernel_v_fp32_32x7_b1`, the beta=1 compute step generated for this
tile. Counting the vector values live at once, at the peak (right before the store-back):

| what | count | why |
|---|---|---|
| A-tile loads (`vle32`, kept alive across all 7 NR columns) | 4 | `MR/lane_width` = 32/8 |
| C-tile loads (`vle32`, beta=1 loads existing C before accumulating) | 28 | `MR/8 × NR` = 4×7 |
| `vfmacc` (C += A·b, in place) | 28 | one per C register |
| `vse32` stores (C tile written back) | 28 | one per C register |

Peak live vector values = 4 (A) + 28 (C) = **32 of 32** vector registers, with nothing free. This is a
source-level count (MLIR → C++ intrinsics, before the backend compiler's own register allocation), so
it is not proof of an actual spill — but it is the same kind of count [0006] used to confirm 32x6 is
spill-free (28 of 32, 4 spare) and [0014] used to explain why 32x8 spills (would need 36). Laid out
together:

| tile | live A+C registers | spare | measured outcome |
|---|---|---|---|
| 32x6 | 4 + 24 = 28 | 4 | confirmed zero spills, byte-identical to reference [0006] |
| **32x7** | **4 + 28 = 32** | **0** | **not checked — this doubt** |
| 32x8 | 4 + 32 = 36 | −4 (over) | confirmed spills (reasoned), runs at 51-67% of 32x6 [0014] |

32x7 sits exactly on the boundary between a tile that's proven fine and one that's proven to spill and
lose half its speed. Zero spare registers doesn't only risk an outright spill — it also removes any
room the backend compiler has to reorder loads and `vfmacc`s for latency hiding (e.g. starting next
k-iteration's A loads while still consuming this one's C), which 32x6's 4-register margin allows. That
could show up as a real slowdown with no actual spill instruction anywhere, which is why this needs
looking at from more than one angle (§4).

This also matters beyond B4: S1-S5 are pinned to the same `32x7` tile (`issue2_square_bert_32x7.dat`),
so whatever is true of this register count is true of six of the ten benchmark rows, not just B4.

## 3. What we already know, so the plan doesn't re-derive it

- **B4's dims**: M=4096, N=384, K=1024, reuse≥24 (BERT-shaped, not square) — from `issue26_b4.dat`.
- **Pack weight, B4, 2026-09-16** [0003 §3.1]: 16.1% scalar-pack / 14.8% vector-pack of total time —
  a real with/without-packing split already exists, just nine days old and from before several since-
  landed changes (issue #13's metric fix, issue #27's reference protocol, the pack-swap work in #26).
  The plan below re-derives it fresh rather than trusting a number this old.
- **Most recent B4 GFLOPS** [0014 §5], packing included, uninstrumented, reference protocol, 16-run
  sweep: vector pack 7.80, scalar pack 7.58. Reference's nearest built kernel (32x6, not true 32x7) —
  6.91. Published table figure — ~8.5, which *nothing* reaches, including the reference itself at its
  own near kernel (6.91, −18.7% off the published number). So "B4 underperforming" is already on the
  record; this plan is about whether the register-boundary story in §2 is *why*.
- B4 has never been run at any tile other than 32x7 — `issue2_square_bert_32x7.dat`'s header says
  explicitly "B4 is the only BERT shape pinned to MR32_NR7." There is no same-shape 32x6 or 32x8 number
  for B4 to compare against; only the 32x6/32x8 numbers from *different* shapes (S1-S5, or the
  reference's brackets in [0014]) exist today.

## 4. The evaluation plan

Four checks, cheapest and least board-time first. Each is independent — later ones only make sense to
run if an earlier one doesn't already answer the question.

**4.1 — Does the compiled kernel actually spill? (no board time, needs the cross toolchain only)**

Compile `gemm_b1_xdsl.cpp` (and the beta=0 counterpart) for the MR32NR7 project to assembly
(`-O3 -march=rv64gcv -mabi=lp64d`, matching [0006]'s flags) and grep the compute-nest region for
`vs1r.v`/`vl1re32.v` (the spill/reload pair [0004] used to detect this in the packer) or any stack
`sd`/`ld` of a vector-sized value adjacent to the `vfmacc` block. Compare instruction count and mix
against 32x6's already-confirmed-clean disassembly [0006]. This directly answers "is the doubt real,"
no hardware needed, and is the single most informative check in this plan.

**4.2 — Fresh with/without-packing split for B4 specifically**

Re-run B4 alone (not the full matrix) on the `INSTRUMENTED=1` build, which already reports
`Vector_Pack_Time_ms`, `Scalar_Pack_Time_ms`, and `Compute_Time_ms` per case [0003 §2] — no new
instrumentation needed, just a fresh run since the one on record is nine days old and pre-dates #13/#26/
#27. Report, per the house rule, which build the number came from (`docs/metrics.md`'s "always state
which build"): this is necessarily the `INSTRUMENTED=1` number, since that's the only build that
produces the split at all — it is not comparable to the `INSTRUMENTED=0` headline GFLOPS number and
should be reported as its own line, not blended into it.

Read the result two ways:
- **Pack fraction vs. the other BERT shapes.** If B4's pack share (currently 14.8-16.1%) is now
  noticeably higher than B1/B5 (both ~15-17% at 32x6 [0003 §3.1]) that would say packing itself, not
  the compute kernel, moved. If instead the *compute* time per rep grew relative to what raw FLOP count
  alone predicts (i.e. `Compute_Time_ms` scaled worse than M×N×K would suggest, compared to 32x6's B1/
  B5), that points at the compute kernel — consistent with §4.1's register story.
- **Perf counters.** The same `INSTRUMENTED=1` run also writes `perf_*.txt` (hardware counters, per
  [0006]/[0004]'s method) — check IPC and L1 load/store counts on the compute nest specifically. A
  spill or reload shows up as extra loads/stores beyond what the instruction count in §4.1 predicts, and
  as a lower IPC than 32x6's compute nest at a comparable K.

**4.3 — Bracket B4 itself, not a different shape**

No B4 number exists today at any tile but 32x7 (§3). Add two more `config.json` rows pinning B4's own
`M=4096, N=384, K=1024` to `MR32_NR6` and `MR32_NR8`, using the same `--no-nc-overrides --cpu 0 --tmin 10`
head-to-head protocol as every other cross-comparison quoted against the reference. This gives a true bracket
on B4's own shape instead of borrowing S1-S5's or the reference's numbers for a different M/N/K, and
directly tests whether B4 would be faster at a tile with register slack, all else equal. Expected board
time: small — these are two extra single-shape rows, not a full matrix sweep.

**4.4 — Only if 4.1-4.3 confirm the doubt: redesign candidates**

Not run yet, listed here so the plan doesn't stop short of the user's actual ask. If the checks above
show 32x7 is genuinely register-starved on B4's shape, the candidates worth generating and benchmarking
next are ones that keep roughly the same MR×NR compute-to-load ratio while landing inside the 28-
register band 32x6 already proved clean:
- `MR=24, NR=7` (3+28=... recompute: 24/8=3 A regs, 3×7=21 C regs, 24 live, more slack) or
- `MR=32, NR=6` / a search over `NR` at fixed `MR=32` bounded to stay ≤28 live registers, which is
  exactly what §4.3 already benchmarks on B4's real shape — so 4.4 may turn out to be "use one of 4.3's
  brackets" rather than a new kernel at all, if one of them beats 7.80 GFLOPS.

Whether a redesign is worth doing at all depends on 4.1-4.3's answer: if no spill and no compute-side
slowdown shows up, the doubt is unfounded and B4's current gap to the reference/published numbers
(§3) has some other cause, not this one.

## 5. What this plan cannot show

- §4.1's disassembly read is static — it shows whether the compiler chose to spill, not whether a
  non-spilling-but-tight schedule still costs cycles (the "zero scheduling slack" risk in §2). Only
  §4.2's perf counters and §4.3's real bracket numbers speak to that.
- §4.3's bracket is B4 alone, one shape. It does not say whether the same is true for S1-S5, which
  share the 32x7 tile but have very different reuse (square, not BERT-shaped) — a separate question if
  this one comes back positive.
- None of this touches the reference design's own register behavior at 32x7, because it has no true
  32x7 kernel to inspect [0014 §4] — the reference side of this stays a bracket forever, not just until
  this plan runs.
  **Correction (§6 below): this turned out to be wrong.** A true 32x7 kernel exists in the reference's
  own sources, just not as a pinned top-level build — [0014]'s claim was right for the *build the
  sweep used*, not for the reference's full kernel set.
- Board time and time-of-day are not controlled between these checks and the numbers they're compared
  against ([0014] §6) — same caveat as every other cross-run comparison in this project.

## 6. The compute-kernel swap experiment (2026-09-24)

The user asked to go further than reading assembly (§4.1): find the reference's own 32x7 kernel if one
exists anywhere in its sources, drop it into our harness in place of ours, and run it, to see whether
the register-pressure story in §2 is a property of the *tile* (shared by both designs) or of *our
code generation specifically*.

### 6.1 — The reference's own true 32x7 kernel exists — buried, not built

`ext/RVV_ukernels_benchmark/kernels/RVV_256_LDX/fp32/` has a directory per `MRxNR`, but only even `NR`
(`32x6`, `32x8`, `32x10`, ... every `MR`, every `NR` in the whole tree) — no `32x7` directory, matching
[0014]'s "the reference has no true 32x7." But `32x8`'s kernel file
(`.../32x8/loadAB/macc/kernels_RVV_32x8_fp32.c`) turns out to contain a **complete, dense fringe table**
— every `(mr, nr)` pair from 1 up to the built tile, Exo-generated, used to handle remainder tiles when
M or N isn't a multiple of MR/NR. That table includes `gemm_RVV_32x7_b0_col_fp32` (line 28883) and
`gemm_RVV_32x7_b1_col_fp32` (line 29019) — a true 32x7, generated by the reference's own Exo scheduler,
just never exposed as its own pinned build.

Reading it settles the register-count question directly, no assembly needed: same structure as ours —
`A_reg_0..3` (4, one per `MR/8` chunk) + `C_reg_{0..6}_{0..3}` (28, one per `NR × MR/8`) declared as
`vfloat32m1_t`, the `b1` (beta=1) variant loading existing C before the k-loop, `b0` zero-initializing
it with `vfmv_v_f`. **4 + 28 = 32 of 32 registers live, zero spare — identical to §2's count for our
own kernel.** This is generated independently, by a different toolchain (Exo, not xDSL), so the
register-pressure boundary in §2 is now confirmed to be a property of the (32,7) tile under VLEN=256,
not an artifact of our lowering path.

### 6.2 — Extracting it and swapping it in

Verbatim copy of both functions (`sed -n '28875,29145p'` of `kernels_RVV_32x8_fp32.c`), renamed
`ref_gemm_RVV_32x7_{b0,b1}_col_fp32` to avoid clashing with our own names, appended to the generated
`pj_gen/fp32_AXPY_FP32_Pack_vlen256_MR32NR7/gemm_{b0,b1}_xdsl.cpp`. Our own
`gemm_microkernel_v_fp32_32x7_{b0,b1}` bodies were replaced with a one-line call-through:

```cpp
void gemm_microkernel_v_fp32_32x7_b1(size_t v1, float* v2, float* v3, size_t v4, float* v5, size_t v6, float* v7, float* v8, size_t v9) {
  ref_gemm_RVV_32x7_b1_col_fp32(nullptr, (int_fast32_t)v1, v2, v3, (int)v4, v5, (int)v6, v7, v8, (int)v9);
}
```

The dispatch call site (`gemm_dispatch_xdsl.cpp:9`) was not touched, and neither was the packer
(`gemm_pack_xdsl.cpp`), the loop nest, or the driver — the argument mapping (`v1`=kc, `v2`=`&alpha`,
`v3`=Ac the packed-A buffer, `v4`=`max_mr` (lda), `v5`=Bc, `v6`=`max_nr` (ldb), `v7`=`&beta`, `v8`=C,
`v9`=ldc) lines up 1:1 with the reference function's own `(ctxt, KC, alpha, A, lda, B, ldb, beta, C,
ldc)` signature, so this isolates the compute microkernel alone. `pj_gen/` is gitignored and
regenerated from `config.json` on the next normal run — this patch lives only in the working tree, not
committed.

### 6.3 — Correctness first: verified bit-exact before trusting any speed number

Built a standalone test on `jbpi2` linking three things into one binary: our original generated
kernel (renamed `orig_gemm_microkernel_v_fp32_32x7_{b0,b1}`, from a pre-swap backup), the swapped
`gemm_microkernel_v_fp32_32x7_{b0,b1}` (now calling the reference kernel), and a `main()` that feeds
both identical synthetic packed buffers (`kc` ∈ {1, 3, 8, 17}, both `beta`∈{0,1}) and diffs the output
C tile. **All 8 cases: max abs diff = 0.000e+00 (bit-exact).** The swap changes nothing about what's
computed, only what code computes it — the performance numbers below are trustworthy on that basis.

### 6.4 — Result, core 0: the reference's own kernel is *slower*, not faster

Same board, same session, back to back, `INSTRUMENTED=0` (the benchmarking build), reference protocol,
`--no-nc-overrides --tmin 10`, `taskset -c 0`, one run each (not yet a 16-run sweep — see §6.8):

| case | ours (baseline) | reference's own 32x7, swapped in | Δ |
|---|---|---|---|
| B4 (4096×384×1024) | 7.77 GOPs | 7.22 GOPs | **−7.1%** |
| S1 (1000³) | 8.34 | 8.30 | −0.5% |
| S2 (2000³) | 8.76 | 8.21 | −6.3% |
| S3 (3000³) | 8.99 | 8.57 | −4.7% |
| S4 (4000³) | 8.69 | 8.20 | −5.6% |
| S5 (5000³) | 9.12 | 8.61 | −5.6% |

Per-rep noise is tight on both sides (std-dev/median 0-1% where more than a couple of reps fit in
`tmin`), and the gap grows from near-zero at S1 (smallest K, least loop-body exposure) up through
S2-S5 (deeper K), consistent with the difference living in the k-loop body itself rather than a
fixed per-call cost. Raw data: `tests/pure_xdsl_blis_rvv/logs/.../raw_fp32_AXPY_FP32_Pack_noinstr_refproto_no_nc_override.txt` (not committed — this is a scratch benchmark run, not a tracked log).

> **Audit note (2026-09-27, §8):** the "gap grows with deeper K" reading does not hold. Every case here
> uses `kc`=128 (see the cache-blocking line in `logs/run_2026-09-24_143441`), so the kernel's k-loop has
> the same length for S1-S5. The core-0 gap it explains also did not repeat on core 4 (§6.7).

This is the opposite of what §2 predicted. Both kernels hold the identical 32-of-32 live registers
(§6.1); if the register boundary alone explained our B4 gap to the published/reference numbers, the
reference's *own* equally-constrained kernel should be at least as fast as ours, not 5-7% slower.
It isn't. **The register-pressure doubt in §2 is confirmed as a real, shared property of the tile, but
it does not explain "our kernel underperforms" — on this test, our generated code is faster than the
reference's own Exo output at the exact same register count.**

Two explanations, not yet distinguished:
- **Compiler scheduling.** Our generated loop body is far noisier at the source level (every
  loop-carried C register goes through a redundant `vfloat32m1_t vXXX; vXXX = vYYY;` copy per
  iteration — an artifact of the MLIR→C++ lowering, see `gemm_b1_xdsl.cpp:201-224` for an example)
  while the reference's is a tight, hand-shaped loop. It's possible GCC's scheduler, given a real
  32-live-register problem, coincidentally makes better instruction-ordering choices when it first has
  to clean up our redundant copies than when handed the reference's already-tight loop directly. This
  would mean the *result* generalizes (our codegen isn't worse here) but the *reason* is closer to
  lucky compiler behavior than a deliberate advantage — worth being honest about rather than reading it
  as "our scheduling is better."
- **Not yet ruled out**: a confound from board conditions between the two runs (see §6.6) — the core-0
  pair above is back-to-back and same-session, which controls for time-of-day, but not for anything
  that changed on the shared board mid-comparison.

### 6.5 — What this changes about §3's "B4 underperforms" story

§3 recorded that B4 misses the published ~8.5 GFLOPS figure, and that even the reference's own nearest
built kernel (32x6 bracket) misses it too. §6.4 adds: **the shortfall is not a 32x7 register-pressure
code-generation gap**, because the reference's own equally register-constrained 32x7 kernel is slower
than ours, not faster. Whatever is behind the published-number gap, it is not "our xDSL lowering
handles the 32-register boundary worse than Exo does" — on this evidence it's the other way round.

### 6.6 — Board is shared; switched from core 0 to core 4; one rerun still pending

Mid-experiment, found `jbpi2` is a shared machine: another user's process (`psiwins`,
`mixed-precision-ir reference ... SpMV/matrix/BenElechi1/BenElechi1.mtx`) was pinned to core 3 at 100%
CPU throughout (started 12:12, still running). Cores 4-7 were idle. `board_pin.sh` already avoids core
0 by default for an unrelated reason (BENCH_CPU_DEFAULT=4 — core 0 absorbs every device interrupt);
this project's runs had been using `--cpu 0` per a recorded default-run-command preference, which
predates discovering this. Re-ran the swapped kernel alone on core 4 (`taskset -c 4`, same
`--no-nc-overrides --tmin 10`) to check: **core 0 itself runs 5-10% slower than core 4 for the
identical swapped binary** (B4 7.22 on core 0 vs 7.74-7.85 across two core-4 repeats; S3-S5 similarly
+9-10% on core 4). So core 0's interrupt tax is real and sizeable — confirming `board_pin.sh`'s own
comment — and core 0 numbers and core 4 numbers must never be compared to each other.

**What's still open**: only the *swapped* kernel has been re-run on core 4 (twice, consistent with
itself: B4 7.74 and 7.85). The *baseline* (our original, unswapped kernel) has not yet been re-run on
core 4 — the core-0 pair in §6.4 is internally valid (both sides paid the same core-0 tax, back to
back, same session) and the interrupt tax is a plausible flat overhead rather than something that would
flip the *direction* of a 5-7% gap, but that is an assumption, not something checked. **A same-core-4
baseline run is the one piece needed to confirm §6.4's numbers at full board fidelity — left as a
pending follow-up rather than run immediately, because the board is a shared resource** and re-running
under contention (or racing the other user's job) isn't a run worth trusting more than what's already
here. Two mechanical accidents during this session are also worth recording so a future run avoids
them: (1) a local `timeout` wrapped around the double-hop `ssh fej "ssh jbpi2 ..."` command kills the
*local* ssh client at the deadline, which can tear down the remote command mid-run without warning —
use `nohup ... > log 2>&1 < /dev/null & disown` on the remote side and poll the log file instead of
wrapping the whole thing in a client-side timeout; (2) `INSTRUMENTED=1` runs default to the warm-up
protocol and reject `--tmin` outright (the driver's own check, not just the wrapper script's) — drop
`--tmin` for any instrumented run.

### 6.7 — The pending core-4 baseline run (2026-09-24, later same day): B4's core-0 result does not reproduce

Board check before running: `top`/`ps` on `jbpi2` showed load average 2.00 but no active process using
meaningful CPU at the moment of the check (the earlier `psiwins` job from §6.6 was not visible in
`ps -eo pid,psr,pcpu` at that time) — read as "clear enough to run," not as a guarantee the board stayed
idle throughout, since this project's runs never lock the board against other users (§6.6).

Re-ran our original (unswapped) kernel — `pj_gen/` regenerated fresh from `config.json`, which wipes
the §6.2 swap patch automatically since it's gitignored and rebuilt on every run — on `jbpi2`,
`taskset -c 4`, same protocol as every other number in §6 (`INSTRUMENTED=0`, reference protocol,
`--no-nc-overrides --tmin 10`), one run:

| case | baseline, core 4 (mean / median, reps) | swapped, core 4 (§6.6) | Δ vs. swapped, core 4 | Δ vs. swapped, core 0 (§6.4) |
|---|---|---|---|---|
| B4 (4096×384×1024) | 7.72 / 7.73 GOPs, 24 reps | 7.74 and 7.85 (two single runs) | **≈ 0 to +1.7%** (swapped ≥ baseline) | −7.1% (baseline faster) |
| S1 (1000³) | 8.64 / 8.64, 44 reps | not re-run on core 4 | n/a | −0.5% |
| S2 (2000³) | 8.96 / 8.96, 6 reps | not re-run on core 4 | n/a | −6.3% |
| S3 (3000³) | 9.35 / 9.35, **2 reps — WARN, near-single-cold-sample** | ~9.34–9.43 (est. from core-0 8.57 × "+9-10%") | roughly flat, within the estimate's own width | −4.7% |
| S4 (4000³) | 9.07 / 9.07, **1 rep — WARN** | ~8.94–9.02 (est. from core-0 8.20 × "+9-10%") | roughly flat to slightly baseline-favoring | −5.6% |
| S5 (5000³) | 9.40 / 9.40, **1 rep — WARN** | ~9.38–9.47 (est. from core-0 8.61 × "+9-10%") | roughly flat, within the estimate's own width | −5.6% |

**B4 is the headline change.** On core 0, our kernel was clearly faster than the reference's swapped-in
32x7 (−7.1%, tight noise on both sides, 24-rep and multi-rep samples). On core 4, the gap all but
disappears: baseline (7.72 mean / 7.73 median, 24 reps, std-dev/median 0%) sits at or slightly below the
swapped kernel's own two single-run numbers (7.74, 7.85 — themselves 1.4% apart from each other, so not
a tight pair). Read plainly: **the direction of the B4 result is not settled by what's been run so
far** — core 0 said "ours is faster by a real margin," core 4 says "roughly a wash, maybe the other way."
This directly qualifies §6.5's claim ("on this evidence it's the other way round") — that claim held for
the core-0 evidence only; the core-4 evidence does not confirm it for B4.

S3-S5 don't show the same collapse — the fresh core-4 baseline numbers land inside the range the §6.6
qualitative "+9-10%" bump on the swapped kernel would predict — but this is a weak comparison: the
swapped-core-4 values for S3-S5 were never recorded exactly (only "+9-10%" relative to the core-0
numbers, no raw log kept), and the new baseline numbers themselves are 1-2 reps each (S3-S5's `tmin=10s`
budget barely fits a rep at this problem size), same limitation §6.8 already flagged for the core-0 run.
S1/S2 have no swapped-kernel core-4 counterpart at all — §6.6 only re-ran the swapped kernel on core 4
for B4 and "similarly" for S3-S5, not S1/S2 — so there is nothing to compare their fresh core-4 baseline
numbers (8.64, 8.96) against.

**What this changes:** §6.6's specific pending item (get a baseline number on core 4) is done. It does
not close the broader question — if anything it reopens the B4-specific piece of §6.5's conclusion. A
trustworthy answer for B4 needs the *swapped* kernel re-run on core 4 with the same rep depth as this
baseline (24 reps, not 2 single runs) before the direction can be called either way; that swap patch no
longer exists in the working tree (wiped by this run's `pj_gen` regeneration) and isn't committed
anywhere (§6.2), so re-running it means redoing §6.2's extraction-and-splice, not just re-running a
script. Not done here — flagging it as the next concrete step rather than guessing.

Raw data: `tests/pure_xdsl_blis_rvv/logs/run_2026-09-24_143441/` (gitignored, not committed — a scratch
benchmark run like the rest of §6, per the house rule).

### 6.8 — What §6 cannot show yet

- **One run per condition**, not the project's usual 16-run sweep — the core-0 pair's per-rep noise is
  tight (0-1% std-dev/median where multiple reps fit) but board-to-board and hour-to-hour spread is
  unmeasured here.
- **Baseline-on-core-4 is now in (§6.7), but the swapped kernel isn't re-run at matching rep depth
  there.** B4's core-4 comparison rests on a 24-rep baseline against two single-run swapped numbers
  (7.74, 7.85) — enough to say the core-0 result doesn't hold up, not enough to say what the true core-4
  gap is or which direction it runs. S1/S2 have no core-4 swapped number at all (§6.7).
- **No IPC/perf-counter comparison between the two kernels.** An `INSTRUMENTED=1` run of the swapped
  kernel alone was captured (core 0, warm-up protocol: B4 6.48 GOPs mean, compute IPC 0.54 — the S5 row
  of that run did not complete, likely another instance of the ssh-timeout problem in §6.6) but no
  matching instrumented run of the *original* kernel exists yet, so there's no IPC-level confirmation
  of the "compiler scheduling" hypothesis in §6.4 — only the GOPs-level result.
- **Still no disassembly read** (§4.1's original proposal) — §6 answers a different, more direct
  question (does a same-register-count kernel from the other toolchain actually run faster or slower?)
  but doesn't explain *why* at the instruction level. Worth doing if the scheduling hypothesis in §6.4
  needs settling.

## 7. Conclusion: the tile is not the problem — the open question moves to #34

### 7.1 — The square family's own stability is itself evidence

Separately from the swap experiment, this issue's own §6.7 run put five square shapes (S1-S5), all on
the identical `32x7` tile B4 uses, side by side, on the same core, same session:

| case | M=N=K | KerGOPs (median rep) | reps | jitter |
|---|---|---|---|---|
| S1 | 1000 | 8.64 | 44 | CV 0% |
| S2 | 2000 | 8.96 | 6 | CV 0% |
| S3 | 3000 | 9.35 | 2 (thin — §6.8) | CV 0% |
| S4 | 4000 | 9.07 | 1 (thin — §6.8) | CV 0% |
| S5 | 5000 | 9.40 | 1 (thin — §6.8) | CV 0% |
| **B4** | **4096×384×1024** | **7.72** | 24 | CV 0% |

Across a 5x range of problem size (S1→S5), performance is stable and roughly size-ordered, with tight
per-rep jitter throughout (the thin rep counts at S3-S5 are a `tmin` budget artifact, not instability —
what reps *did* run agree with each other to within 0.01-0.02 GOPs). This is the kernel behaving exactly
as a non-spilling, well-scheduled design should. B4 sits below all five despite a FLOP count
(2×4096×384×1024 ≈ 3.2 GFLOP) that lands *between* S1's (2.0 GFLOP) and S2's (16.1 GFLOP) — so it isn't
explained by "smaller problems run slower" either.

### 7.2 — Why this closes the register-pressure doubt

§2 predicted that `32x7`'s zero-spare-register profile could cost real cycles even without an outright
spill, through lost instruction-scheduling slack. Two independent lines of evidence now say that isn't
happening in a way that matters:

- **The swap experiment (§6.2-§6.7):** the reference's own kernel, generated independently and holding
  the identical 32-of-32 register count, was never meaningfully *slower to use than ours* — core 0 said
  ours was 7.1% faster, core 4 said the two are within noise of each other. If the zero-spare-register
  boundary were costing us specifically, the reference's own equally-constrained kernel should have
  been the fast one. It wasn't, on either core.
- **The square family (§7.1):** the same tile, the same register profile, swept over a 5x size range,
  produces stable, low-jitter, size-ordered numbers — not the kind of behavior a scheduling-starved
  kernel would produce.

**Conclusion: `32x7`'s register pressure is real (§2's count was correct — 32 of 32, zero spare) but it
is not the cause of any observed underperformance.** The kernel performs as intended at this tile.

### 7.3 — What's left, and why it's a new issue

B4 underperforms every one of S1-S5 despite sharing their exact tile and register profile, and despite
a FLOP count between S1 and S2's. With the tile ruled out (§7.2), what's left is B4's *shape* —
M=4096, N=384, K=1024, extremely non-square, vs. S1-S5's M=N=K — which is a different question:
packing/reuse pattern, cache-blocking behavior (B4's `nc=384` exactly equals `N`, unlike any square
case), and loop-iteration counts, not tile/register design. That question doesn't belong in an issue
titled "register-pressure doubt," so it moves to
**[#34](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/34)**
([[0019-b4-bert-shape-underperformance]]), seeded with:
- the B4-vs-S1-S5 table above as its starting evidence (not to be re-derived),
- the swap-experiment's confirmation that the tile/kernel design itself is not the cause,
- a refined measurement plan aimed at the shape question specifically (instrumented pack/compute split
  for B4 vs. a FLOP-matched square; B1/B5 — BERT-shaped but on the already-proven-clean `32x6` tile —
  vs. squares, to check whether the effect is tile-independent; a blocking/fringe-tile breakdown; an
  N-only sweep holding M and K at B4's values).

This issue (#31) is resolved on the question it was opened to answer.

---

## 8. Audit (2026-09-27)

Audited against the logs by a later model version; full brief in the audit report linked at the top.

- **Holds.** The 32-of-32 register count (the reference's own 32x7 kernel has the same). The §6.7 core-4
  baseline table matches `logs/run_2026-09-24_143441` exactly.
- **Weakly argued, but the conclusion stands.** §7's "the tile is not the problem" rests on two
  indirect arguments. The swap compares two kernels with the *same* register pressure, so it cannot
  show what that pressure costs. The square family has no lower-pressure tile to compare against. The
  direct test (§4.3: B4 at 32x6 and 32x8) was never run here. It was run later: #32 (32x8 falls 44%,
  32x6 within ±3% of 32x7) and #34 §5.5 (the gap follows N=384 at 32x6 too). Both support §7.
- **Does not hold.** §6.4's "gap grows with deeper K" argument (see the note there).
- **Records.** The core-0 swap runs (§6.4) and the swapped core-4 runs (§6.6) left no logs. S3-S5's
  "CV 0%" comes from 1-2 reps and says nothing about spread.
- **Follow-ups filed:** 32x6 vs 32x7 inside one launch → #37; benchmark house rules (no core 0, keep
  every quoted log) → #39.
