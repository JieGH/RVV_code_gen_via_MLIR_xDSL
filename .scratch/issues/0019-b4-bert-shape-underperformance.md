# Issue #34: Why does B4 (BERT-shaped) underperform the square family at the identical tile?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/34

**Role / Label:** `needs-triage` — spun out of
[[0016-b4-32x7-register-pressure-eval-plan]] (#31) once that issue's own question (was `32x7`'s
register pressure the cause?) was answered **no**. This issue starts from that answer rather than
re-deriving it: the tile is confirmed fine, so this is about B4's *shape*.
**Status:** 🔴 Not started. Evidence recapped below (§1-2), a measurement plan proposed (§3), nothing
run yet.
**Target:** `jbpi2` (BananaPi M2 Pro, VLEN=256), reached through jump host `fej`, core 4 (not core 0 —
see [[0016-b4-32x7-register-pressure-eval-plan]] §6.6: core 0 absorbs every device interrupt and runs
5-10% slower for identical binaries). The board is shared with other users; check `ps -eo
pid,psr,pcpu,user,cmd --sort=-pcpu` before a timed run, same as [[0016-...]] §6.7.
**Primary Sources:**
- [[0016-b4-32x7-register-pressure-eval-plan]] (#31) — the parent investigation. §7 there is the
  closing argument this issue starts from: two independent lines of evidence (a kernel-swap experiment
  and the square family's own stability, both on the identical `32x7` tile B4 uses) say `32x7`'s
  register pressure is not the cause of B4's gap. Read §7 before re-litigating the tile question here.
- `tests/pure_xdsl_blis_rvv/config.json:60`, `datasets/issue2_square_bert_32x7.dat` — B4
  (M=4096, N=384, K=1024) and S1-S5 (M=N=K, 1000-5000), all pinned to `MR32_NR7`.
- `datasets/issue26_b4.dat` — B4's own dataset row; its header calls B4 "reuse≥24," a project-computed
  metric not yet pulled into this comparison (§3.5 below).
- [[0003-scalar-vs-vector-pack-timing]] §3 — the existing (9-day-old as of #31, now further out of
  date) with/without-packing split for B4: 16.1% scalar-pack / 14.8% vector-pack of total time. Was
  never refreshed under current protocol/build (issue #31 §4.2 proposed this and it was never run —
  §6 detoured into the tile-swap experiment instead).
- `docs/metrics.md` — `KerGOPs` includes packing since issue #13; state which build
  (`INSTRUMENTED=0`/`=1`) every number here comes from, per house rule.

---

## 1. What's being asked

The user's read of [[0016-b4-32x7-register-pressure-eval-plan]]'s outcome: the kernel performs as
intended and is stable across problem size for the square family (the tile/register-pressure doubt
that issue was opened to check is resolved, not confirmed). The question that's left is why B4
specifically — the one BERT-shaped case pinned to this tile — runs slower than every square case,
since that's no longer explained by the tile itself.

## 2. What's already known (recap from #31 — not to be re-derived)

From [[0016-b4-32x7-register-pressure-eval-plan]] §6.4 and §6.7, same board/session/protocol
(`INSTRUMENTED=0`, reference protocol, `--no-nc-overrides --tmin 10`), all six cases on the identical
`32x7` tile:

| case | shape (M×N×K) | core 0 KerGOPs | core 4 KerGOPs (median rep, reps) |
|---|---|---|---|
| B4 | 4096×384×1024 | 7.77 | 7.73 (24) |
| S1 | 1000³ | 8.34 | 8.64 (44) |
| S2 | 2000³ | 8.76 | 8.96 (6) |
| S3 | 3000³ | 8.99 | 9.35 (2 — thin, `tmin`-limited) |
| S4 | 4000³ | 8.69 | 9.07 (1 — thin) |
| S5 | 5000³ | 9.12 | 9.40 (1 — thin) |

B4 is the slowest case on both cores, by 7-19%, despite:
- **Identical tile and register profile** — 32 of 32 vector registers live, zero spare, same as every
  S1-S5 case. [[0016-...]] §7.2 confirms this isn't the cause: the reference's own equally-constrained
  `32x7` kernel, swapped into our harness, was never meaningfully slower to use than ours on either
  core — if the register boundary were the problem, that swap should have made things *faster*, not
  roughly a wash.
- **A FLOP count between two square cases that both beat it** — B4 is 2×4096×384×1024 ≈ 3.2 GFLOP,
  between S1's 2.0 GFLOP and S2's 16.1 GFLOP, yet slower than both. Not explained by "small problems
  run slower."
- **Stable, low-jitter square-family performance across a 5x size range** (CV 0% throughout, S1-S5) —
  the kernel design itself behaves as intended; B4 is the outlier, not the family.

What's structurally different about B4: it is not square. M=4096 is over 10x N=384, and the cache
blocker's own output reflects it — B4's `nc=384` lands exactly on `N` (a single `nc`/jc-loop block,
no reuse of a packed B panel across multiple jc iterations), which none of S1-S5 hit at this tile (their
`N` ranges 1000-5000 against an `nc` heuristic of ~1000-1024, so most have 2+ `nc` blocks; S1's
`N`=`nc`=1000 is the closest analog and it still beats B4 by 7-12%, so "single nc block" alone isn't
the whole story either — worth checking directly, not assumed, per §3.3).

## 3. Measurement plan

Ordered cheapest/most isolating first. Items 1 and 2 need no new `config.json`/dataset rows; 3 is
static/no board time; 4 and 5 need a few new dataset rows (not a new kernel).

**3.1 — Instrumented pack/compute split, B4 vs. a FLOP-matched square (S1)**

`--instr` build reports `Vector_Pack_Time_ms`/`Scalar_Pack_Time_ms`/`Compute_Time_ms` per case — never
actually run for this comparison (proposed as #31 §4.2, not executed there). Run both on core 4,
`INSTRUMENTED=1` (warm-up protocol — the reference protocol doesn't run instrumented, per
[[0016-...]] §6.6's mechanical note: drop `--tmin` for this build). Read two ways:
- If B4's pack-time *fraction* is much higher than S1's, packing itself (not compute) is the driver —
  consistent with the "no B-panel reuse" structural difference noted in §2.
- If `Compute_Time_ms` scales worse than raw FLOP count predicts relative to S1, the compute nest
  itself is the driver — would need a disassembly-level look (§3.6) to say why, since register
  pressure itself is ruled out (§2).
- The same run's `perf_*.txt` gives IPC and L1 load/store counts on the compute nest specifically —
  compare against S1's.

**3.2 — B1/B5 (BERT-shaped, but the proven-clean `32x6` tile) vs. squares**

B1/B5 share B4's non-square BERT proportions but use `MR32_NR6` — the tile already confirmed
zero-spill and byte-identical to the reference
([[0006-match-reference-design-everything-but-the-two-intended-differences]]). No B1/B5-vs-square
comparison exists today (S1-S5 are pinned to `32x7`, not `32x6`, so there's no existing same-tile
square number to compare against — this needs a same-tile square row added, e.g. a square shape
benchmarked at `32x6`, or reuse existing `32x6` square numbers if any exist elsewhere in the project's
history — check before assuming none do).

This is the most decisive check in this plan: **if B1/B5 also underperform relative to squares at
their own tile, that confirms the effect is BERT-shape-driven, independent of which tile is used at
all** — ruling out anything `32x7`-specific once and for all, and pointing squarely at packing/blocking/
reuse as the mechanism. If B1/B5 do *not* underperform relative to squares, that narrows the effect
back toward something `32x7`-specific after all, reopening (not closing) the question §7.2 in #31
answered — worth being honest about that possibility rather than assuming the answer.

**3.3 — Blocking/loop-iteration breakdown (no board time)**

For B4 and each of S1-S5: pull `mc`/`nc`/`kc` from the cache-blocking heuristic
(`get_optim_mc_nc_kc`, already printed per-run — see the "cache blocking" line in every benchmark run's
output), then compute the actual block counts (`⌈M/mc⌉ × ⌈N/nc⌉ × ⌈K/kc⌉`) and what fraction of each
axis's last block is a fringe (partial-tile) remainder vs. a full tile. B4's `nc=384=N` (single `nc`
block, no B-panel reuse across a jc loop) is the one difference already spotted (§2) — check whether
S1 (whose `nc` also lands close to its own `N`) shares that property, and if it does, whether it's
still measurably better off than B4, which would say "single nc block" isn't sufficient on its own and
something else in B4's shape matters too.

**3.4 — N-only sweep, M=4096 and K=1024 held at B4's own values**

Cleanest isolation of the aspect-ratio question: add a few `config.json`/dataset rows stepping N from
384 up toward something square-scaled (e.g. 384, 1024, 2048, 4096), M and K fixed. If speed recovers as
N grows, that pins the effect on small-N (or on the lack of `nc`-block reuse it causes, per §3.3) rather
than on B4's specific numbers. Small addition to the dataset, not a new kernel or tile.

**3.5 — What "reuse≥24" (from `issue26_b4.dat`'s header) actually measures**

Pull the reuse figure for B4 and, if the same metric is computed for S1-S5, for them too — check
`generate_pure_blis_project.py`/`model_level.c` for where this number comes from. If it's a cheap,
already-available metric, free to check before spending more board time; if it correlates with the
slowdown, it may already point at the mechanism (e.g., panel reuse count tied directly to the
`nc`-block story in §3.3).

**3.6 — Disassembly read, only if 3.1-3.5 point at the compute nest specifically**

If §3.1's `Compute_Time_ms`/IPC comparison shows B4's compute nest itself running colder than S1's
(not just paying a bigger relative pack cost), a disassembly read of `gemm_b1_xdsl.cpp` for B4 vs. S1's
own generated kernel at their respective tiles (`32x7` vs. whatever S1 would use if it weren't also
pinned to `32x7` — may need a same-tile compute-only comparison instead) would say whether it's an
instruction-mix difference or purely a data-size/cache-residency effect. Held back until 3.1 motivates
it — no point reading assembly before knowing which region of the call is actually slow.

## 4. What this plan cannot show

- Everything here compares B4 against the square family; it does not yet check B4 against B1/B2/B3/B5
  (the rest of the BERT family), which have their own tiles and shapes and might independently confirm
  or complicate a "BERT shape is inherently harder" story — worth a look once this narrower B4-vs-square
  question has an answer.
- §3.2's B1/B5 check needs a `32x6` square baseline that may not exist yet — confirm before assuming a
  fresh board run is required for it.
- As with every board measurement in this project, board time and time-of-day are not controlled
  between separate runs; keep same-session, same-core pairs wherever the comparison matters
  ([[0016-b4-32x7-register-pressure-eval-plan]] §6.6/§6.7's discipline carries over here).
