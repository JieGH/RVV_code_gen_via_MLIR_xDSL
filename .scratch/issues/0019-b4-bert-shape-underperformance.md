# Issue #34: Why does B4 (BERT-shaped) underperform the square family at the identical tile?

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/34

**Engineering report (#31-#44, all issues together):** [issues-31-43-engineering-report.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-43-engineering-report.tex)), 2026-09-28.

**Closed 2026-09-27.** **Audit report:** [issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.tex)) — brief and audit of #31-#34, 2026-09-27.
**Performance results per test:** Appendix A of the audit report, rows 16-22 (this issue's tests).
**Role / Label:** resolved — spun out of
[[0016-b4-32x7-register-pressure-eval-plan]] (#31) once that issue's own question (was `32x7`'s
register pressure the cause?) was answered **no**. This issue starts from that answer rather than
re-deriving it: the tile is confirmed fine, so this is about B4's *shape*.
**Status:** 🟢 Answered (2026-09-26). §3.1-§3.5 run; §3.6 (disassembly) not needed, see §5.1. Bottom
line: **B4 is slow because N=384 is small, not because of the tile or the compute loop.** With `nc`
capped at N=384, every packed A panel serves only 384 columns, so B4 packs 2.3x more elements per flop
than a square case. Its compute loop runs within 2% of S1's. Widening N alone (M, K kept) brings B4 back
to square speed (+9.5 to +11.4%). A second, smaller cost (2-5%) comes from M=4096 being a power of two.
Results in §5. Follow-up §7 (2026-09-27): which layers are unstable (B3, B2), and a first B3 test
showing B fitting in L2 does not fix B3; with the scalar packer the slow case is N=`nc`=384.
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

---

## 5. Results (2026-09-26)

**Inputs, all new in this issue:** `tests/pure_xdsl_blis_rvv/config_issue34_shape.json` (two rows,
`32x7` and `32x6`), `datasets/issue34_b4_shape_32x7.dat`, `datasets/issue34_bert_square_32x6.dat`, and
`shape_breakdown.py` (+ `test_shape_breakdown.py`) for the no-board numbers in §5.3.

**Runs, all on `jbpi2` (board idle before each; load average at its usual ~2.0 baseline), same evening:**

| run | build / protocol | core | log dir |
|---|---|---|---|
| A | `INSTRUMENTED=0`, reference protocol, `--no-nc-overrides --tmin 10` | 4 | `logs/run_2026-09-26_230814` |
| B | `INSTRUMENTED=1` (`--instr`), warm-up protocol, `--no-nc-overrides` | 4 | `logs/run_2026-09-26_231548` |
| C | same as A (repeat) | 2 | `logs/run_2026-09-26_232909` |

Command for A (C is the same with `--cpu 2`; B drops `--tmin` and adds `--instr`):

```bash
source .venv/bin/activate && REMOTE_PASS="" REMOTE_HOST="fej" REMOTE_PORT="3322" \
  ./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh --jump \
  --riscv-user jlei --riscv-ip jbpi2 --riscv-port 22 \
  --no-nc-overrides --cpu 4 --tmin 10 --config config_issue34_shape.json
```

Headline GOPs below are runs A and C (uninstrumented), shown as "by mean time / by median rep". Every
row is tight inside its run: p5-p95 within 1%. Percent changes are always new/old − 1, by mean time.

### 5.1 — §3.1: pack/compute split (run B, instrumented — compare only within this table)

| case | M×N×K | pack share of time | compute-only GFLOP/s | pack ns per packed element |
|---|---|---|---|---|
| B4 | 4096×384×1024 | **14.7%** | 8.46 | 14.8 |
| B4_N1024 | 4096×1024×1024 | 7.0% | 8.51 | 14.8 |
| B4_N4096 | 4096×4096×1024 | 7.0% | 8.49 | — |
| S1 | 1000³ | 8.1% | 8.60 | 10.5 |

(All `32x7`. Compute-only = 2MNK / `Compute_Time_ms`; the per-tile timers slow the whole build by
11-17%, so these are for comparing with each other only.)

- **The compute loop is not where B4 loses.** B4's compute-only rate is within 0.6% of B4_N1024's and
  1.6% of S1's. Compute IPC (0.57-0.60) and compute L1 read-miss rate (1.34-1.43%) are the same for
  all of them. Compute L1 write-miss is 2.14% for B4 against 1.84% for S1 (0.11% for B4_N1024); §5.4
  covers where B4's comes from and why it costs little. So §3.6's disassembly read is not needed.
- **Packing is.** B4 spends twice the share of its time packing (14.7% vs 7.0%). The cost *per packed
  element* is the same for B4 and B4_N1024 (14.76 vs 14.82 ns). B4 simply packs more elements for each
  flop it does (§5.3).

### 5.2 — §3.4: N-only sweep, M=4096 and K=1024 kept (runs A and C, `32x7`)

| case | N | core 4 (A), mean / median | core 2 (C), mean / median | vs B4, A / C |
|---|---|---|---|---|
| B4 | 384 | 8.10 / 8.10 | 7.88 / 7.88 | — |
| B4_N1024 | 1024 | **9.02 / 9.02** | **8.63 / 8.63** | +11.3% / +9.5% |
| B4_N2048 | 2048 | 9.02 / 9.02 | 8.70 / 8.70 | +11.4% / +10.3% |
| B4_N4096 | 4096 | 8.97 / 8.97 | 8.75 / 8.75 | +10.8% / +11.0% |
| S1 (anchor) | 1000³ | 8.73 / 8.73 | 8.57 / 8.57 | +7.8% / +8.7% |

N alone moves B4 from below every square case to above S1. The jump is complete by N=1024, which is
where the cache model's `nc` stops growing (1024). Past that point, more N adds jc blocks but no more
reuse per packed A panel, and the curve is flat.

### 5.3 — §3.3 and §3.5: what the shape changes, no board time (`shape_breakdown.py`)

```
python3 tests/pure_xdsl_blis_rvv/shape_breakdown.py datasets/issue34_b4_shape_32x7.dat --mr 32 --nr 7
```

| case | mc/nc/kc | A packed | Ac reuse (tiles per packed A panel) | Bc reuse (mc/mr) | packed elements per 1000 flops | C-tile lines per L1 set (4 ways) |
|---|---|---|---|---|---|---|
| B4 | 768/384/128 | 1× | **55** | 24 | **1.424** | **7 — over** |
| B4_M4064 | 768/384/128 | 1× | 55 | 24 | 1.425 | 1 |
| B4_N1024 | 768/1024/128 | 1× | 147 | 24 | 0.610 | 7 — over |
| S1 | 768/1000/128 | 1× | 143 | 24 | 1.000 | 1 |
| S2-S5 | 768/1024/128 | 2-5× | 147 | 24 | 0.60-0.75 | 1 |

- **§3.5: "reuse≥24" is `mc/mr`** (how many tiles use one packed B panel), from issue #1/#3 — not a
  dataset field; the dataset header this plan cited doesn't carry it, and neither
  `generate_pure_blis_project.py` nor `modelLevel/model_level.c` contains the word (grep).
  It comes from `.scratch/issues/0001-*.md` ("reuse factor = mc/mr") and `0003-*.md` §3.1. It is 24 for *every* case here,
  B4 and S1-S5 alike, so it cannot explain the gap. The reuse number that does differ is the other one:
  how many tiles use one packed A panel, `nc/nr` — 55 for B4 against 143-147 for the squares.
- **§3.3: single `nc` block.** B4 and S1 both have one jc block (nc = N), as §2 suspected. It isn't
  the single block that hurts: it's that B4's single block is only 384 wide, so the whole 16 MiB of A
  is packed for 384 columns of work. Packed elements per flop = 1/(2N) + 1/(2M) when A is packed once:
  1.30 + 0.12 per 1000 flops for B4, 0.50 + 0.50 for S1.
- **Fringe tiles are not it**: extra padded work is 0.26% for B4, against 2.50% (S1), 1.25% (S2),
  0.60% (S3), 0.45% (S4), 1.00% (S5) (`ref_baseline/fringe.py`, `fringe_for_case`, `32x7`).

### 5.4 — Second cause, smaller: M=4096 is a power of two

The driver sets `lda = ldc = M`. At M=4096 the columns of one C tile sit 16 KiB apart, a multiple of the
L1's 8 KiB set period (128 sets × 64 B), so all 7 columns of a `32x7` tile fall into the same two L1
sets, which are only 4-way. `shape_breakdown.py` counts 7 lines per set (6 at `32x6`); no square case
has a power-of-two M. Moving M by one tile turns it off without changing anything else:

| change | tile | core 4 (A) | core 2 (C) |
|---|---|---|---|
| B4 → B4_M4064 | 32x7 | +4.6% | +3.4% |
| B4 → B4_M4128 | 32x7 | +2.1% | +2.9% |
| B4 → B4_M4064 | 32x6 | +3.4% | +2.8% |
| S1_M2000 → S1_M2048 (turns it *on*) | 32x7 | −1.8% | −1.2% |

The counters agree for B4: compute L1 *write*-miss rate is 2.14% at M=4096, 0.55% at 4064, 0.73% at
4128 (run B). They do **not** agree for S1_M2048 (0.88%, lower than plain S1's 1.84%), and in the
instrumented build the extra misses cost no measurable compute time (8.46 for all three B4 rows). So:
real, small (2-5%), and only shown clearly for B4. This was not in the §3 plan; it came out of §3.3,
and is listed in §6 as a second, minor cause because it held in both runs on both tiles.

### 5.5 — §3.2: BERT vs Square on one tile, `32x6` (no Square number existed at 32x6 before; checked all 263 square rows in `logs/`, all `32x7`)

| case | M×N×K | core 4 (A), mean / median | core 2 (C), mean / median |
|---|---|---|---|
| B1 | 1024×384×1024 | 7.97 / 7.97 | 8.21 / 8.21 |
| B5 | 1024×384×4096 | 7.79 / 7.79 | 7.87 / 7.87 |
| B4 | 4096×384×1024 | 7.62 / 7.62 | 7.98 / 7.97 |
| S1 | 1000³ | 8.25 / 8.25 | 8.35 / 8.35 |
| S2 | 2000³ | 8.56 / 8.56 | 8.80 / 8.81 |

All three N=384 shapes sit below both squares on both cores. B1 and B5 have M=1024 (C-tile lines per
set = 3, under the 4 ways, so no power-of-two cost) and still trail. **So the gap follows the BERT
shape (N=384), not the `32x7` tile** — the outcome §3.2 called decisive. The run B split says the same:
pack share is 14.8% (B1), 15.1% (B5), 13.4% (B4) at 32x6, against 7.4% (S1) and 6.8% (S2).

### 5.6 — Side finding: absolute numbers move a few % from one run to the next

Every row in A and C is tight inside its run (p5-p95 within 1%), but the same row differs between the
two runs by −4.3% to +4.7%, in opposite directions for the two tiles (32x6 faster in C, 32x7 slower).
That is enough to flip which tile wins for B4: 32x7 by 6% in A, 32x6 by 1% in C — [[0017-b3-b4-nr-sweep-eval-plan]]
(#32, nine core-0 launches) saw 32x6 ahead by ~2% and the same binary range 7.63-7.94 across
launches. The board has two clusters (cores 0-3 and 4-7, one 512 KiB L2 each, from sysfs), and A and
C ran on different ones, but with one launch per cluster a cluster effect can't be told apart from
plain launch-to-launch spread. What it does mean: **compare shapes inside one run, never across runs**
— every conclusion in §5.1-§5.5 is a within-run difference that held in both A and C.

(Also from sysfs: the L2 is 16-way; `cache_config.py`/`bananapi.h` use `WL2 = 8`. Not chased here.)

## 6. Conclusion

B4's gap to the square family is a property of its shape:

1. **Main cause — small N (≈10%).** `nc` is capped at N=384, so each packed A panel is used by 55 tiles
   instead of ~145, and B4 packs 2.3× more elements per flop than S2-S5. Packing takes 14.7% of B4's
   time vs 7-8% for squares; the compute loop itself runs at square speed. Widening N to 1024 alone
   closes the gap.
2. **Smaller cause — power-of-two M (2-5%).** C-tile columns 16 KiB apart share L1 sets.
3. **Ruled out:** the tile (#31, and §5.5 here), the compute loop (§5.1), fringe tiles (§5.3), and
   the `mc/mr` reuse (same for all cases).

Fixing it is outside this issue (no redesign in scope, same boundary as #31-#33). A is already packed
only once for B4, so there is no repeat packing to remove; the levers left are a cheaper A-pack (it
runs at ~0.3 GB/s, IPC 0.03-0.04 in run B) and padding `lda`/`ldc` off powers of two. The reference
design blocks and packs the same way, so it should pay the same small-N cost.

### Closure checklist

- [x] §3.1 instrumented pack/compute split (run B)
- [x] §3.2 BERT vs Square on `32x6` (runs A, C)
- [x] §3.3 blocking breakdown (`shape_breakdown.py`, tested)
- [x] §3.4 N-only sweep (runs A, C)
- [x] §3.5 what "reuse≥24" is
- [x] §3.6 disassembly — not needed, compute loop runs at square speed (§5.1)
- [x] repeat on a second core, same session (run C)
- [ ] not done: a second board (`jbpi`), or several launches per core to size the §5.6 spread

---

## 7. Follow-up: why some layers are not stable (2026-09-26/27)

Asked after §6: B4 is slow but steady, so which layers are actually *unstable*, and why? This section
reads the #32 sweep logs, the #33 2048³ logs and this issue's own runs together, then adds one new
experiment on B3 (§7.2). All numbers `INSTRUMENTED=0`, reference protocol, `--no-nc-overrides --tmin 10`.

### 7.1 What the existing logs already say

1. **Slow is not the same as unstable.** B4, S1-S5 and the 2048³ cases from #33 are tight inside a run
   (p5-p95 within 1%). B4 is steadily slower, which is the small-N effect in §6, not noise.
2. **The unstable layers are the small, pack-heavy ones.** B3 (and sometimes B2) do about 19 MFLOP per
   call, a few milliseconds each. B3 spends 37-40% of its time packing ([[0003-scalar-vs-vector-pack-timing]]).
3. **B3 switches between two speeds inside one run.** In every #32 vector-pack run, B3 at `16x15` has a
   median rep near 4.4 GOPs but a best rep near 9.9. That best rep beats the scalar packer's best (~9.3),
   so the vector packer is not slow in itself: most reps take a slow path, and with the scalar packer
   fewer reps do. Same two speeds [[0013-issue-26-pack-swap-end-to-end]] §3 found.
4. **It is not the tile or the registers.** In #32 no tile at or under 32 live registers lost speed; in
   #33 our kernels spill exactly as often as the reference's.
5. **The whole level also moves 2-5% from one launch to the next** (§5.6). The rule stays: compare
   shapes inside one run, never across runs.

### 7.2 New: does B fitting in L2 make B3 stable? (`config_issue34_b3_l2_fit.json`)

Guess tested: B3's B is 384×384 floats = 576 KiB, over the 512 KiB L2 (B2's is 96 KiB and B2 is
stable). Dataset `datasets/issue34_b3_l2_fit.dat` keeps M=64 and the `16x15` tile, and shrinks B under
the L2. `jbpi2`, core 4, three launches back to back: vector (`logs/run_2026-09-26_231823`), scalar
(`..._232047`), vector (`..._232311`). GOPs by median rep; mean in brackets; CV across reps.

| case | M×N×K | B size | vector, launch 1 | vector, launch 2 | scalar |
|---|---|---|---|---|---|
| B3 | 64×384×384 | 576 KiB | 4.36 (4.65), CV 27% | 4.38, CV 43% | 6.06, CV 29% |
| B3_N320 | 64×320×384 | 480 KiB | 4.69 (5.32), CV 35% | 4.68, CV 34% | **8.75, CV 7%** |
| B3_N256 | 64×256×384 | 384 KiB | 4.56 (4.82), CV 23% | 4.74, CV 24% | **8.48, CV 14%** |
| B3_K256 | 64×384×256 | 384 KiB | 4.59 (4.63), CV 14% | 4.54, CV 15% | 4.49, CV 19% |
| B3_N192 | 64×192×384 | 288 KiB | 5.82 (6.58), CV 29% | 5.58, CV 21% | 8.14, CV 14% |

- **The L2 guess is wrong.** With the vector packer every row keeps both speeds (best rep 9.7-10.2,
  median 4.4-5.8), even at half the L2 size. The two vector launches agree to within 0.24.
- **With the scalar packer, what matters is N=384, not B's size.** B3_N256 and B3_K256 have the same
  384 KiB B. N256 runs fast and steady (8.48); K256, which keeps N=384, stays on the slow path (4.49),
  and so does B3 itself (6.06). Every row with N≤320 is fast. With `--no-nc-overrides`, `nc` = N here,
  so "N=384" is also "`nc`=384". That fits issue #1's shipped fix for B3, the `{"B3": 128}` `nc`
  override, which the uncommitted `config.json` change from #32 removed.
- **Limits.** One scalar launch only, one board, one core. Nothing here says *why* `nc`=384 is the
  slow case, or why the vector packer stays slow at every N.

### 7.3 Next steps (not run — each now filed as its own issue, implementation on hold)

Filed 2026-09-27: B3 `nc` override decision → #35; B3 vector packer (items 1-2 below) → #36;
32x6 vs 32x7 → #37; B4 shape-cost levers (§6) → #38; benchmark house rules → #39.


- Instrumented pack/compute split for B3 at `16x14`, `16x15`, `16x16` (the step the #32 comment
  proposed), and the same for B3_N320 vs B3_K256 with the scalar packer.
- Save per-rep times, to see whether slow reps come in runs or at random.
- Repeat §7.2's scalar launch, and add a `nc`=128 row, before deciding whether to restore the B3
  override in `config.json` (5 tests fail without it; the call is the user's).
- Side note from sysfs, not chased: the L2 is 16-way, `cache_config.py`/`bananapi.h` use `WL2 = 8`.
