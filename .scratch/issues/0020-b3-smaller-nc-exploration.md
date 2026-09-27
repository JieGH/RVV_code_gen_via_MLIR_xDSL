# Issue #35: How does a smaller `nc` affect B3 and the M-small layers? (and what to do with the B3 override)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/35

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 1.
Expanded 2026-09-27 from "keep or drop the override?" into a study of `nc`, after the discussion in §2.
**Performance results per test:** Appendix A of the audit report, rows 6 and 22 (B3 sweep; B3 with B shrunk under L2).
**Brief:** [issues-35-36-39-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.pdf) ([LaTeX source](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-35-36-39-brief.tex)) — results of #35, #36 and #39, 2026-09-27.
**Status:** ✅ Study run 2026-09-27 (§6). **Decided 2026-09-27, then changed the same day:** the override was first dropped (683dd24), then **put back as the baseline** (`{"B3": 128}` in `config.json`, user's decision after #41/#42 showed the slow state is the L1D prefetcher backing off). B3 at `nc`=128 stays two-speed with the vector packer; the fixes being specified are #43 (`cbo.zero` on the pack buffers) and #44 (fewer B streams at once).
**Primary Sources:**
- [[0001-b3-m7-packing-instability]] (#1) §0, §7 — where `nc`=128 came from.
- [[0019-b4-bert-shape-underperformance]] (#34) §7.2 — B3 with B shrunk under the L2.
- [[0017-b3-b4-nr-sweep-eval-plan]] (#32) §6 — B3 and B2 at every NR, `nc`=384.
- `tests/pure_xdsl_blis_rvv/modelLevel/model_level.c` (`get_optim_mc_nc_kc`), `modelLevel/archs/bananapi.h`.
- `tests/pure_xdsl_blis_rvv/config.json` (the B3 row), `templates/driver_benchmark.cpp:448` (`RVV_FORCE_NC`).

---

## 1. Question

B3 (64×384×384) runs faster and steadier with a smaller `nc` (the width of the packed B block), but
the cache model never picks a small `nc`. We want to know:

1. How speed and stability change as `nc` shrinks, for B3 and the other layers with a small M.
2. *Why* a smaller `nc` helps, since cache capacity does not explain it (§2).
3. Given 1 and 2, what the shipped setting for B3 should be.

## 2. What is known

**Where `nc` comes from.** `get_optim_mc_nc_kc` sizes `nc` so the packed B block fits in an
**L3** (`bananapi.h`: 4096 sets × 64 B × 16 ways ≈ 4 MiB), then caps it at N. The board has **no L3**
(sysfs, #1 §6). For B3 (`kc`=128, `mc`=64) the model asks for ~5,120 columns, so `nc` = N = 384.
Using the real last cache (512 KiB L2) instead still gives 512 (8-way) or 640 (16-way), so also 384
after the cap (hand calculation, not run). **No version of the model gives 128.**

**Where 128 came from.** Issue #1 tried 384, 192 and 128 by hand and shipped the smallest, because CV
fell as `nc` fell on every layer in the family. It is a hand-tuned number, applied only to B3 through
the `config.json` override (`{"B3": 128}`).

**Cache size does not explain it.** At `nc`=384 the packed B block is 128 × 384 × 4 B = 192 KiB,
which already fits in the 512 KiB L2.

**#1's numbers (`jbpi`, 2026-09-16) — median KerGOPs and CV by `nc`:**

| layer | M×N×K | `mc/mr` | nc=384 | nc=192 | nc=128 |
|---|---|---|---|---|---|
| B3_M7 | 64×384×384 | 4 | 7.15-7.41, CV 16-27% | 8.78-8.89, CV 11-16% | 8.75-10.55, CV 11-15% |
| Reuse8 | 128×384×384 | 8 | 7.99-8.05, CV 17-21% | 8.97-9.14, CV 6-12% | 9.14-10.47, CV 6-11% |
| Reuse12 | 192×384×384 | 12 | 8.57-8.65, CV 14-15% | 8.84-10.68, CV 11-12% | 10.49-11.07, CV 4-10% |
| Reuse24 | 384×384×384 | 24 | 9.57-9.81, CV 7-9% | 8.26-9.36, CV 7-9% | 9.15-9.42, CV 3-6% |

These predate issue #13 (KerGOPs then left out pack time), the reference protocol (#27) and the
uninstrumented build, so **they are not comparable to current numbers**. They show the direction only.

**Recent results (current protocol, `nc` = N unless stated):**
- #34 §7.2 (one scalar launch, core 4): B3 fast and steady (8.1-8.8 GOPs) whenever N ≤ 320, slow when
  N = 384 (4.5-6.1), even when B is small (K=256). With the vector packer every N stays slow.
- #32: at `nc`=384 the scalar packer beats the vector packer on B3 by 21-78%; B2 (M=384, K=64) is
  steady and the vector packer is 0-6% ahead there.

**Why it matters beyond speed.** The reference design uses the same model and so the same `nc`=384.
Head-to-head comparisons (#33) rely on identical blocking on both sides; a hand-set `nc` for B3
breaks that. Discussion 2026-09-27: without a principled way to derive a small `nc`, the leaning is to
**drop the override by default** and treat B3's `nc` sensitivity as a symptom to explain, reporting any
`nc`=128 figure only as a separately labelled "hand-tuned" number. Final decision waits for §3.

## 3. Proposed study (run 2026-09-27, results in §6)

All `INSTRUMENTED=0` unless stated, reference protocol, `--tmin 10`, a non-zero core (#39), all
settings of one sweep inside **one** launch, repeated over at least 3 launches. `nc` set with
`RVV_FORCE_NC` or the `config.json` override field (no code change needed).

1. **`nc` sweep.** `nc` ∈ {32, 64, 96, 128, 192, 256, 320, 384}, with multiples of NR=15 (e.g. 120, 240,
   375) added to separate "small" from "no partial tile". Layers: B3_M7, Reuse8, Reuse12, Reuse24
   (`datasets/testCase_repeat.dat`), B2 as the steady control. Both packers. Report mean and median
   GOPs, CV, and best rep.
2. **Where the time goes.** `INSTRUMENTED=1` pack/compute split and L1/L2 miss counts at `nc` = 128 and
   384 for B3 and Reuse24, both packers. Does a small `nc` shrink pack time, compute time, or the share
   of slow reps?
3. **Per-rep times** at `nc` = 128 and 384 (B3, both packers), to see whether slow reps come in runs
   or at random.
4. **Is it `nc` or N?** Repeat #34 §7.2's N rows (N = 192-384) with `nc` held at 128, to check whether
   N itself still matters once `nc` is fixed.
5. **Model check (no board time).** Run `get_optim_mc_nc_kc` with the L2 in place of the missing L3 for
   all layers, to confirm §2's hand calculation and see which other shapes it would change.

## 4. How to read it

- If speed and stability improve smoothly as `nc` shrinks and the gain comes from fewer slow reps, the
  effect is about how the packer and kernel interact, not cache size: tie it to #36 (vector packer on
  B3), and look for a rule based on something measurable (for example packed bytes per call, or `mc/mr`).
- If there is a clear threshold that a real cache limit explains (L1, L2 ways, TLB reach), write that
  limit into the model instead of a per-layer number.
- Either way, decide the shipped B3 setting from this data, not from #1's pre-#13 numbers.

## 5. Out of scope

- Changing the packer (#36) or the cache model itself. This issue only measures and recommends.
- The 5 tests that expect `nc`=128 are left as they are until the decision.

---

## 6. Results (2026-09-27)

**Setup.** Board `jbpi`, **core 4**, uninstrumented build (`INSTRUMENTED=0`), reference protocol, `--tmin 10`,
tile MR16_NR15. All eleven `nc` values, the Reuse layers, B2 and the N/K rows ran **inside one launch**
(`config_issue35_nc_sweep.json` + `datasets/issue35_nc_sweep.dat`: one label per setting, `nc` set by a
per-label override). Six launches in the order vector, scalar, scalar, vector, vector, scalar (3 per packer).
Every timed rep saved with the new `--save-reps` flag and summarised by `reps_report.py`.

Logs (all committed, `tests/pure_xdsl_blis_rvv/logs/`):

| launch | packer | run dir |
|---|---|---|
| 1 | vector | `run_2026-09-27_104338` |
| 2 | scalar | `run_2026-09-27_105417` |
| 3 | scalar | `run_2026-09-27_110456` |
| 4 | vector | `run_2026-09-27_111536` |
| 5 | vector | `run_2026-09-27_112614` |
| 6 | scalar | `run_2026-09-27_113653` |
| 7 | vector, `--instr` | `run_2026-09-27_114341` (step 2, see #36) |
| 8 | scalar, `--instr` | `run_2026-09-27_115023` (step 2, see #36) |

How to read the columns: *mean-time GOPs* is the reference's formula (all ops / all time), averaged over the
3 launches, with the lowest and highest launch in brackets; *median rep* is the median rep's GOPs; *slow %*
is the share of reps in the slower of two clearly separate speeds (none when the reps have one speed);
*fast start* is how long a case runs, from its first rep, before its first slow rep (10 s = never went slow).
`Bc` = packed B block, `kc` × `nc` × 4 B (`kc` = 128).

#### B3 (64x384x384) by nc

| nc | Bc KiB | vector: mean-time GOPs (lo-hi) | median rep | CV % | slow % | fast start s | scalar: mean-time GOPs (lo-hi) | median rep | CV % | slow % | fast start s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 16 | 6.53 (6.43-6.59) | 6.53 | 0.9 | 0 | 10.00 | 5.31 (5.21-5.48) | 5.31 | 0.6 | 0 | 6.83 |
| 64 | 32 | 6.71 (6.50-7.01) | 7.55 | 16.0 | 22 | 1.94 | 6.65 (6.60-6.74) | 6.65 | 1.1 | 0 | 10.00 |
| 96 | 48 | 5.81 (5.48-6.34) | 5.59 | 22.4 | 66 | 0.37 | 7.27 (7.16-7.42) | 7.28 | 1.6 | 0 | 10.00 |
| 120 | 60 | 5.67 (5.45-5.85) | 5.31 | 23.6 | 77 | 0.22 | 7.26 (7.05-7.41) | 7.62 | 9.4 | 10 | 5.88 |
| 128 | 64 | 5.52 (5.37-5.73) | 5.16 | 24.4 | 76 | 0.09 | 7.36 (7.31-7.43) | 7.58 | 7.1 | 6 | 8.23 |
| 192 | 96 | 5.55 (5.44-5.69) | 4.83 | 27.4 | 76 | 0.20 | 7.04 (6.68-7.65) | 7.48 | 15.0 | 37 | 0.86 |
| 240 | 120 | 5.25 (5.23-5.28) | 4.76 | 27.1 | 82 | 0.26 | 6.81 (6.53-7.04) | 7.46 | 16.9 | 40 | 1.57 |
| 256 | 128 | 5.10 (5.04-5.18) | 4.55 | 26.9 | 79 | 0.18 | 7.07 (6.57-7.54) | 7.63 | 13.3 | 29 | 2.15 |
| 320 | 160 | 4.89 (4.65-5.07) | 4.34 | 28.4 | 81 | 0.14 | 7.14 (6.75-7.76) | 7.82 | 13.8 | 23 | 1.82 |
| 375 | 187 | 4.92 (4.80-4.99) | 4.21 | 33.4 | 77 | 0.11 | 6.81 (6.71-6.90) | 7.63 | 18.1 | 38 | 2.18 |
| 384 | 192 | 4.73 (4.65-4.79) | 4.27 | 30.7 | 86 | 0.14 | 6.65 (6.36-7.06) | 7.21 | 21.0 | 46 | 0.49 |

#### Reuse layers and B2: GOPs by mean time (avg of 3 launches), slow-rep % in brackets

| nc | vector R8 | vector R12 | vector R24 | scalar R8 | scalar R12 | scalar R24 |
|---|---|---|---|---|---|---|
| 32 | 6.80 (0%) | 6.59 (0%) | 5.93 (0%) | 5.78 (0%) | 5.70 (0%) | 5.43 (0%) |
| 64 | 8.15 (2%) | 8.14 (3%) | 7.52 (0%) | 7.37 (0%) | 7.47 (0%) | 7.12 (0%) |
| 96 | 7.58 (30%) | 8.55 (14%) | 8.01 (0%) | 8.12 (0%) | 8.36 (0%) | 7.95 (0%) |
| 120 | 6.94 (57%) | 7.70 (45%) | 8.30 (0%) | 8.43 (0%) | 8.71 (0%) | 8.29 (0%) |
| 128 | 6.58 (62%) | 7.53 (47%) | 7.58 (9%) | 8.39 (0%) | 8.82 (0%) | 8.29 (0%) |
| 192 | 6.43 (74%) | 7.40 (54%) | 8.25 (13%) | 8.97 (0%) | 9.30 (0%) | 8.85 (0%) |
| 240 | 6.89 (60%) | 7.91 (47%) | 8.56 (0%) | 8.54 (9%) | 9.26 (1%) | 8.90 (0%) |
| 256 | 6.58 (58%) | 7.28 (55%) | 8.10 (7%) | 8.63 (9%) | 9.11 (0%) | 8.69 (0%) |
| 320 | 6.06 (76%) | 7.49 (49%) | 8.02 (12%) | 8.50 (12%) | 9.02 (2%) | 8.68 (0%) |
| 375 | 6.32 (63%) | 6.80 (68%) | 8.43 (8%) | 8.64 (6%) | 9.31 (0%) | 8.95 (0%) |
| 384 | 6.44 (66%) | 7.29 (64%) | 8.53 (6%) | 8.76 (14%) | 9.23 (7%) | 9.29 (0%) |

B2 nc=128: vector 11.45 (median 11.49, CV 1.0%), scalar 10.51 (median 10.53, CV 1.4%)

B2 nc=384: vector 11.98 (median 12.37, CV 6.2%), scalar 11.82 (median 11.90, CV 2.1%)

#### Is it nc or N? B3 with N shrunk, at nc = N and at nc = 128

| shape | packer | nc = N: mean-time GOPs (lo-hi) | median | slow % | nc = 128: mean-time GOPs (lo-hi) | median | slow % |
|---|---|---|---|---|---|---|---|
| B3_N192 (64x192x384) | vector | 6.09 (6.07-6.10) | 5.58 | 80 | 6.19 (6.01-6.32) | 5.89 | 66 |
| B3_N192 (64x192x384) | scalar | 7.72 (7.37-7.94) | 8.17 | 17 | 7.47 (7.42-7.53) | 7.49 | 0 |
| B3_N256 (64x256x384) | vector | 5.36 (5.17-5.46) | 4.85 | 83 | 5.93 (5.56-6.60) | 6.16 | 68 |
| B3_N256 (64x256x384) | scalar | 7.48 (6.96-7.83) | 8.11 | 16 | 7.54 (7.49-7.63) | 7.73 | 0 |
| B3_N320 (64x320x384) | vector | 5.19 (4.85-5.53) | 4.51 | 75 | 5.90 (5.55-6.61) | 5.92 | 61 |
| B3_N320 (64x320x384) | scalar | 6.99 (6.63-7.17) | 7.87 | 32 | 7.22 (7.08-7.32) | 7.38 | 3 |
| B3_K256 (64x384x256) | vector | 4.72 (4.66-4.74) | 4.66 | 98 | 5.69 (5.45-6.16) | 5.44 | 89 |
| B3_K256 (64x384x256) | scalar | 4.54 (4.49-4.62) | 4.39 | 93 | 5.64 (5.53-5.78) | 5.25 | 74 |


### 6.1 What the sweep shows

1. **The slow reps are the whole story.** Every B3 row that is steady (one speed) is at or near its best.
   Where B3 loses speed, it is because most reps fall into a second, slow speed (~4.3-5 GOPs against ~8-10).
2. **Scalar packer: a small `nc` works.** B3 is steady from `nc` = 32 to 128 (CV 1-7%, 0-10% slow reps) and
   fastest at 96-128 (**7.27-7.36 GOPs**). From 192 up it is two-speed (23-46% slow), ending at **6.65** at 384.
   So for the scalar packer `nc` = 128 is worth about **+11%** over 384 and makes B3 steady. This is #1's
   finding, and it holds today — but only for the scalar packer.
3. **Vector packer (the shipped one): a small `nc` does not fix B3.** B3 is two-speed at every `nc` from 96 to
   384 (66-86% slow reps). `nc` = 128 gives **5.52** against **4.73** at 384 (+17%), still with CV 24%. Only
   `nc` ≤ 64 is steady; the best mean is `nc` = 64 (6.71, 22% slow) or 32 (6.53, steady).
4. **The vector threshold lines up with the L1.** With the vector packer every layer is steady at `nc` = 32
   (`Bc` 16 KiB); at 64 (`Bc` 32 KiB, the L1's size) R8/R12/R24 are steady (0-3% slow) and B3 mostly is
   (22% slow); from 96 (48 KiB) up B3, R8 and R12 are two-speed. With the
   scalar packer the threshold is higher (B3: between 128 and 192; R8/R12 mostly steady throughout). This is
   a pattern, not a proven cause.
5. **More reuse (bigger M) protects against the slow mode.** R24 (mc/mr = 24) is steady with the scalar
   packer at every `nc` and mostly steady with the vector one (0-13% slow); it is fastest at a large `nc`
   (scalar 9.29 at 384; vector 8.53-8.56 at 240-384). R8 and R12 sit between B3 and R24.
6. **When there is no slow mode, the vector packer is as fast or faster** (e.g. B3 `nc` = 32: 6.53 vs 5.31;
   R24 `nc` = 64: 7.52 vs 7.12; B2: 11.45-11.98 vs 10.51-11.82). Its loss on B3 comes entirely from the slow mode.
7. **B2 (control) is steady** under both packers (CV 1-6%, no slow reps) and faster at `nc` = 384 than 128.
8. **Slow reps come in long stretches, not at random.** Clumping is 0.01-0.1 everywhere (runs of slow reps are
   10-100 times fewer than chance would give). **Each case starts fast** and falls into the slow speed after
   0.1-0.5 s with the vector packer, 0.5-8 s with the scalar one; then it mostly stays there, with occasional
   fast stretches. Example (launch 1, vector, B3 `nc` = 384, first 150 reps): 44 fast, 84 slow, 22 fast.

### 6.2 Step 2: where the time goes (instrumented build)

See #36 §Results for the table. In short: in the slow reps **both** the pack and the compute time go up (vector
B3 `nc` = 384: pack 0.75 → 1.46 ms, compute 1.88 → 2.59 ms), and the core's clock stays at ~1.65 GHz in both
speeds, so the core is waiting on memory, not running slower. At `nc` = 128 the scalar packer's B3 never went
slow in 1000 reps; the vector one was slow in 533 of 1000.

### 6.3 Step 4: is it `nc` or N?

With the **scalar** packer and `nc` held at 128, N barely matters: B3 with N = 192/256/320/384 gives
7.47/7.54/7.22/7.36 GOPs, all nearly steady (0-6% slow). With `nc` = N instead, the same shapes are two-speed
(16-46% slow). So #34 §7.2's "slow when N = 384" was `nc` = N = 384: it is **`nc`, not N**. With the **vector**
packer every row is two-speed whatever N and `nc` (61-86% slow).

B3_K256 (K = 256, so B is 384 KiB and fits the L2) is the slowest row of all, with both packers (89-98% slow
at `nc` = 384, 74-89% at 128). A B that fits in L2 does not protect B3. (Its `ldb` is 256 floats, a power of
two; not tested whether that matters.)

### 6.4 Step 5: the cache model with the real L2 (host, no board)

`get_optim_mc_nc_kc` run on the host for every shipped layer, with the missing L3 replaced by the 512 KiB L2
(program and output: `.scratch/issues/0020-model-check.c`, `-model-check.txt`). sysfs on `jbpi` says the L2 is
**16-way**, shared by cores 4-7 (`bananapi.h` has 8-way):

| layer | M×N×K | tile | kc | mc | nc as shipped (L3) | nc with L2, 8-way | nc with L2, 16-way |
|---|---|---|---|---|---|---|---|
| B2 | 384×384×64 | 16x15 | 64 | 384 | 384 | 256 | 256 |
| B3 | 64×384×384 | 16x15 | 128 | 64 | 384 | 384 (raw 512) | 384 (raw 640) |
| Reuse8 | 128×384×384 | 16x15 | 128 | 128 | 384 | 384 | 384 |
| Reuse12 | 192×384×384 | 16x15 | 128 | 192 | 384 | 256 | 384 |
| Reuse24 | 384×384×384 | 16x15 | 128 | 384 | 384 | 128 | 192 |
| B1, B5 | 1024×384×1024/4096 | 32x6 | 128 | 768 | 384 | 128 | 128 |
| B4 | 4096×384×1024 | 32x7 | 128 | 768 | 384 | 128 | 128 |
| S1-S5 | 1000³-5000³ | 32x7 | 128 | 768 | 1000-1024 | 128 | 128 |

This confirms §2's hand calculation: **no version of the model gives B3 a small `nc`** (the L2 gives 512/640,
capped at 384). It would instead move the big layers and Reuse24 to 128-192, which the sweep shows is the
wrong direction for Reuse24 (fastest at 384). Swapping the L3 for the L2 is not a fix.

## 7. Recommendation

1. **Do not ship the B3 override on the strength of #1.** With the shipped vector packer, `nc` = 128 is worth
   +17% on B3 but leaves it two-speed (CV 24%); the steadiness #1 saw is real only with the scalar packer. No
   form of the cache model produces 128, and head-to-head comparisons need the reference's blocking. This
   supports the 2026-09-27 leaning: **drop the override by default**; quote any `nc` = 128 figure only as a
   separately labelled hand-tuned number. (An uncommitted `config.json` edit in the working tree, made
   outside this study, already removes the override; with it the 5 tests that expect `nc` = 128 fail. They were
   left as §5 asks, to be updated in the same commit as that decision.)
2. **Treat the slow mode as the thing to fix** (#36). It is a memory-side state that sets in shortly after a
   case starts and lasts; a small `nc` only lowers the chance of entering it, and only for the scalar packer.
3. If a rule-based small `nc` is wanted later, the candidate the data suggests is "keep `Bc` within the L1"
   (`nc` ≤ 64 here) for small-M layers with the vector packer. It costs the Reuse layers speed (R24: 7.52 at
   64 vs 8.53 at 384), so it would have to depend on `mc/mr`. Not proposed yet: the cause is not known.

**Not done:** the second board (`jbpi2`), other tiles for the sweep, per-rep L2 miss counters (the perf log
has no L2 events and only whole-case averages), and a test of whether `ldb` being a power of two matters.
