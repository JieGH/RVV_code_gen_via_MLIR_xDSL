# Issue #35: How does a smaller `nc` affect B3 and the M-small layers? (and what to do with the B3 override)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/35

**Role / Label:** `needs-triage` — from the #31-#34 audit report ([issues-31-34-brief.pdf](https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/blob/pure-xdsl/.scratch/reports/issues-31-34-brief.pdf)), next step 1.
Expanded 2026-09-27 from "keep or drop the override?" into a study of `nc`, after the discussion in §2.
**Performance results per test:** Appendix A of the audit report, rows 6 and 22 (B3 sweep; B3 with B shrunk under L2).
**Status:** ⏸️ Issue only. **Implementation on hold.**
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

## 3. Proposed study (not started)

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
