# Issue #26: scalar pack live instead of RVV pack, measured end to end (B3, S1)

**Role / Label:** `done` — measured on `jbpi`, tooling and tests written. Two shapes only; question 3 (a block-size threshold) is not answered.
**Status:** ✅ B3: the scalar-pack GEMM is **44.7% faster end to end** (4.48 → 6.48 GOPs by mean time), much more than the isolated pack times can account for. S1: no difference (8.51 vs 8.51, ±0.3%). The B3 reps are bimodal (see §3), so the mean is the only stable figure, and *why* it moves is not known; see §5.
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256), reached via `fej`, one session 2026-09-20 12:06–12:21, core 0, governor `performance`.
**Primary Sources:**
- [[0005-swap-pack-impl-bottleneck-isolation]] — the local plan this issue came from.
- [[0003-scalar-vs-vector-pack-timing]] — the isolated pack times (vector 12.7% slower on B3).
- [[0004-reference-comparison-metric-mismatch]], [[0009-issue-11-midpoint-the-result-flipped]] — B3's 40.6% pack share and the B3 problem.
- [[0001-b3-m7-packing-instability]] — B3's bimodal reps, which turn up again here.
- `tests/pure_xdsl_blis_rvv/logs/run_2026-09-20_{120657,121022,121347,121713}/` and `logs/pack_swap_2026-09-20_120657.txt`.

---

## 1. What was rewritten in the issue first

The issue text was filed before issues #14 and #27 and asked for a new "scalar compute time" column from the scalar-pack pass. That pass no longer exists in the default run (the reference protocol has none), and `--scalar-pack` already forces the scalar packer on every timed call, so `RVV_Time_ms` is already the end-to-end time for whichever packer is live. No new timing column was needed. What was missing was that a `--scalar-pack` run wrote into the same log files as a vector run. The issue on GitHub was rewritten to say so.

The reference design is now in the repo at `.ext/RVV_ukernels_benchmark` (git-ignored, identical to the old `../ext` copy), and its packers (`pack_RB`, `pack_CB`) are plain scalar C loops, so scalar-pack-live is also the reference's packing strategy on our microkernel. `ref_baseline/run_ref_baseline.sh` still reads `../ext/...`; that was left alone because this issue does not run the reference.

## 2. What was built

- `--scalar-pack` runs write `*_noinstr_refproto_scalarpack[_no_nc_override].txt` (driver, `templates/Makefile`, `compile_macro_bench.sh`). Tested in `test_ref_protocol.py`: own log names, the environment variable does the same, every timed rep really uses the scalar packer, and the name combines with `_no_nc_override`.
- `run_pack_swap.sh`: vector, scalar, scalar, vector on one pinned core, B3 then S1 (`config_pack_swap.json`, datasets `issue26_b3.dat`, `issue26_s1.dat`), default run command flags (`--no-nc-overrides --cpu 0 --tmin 10`).
- `compare_pack_swap.py`: e2e GOPs by mean time per side, the difference, and the spread between repeats of one side. Not covered by tests (no test seam was agreed for it; it was checked by hand on made-up logs and on the real ones below).
- Docs: `docs/metrics.md`, `docs/agents/execution.md`, `README.md`.

## 3. Result

Uninstrumented build, reference protocol (no warm-up, reps until `tmin` = 10 s), `--no-nc-overrides` (B3 at `nc=384`), core 0. e2e GOPs = 2·M·N·K / mean time per GEMM call, packing included.

| Case | vector, run 1 / run 4 | scalar, run 2 / run 3 | scalar vs vector | spread vec / scl |
|---|---|---|---|---|
| B3 `64x384x384`, MR16_NR15 | 4.47 / 4.49 (mean 4.48) | 6.50 / 6.47 (mean 6.48) | **+44.7%** | 0.5% / 0.4% |
| S1 `1000x1000x1000`, MR32_NR7 | 8.50 / 8.52 (mean 8.51) | 8.50 / 8.52 (mean 8.51) | +0.0% | 0.3% / 0.2% |

The order of the runs does not explain it: the two vector runs sit at both ends of the session, agree to 0.5%, and both are far from the two scalar runs between them. B3 passed the correctness check in all four runs (relative error over the whole C matrix within tolerance; the console also prints one sample entry, `C[40,96]: 104.56 == 104.56`); S1's oracle was skipped as configured, so the scalar packer's S1 output was not checked in these runs (`test_pack_correctness.py` covers the packer itself).

**B3's reps are bimodal.** The per-rep figures (GOPs) for B3:

| run | packer | median | 5th percentile | 95th percentile | fastest | std-dev |
|---|---|---|---|---|---|---|
| 1 | vector | 4.30 | 4.14 | 6.33 | 9.88 | 0.92 |
| 4 | vector | 4.36 | 4.19 | 6.23 | 9.83 | 0.78 |
| 2 | scalar | 8.51 | 4.23 | 9.04 | 9.18 | 1.95 |
| 3 | scalar | 5.94 | 4.27 | 9.07 | 9.25 | 1.65 |

B3 has two speeds, about 4.2 and about 9 GOPs (the bimodal reps of [[0001-b3-m7-packing-instability]]). This is read from percentiles only; no histogram was made. In the scalar runs the 5th percentile is 4.2-4.3 and the 95th 9.0-9.1, so both speeds are common. In the vector runs the 95th percentile is only 6.2-6.3 and the fastest rep about 9.9, so the fast speed is rare there (under 5% of reps). The mean of 3400 reps is stable to 0.4% (that is what the spread column shows), but the median flips from 8.51 to 5.94 between two identical scalar runs, because it sits on the boundary between the two speeds. So: quote B3 by mean time, and do not quote its median in either direction. The most likely reading of the +44.7% is that the scalar pack puts many more reps in the fast speed, not that every rep is faster; the data here does not prove it.

## 4. Answers to the three questions

1. **Faster or slower end to end?** B3: scalar pack faster by 44.7%. S1: the same speed.
2. **Same sign as the isolated pack time?** Same sign, far larger size. Taking #7's isolated result (vector pack 12.7% slower than scalar) and #10's pack share (40.6% of the vector run), the pack time alone predicts about +5% (vector total = 0.406 + 0.594 = 1.0; scalar total = 0.406/1.127 + 0.594 = 0.954; 1/0.954 = 1.048). This is only a rough figure: both inputs come from other, instrumented runs under the warm-up protocol, and this session's vector-live B3 (4.22 ms per call) is not the same operating point as theirs. A check that does not lean on them: the scalar-live call is 1.32 ms shorter (4.22 → 2.90 ms). If the vector pack really is about 40% of a call (about 1.7 ms), the scalar pack would have to cut its time by about three quarters, and the isolated pack times say 12.7%. So a pack-time-only explanation does not fit; the pack must be changing something the microkernel sees afterwards. Cache warmth is the second-order effect to chase, as the issue said; §5.
3. **A block-size threshold below which to skip the vector pack?** Not answered. Two shapes give one large effect (B3, whose blocks are `mc=64, nc=384, kc=128`) and none (S1, `mc=768, nc=1000, kc=128`). Where it turns on is between them, and a sweep is needed to find it.

## 5. What this does not establish

- **Why.** Nothing here shows what the scalar pack leaves in cache, or in what order, that makes the fast mode more common. The uninstrumented build cannot say. The step named in the issue is an instrumented pair, `--instr` and `--instr --scalar-pack`, comparing `Compute_Time_ms` and the L1D miss rate of the compute region (in the scalar-live log `Vector_Pack_Time_ms` holds the scalar packer's time; `docs/metrics.md` says so). Not done: the instrumented build runs the warm-up protocol with the 5 reps from `config_pack_swap.json`, too few for a shape this bimodal, so the rep count would be raised first.
- **The shipped configuration.** Everything above is `--no-nc-overrides` (B3 at `nc=384`). B3 ships at `nc=128`; the scalar-pack effect there was not measured (`run_pack_swap.sh --shipped-nc` runs it).
- **Against the reference design.** No reference run was made in this session, so no number here is a gap to the reference. The reference packs with scalar loops, which makes the B3 result a candidate explanation for part of the B3 gap in [[0009-issue-11-midpoint-the-result-flipped]], but only a same-session head-to-head can say how much.
- **Board conditions.** The load average on `jbpi` before each run was about 2.0-2.9 (`provenance.txt`), so something else was running on the board, and core 0 also takes the device interrupts. Either could feed B3's switching between speeds. The four runs were made back to back and the vector runs at both ends agree, but a quieter or different-core session was not tried.
- **The noise estimate is thin.** Two runs per side, and the spread only shows the mean's stability, not B3's bimodality. S1 drifts upward by 0.3% across the four runs, the same size as its vector/scalar gap, so "same speed" means "within ±0.3%".
- **Other shapes and the other board.** Two shapes, one board (`jbpi`).
- **What to do about it.** This issue only measures. Making B3 use the scalar packer (a block-size rule, or fixing whatever makes the vector pack leave the caches cold) is a separate change.
