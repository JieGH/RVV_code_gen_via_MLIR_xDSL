# When does vector packing pay off? Sweep m, n, k, tile and threads, vector against scalar packer (sub-issue of #83)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/85
**Parent:** #83 (scalar packing rerun). Follows its T4 figure "Does the pack share explain the slowdown?".
**Role / Label:** `needs-triage`. Board runs (jbpi / jbpi2), idle boards only for timing.
**Status:** spec written 2026-11-17. Not started.
**Branch:** `when-does-vector-packing-pay-off`, cut from `pure-xdsl` (after PR #84).
**Report:** `.scratch/reports/issue-85/` (README + brief PDF/LaTeX, pgfplots charts, glossary).

---

## 1. Question

#83 found, on 11 shapes picked from earlier sweeps:

- Total slowdown from scalar packing = **pack share x extra pack cost**. It predicts the 1T ratio within 1% on 9 of 11 shapes.
- Pack share follows $n_c$: 49-83% at $n_c$ = 4-80, 7-23% at $n_c \ge 256$.
- Scalar pack costs 1.07-1.16x on large packs and 1.55-1.94x on small packs that stay in cache.
- At 8T the $n = 4$ gap shrinks (Y14 0.60 → 0.70). Inferred only: the 8T path has no timers.

These shapes were not chosen to test the cause. They vary m, n, k and tile all at once. This issue tests the cause directly:

1. **Which parameter** sets the pack share: n (A packed once per $n_c$ block), m (B packed once, reused over m), or k?
2. **Which parameter** sets the extra cost of scalar packing: panel size (in cache or not), k, tile width $m_r$ / $n_r$?
3. **How threads change both**, at 1T and 8T.
4. **Decision rule:** for which (m, n, k, tile, threads) does the vector packer give a gain above noise, and is it ever slower?

## 2. Runs: one-factor sweeps around a base point

Same design as #83; the only switch is `--scalar-pack`. Each sweep runs at **1T (core 4)** and **8T**.

| Sweep | Varies | Held fixed | Points |
|---|---|---|---|
| P-n | n ∈ {4, 8, 16, 32, 64, 128, 256, 512, 1024} | m = 1200, k = 256 | 9 |
| P-m | m ∈ {64, 128, 300, 600, 1200, 2400, 4800} | k = 256; n = 16 and n = 256 | 14 |
| P-k | k ∈ {16, 32, 64, 144, 256, 576, 1152} | m = 1200; n = 16 and n = 256 | 14 |
| P-tile | tile ∈ {16x4, 24x8, 32x4, 16x15, 32x7} | (1200, 4, 16), (1200, 16, 256), (1200, 256, 256) | 15 |

- Tile for P-n / P-m / P-k: one fixed tile (24x8) so only the swept dimension moves. P-tile covers the tile.
- Cache blocking: the analytical model, `--no-nc-overrides`, as #83. The log records the $m_c$, $k_c$, $n_c$ chosen per shape.

## 3. Measurements

- **M1, GOPs (the answer):** uninstrumented build, reference protocol, 3 launches A-B-B-A (vector, scalar, scalar, vector order), core 4 at 1T. Report scalar/vector ratio by mean time and by median rep, with the launch-to-launch range as the noise band.
- **M2, 1T pack/compute split (the cause):** instrumented build, warm-up protocol, 200 reps, as #83 T4. Pack A and pack B times reported separately.
- **M3, 8T pack/compute split:** the 8T path has no timers (#56). **Code change:** add pack A / pack B / compute timers to the parallel nest in the instrumented build only (per thread, report the max and mean over threads). Uninstrumented 8T code stays unchanged. Host test: the instrumented 8T build still matches the double-precision reference, and timers sum to within 5% of the wall time.

## 4. Output

- Per sweep, one chart per thread count: scalar/vector GOPs ratio against the swept dimension, noise band shaded; pack share on the same rows.
- Check of the #83 model on every point: predicted (from M2/M3) against measured (M1).
- **Decision table:** region of (n, m, k, tile, threads) where vector packing gains > noise, where it is neutral, and any point where it loses.
- Brief PDF + LaTeX (pgfplots, glossary of launch, A-B-B-A, tile, pack share, GOPs by mean/median), linked from this issue and from #83.

## 5. Limits to state with the numbers

- M2/M3 are instrumented (timers cost 11-17% on short calls): use their shares and ratios, never their GOPs.
- One board per series; name board, build, protocol and core for every number.
- Synthetic shapes: the base point is YOLO-like (m = 1200), not a real layer.

## 6. Closure

- [ ] M3 timers in the 8T instrumented nest + host test
- [ ] Configs and driver for P-n, P-m, P-k, P-tile
- [ ] Correctness pass (vector and scalar, 1T and 8T) on every shape
- [ ] M1, M2, M3 runs on idle boards, logs committed
- [ ] Report + brief, linked here and in #83
- [ ] PR → `pure-xdsl`
