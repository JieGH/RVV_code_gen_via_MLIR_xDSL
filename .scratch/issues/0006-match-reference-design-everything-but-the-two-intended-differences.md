# 0006 — Make everything match the reference except the two things we meant to change

Status: documented as GitHub issue #11 — https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/11
Date: 2026-09-18
Related: #10 (metric mismatch), .scratch/issues/0004, 0005

## Premise

Our design is meant to differ from `ext/RVV_ukernels_benchmark` in exactly two ways:

1. vectorised (RVV) packing instead of scalar packing
2. auto-generated microkernel from xDSL instead of Exo-generated C

Everything else — the BLIS loop nest, the blocking, the data layout, the build, the
measurement — should be identical, so that any performance delta is attributable to (1)
and (2). This issue records every place where that is currently not true.

Source: four read-only audits (loop nest / packing / measurement / build+kernel),
2026-09-18.

## Confirmed matching (no work needed)

- 5-level loop nest `jc→pc→ic→jr→ir`, and which operand is packed at which level
- Packed-panel index formulas for Ac and Bc, including fringe panels and both
  order/trans branches
- CCC / NN column-major layout, `lda=m`, `ldb=k`, `ldc=m`
- alpha/beta semantics: both sides ignore alpha, both split on `beta==0` via b0/b1
- `get_optim_mc_nc_kc` formula (byte-identical port)
- VLEN=256, LMUL=m1, single hoisted `vsetivli`
- Vector instruction mix at 32x6: 4 `vle32.v` + 24 `vfmacc.vf` per k-iteration, both sides
- Register pressure: 28 of 32 vector registers, zero spills, both sides
- `-march=rv64gcvzfh` vs `rv64gcv`: verified byte-identical disassembly (zfh is fp16 only)
- Microkernel signature maps 1:1 (ref has an extra unused `void *ctxt`)
- Neither side is threaded; neither allocates inside a timed region

## Mismatches, ranked

### 1. `WL2 = 16` (ours) vs `WL2 = 8` (reference)
`tests/pure_xdsl_blis_rvv/modelLevel/archs/bananapi.h:25` vs
`ext/.../modelLevel/archs/bananapi.h:26`. Flipped in commit `40d1ea8`.
Changes MC (and through it NC) on 8 of 10 shapes. The two sides run different blockings.
`cache_config.py:14` still says 8, so the harness's own predicted-blocking cross-check
reports agreement while the binaries disagree. `ref_baseline/README.md:73-79` is now wrong.

### 2. Metric: `KerGOPs` excludes packing, reference GFLOPS includes it
Worth +16% on B1/B5, +41% on B3. Known (#10), `e2e` column exists, but
`summarize_run.py:116` and the dashboards still lead with `KerGOPs_Median`.

### 3. Per-tile `chrono::now()` inside the innermost loop
`gemm_dispatch_xdsl.cpp:126,143`. ~125-160 ns/tile; 9.9M clock reads in one S5 rep vs the
reference's one `gettimeofday` per rep. Half the cost falls outside both our timers
(2.0-8.7% of wall time, unattributed), half inflates `Compute_Time_ms`.
Consequence: `e2e = Vector_Pack + Compute` under-counts. Should use `RVV_Time_ms`.

### 4. Fringe tiles: full MR×NR kernel + `tmp_C` round-trip
Reference dispatches `ukrmatrix[mr][nr][be]` — a right-sized specialised kernel, writing C
in place, no copy. We always run the full kernel, plus `memset(tmp_C)` + copy-in + copy-out
(`gemm_dispatch_xdsl.cpp:127-139`). Extra multiply-adds: +1.2% (S3) to +5.5% (B3).
Structural, costs us.

### 5. `nc = 128` override on B3, ours only
`config.json:45` → `generated_kernels.h:16` → `driver_benchmark.cpp:530-542`.
Reference uses nc=384 there. 3× the Ac and Bc packs, and ~4pp of B3's fringe penalty.
The B3 row is comparing two different configurations. Also: plain prefix match, so a
future `B30` label would inherit it. Plus an undocumented `RVV_FORCE_NC` env escape hatch.

**Status (GitHub #16): fixed by option (a).** `compile_macro_bench.sh --no-nc-overrides`
(driver `--no-nc-overrides`) ignores `CONFIG_NC_OVERRIDES`, so B3 runs at nc=384 like the
reference; the run's logs get a `_no_nc_override` suffix so they never mix with the shipped
(nc=128) run, which is reported separately. The override now matches the label exactly
(`B30` no longer inherits `B3`). `RVV_FORCE_NC` was already in `docs/metrics.md` and the run
overview; it is unchanged. No board run was made for this change; the host-side driver
tests cover the nc selection, log names and overview.

### 6. No correctness check is running, and the oracle is broken
`SKIP_REF_RUN` set in all three projects → every row logs `Status=SKIP`.
Separately, `driver_benchmark.cpp:191-330` (`gemm_ref`) has unbalanced braces: beta is
applied unconditionally. Harmless at beta=0, wrong the moment the check is re-enabled.

### 7. Zero-padding memsets land in neither timer
`gemm_dispatch_xdsl.cpp:63-67, 90-94`. Extra work the reference never does (it relies on
Ac/Bc being calloc'd once). Sits before both `t_pack_start_*`, so invisible in
`Vector_Pack_Time_ms` and `Compute_Time_ms`, visible only in wall clock. Probably
removable — our fringe path already routes C through `tmp_C`, so the pad values never
reach C.

### 8. Ac/Bc buffer sizes differ from the reference
`driver_benchmark.cpp:557-558` substitutes `mr`/`nr` where the reference uses `KR=1`.
At 32x6 our Ac is 512 KB vs the reference's 412 KB — 24% larger footprint, against us.
Alignment 64 vs 32. The comment at `:556` claiming the sizes match is inaccurate.

### 9. Estimator asymmetry in the comparison script
`compare_ref_baseline.py:74-75` takes best-of-K variants for the reference;
`:103` takes mean-of-medians over our last 5 run dirs. Not the same statistic.

### 10. No CPU pinning, no governor control, measurements 12h apart
Our cron runs 00:00-04:41; `refbase_2026-09-18_164740` was taken at 16:47. Unpinned,
shared board, unknown thermal state. Uncontrolled variable under every percentage.

### 11. `-O2` + `-fno-strict-aliasing` (ours) vs `-O3` (reference)
Measured, not assumed: the 32x6 microkernel is **146 instructions at both -O2 and -O3**,
so this is worth nothing on the hot kernel. Pack kernels differ ~2% in instruction count.
Low impact — but free to fix and removes an objection.

### 12. Reference nreps collapses to 1 on S3/S4/S5
At `TIMIN=3.0` with 7/17/31 s per rep, the reference number for those three shapes is a
single cold run. Those rows need an explicit `n=1` marker.

### 13. MR=32 pack path is untested
`test_pack_correctness.py:384` hardcodes RR=15. MR=32 is the only shipped config needing
4 vector chunks; chunks 3 and 4 have never been compared against the oracle.

### 14. Minor
- `N_REPS=5` for the 32x7 project vs 1000 for the other two (`config.json:45-49`)
- `_if` pack variants hardcoded (`gemm_dispatch_xdsl.cpp:46-47`) — fine for CCC/NN only
- Board `gcc --version` not recorded in either provenance file
- `-DCHECK` runs inside the reference's timed region (<0.1%, cannot fix, disclose)
- Dead code: `kc_pad`/`padded_mc`/`padded_nc` in the driver disagree with what dispatch
  recomputes; would silently diverge if a DOT method is ever enabled

## Settled: we are LDX, not OPT

Previously ambiguous. Resolved by disassembly:

- Our kernel signature takes `lda`/`ldb`/`ldc` as runtime args and the inner loop
  multiplies by them (`gemm_b0_xdsl.cpp:115,128`)
- The `emitc.constant 32` in the MLIR is **dead** — `v43` is defined and never read
- No LTO, and the kernel is a separate TU from the dispatch, so gcc cannot recover 32/6
- Per-k-iteration: ref OPT 40 insns, ref LDX 41, ours 41 — ours is structurally identical
  to ref LDX with the same register allocation
- Whole-function b0: ref OPT 147, ref LDX 149, ours 146

**So `--opt LDX` is the correct baseline.** `run_ref_baseline.sh:26` still defaults to
`OPT`, which hands the reference ~1 scalar instruction per k-iteration. The
`refbase_2026-09-18_164740` run used LDX and is fair; the default is not.

## Where the numbers actually land

Correcting B1 for pack inclusion and the per-tile timer gap: roughly 6.7-6.9 GOPs against
the reference's 7.02 — parity to a few percent behind, not the `+16.3%` the KerGOPs column
shows. Still unaccounted for, in our favour: `-O2`, `-fno-strict-aliasing`, fringe-tile
handling, the 24% larger Ac. In the reference's favour: no warm-up, `-DCHECK`.
