# Benchmark metrics glossary

Every column written by the pure xDSL BLIS macrokernel benchmark
(`tests/pure_xdsl_blis_rvv/`), across the console table, the raw/perf CSV logs, and
the per-run comparison table. Pointed to from the console banner at the start of
every benchmark run.

## Where results land

`compile_macro_bench.sh`'s `[4/4] Fetch Logs` step writes one timestamped directory
per invocation:

```
tests/pure_xdsl_blis_rvv/logs/run_<timestamp>/
  <project>/                       # one per MRxNR pinning, e.g. fp32_AXPY_FP32_Pack_vlen256_MR16NR15
    raw_<dtype>_<method>.txt       # one row per (case, variant): every metric below
    perf_<dtype>_<method>.txt      # one row per (case, variant): hardware counters
    summary_<dtype>_<method>.txt   # one human-readable line per case (best variant)
  all_raw_<dtype>_<method>.txt     # every project's raw_*.txt concatenated, tagged by a leading Project column
  all_perf_<dtype>_<method>.txt    # same, for perf_*.txt
  all_summary_<dtype>_<method>.txt # same, for summary_*.txt (grouped under "=== <project> ===" headers)
  comparison.md                    # per-case table built from all_raw_*.txt, see below
```

`comparison.md` is generated automatically by `summarize_run.py` at the end of the
fetch step. To regenerate it by hand against an existing run directory:

```bash
python3 tests/pure_xdsl_blis_rvv/summarize_run.py tests/pure_xdsl_blis_rvv/logs/run_<timestamp>
```

## Timing / GOPs metrics (raw log + console table)

Each case runs `N_REPS` times (from `config.json`'s `iterations` field, or 5 by
default). `KerGOPs_*` are computed per-rep from
`2*M*N*K / (kernel_time + rvv_pack_time)`, i.e. microkernel compute **plus** the
RVV pack. Since issue #13 packing time is included, so the number is comparable to
the reference design's `2mnk / wall(gemm_blis)`. Before #13 it was
`2*M*N*K / kernel_time` (compute only), which overstated GOPs by
`1 / (1 - pack_fraction)` — about +7-9% on Square and +18-20% on B1/B4/B5 (see
`.scratch/issues/0004-reference-comparison-metric-mismatch.md`). Raw logs from
before that fix are not comparable to newer ones. The column keeps the name
`KerGOPs` so existing tooling still reads it.

### Instrumented vs. uninstrumented builds (issue #14)

The timers above are not free. Every microkernel tile is bracketed by a
`chrono::now()` pair (~160 ns on jbpi) and every pack call and compute nest by a
`perf_event_open()` read. `make INSTRUMENTED=0` — or `compile_macro_bench.sh
--no-instr` — compiles all of it out.

|  | `INSTRUMENTED=1` (default) | `INSTRUMENTED=0` |
|---|---|---|
| binary / logs | `test_gemm_*.x`, `raw_*.txt` | `test_gemm_*_noinstr.x`, `raw_*_noinstr.txt` |
| `RVV_Time_ms` | real, but 11-17% inflated by the timers | real, no timer tax |
| `KerGOPs_*` denominator | `kernel_time + rvv_pack_time` | that rep's own wall clock |
| pack/compute breakdown, `*_Pack_GBs`, perf counters | yes | `nan` — they do not exist |

Both builds do exactly the same arithmetic (verified `PASS` against the scalar
reference either way) and run the same passes, including the scalar-pack pass,
which doubles as the RVV loop's warm-up. They differ in the timers and nothing
else, which is what makes an A/B between them meaningful.

Use the default build for any question about *where* time goes, and the
uninstrumented build for any headline number compared against the reference
design. See `ref_baseline/README.md` for the measured tax per case.

| Column | Meaning |
|---|---|
| `KerGOPs_Median` / `_Mean` / `_Std` | Median / mean / standard deviation of per-rep KerGOPs across all `N_REPS` reps |
| `KerGOPs_Min` / `_Max` | Slowest / fastest single rep |
| `KerGOPs_P5` / `_P95` | 5th / 95th percentile — ignores bottom 5% OS-interrupt stalls and top 5% lucky spikes |
| `KerGOPs_IQR` | Interquartile range (P75 − P25): how tightly the middle 50% of reps cluster |
| `KerGOPs_Harmonic` | Harmonic mean of per-rep GOPs (weights slow reps more than the arithmetic mean) |
| `CV%` (console only) | `Std / Median × 100`. Above ~15% indicates severe run-to-run instability |
| `mx/md` (console only) | `Max / Median`. Above ~1.25 indicates extreme cache sensitivity (thrashing) — see `.scratch/issues/0001-b3-m7-packing-instability.md` |
| `RVV_Pack_GBs` / `Scl_Pack_GBs` | Pack throughput (bytes packed / pack time), vector vs. scalar packer |
| `RVV_Time_ms` | Total wall-clock per rep for the RVV pass (pack + compute + loop overhead + every timer and counter read the build contains). The only column that cannot lose time to the gaps between the inner timers, so it — not `Vector_Pack_Time_ms + Compute_Time_ms` — is what `compare_ref_baseline.py` puts against the reference (issue #14) |
| `Ref_Time_ms` | Wall-clock for the single scalar reference GEMM run (`gemm_ref`), used only for correctness checking. `0` whenever `skip_ref_run: true` in `config.json` (true for every case in the current config, to keep board time bounded) |
| `Vector_Pack_Time_ms` / `Scalar_Pack_Time_ms` | Per-rep-averaged wall-clock pack time, RVV vs. scalar packer (issue #7) |
| `Compute_Time_ms` | Per-rep-averaged wall-clock microkernel time, excluding packing (issue #2) |
| `Status` | `PASS` (matched the scalar reference), `FAIL` (mismatch beyond tolerance), or `SKIP` (`skip_ref_run: true`, so no reference was computed) |

## Hardware perf counters (perf log, issue #8)

Supplements, never replaces, the timing above — every column here is read from one
`perf_event_open()` counter group (see `templates/perf_counters.h`), snapshotted
around each pack call and around the whole microkernel loop nest, **separately**
for the Vector_Pack, Scalar_Pack, and Compute regions (each gets its own set of the
columns below, e.g. `Vector_Pack_IPC`, `Scalar_Pack_IPC`, `Compute_IPC`).

`rdcycle`/`rdinstret` and the `perf stat` CLI don't work on `jbpi` (see the issue #8
comment thread); this is the raw `perf_event_open()` syscall instead.

| Column suffix | Meaning |
|---|---|
| `_Cycles` / `_Instructions` | Per-rep-averaged raw counts |
| `_IPC` | Instructions per cycle (`_Instructions / _Cycles`). Roughly: higher is better; <1 usually means the core is stalled (memory-bound), not compute-bound |
| `_L1D_Read_Miss_Pct` / `_L1D_Write_Miss_Pct` | L1 data cache miss rate, as a percentage of *loads* (not of cycles) — read and write misses are tracked and reported separately |
| `_dTLB_Misses` | Data TLB misses. Plausible contributor for packed buffers (`Ac`/`Bc`) that span multiple pages |
| `_Stall_BE_Pct` / `_Stall_FE_Pct` | % of cycles the core was stalled waiting on backend (execution/memory) vs. frontend (fetch/decode) resources |
| `_Branch_Misses` | Raw branch mispredict count. Small/noisy on this kernel shape so far — treat as secondary |
| `Counters_Valid` | `1` if every counter read in this row was valid, `0` if the group was multiplexed or a read failed — treat every `*_` column as unreliable when `0` (columns are written as `nan`, not a misleading `0`) |
| `Vector_Pack_Cycles_CV_Pct` / `Compute_Cycles_CV_Pct` | Coefficient of variation of per-rep cycle counts — the hardware-counter analog of `KerGOPs`' `CV%`, but far tighter in practice (counters are far less noisy than wall-clock on this board) |
| `Compute_FLOPs_per_Cycle` | `2*M*N*K / Compute_Cycles` — decouples from clock-rate variance, unlike GOPs/second |
| `Arith_Intensity_FLOPs_per_Byte` | FLOPs per byte packed, computed analytically from `m,n,k,mc,nc,kc` (not measured) — compare against the board's machine balance point to judge compute- vs. memory-bound |

**Known limitations** (see the issue #8 tracking comment for detail):
- This board's PMU does not expose L2/LL cache counters at all — an unsupported
  event added to the group makes the *whole* group unschedulable (every count reads
  0), which is why validity is tracked per-pass rather than per-event.
- The `Compute_*` counters include the per-tile `chrono::now()` timing pair
  (~200 instructions / ~257 cycles on `jbpi`) that `Compute_Time_ms` excludes —
  measurable on small-tile shapes, negligible on large ones.

## Live per-variant console block (issue #9)

As each variant finishes benchmarking, `driver_benchmark.cpp` prints a multi-line
block to the console (in addition to the existing one-line best-of-case summary,
which is unchanged and still what `summary_<dtype>_<method>.txt` records). The block
surfaces the same fields as a `comparison.md` row, just reflowed as text instead of a
wide table so it's readable live during a long run rather than only after
`summarize_run.py` runs at the end:

```
  +- B2 (384x384x64) 16x15 [MR16_NR15]
  |  Pack:  Scalar 3.10ms (32.6%) | Vector 1.05ms (14.1%) | Compute 6.40ms | Pack Gain +66.1%
  |  GOPs:  12.73 (this run) [min:11.6, p5:12.5, p95:12.8, max:12.9, iqr:0.2, s:0.1, CV:1%, mx/md:1.01] | Ref N/A | Delta N/A
  |  Counters valid: yes
  |    Vector Pack: IPC 0.72 | L1D Rd Miss 1.2% | L1D Wr Miss 3.4% | Stall BE 18.0%
  |    Scalar Pack: IPC 0.55 | L1D Rd Miss 2.1% | L1D Wr Miss 6.6% | Stall BE 25.0%
  |    Compute:     IPC 0.95 | L1D Rd Miss 0.5% | L1D Wr Miss 0.1% | Stall BE 8.0% | FLOPs/Cyc 20.97
  +- ✅ PASS (mc:384 nc:384 kc:64)
```

Every field means exactly what its `comparison.md` counterpart means (see below) —
`Ref`/`Delta` read `N/A` under `skip_ref_run: true` and `Counters valid: no` blanks
every counter field to `N/A`, same rules as the table. Printed once per `(case,
variant)`, i.e. one block per row that would appear in `comparison.md`, not just for
the case's winning variant. `short_variant_name`/`jitter_bracket` in
`driver_benchmark.cpp` are the single source of formatting shared with the top-1
summary line; `summarize_run.py`'s identically-named Python helpers are kept in sync
by convention (cross-language, not shared code) — see that file's `_jitter_bracket`
comment.

## Comparison table (`comparison.md`)

One row per case (the pinned MR×NR's single variant), built from `all_raw_*.txt` by
`summarize_run.py`. When the matching `all_perf_*.txt` exists in the same run
directory, its counters are joined in too (by `Project, Layer, M, N, K, Variant`);
otherwise every perf column below reads `N/A`.

| Column | Meaning |
|---|---|
| `Case` | Dataset row label (e.g. `S1`, `B3`) |
| `Shape (m×n×k)` | Problem dimensions |
| `MRxNR` | Pinned microkernel register tile |
| `Scalar Pack ms` / `Vector Pack ms` / `Compute ms` | Same as the raw log's `*_Time_ms` columns |
| `Scalar Pack % of total` / `Vector Pack % of total` | `pack_ms / (pack_ms + Compute_Time_ms) × 100` — both use the same shared `Compute_Time_ms`, since scalar-pack-only reps don't separately measure kernel time |
| `KerGOPs (this run)` | `KerGOPs_Median` from the raw log |
| `KerGOPs Jitter [min, p5, p95, max, iqr, s, CV%, mx/md]` | Same bracket the console prints per case, rebuilt from the raw log's `KerGOPs_Min/P5/P95/Max/IQR/Std` — see the "Timing / GOPs metrics" section above for what each of these means |
| `Ref GOPs` | `2*M*N*K / Ref_Time_ms`, or `N/A` when `skip_ref_run: true` (currently every case) |
| `Δ vs ref` | `KerGOPs (this run) / Ref GOPs`, as a speedup multiplier (e.g. `10.0×`); `N/A` alongside `Ref GOPs` |
| `Vector Pack Gain %` | `(Scalar Pack ms − Vector Pack ms) / Scalar Pack ms × 100`. Positive = vector packing faster; negative = vector packing slower (see `.scratch/issues/0003-scalar-vs-vector-pack-timing.md` for the B3 case where it's negative) |
| `Counters Valid` | `yes`/`no` from the perf log's `Counters_Valid`, or `N/A` if no perf log was found for this run |
| `{Vector Pack, Scalar Pack, Compute} IPC` / `L1D Read Miss %` / `L1D Write Miss %` / `Stall BE %` | Same metrics as the "Hardware perf counters" section above, one set per region — `N/A` whenever `Counters Valid` is `no` or `N/A` |
| `Compute FLOPs/Cycle` | Same as the perf log's `Compute_FLOPs_per_Cycle` |
| `additional info` | `Status`, suffixed with `(skip_ref_run)` when `Ref GOPs` is `N/A` for that reason |
