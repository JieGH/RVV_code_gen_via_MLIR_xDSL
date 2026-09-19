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
default), or, under the reference protocol (the default in the uninstrumented build, issue #27), for as many reps as fit in `tmin`, in which case `iterations` is unused. `KerGOPs_*` are computed per-rep from
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

Which *protocol* a run used is a separate switch (issue #27, below): every number
names its build **and** its protocol.

**Which build to use.** Benchmarking means `INSTRUMENTED=0` — that is the build whose
numbers are comparable to the reference design, and it is the default for any
figure quoted anywhere. Plain `make` is for debugging our own internals (pack vs
compute, IPC, cache misses), which is the only thing it can do that the other
cannot. Neither build does both jobs. See `ref_baseline/README.md` for the
measured tax per case.

**Always state which build a number came from.** In particular, `KerGOPs` does not
mean the same thing in the two builds: instrumented divides by `kernel_time +
pack_time`, uninstrumented by that rep's wall clock. Never put the two in one
table or one plot. `RVV_Time_ms` is the only column that means the same thing in
both, which is why the cross-design comparison runs on `e2e` and not on `Ker`.

### How close the uninstrumented build is to the reference protocol

The reference times its GEMM like this
(`ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c:339-361`):

```c
time = 0.0;  t1 = dclock();  nreps = 0;
while (time <= tmin) {
    nreps++;
    gemm_blis_B3A2C0(...);        // packing is inside
    t2 = dclock();                // ONE clock read per rep
    time = (t2 > t1 ? t2 - t1 : 0.0);
}
time = time/nreps;                // mean
GFLOPS = 2.0*m*n*k / (1.0e+9 * time);
```

Matching, as of issue #14 (any uninstrumented run) and issue #27 (the reference
protocol, which is the uninstrumented build's default; `--no-ref-protocol` turns it off):

| | reference | uninstrumented, `--no-ref-protocol` | uninstrumented, default (reference protocol) |
|---|---|---|---|
| formula | `2mnk / mean per-rep wall` | same | same |
| packing inside the timed region | yes | yes | yes |
| anything else inside the region | nothing | nothing | nothing |
| statistic | mean | mean (`RVV_Time_ms`) | same |
| clock reads per rep | 1, differenced against a single pre-loop `t1` | 1, against a single pre-loop `start` | same |
| warm-up | none | 5 kernel calls, then a scalar-pack pass | **none** |
| rep count | until `tmin` (3.0 s in our baseline runs) | fixed `iterations` | **until `--tmin`** (default 3.0) |
| correctness check | after timing | before the benchmark loops | **after timing** |

Still differing — quote these alongside any cross-design number:

| difference | direction |
|---|---|
| `compare_ref_baseline.py` keeps the reference's *best* kernel variant per case; we report one pinned kernel | favours the reference |

With `--no-ref-protocol` the three rows in bold above (warm-up, rep count,
correctness-check timing) differ, so that run is **not** a like-for-like protocol match.
The instrumented build always runs the warm-up protocol.

### The reference protocol (issue #27) is the default

No flag needed: `compile_macro_bench.sh` and `./test_gemm_*_noinstr.x <dataset>` time
the way the reference does. `--tmin <s>` sets the time budget (default 3.0).

- `--no-ref-protocol` gives the older **warm-up protocol** instead: 5 warm-up calls and
  a scalar-pack pass, then the fixed `iterations` from `config.json`. Use it when you
  want steadier numbers on short shapes: dropping the warm-up by accident made B3 14%
  slower and swing 2.07–3.94 ms during issue #14, and B3 is noisy under either
  protocol (see the measurement below).
- The **instrumented build** cannot run the reference protocol (its timers make the
  numbers incomparable), so its default is the warm-up protocol. Explicitly asking it
  for `--ref-protocol` is refused, as is `--tmin` wherever the reference protocol is not
  running.
- **Every run starts with an overview** of the settings in effect, defaults included,
  each marked `[default]` or with what set it (a flag, an environment variable, or
  `config.json`): build, protocol, warm-up, reps, correctness check, packing, kernel
  family, variants, cache blocking (`nc` overrides, `RVV_FORCE_NC`), dataset, log
  names. The script prints one at the top; each project's driver prints its fuller one
  on the board. `compile_macro_bench.sh --dry-run` prints the script's and stops
  without touching the board.
- **`--no-nc-overrides`** (issue #16) ignores the per-shape `nc` overrides in
  `config.json`. The shipped configuration pins B3 to `nc=128`; the reference design has
  no such override and runs B3 at the heuristic's `nc=384`. That triples B3's `jc`
  iterations (3x the Ac/Bc packs) and adds ~4 percentage points of fringe work, so a B3
  number from the shipped configuration is *not* a like-for-like comparison. For the
  head-to-head number run with the flag; keep the run without it as the
  shipped-configuration number and quote the two separately. An override now names a
  label exactly (`B3` does not match `B30`). `RVV_FORCE_NC` is a separate ad hoc override
  and still wins over everything.
- Log names carry the protocol: `raw_*_noinstr_refproto.txt` (reference protocol) vs
  `raw_*_noinstr.txt` (warm-up protocol). `compare_ref_baseline.py` defaults to the
  instrumented name, so pass `--raw-name all_raw_<...>_noinstr_refproto.txt`. A
  `--no-nc-overrides` run adds `_no_nc_override` (`..._refproto_no_nc_override.txt`), so
  it never lands in the shipped configuration's file; give `compare_ref_baseline.py` that
  name and it computes its fringe column at the same `nc` on its own. `--force-nc` cannot
  be combined with `--no-nc-overrides` (the script refuses).

- **No warm-up at all**: neither the 5 kernel calls nor the scalar-pack pass.
  (The issue text named only the scalar-pack pass; the 5 calls before it are also a
  warm-up, and they are what fills `C` for the correctness check, so that check
  moves after the timed loop, as in `test_gemm.c`.)
- **Rep count** is whatever fits in `tmin`. It is printed (`Reps:` line) and logged
  in a new trailing `N_Reps` column. The column is present in warm-up-protocol logs too
  (there it equals `iterations`, or 0 if the case FAILed before timing). A warning prints when `nreps < 3`: the largest
  Square shapes finish in one rep at `tmin=3.0`, so the number is a single cold
  sample.
- **Logs** are named `raw_*_noinstr_refproto.txt` (and `summary_`/`perf_`,
  `results_*_noinstr_refproto.csv`), so they cannot be merged into a warm-up run.
- The timed loop body is the same code in both protocols; only the stop test differs.

**Measured effect of the warm-up** (`jbpi`, uninstrumented, warm-up protocol
(then called "default") vs reference protocol, 6 alternating runs each, `tmin=3.0`; per-rep `RVV_Time_ms`,
mean of the 6):

| case | warm-up protocol | reference protocol | change | run-to-run CV (default / ref) |
|---|---|---|---|---|
| B1 | 101.37 ms | 101.13 ms (30–31 reps) | −0.2% | 1.0% / 1.1% |
| B5 | 421.78 ms | 419.18 ms (8 reps) | −0.6% | 0.8% / 1.6% |
| B2 | 1.491 ms | 1.498 ms (~2000 reps) | +0.4% | 0.5% / 0.9% |
| B3 | 3.162 ms | 2.734 ms (972–1204 reps) | −13.5% | 7.7% / 7.2% |

For B1, B5 and B2 the gap is under 1%, inside the run-to-run noise: the warm-up
does not measurably favour us there, so the old "favours us, likely <1%" note is
now a measurement. **B3 is not quoted** (issue #1 is unresolved): it swings
2.88–3.49 ms per run by default and 2.49–3.09 ms under `--ref-protocol`, its
default-vs-reference-protocol gap is larger than either spread, and the direction is the opposite of the "expect `--ref-protocol` to
be noisier" guess. The 32x6 default runs used `iterations=30`, not the
`config.json` 1000, to keep board time bounded; B2 and B3 used 1000. Write-up:
`.scratch/issues/0008-ref-protocol-warmup-effect-measured.md`.

| Column | Meaning |
|---|---|
| `KerGOPs_Median` / `_Mean` / `_Std` | Median / mean / standard deviation of per-rep KerGOPs across all `N_REPS` reps |
| `KerGOPs_Min` / `_Max` | Slowest / fastest single rep |
| `KerGOPs_P5` / `_P95` | 5th / 95th percentile — ignores bottom 5% OS-interrupt stalls and top 5% lucky spikes |
| `KerGOPs_IQR` | Interquartile range (P75 − P25): how tightly the middle 50% of reps cluster |
| `KerGOPs_Harmonic` | Harmonic mean of per-rep GOPs (weights slow reps more than the arithmetic mean) |
| `std-dev / median` (console only; formerly `CV%`) | `Std / Median × 100`. Above ~15% indicates severe run-to-run instability |
| `fastest / median` (console only; formerly `mx/md`) | `Max / Median`. Above ~1.25 indicates extreme cache sensitivity (thrashing) — see `.scratch/issues/0001-b3-m7-packing-instability.md` |
| `RVV_Pack_GBs` / `Scl_Pack_GBs` | Pack throughput (bytes packed / pack time), vector vs. scalar packer |
| `RVV_Time_ms` | Total wall-clock per rep for the RVV pass (pack + compute + loop overhead + every timer and counter read the build contains). The only column that cannot lose time to the gaps between the inner timers, so it — not `Vector_Pack_Time_ms + Compute_Time_ms` — is what `compare_ref_baseline.py` puts against the reference (issue #14) |
| `Ref_Time_ms` | Wall-clock for the single scalar reference GEMM run (`gemm_ref`), used only for correctness checking. `0` whenever `skip_ref_run: true` in `config.json` (true for every case in the current config, to keep board time bounded) |
| `Vector_Pack_Time_ms` / `Scalar_Pack_Time_ms` | Per-rep-averaged wall-clock pack time, RVV vs. scalar packer (issue #7) |
| `Compute_Time_ms` | Per-rep-averaged wall-clock microkernel time, excluding packing (issue #2) |
| `N_Reps` | Reps the RVV timing loop ran: `iterations` under the warm-up protocol, whatever fit in `tmin` under the reference protocol (issue #27) |
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

## Terminal report (console block per variant, one-line summary per case)

As each variant finishes, `driver_benchmark.cpp` prints a block (issue #9), then a
one-line best-of-case summary that is also what `summary_<dtype>_<method>[_noinstr[_refproto]].txt`
records. The wording was rewritten so that every figure says what it measures and a
`[bracketed note]` says what is inside it, instead of bare abbreviations. In
particular the old label `KerGOPs` is gone from the terminal: it looked like
kernel-only speed, but since issue #13 it has included packing. **The log columns
keep their old names** (`KerGOPs_*`, `RVV_Time_ms`, ...) so existing tooling still
reads them; this table maps one to the other.

Uninstrumented build (the default): only real numbers are printed, and the pack/compute
split and counters are replaced by one line saying they need `--instr`:

```
  +- B2 (384x384x64) 16x15 [MR16_NR15: 16 rows x 15 columns of C per microkernel tile]
  |  Speed [GOPs = billions of operations per second = 2*M*N*K / time; higher is better]
  |    by mean time      : 12.52 GOPs  [mean of 664 reps = 1.508 ms per GEMM call; the reference design's own formula]
  |    by median rep     : 12.60 GOPs  [median of the per-rep GOPs; not thrown off by a few slow reps]
  |    what is timed     : the whole GEMM call per rep  [pack A and B + microkernel compute + loop overhead; same region the reference design times]
  |    spread per rep    : slowest 10.09 | 5th percentile 11.89 | 95th percentile 12.73 | fastest 12.77  [GOPs of single reps]
  |                        middle-50% width 0.18 | std-dev 0.26 | std-dev / median 2% [above 15% = unstable] | fastest / median 1.01x [above 1.25 = cache thrashing]
  |    reps              : 664  [until tmin=1 s, reference protocol]
  |    naive scalar GEMM : not run  [skip_ref_run in config.json]
  |    pack vs compute   : not available  [per-tile timers are compiled out of this build; rerun with --instr ...]
  +- ⏭️ SKIP [correctness not checked: skip_ref_run in config.json] [cache blocking: mc=384 rows of A per block, ...]
▶ B2 (384x384x64) | best variant 16x15 [...] | 12.52 GOPs by mean time [whole GEMM call, packing included] | 12.60 GOPs median rep | 664 reps | ...
```

The instrumented build (`--instr`) adds a "Time split per rep" table (vector pack, scalar
pack, compute, vector-pack gain) and a "Hardware counters" table (instructions per
cycle, L1 data-cache miss rates, back-end stall share, FLOPs per cycle), and labels its
speed rows as a DEBUG number that must not be compared to the reference. Counters that
cannot be read print `unavailable` instead of rows of `N/A`.

| terminal label | log column it comes from | note |
|---|---|---|
| by mean time (GOPs) | `RVV_Time_ms` -> `2*M*N*K / (RVV_Time_ms * 1e6)` | the reference's formula, and `compare_ref_baseline.py`'s `e2e`. **The headline number** |
| by median rep (GOPs) | `KerGOPs_Median` | robust to slow outliers; previously printed as "KerGOPs" |
| slowest / 5th percentile / 95th percentile / fastest | `KerGOPs_Min` / `_P5` / `_P95` / `_Max` | GOPs of single reps |
| middle-50% width / std-dev | `KerGOPs_IQR` / `KerGOPs_Std` | |
| std-dev / median, fastest / median | `Std / Median`, `Max / Median` (console only) | printed with their thresholds (15%, 1.25) |
| reps | `N_Reps` | |
| naive scalar GEMM | `Ref_Time_ms` | our own slow triple loop, used only to check the answer. Previously printed as "Ref" / "Delta", which read like the reference design and are not |
| vector pack / scalar pack / compute (ms) | `Vector_Pack_Time_ms` / `Scalar_Pack_Time_ms` / `Compute_Time_ms` | instrumented only |
| GB/s [bytes packed / pack time] | `RVV_Pack_GBs` / `Scl_Pack_GBs` | instrumented only |

Printed once per `(case, variant)`, not just for the case's winning variant. The
terminal wording and `summarize_run.py`'s `comparison.md` columns are no longer
kept identical; `comparison.md` still uses the compact names (`KerGOPs`, `Ref GOPs`,
`Δ vs ref`) and has the same ambiguity, which is a possible follow-up.

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
