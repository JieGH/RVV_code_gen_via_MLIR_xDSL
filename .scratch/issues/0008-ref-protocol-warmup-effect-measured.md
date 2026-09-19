# Issue #27 (local): `--ref-protocol`, and what the warm-up was worth (under 1% on B1/B5/B2, unsettled on B3)

**Role / Label:** `done` — implemented, measured on hardware, GitHub #27 closed
**Follow-up:** section 5 below: the reference protocol was made the *default* of the
uninstrumented build (`--no-ref-protocol` brings the old one back) and every run prints an
overview of its settings. Sections 1–4 describe the first commit (`--ref-protocol` as an
opt-in flag); there, "default" means the protocol that was the default *at measurement
time*, now called the warm-up protocol.
**Status:** 🟢 `--ref-protocol` shipped, then made the default. This note records the measurement that replaces the
"under 1%, favours us" estimate in `docs/metrics.md` and [[0006-match-reference-design-everything-but-the-two-intended-differences]].
**Target:** `jbpi` (BananaPi M2 Pro, VLEN=256), reached via `fej`. Uninstrumented build (`INSTRUMENTED=0`) throughout.
**Primary Sources:**
- GitHub #27, child of #14 and #11.
- `ext/RVV_ukernels_benchmark/gemm_blis_family/src/test_gemm.c:339-361` — the protocol matched.
- [[0007-timer-tax-measured-not-estimated]] — the previous step, same build switch.
- `templates/driver_benchmark.cpp`, `compile_macro_bench.sh`, `templates/Makefile`, `test_ref_protocol.py`.

---

## 1. What changed

`--ref-protocol [--tmin S]` on the driver (and `compile_macro_bench.sh`). Off by default.

| | default | `--ref-protocol` |
|---|---|---|
| 5 warm-up kernel calls | yes | **no** |
| scalar-pack pass (N_REPS calls) | yes | **no** |
| reps | `iterations` | until `tmin` (3.0 s), one clock read per rep |
| correctness check | before timing | **after** timing |
| logs | `raw_*_noinstr.txt` | `raw_*_noinstr_refproto.txt` |

The issue named only the scalar-pack pass as the warm-up. There is a second one: five plain
kernel calls that come first and also fill `C` for the correctness check. Both are gone under
the flag; that is why the check moves after the timed loop (the reference does the same).
Refused in the instrumented build, by the driver and, earlier, by the script. Warns when
`n_reps < 3`. A trailing `N_Reps` column is added to the raw log in both protocols.

## 2. Measurement

6 rounds, alternating which protocol goes first, same binary for both, `tmin=3.0`.
Per-rep `RVV_Time_ms` (mean of 6 runs):

| case | default | `--ref-protocol` | change | CV default / ref | reps under flag |
|---|---|---|---|---|---|
| B1 1024×384×1024 | 101.369 | 101.130 | −0.24% | 1.0% / 1.1% | 30–31 |
| B5 1024×384×4096 | 421.780 | 419.184 | −0.62% | 0.8% / 1.6% | 8 |
| B2 384×384×64 | 1.491 | 1.498 | +0.43% | 0.5% / 0.9% | 1980–2029 |
| B3 64×384×384 | 3.162 | 2.734 | −13.55% | 7.7% / 7.2% | 972–1204 |

B1, B5, B2: the warm-up gap is under 1% and smaller than the run-to-run spread, in both
directions. The estimate was right in size, but it was never measured until now.

**B3 is not quoted** (#1 is open). Its per-run mean spans 2.88–3.49 ms by default and
2.49–3.09 ms under `--ref-protocol`. `--ref-protocol` was *faster* on average, not noisier,
and the gap is larger than either side's CV. Not explained here.

## 2b. Correctness (acceptance 5)

The timing builds above run with `skip_ref_run` (Status `SKIP`), as `config.json` does.
Correctness was checked separately, on copies of the same two projects with `SKIP_REF_RUN`
removed so the scalar reference GEMM runs and the result is compared. `RVV_Time_ms` and
`N_Reps` from those runs, uninstrumented:

| case | default protocol | `--ref-protocol` |
|---|---|---|
| B1 | not captured | PASS, 101.746 ms, 30 reps |
| B5 | PASS (console line seen; row not captured) | PASS, 409.027 ms, 8 reps |
| B2 | PASS, 1.50437 ms, 1000 reps | PASS, 1.4881 ms, 2017 reps |
| B3 | PASS, 3.33485 ms, 1000 reps | PASS, 3.14111 ms, 956 reps |

So `--ref-protocol` is `PASS` on all four. The default protocol's B1 result was not
captured in this session; its check is unchanged code and covered by the host test. The
host test (`test_ref_protocol.py`) also shows the check is not vacuous after the move: a
deliberately wrong kernel gets `FAIL` under the flag.

## 3. Caveats

- The 32×6 project (B1, B5) ran the *default* protocol at `iterations=30`, not the
  `config.json` value of 1000, so B5's default run would not take about an hour. The
  16×15 project (B2, B3) used 1000. Neither changes the mean being compared, but the
  30-rep default is a shorter run than the published ones.
- Both binaries were built once and run alternately, on an otherwise idle board.
- The timing runs used a build from before one last edit: the timing vector is now reserved
  up front under `--ref-protocol` so no reallocation lands inside a rep. It only changes a
  handful of sub-microsecond events over 30–2000 reps, far below the noise here, and the
  numbers were not re-taken.
- B4 and the Square shapes were not run. At `tmin=3.0` the large Square shapes finish in 1–2
  reps, which is what the under-sampling warning is for.

## 5. Follow-up: the reference protocol became the default

Requested after the first commit: nobody should have to type the flag, and a run should say
up front what it is going to do.

| | before | now |
|---|---|---|
| uninstrumented build, no flag | warm-up protocol | **reference protocol** |
| old protocol | the default | `--no-ref-protocol` |
| instrumented build, no flag | warm-up protocol | warm-up protocol (unchanged) |
| `--ref-protocol` with `--instr` | refused | refused (unchanged) |
| `--tmin` where it would be ignored | accepted silently | **refused** (driver and script) |
| `iterations` in `config.json` | every run | warm-up protocol only; the reference protocol ignores it |

The instrumented build has to keep the old default: it cannot run the reference protocol (timer
tax, #14), so a global default of "reference" would refuse every `--instr` debugging run. The
default is therefore chosen per build, and the overview says when that is why.

**Run overview.** The driver prints, once per project at the start, every setting in effect,
defaults included, each tagged `[default]` or with what set it (flag, environment variable, or
`config.json`): build, protocol, warm-up, reps, correctness check and when, packing, kernel
family, variants, cache blocking (`nc` overrides, `RVV_FORCE_NC`), dataset, log names.
`compile_macro_bench.sh` prints a shorter one at the top, and `--dry-run` prints it and stops
(added so the overview can be tested without the board).

**Checked** on `jbpi`, real kernels, no flags, uninstrumented: overview printed; B2 ran 2021 reps
and B3 989 (both `SKIP`, as `config.json` has `skip_ref_run`); logs written as
`*_noinstr_refproto.txt`. Host tests cover the rest with a stub kernel (51 tests, all
passing). The driver also compiles against all three real generated headers in both builds.

**Not re-measured.** Sections 2–3 compare the two protocols and were taken before this
change; flipping which one is the default does not change what either one does.

**Consequence to remember.** Every default run now spends about `tmin` (3 s) per shape per
variant, and large Square shapes get 1–2 reps, so the under-3-reps warning will fire on them
by design. B3 is noisy under the reference protocol as well (CV 36% in the smoke run).
`compare_ref_baseline.py` still defaults to the instrumented log name, so it needs
`--raw-name all_raw_<project>_noinstr_refproto.txt` for a default run.

## 4. Raw data

```
mode,round,Layer,Status,KerGOPs_Min,KerGOPs_Max,KerGOPs_Median,RVV_Time_ms,N_Reps
default,1,B1,SKIP,7.90133,7.97471,7.93642,101.455,30
default,1,B5,SKIP,7.44163,7.77756,7.72748,418.682,30
default,1,B2,SKIP,11.687,12.8468,12.6959,1.49011,1000
default,1,B3,SKIP,4.42997,9.20351,5.06144,3.3926,1000
ref,1,B1,SKIP,7.65984,7.94635,7.89945,102.23,30
ref,1,B5,SKIP,7.61594,7.65292,7.64153,421.866,8
ref,1,B2,SKIP,10.1567,12.8188,12.6537,1.49939,2001
ref,1,B3,SKIP,4.36168,9.06682,8.06255,2.65187,1132
ref,2,B1,SKIP,7.8674,7.94198,7.91984,101.738,30
ref,2,B5,SKIP,7.55164,7.60121,7.58851,424.834,8
ref,2,B2,SKIP,10.1315,12.9124,12.7675,1.48069,2027
ref,2,B3,SKIP,4.48684,8.98909,7.50622,2.84442,1055
default,2,B1,SKIP,7.64009,7.94443,7.90145,102.181,30
default,2,B5,SKIP,7.46277,7.61146,7.57505,425.66,30
default,2,B2,SKIP,3.38243,12.9752,12.7909,1.48743,1000
default,2,B3,SKIP,4.68776,9.26146,8.38895,2.88311,1000
default,3,B1,SKIP,7.85635,7.94812,7.91508,101.797,30
default,3,B5,SKIP,7.45977,7.664,7.64593,421.76,30
default,3,B2,SKIP,11.3731,12.7844,12.5824,1.50451,1000
default,3,B3,SKIP,4.67354,9.27796,6.54441,2.98277,1000
ref,3,B1,SKIP,8.03167,8.15171,8.09509,99.434,31
ref,3,B5,SKIP,7.89867,7.96202,7.94513,406.012,8
ref,3,B2,SKIP,3.42537,12.8308,12.7105,1.50049,2000
ref,3,B3,SKIP,4.68301,9.09339,8.26497,2.49242,1204
ref,4,B1,SKIP,7.89455,8.01126,7.93809,101.383,30
ref,4,B5,SKIP,7.49039,7.61074,7.5954,425.446,8
ref,4,B2,SKIP,6.02221,12.9752,12.8127,1.47915,2029
ref,4,B3,SKIP,4.74452,8.90147,8.27312,2.57387,1166
default,4,B1,SKIP,8.0675,8.14995,8.11353,99.2691,30
default,4,B5,SKIP,7.51683,7.78501,7.75556,416.084,30
default,4,B2,SKIP,11.9261,12.8848,12.7227,1.48474,1000
default,4,B3,SKIP,4.54761,9.2911,5.05906,3.30229,1000
default,5,B1,SKIP,7.85946,7.93596,7.90588,101.911,30
default,5,B5,SKIP,7.37624,7.65115,7.61058,424.011,30
default,5,B2,SKIP,11.2536,12.9448,12.7653,1.48237,1000
default,5,B3,SKIP,4.10824,9.36912,5.04689,3.49396,1000
ref,5,B1,SKIP,7.28439,7.95906,7.89799,102.228,30
ref,5,B5,SKIP,7.6348,7.68721,7.66375,420.603,8
ref,5,B2,SKIP,10.085,12.8192,12.6322,1.51062,1986
ref,5,B3,SKIP,4.7227,9.18373,6.29189,2.75178,1091
ref,6,B1,SKIP,7.98167,8.09741,8.07604,99.7646,31
ref,6,B5,SKIP,7.55592,7.78845,7.76882,416.343,8
ref,6,B2,SKIP,10.4224,12.5915,12.5008,1.5153,1980
ref,6,B3,SKIP,4.56765,8.73986,6.1263,3.08931,972
default,6,B1,SKIP,7.89096,8.03178,7.91947,101.6,30
default,6,B5,SKIP,7.40134,7.63692,7.59993,424.482,30
default,6,B2,SKIP,11.2987,12.7711,12.616,1.4978,1000
default,6,B3,SKIP,4.58271,9.21643,6.63558,2.91907,1000
DONE
```

Columns: `mode` (`ref` = `--ref-protocol`), round, layer, status (`SKIP`: these builds ran with
`skip_ref_run`, as `config.json` does), min and max and median KerGOPs, `RVV_Time_ms`, `N_Reps`.
