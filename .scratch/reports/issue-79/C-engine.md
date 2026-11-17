# Issue #79, Part C: is the parallel engine (#56) sound?

Audit date 2026-11-17.

| item | verdict |
|---|---|
| C1 host tests | PASS (21 of 21 parallel tests, 623 of 623 in the full suite) |
| C2 code read against #56 §5 | PASS |
| C3 board correctness, 19 tiles x 50 shapes x 8 threads | PASS (950 of 950) |

---

## C1. Host tests

- `test_parallel_gemm.py`: **21 passed** (4.3 s). It covers:
  - `--threads 1` byte-identical to no flag;
  - OpenMP only when N > 1;
  - 3- and 8-thread dispatch correct on the host with stub kernels (16x6), 20 reps per case;
  - pack bytes equal to 1 thread;
  - the script refusals and the provenance line.
- Full suite (`cd tests/pure_xdsl_blis_rvv && python -m pytest .`, 2026-11-17 08:17): **623 passed** in 15 min 50 s. It rewrote three tracked files under `ref_baseline/generated/`; I restored them with `git checkout` before any board run.
- Note: the tests only import when pytest runs from inside `tests/pure_xdsl_blis_rvv` (`python -m pytest`). Run from the repo root, `test_parallel_gemm.py` fails to collect (`ModuleNotFoundError: test_pad_zeroing`, because `--import-mode=importlib` is set in `pyproject.toml`). Several other test files have the same import, so this is an existing convention, not a #56 bug. Left as is.

## C2. Code read: `render_parallel_nest` (`generate_pure_blis_project.py:278-346`) against #56 §5: PASS

| §5 promise | code | ok? |
|---|---|---|
| one `omp parallel` region per call | `#pragma omp parallel num_threads(N)` wraps the jc/pc nest | yes |
| every thread runs jc/pc | the loops sit inside the region, not split | yes |
| B pack split over (panel x k-chunk), small n still feeds all threads | `k_chunks = min(ceil(nth / n_panels), cur_kc)`; units dealt round robin; each unit writes `Bc + j0*kc_pad + k0*nr`, rows `k0..k1` of panel `j0`, so units don't overlap | yes |
| one barrier after the B pack | `#pragma omp barrier` after the unit loop | yes |
| ir split: each thread packs and computes its own MR panels of each `mc` block | `ir0/ir1` from `m_panels * tid / nth`; the thread packs `Ac + ir0*kc_pad` and reads only `[ir0, ir1)` | yes |
| no barrier per full `ic` block | full blocks split the same way each time, so a thread only overwrites Ac rows it alone reads | yes |
| extra barrier before a short last `ic` block | `if (cur_mc < mc && ic > 0) barrier`. Every thread computes the same condition, so all reach it (an OpenMP requirement). It comes before the `continue` of a thread with no panels | yes |
| barrier before the next `pc` repacks Bc | `#pragma omp barrier` at the end of each `pc` | yes; it also covers Ac and the next `jc` |
| beta only on the first `pc` | `GET_BETA(pc, beta)` | yes |
| counters | `pack_bytes` added by thread 0 only, from the whole-block sizes; the other counters are `(void)` | yes (matches the `test_pack_bytes_match_one_thread` test) |

No race found. The host tests use one tile, though, so the board pass (C3) is what checks all 19 generated kernels, fringe paths included, under 8 threads.

## C3. Board correctness pass at 8 threads (the fix for B1): PASS

- **Command** (repo root): `compile_macro_bench.sh --jump --riscv-ip jbpi --threads 8 --no-nc-overrides --tmin 0.2 --config tests/pure_xdsl_blis_rvv/config_yolo_correctness_8t.json`.
- **Config:** the new `config_yolo_correctness_8t.json`, the 19 tiles of `config_yolo_search.json` with the scalar oracle on, on the full `datasets/yolo26_k1.dat`.
- **Run conditions:** `jbpi`, uninstrumented, reference protocol, 8 threads on cores 0-7 (`OMP_PROC_BIND=close`).
- **When:** 2026-11-17, 08:34 → 09:24. Other users: none before (load average 2.26, the board's steady 2.0).
- **Result: 950 of 950 rows PASS** (19 tiles x 50 shapes), no FAIL. Log: `tests/pure_xdsl_blis_rvv/logs/run_2026-11-17_092354/`.
- **What this covers:** every tile #63 timed, now checked at 8 threads against the scalar oracle on every YOLO26 shape, with the real cache-model blocking. That includes the short-last-`ic`-block barrier path wherever `m` isn't a multiple of `mc`, and the MR fringe rows (m = 300 against MR = 16, 24, 32).
- **This is a correctness run only.** Its timings (`--tmin 0.2`) are not quoted anywhere.

**Part C verdict: PASS.** #63's timings came from kernels that compute the right answer at 8 threads.
