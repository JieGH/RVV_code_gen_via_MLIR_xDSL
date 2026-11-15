# S2: Add microkernel size search to our design: time a chosen list of tiles on each shape and report the best

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/60

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Spec + build in **our design** (Project Generator), no board until the correctness check.
**Status:** 🔨 Built 2026-11-15 (decisions §3). Waiting on the correctness pass on a board.
**Blocked by:** #59 (S1, dataset and tile list).
**Primary sources:**
- `tests/pure_xdsl_blis_rvv/config.py:54` (8th field present = pinned, absent = search)
- `generate_pure_blis_project.py:330-445` (search mode builds every tile 1x1 .. max MR x max NR)
- `config_issue55_tiles.json` + `test_run_issue55.py` (several pinned rows in one config already run in one `compile_macro_bench.sh` call, one `pj_gen/..._MR<m>NR<n>` folder each)
- `templates/driver_benchmark.cpp:1335-1350` (prints the best tile per shape, ranked by `ker_gflops`)
- `config_issue48_sweep.json`, `run_issue48.sh`, `issue48_report.py`, `reps_report.py` (the pinned-row sweep we use today)

---

## 1. Goal

Our design should find the best microkernel size for each shape by itself. A user gives a list of candidate tiles and a dataset. The design builds those tiles, times each one on every shape with the same protocol as today, and reports the fastest tile per shape.

This is built **in our design**. Héctor's code plays no part in it.

## 2. What exists (read 2026-11-15)

**Search mode** (a row without `iterations`):
- builds every tile from 1x1 up to the largest MR and NR in the row. A list of `[[16,8],[32,7]]` builds all 32 x 8 = 256 tiles, each in beta 0 and beta 1. It cannot build only the tiles we listed.
- uses the small tiles as fringe kernels through a dispatch table. Pinned mode runs the full tile into a temporary C buffer instead. The two modes therefore time different fringe code.
- ranks by kernel GFLOPS, not by the whole-call time.
- has not been run in recent board work. Only `test_config.py` sets `is_pinned = False`.

**Pinned rows** (what #32, #48 and #51 used): one config row per tile, each in its own `pj_gen/..._MR<m>NR<n>` directory. Each row runs every shape in its datasets. Per-rep logs and `reps_report.py` already work with this, and so do the house rules (#39).

## 3. Decisions (grilling, 2026-11-15)

1. **No new row kind, no generator change.** A config with several pinned rows already runs today: one `compile_macro_bench.sh --config` call builds and runs every row, each in its own `pj_gen/..._MR<m>NR<n>` folder (`config_issue55_tiles.json`). So the search config is one pinned row per Candidate Tile. Each tile times exactly as it does alone. The "list of tiles on a pinned row" feature and its byte-identical test are dropped: nothing that runs today changes.
2. **The old search mode stays as it is.** Out of scope. Retiring it would be its own issue.
3. **`--tmin` moved to #61** (§2b there). It is a run flag, not part of what this issue builds.
4. **Blocking: our cache model only.** Our `nc` overrides set `nc` only, so Héctor's MC=256/KC=256 cannot be reproduced without new code. Noted as a limit in #63.
5. **`search_report.py` builds on `reps_report.py`** and needs the per-rep logs (on by default). A run folder without them is an error naming `--no-save-reps`.
6. **Glossary:** "Tile Search" added to `CONTEXT.md` (avoid "sweep", the old mode).
7. **Report content** as proposed: per shape the best tile by GOPs by mean time and by median rep (whole call, packing included), the launch spread, an optional compare tile, a WARNING under 3 launches; then the best single tile over the dataset and the best tile per n, by total measured time with each shape counted once.

## 4. What was built (2026-11-15)

- `tests/pure_xdsl_blis_rvv/config_yolo_search.json`: 19 pinned rows (the #59 tiles) on `yolo26_k1.dat`, `skip_ref_run: true`.
- `tests/pure_xdsl_blis_rvv/config_yolo_correctness.json` + `datasets/yolo_correctness_small.dat`: the same 19 tiles, oracle on, on 4 shapes (m = 300; n = 4, 16; k = 9, 27). Added to the B2-limit test in `test_correctness_pass.py`.
- `tests/pure_xdsl_blis_rvv/search_report.py`: usage `search_report.py <run folder> [...] --compare 16x8 --compare n=4:16x4`. Checked on a real run folder (`logs/run_2026-10-23_125339`, 3 tiles x 4 shapes).
- `tests/pure_xdsl_blis_rvv/test_tile_search.py`: both configs, and the report on hand-made run folders (best tile per shape by both metrics, compare tile, summary arithmetic, the WARNING under 3 launches, the missing-reps error).
- Docs: `README.md` ("Finding the best microkernel size"), `docs/agents/execution.md` (entry point), `CONTEXT.md` (Tile Search).

## 5. Later, not in this issue

- Let the search pick its candidates from the register budget instead of a hand list.
- Rank with 8 threads once #56 exists.

## 6. Closure criteria

- [x] Decisions recorded here (§3).
- [x] ~~Search built in `config.py` / the generator~~ Not needed (§3.1): plain pinned rows.
- [x] `search_report.py` and its test committed.
- [x] `config_yolo_search.json` and `config_yolo_correctness.json` committed.
- [ ] The correctness config passes on one board (all 19 x 4 rows PASS).
- [x] Docs updated. Tests pass.
