# S2: Add microkernel size search to our design: time a chosen list of tiles on each shape and report the best

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/60

**Role / Label:** `needs-triage`. Sub-issue of #58 ([[0041-yolo-best-microkernel-size-parent]]). Spec + build in **our design** (Project Generator), no board until the correctness check.
**Status:** 📝 Open. Not built. Updated 2026-11-15: the search is a feature of our design, not a set of side scripts.
**Blocked by:** #59 (S1, dataset and tile list).
**Primary sources:**
- `tests/pure_xdsl_blis_rvv/config.py:54` (8th field present = pinned, absent = search)
- `generate_pure_blis_project.py:330-445` (search mode builds every tile 1x1 .. max MR x max NR)
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

## 3. Rule: nothing that runs today changes

- A pinned row with one tile builds and times exactly as today: same generated code, same driver path, same log names. Every result since #39 stays valid.
- The old search mode (a row without `iterations`) stays as it is.
- The new search is a new kind of row. It is turned on only by that row.
- **Test:** the generated project for an existing pinned row is byte-identical before and after this change (e.g. `config_issue48_sweep.json`, `config.json`).

## 4. Design (proposal for review)

1. **Config: a list of tiles on a pinned row.** Today a row with 8 fields and a list of tiles silently keeps only the first tile (`config.py:115`, `mr_nr_family[0]`). Change it so that such a row means "search over exactly these tiles". Each tile uses the same pinned fringe path as today (full tile into the temporary C buffer), so a tile times the same whether it runs alone or in a search.
   - **Option (a):** one project directory per tile, run one after another (the generator loops over today's pinned path). The least new code.
   - **Option (b):** one project holding all listed tiles as variants of one binary. The driver already loops over variants and prints a "best variant". One build, but the generated code differs from a single-tile project.
   - Proposed: **(a)**. It reuses the pinned path unchanged and keeps rule §3 easy to prove.
2. **Ranking:** by GOPs by mean time of the whole call (packing included), with GOPs by median rep beside it. Not by kernel GFLOPS.
3. **Report `search_report.py`** (general, not YOLO-only). It reads the run folder(s) of a search and prints, per shape:
   - the best tile by both metrics
   - the spread over launches, and a WARNING under 3 launches (reuse `reps_report.py`)
   - an optional "compare with" tile (for YOLO: Héctor's 16x8, or 16x4 when n = 4)
   - a summary: the best single tile for the whole dataset (FLOP-weighted), and the best tile per group of shapes (e.g. per n)

   Add a test with a fake log folder.
4. **Entry point:** `compile_macro_bench.sh --config <search config>` runs it with no new flag. Document it in `README.md` and `docs/agents/execution.md`.
5. **Configs for YOLO:**
   - `config_yolo_search.json`: one row, the 19 S1 tiles, `yolo26_k1.dat`, `skip_ref_run: true`
   - `config_yolo_correctness.json`: the same tiles on 3-4 small YOLO-like shapes (n = 4 and 16, m = 300, k = 9 and 27), every shape under the B2 limit of the scalar check (`test_correctness_pass.py`)
6. **Later, not in this issue:** let the search pick its candidates from the register budget instead of a hand list, and rank with 8 threads once #56 exists.

## 5. Open points for the user

- Option (a) or (b) in §4.1?
- `--tmin`: Héctor uses `TIMIN=2.0`. Our default for decisions is 10 s. With 50 small shapes x 19 tiles, `--tmin 10` costs about 19 x 50 x 10 s = 2.6 h per launch, against about 32 min at `--tmin 2`.
- Blocking: keep our cache-model `mc/nc/kc` (proposed), or also time with Héctor's fixed MC=256, KC=256, NC=n through `nc` overrides?
- The old search mode: keep it, or retire it once the new one works?

## 6. Closure criteria

- [ ] Decisions on §5 recorded here.
- [ ] Search built in `config.py` / the generator, with the byte-identical test for existing pinned rows (§3).
- [ ] `search_report.py` and its test committed.
- [ ] `config_yolo_search.json` and `config_yolo_correctness.json` committed. The correctness config passes on one board (all tiles PASS).
- [ ] Docs updated. Tests pass. Run them before or after a board series, never during one.
