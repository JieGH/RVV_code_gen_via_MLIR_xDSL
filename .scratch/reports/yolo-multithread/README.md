# YOLO multi-thread: all findings on Héctor's YOLO tests

Owner issue: #58 (parent, https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/58). This folder holds the one report to read first. The per-issue folders (`../issue-<N>/`) keep their own reports, data and logs. Nothing was moved; the combined report reads their data in place.

## Read this

- **Combined report v2 (PDF, A4):** [`report/yolo_report_v2.pdf`](report/yolo_report_v2.pdf) ([LaTeX](report/yolo_report_v2.tex)). §1.3 holds the #109 head-to-head (series `issue94_2026-11-19_093036`).
  - Part 1: the best single tile per thread count (#109 re-search: 1T 24x8, 8T 32x7), the earlier fixed-24x8 run (#94), the 1T / 8T tile search with accumulated-GOPs summary tables, Héctor's GEMM.
  - Parts 2–4: the paper shapes, vector against scalar packing, what to tell Héctor.
  - Part 5 comes from #95: why ours trails Héctor's at 1 thread.
- Old v1 (superseded by v2, kept for reference): [`report/yolo_report_v1_old.pdf`](report/yolo_report_v1_old.pdf).

## Index, newest first

| issue | what it answers | report |
|---|---|---|
| #109 | best single tile per thread count, re-searched on all 53 layers (1T 24x8, 8T 32x7); ours at those tiles against Héctor's | v2 Part 1 §1.2 (re-search), §1.3 (head-to-head); logs `../issue-109/` |
| #108 | n = 1 layers (Y06, Y21, Y35); stopped, superseded by #109 | logs `../issue-108/`, not in the report |
| #96 | pack A column by column (the fix for #95) | pending, not started |
| #95 | why ours trails Héctor's at 1T: our A packer's read order | [`../issue-95/report/issue95_report.pdf`](../issue-95/report/issue95_report.pdf); Part 5 here |
| #94 | ours (vector / scalar packer) against Héctor's, 1T and 8T, one series | [`../issue-94/report/issue94_report.pdf`](../issue-94/report/issue94_report.pdf); Part 1 §1.2 here |
| #91 | all findings up to #85 (YOLO, paper shapes, packing) | `../issue-91/report/`; Parts 1–4 here, with §1.2 replaced by #94 |
| #83, #85 | vector against scalar packing | `../issue-83/report/`, `../issue-85/report/`; Part 3 here |
| #71 | paper BERT/Square shapes | Part 2 here |
| #56, #61–#63, #79 | parallel GEMM, 1T and 8T tile search, Héctor's GEMM | `../issue-56/`, `../issue-62/`, `../issue-63/`, `../issue-79/`; Part 1 here |

## Files

- `report/yolo_report_v2.tex`: main file of v2. It inputs `v2_r0_answer.tex`, `v2_r1_yolo.tex` (Part 1, rewritten for #109), `r2_paper.tex`, `r3_packing.tex`, `v2_r4_close.tex`, `v2_r5_appendix.tex`, `r6_appendix95.tex`. Floats are flushed at every (sub)section, so tables and charts stay in their section.
- `report/make_data_v2.py` → `report/data_v2/`: #109 re-search rows (`search1/8_rows.tex`) and the accumulated-GOPs tables of §1.5 / §1.6 (`cum1/8_rows.tex`, from the raw #61 / #63 run logs).
- `report/heatmap_trial/`: s/v heatmaps for §3.6 (n × k at m = 1200, n × m at k = 256; 1T and 8T), drawn from the #85 tables in `report/data/`; only 21 of 63 cells per chart have data. `make_heatmap.py ../data` writes the CSVs that v2 reads through `report/v2_r3_heatmap.tex` (hook `\svheatmaps` in the shared `r3_packing.tex`; v1 and #94 final do not define it, so they are unchanged). `sv_heatmap_trial.tex` is the standalone trial PDF.
- `report/yolo_report_v1_old.tex`: v1 main file. Every GOPs and ms value is by mean time only (user's choice, 2026-11-18; `report/metric_gap.py` shows how far the median rep differs). It inputs:
  - `r0_answer.tex`: copy of #94's, plus the #95 answer and glossary terms.
  - `r1_yolo.tex`, `r2_paper.tex`, `r3_packing.tex`: copies of #94's / #91's Parts 1–3, with the median-rep values removed. The #94 final report reads these too.
  - §1.5 (#108): how the best single tile is picked; charts from `make_data.py` → `data/single_tile_figs.tex` (pass times from the #61 and #63 sweep reports).
  - `r5_appendix.tex`: copy of #94's appendix, reading the mean-only data below.
  - `r4_close.tex`: copy of #94's. The short #95 note now points to Part 5, and the file inputs `r5_why1t.tex`.
  - `r5_why1t.tex`: Part 5, the #95 report body (mean time only).
  - `r6_appendix95.tex`: #95's all-50-shapes table and logs.
- `report/make_data.py` → `report/data/`: mean-only copies of the #91 data files that hold "mean / median" cells (the ~ mark stays).
- `report/metric_gap.py` → `report/data/metric_gap_rows.tex`: GOPs by median rep ÷ GOPs by mean time per case, per run group (min, median, 95th, max). Not in the PDF; the reason only mean time is kept.
- Data read in place: `../issue-91/deck/data`, `../issue-94/report/data`, `../issue-95/report/data`. Build those first with each folder's `make_data.py` (see their READMEs).
- Mean time only also in `../issue-94/report`, `../issue-94/final` and `../issue-95/report` (their generators no longer write median rows). Older reports (`../issue-91`, `../issue-83`, `../issue-85`, #61–#63 READMEs) are kept as they were and still show both; this report supersedes them.

Build v2: from `report/`, run `python3 make_data.py`, then `(cd heatmap_trial && python3 make_heatmap.py ../data)`, then `python3 make_data_v2.py --search-1t ../../../../tests/pure_xdsl_blis_rvv/logs/yolo_sweep_2026-11-19_002730 --search-8t ../../../../tests/pure_xdsl_blis_rvv/logs/yolo_sweep_2026-11-19_014357 --tiles 24x8,32x7 --series ../../../../tests/pure_xdsl_blis_rvv/logs/issue94_2026-11-19_093036` (without it the §1.3 charts and by-n table are not rewritten), then `tectonic -X compile yolo_report_v2.tex`.

## When new YOLO work lands

1. Add a row at the top of the index.
2. Add or replace a Part in `report/`.
3. Rebuild the PDF.
4. Link this folder from the new issue.
