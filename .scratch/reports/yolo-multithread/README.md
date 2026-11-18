# YOLO multi-thread: all findings on Héctor's YOLO tests

Owner issue: #58 (parent, https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/58). This folder holds the one report to read first. The per-issue folders (`../issue-<N>/`) keep their own reports, data and logs. Nothing was moved; the combined report reads their data in place.

## Read this

- **Combined report (PDF, A4, 34 pages):** [`report/yolo_report.pdf`](report/yolo_report.pdf) ([LaTeX](report/yolo_report.tex)).
  - Parts 1–4 come from the #94 final report: the YOLO tile search (1T / 8T), ours against Héctor's (#94 rerun), the paper shapes and vector against scalar packing.
  - Part 5 comes from #95: why ours trails Héctor's at 1 thread.

## Index, newest first

| issue | what it answers | report |
|---|---|---|
| #96 | pack A column by column (the fix for #95) | pending, not started |
| #95 | why ours trails Héctor's at 1T: our A packer's read order | [`../issue-95/report/issue95_report.pdf`](../issue-95/report/issue95_report.pdf); Part 5 here |
| #94 | ours (vector / scalar packer) against Héctor's, 1T and 8T, one series | [`../issue-94/report/issue94_report.pdf`](../issue-94/report/issue94_report.pdf); Part 1 §1.2 here |
| #91 | all findings up to #85 (YOLO, paper shapes, packing) | `../issue-91/report/`; Parts 1–4 here, with §1.2 replaced by #94 |
| #83, #85 | vector against scalar packing | `../issue-83/report/`, `../issue-85/report/`; Part 3 here |
| #71 | paper BERT/Square shapes | Part 2 here |
| #56, #61–#63, #79 | parallel GEMM, 1T and 8T tile search, Héctor's GEMM | `../issue-56/`, `../issue-62/`, `../issue-63/`, `../issue-79/`; Part 1 here |

## Files

- `report/yolo_report.tex`: main file. Every GOPs and ms value is by mean time only (user's choice, 2026-11-18; `report/metric_gap.py` shows how far the median rep differs). It inputs:
  - `r0_answer.tex`: copy of #94's, plus the #95 answer and glossary terms.
  - `r1_yolo.tex`, `r2_paper.tex`, `r3_packing.tex`: copies of #94's / #91's Parts 1–3, with the median-rep values removed. The #94 final report reads these too.
  - `r5_appendix.tex`: copy of #94's appendix, reading the mean-only data below.
  - `r4_close.tex`: copy of #94's. The short #95 note now points to Part 5, and the file inputs `r5_why1t.tex`.
  - `r5_why1t.tex`: Part 5, the #95 report body (mean time only).
  - `r6_appendix95.tex`: #95's all-50-shapes table and logs.
- `report/make_data.py` → `report/data/`: mean-only copies of the #91 data files that hold "mean / median" cells (the ~ mark stays).
- `report/metric_gap.py` → `report/data/metric_gap_rows.tex`: GOPs by median rep ÷ GOPs by mean time per case, per run group (min, median, 95th, max). Not in the PDF; the reason only mean time is kept.
- Data read in place: `../issue-91/deck/data`, `../issue-94/report/data`, `../issue-95/report/data`. Build those first with each folder's `make_data.py` (see their READMEs).
- Mean time only also in `../issue-94/report`, `../issue-94/final` and `../issue-95/report` (their generators no longer write median rows). Older reports (`../issue-91`, `../issue-83`, `../issue-85`, #61–#63 READMEs) are kept as they were and still show both; this report supersedes them.

Build: from `report/`, run `python3 make_data.py`, then `tectonic -X compile yolo_report.tex`.

## When new YOLO work lands

1. Add a row at the top of the index.
2. Add or replace a Part in `report/`.
3. Rebuild the PDF.
4. Link this folder from the new issue.
