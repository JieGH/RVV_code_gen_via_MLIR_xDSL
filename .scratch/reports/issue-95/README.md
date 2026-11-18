# Issue #95: why our YOLO GEMM trails Héctor's at 1 thread

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/95 · issue write-up: `.scratch/issues/0054-why-our-yolo-gemm-is-slower-than-hectors.md` · pending fix: #96 (`.scratch/issues/0055-pack-a-column-by-column.md`).

- **Report (PDF, A4, 5 pages):** [`report/issue95_report.pdf`](report/issue95_report.pdf) ([LaTeX](report/issue95_report.tex)).
- **Data:** [`report/make_data.py`](report/make_data.py) → `report/data/`. It reads only committed logs: `tests/pure_xdsl_blis_rvv/logs/issue95_probe_*`, `logs/issue95_pack_order_50_*.txt`, and #94's `shape_rows.tex`.
- **Tools:**
  - `tests/pure_xdsl_blis_rvv/probe_issue95.sh` runs one shape, ours then Héctor's, at 1T on core 4.
  - `tests/pure_xdsl_blis_rvv/issue95/pack_order.c` times both A read orders on the board.
  - `RVV_FORCE_MC` is an opt-in mc knob, off by default.

Build: from `report/`, `python3 make_data.py`, then `tectonic -X compile issue95_report.tex`.

## Answer

- **Main cause:** our A packer reads A panel by panel. That means a jump of one full column per read. Héctor's packer reads it column by column. Same bytes, same layout.
- **The pack is most of our time** (instrumented build, shares only): 77% on Y10 and 90% on Y05. Our compute on Y10 alone is about 7.8 GOPs, above Héctor's whole call (5.1).
- **Pack test** (both orders on the board, same blocks as our GEMM): 2.19 against 0.71 ms (Y10), 34.69 against 12.13 ms (Y00).
- **Where the gap is:** the 1T gap (#94) is 131.0 of 140.5 ms on the 23 shapes with m > 300 and k ≤ kc. The test's order cost there is 162.2 ms, the same size.
  - The test overstates absolute pack times. On Y01, Y05, Y09 and Y11 it reads above our whole call. So the exact gain needs the in-GEMM change (#96).
- **Not the main cause:**
  - mc: forced to 240, it gives +25% on Y10 only.
  - Our microkernel.
