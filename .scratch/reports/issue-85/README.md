# Issue #85: when does vector packing pay off?

Sub-issue of #83. Vector against scalar packer on one-factor sweeps around a YOLO-like base point (m = 1200, n = 16 or 256, k = 256, tile 24x8), at 1 thread (jbpi, core 4) and 8 threads (jbpi2, cores 0-7). Pack and compute are timed at both thread counts.

- **Brief (PDF):** [`report/issue85_brief.pdf`](report/issue85_brief.pdf) ([LaTeX](report/issue85_brief.tex)): answer, decision table, glossary, one chart per sweep and thread count, model check, appendix with all GOPs.
- **All tables:** [`report/tables.md`](report/tables.md)
- **Script:** [`report/make_tables_and_figures.py`](report/make_tables_and_figures.py) `<this folder>`. Reads `jbpi_1t/` and `jbpi2_8t/` (`runs.tsv` + `runs/<run>/all_raw_*.txt`), writes `tables.md` (stdout) and `report/data/`.

## Answer

- **Vector packing pays off when n is small and the packed blocks are small.** The slowdown from scalar packing = pack share × extra pack cost:
  - **n sets the pack share.** At k = 256 (1T): 87% at n = 4, 77% at n = 16, 48% at n = 64, 7% at n = 1024. 8T is the same within a few points.
  - **k and m set the extra pack cost.** Scalar packing costs 1.2-3.4× the vector pack time at k ≤ 64 (1.9× at m ≤ 128, 1T), but only 1.02-1.12× at k ≥ 144, where both packers wait on memory.
- **Largest gains** (scalar ÷ vector GOPs): 1200×4×16 on every tile, 0.67-0.77 at 1T and 0.73-0.81 at 8T; 1200×16×16, 0.77 / 0.80; 64×16×256 (1T), 0.73.
- **n = 16 with large k:** small, steady gain: 0.92-0.94 (1T), 0.93-0.97 (8T). Pack share is 80%, but the extra cost is only 1.02-1.10×.
- **n ≥ 128 at 8T:** no difference (0.97-1.06, mostly inside the noise), except one case below.
- **Tile hardly matters:** the shape sets the ratio (0.67-0.77 over five tiles at 1200×4×16; 0.90-0.95 at 1200×16×256).
- **The #83 model holds:** predicted (from the timed split) is within 0.02 of measured on 25 of 46 points per thread count, and within 0.05 on 38 (1T) and 36 (8T).
- **Two places where vector packing measures slower.** Both repeat in every launch, and in neither is the pack itself slower:
  1. **1T, tile 24x8, m = 1200-2400, n ≥ 128, k ≥ 144:** scalar is 3-17% faster on 6 points (4 above noise). The vector pack is still 1.05-1.12× faster. The **compute** phase after vector packing is 5-10% slower at n = 128, 512 and 1024. Cause not measured.
  2. **8T, 1200×256×256, tile 16x4:** scalar 35.3 vs vector 24.8 GOPs, in all 6 launches. In the instrumented build both take the same time (5.25 / 5.21 ms). So the loss comes from the uninstrumented binary or the reference protocol. Not tested further.
- **Threads shrink the gain** at small n (n = 16, k = 256: 0.925 at 1T, 0.944 at 8T): each thread packs only its own rows of A.
- **8T/1T above 8×:** only 1200×16×64 (10.0× vector, 9.6× scalar). Its 1T pack share jumps from 42% (k = 32) to 75% (k = 64); the 8T share stays at 47-51%.

### Decision table

| region | scalar ÷ vector | use |
|---|---|---|
| n ≤ 16 and k ≤ 64 (any m, any tile) | 0.67-0.89 | **vector packing** (12-50% more GOPs, 1T and 8T) |
| n ≤ 16, m ≤ 128 (1T) | 0.73-0.74 | **vector packing** |
| n ≤ 32, k ≥ 144 | 0.90-0.97 | **vector packing** (3-11%, above noise) |
| n = 256, m ≤ 300 (1T) | 0.92-0.96 | **vector packing** (4-9%) |
| n = 64 | 0.97-0.98 | either (2-3%, at the noise at 1T) |
| n ≥ 128, 8T | 0.97-1.06 | either (mostly inside the noise) |
| n ≥ 128, k ≥ 144, m = 1200-2400, 1T, tile 24x8 | 1.03-1.17 | scalar measures faster; the loss is in compute, not packing |
| 1200×256×256, 8T, tile 16x4 | 1.42 | scalar measures faster, uninstrumented build only |

## Glossary

| term | meaning |
|---|---|
| **vector / scalar packer** | Copies a block of A or B into a contiguous buffer before the microkernel reads it. Vector: RVV loads/stores, 8 fp32 per instruction. Scalar (`--scalar-pack`): one float per load/store. Same layout and buffers. |
| **tile MRxNR** | The microkernel's block of C: MR rows by NR columns. |
| **m × n × k** | C (m × n) += A (m × k) · B (k × n). nc = n at every point, so each packed A element is used n times. |
| **pack share** | Pack time ÷ (pack + compute), from the instrumented build. A and B together. |
| **extra pack cost** | Scalar pack time ÷ vector pack time. |
| **predicted** | (vector pack + compute) ÷ (scalar pack + compute): the slowdown if only pack time changes. |
| **wait share (8T)** | 1 − (pack + compute) ÷ wall. Pack and compute are means over threads, so the rest is barrier waits. |
| **GOPs by mean time / by median rep** | 2·m·n·k ÷ mean time per call / median over reps of each rep's GOPs. |
| **s/v** | Scalar GOPs ÷ vector GOPs, by mean time, mean of 3 launches each. Below 1: vector packing is faster. |
| **launch** | One full run of the test (every point). M1: 3 launches per packer. |
| **A-B-B-A** | Run order that interleaves the packers. M1: V S S V V S; M2/M3: V S S V. |
| **~ / noise band** | The gap is smaller than the wider of the two packers' launch ranges (min-max), so not a real difference. |
| **1T / 8T** | 1 thread on core 4 (jbpi) / 8 threads on cores 0-7 (jbpi2, user's #56 exception). |
| **uninstrumented build** | `INSTRUMENTED=0`, no timers. All GOPs come from it (M1). |
| **instrumented build** | `--instr`: pack/compute timers (M2 1T, M3 8T). Timers cost 11-17%; only shares and ratios are used. |
| **reference / warm-up protocol** | Reference: no warm-up, reps until `--tmin` (3 s) pass (M1). Warm-up: warm-up, then 200 reps (M2/M3). |

## Runs

Idle boards (other users checked before every launch), `--no-nc-overrides`. Logs: `tests/pure_xdsl_blis_rvv/logs/`, copies of the raw tables in `jbpi_1t/runs/` and `jbpi2_8t/runs/`, launch order in each `runs.tsv`, driver output in [`driver_logs/`](driver_logs/).

| step | board | runs |
|---|---|---|
| correctness 1T (oracle on), vector / scalar | jbpi | `run_2026-11-17_214311` (30/30 PASS), `_215328` (30/30 PASS) |
| correctness 8T | jbpi2 | `run_2026-11-17_214206` (30/30 PASS), `_215120` (30/30 PASS) |
| M1 1T, GOPs, `--tmin 3`, V S S V V S | jbpi | `run_2026-11-17_220548`, `_221808`, `_223028`, `_224248`, `_225509`, `_230730` |
| M1 8T | jbpi2 | `run_2026-11-17_220239`, `_221407`, `_222532`, `_223700`, `_224829`, `_225956` |
| M2 1T, split, V S S V | jbpi | `run_2026-11-17_232351`, `_234012`, `_235635`, `run_2026-11-18_001255` |
| M3 8T, split | jbpi2 | `run_2026-11-17_231101`, `_232200`, `_233258`, `_234355` |

Correctness ran on small shapes under the oracle limit (`datasets/issue85_correctness_small.dat`). A first series at `--tmin 2` was stopped at the user's request and restarted at `--tmin 3`; its partial driver output is in `aborted_2026-11-17_2120_tmin2/`, and no number here comes from it.

## Limits

- M2 and M3 are instrumented (timers cost 11-17%), with 2 launches per packer: shares and ratios only, no launch-range check.
- 1T on jbpi, 8T on jbpi2 (same SoC): the 8T/1T column crosses boards.
- Pack time is A and B together. The sweeps separate them: P-n changes how often A is repacked, P-m how far B's pack is spread.
- Synthetic shapes around a YOLO-like base point; one tile (24x8) outside P-tile.
- Two packers compared, not packing against no packing.
