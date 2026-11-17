# Issue #63: 8-thread Tile Search on YOLO26, and Héctor's GEMMLib at 8 threads (2026-11-16/17)

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/63 · **engineering report: [`report/issue-63-report.md`](report/issue-63-report.md)** · issue write-up: `.scratch/issues/0046-yolo-eight-thread-sweep.md`.

**What ran:** on `jbpi`, 8 threads on cores 0-7 (core 0: the user's 8-thread exception), A-B-B-A: ours L1, his, his, ours L2, ours L3, his. Ours: 19 tiles x 50 shapes, uninstrumented, reference protocol, `--tmin 2`, `--no-nc-overrides`. His: `gemm.x` at `b292c92`, unedited, `TIMIN=2.0`. No other user's job on the board.

## Headline (one pass over the 50 shapes, GOPs by mean time)

| | ms | GOPs |
|---|---|---|
| ours 8T, best single tile 16x12 | 145.6 | 20.43 |
| ours 8T, best tile per shape | 134.1 | 22.17 |
| his 8T | 183.7 | 16.18 |
| ours 8T best per shape, without Y00 | 127.9 | 22.74 |
| his 8T, without Y00 | 123.4 | 23.56 |

His Y00 (76800x16x27) swung 12.9 → 62.6 → 105.5 ms between launches; without it his GEMM is slightly ahead. Best tile per n at 8T: 32x4 (n = 4, 8), 32x7 (16, 32), 32x6 (64), 16x15 (80, 128), 16x14 (256).

## Folders

| folder | what |
|---|---|
| `yolo_sweep_2026-11-16_211823/`, `yolo_sweep_2026-11-16_223529/`, `yolo_sweep_2026-11-16_234634/` | ours, launches 1-3 (each with its one-launch report and heatmap) |
| `hector_gemm_2026-11-16_222943/`, `hector_gemm_2026-11-16_223301/`, `hector_gemm_2026-11-17_005723/` | his, launches 1-3 |
| `report/` | the reports over all 3 launches per side, the side-by-side tables, the heatmap |
