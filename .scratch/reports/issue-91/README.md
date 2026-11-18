# Issue #91: one deck for the YOLO, BERT/Square and packing findings

GitHub: https://github.com/JieGH/RVV_code_gen_via_MLIR_xDSL/issues/91 · issue write-up: `.scratch/issues/0052-present-yolo-bert-square-and-packing-findings.md`.

- **Deck (PDF):** [`deck/issue91_deck.pdf`](deck/issue91_deck.pdf) ([LaTeX](deck/issue91_deck.tex), sections `s1_setup_yolo.tex`, `s2_paper.tex`, `s3_packing.tex`, `s4_close.tex`)
- **Data:** `deck/data/` (CSV files name their source in a `#` line); packing numbers from [`deck/make_data.py`](deck/make_data.py) → `deck/data/packing_tables.md`, `deck/data/p85_tables.md`, `deck/p85/data/`.
- **No new board runs.** Every number comes from committed logs of #56, #61, #62, #63, #71, #79, #83, #85.

Build: from `deck/`, `python3 make_data.py` (run from the repo root, see its docstring), then `tectonic -X compile issue91_deck.tex`.

## Answers

1. **YOLO tiles (Héctor, #58).**
   - 1T: 24x8 for the whole network, 4.26 GOPs, against 3.79 for our GEMM at his 16x8 tile.
   - 8T, in our GEMM, a family by n:
     - MR 32 / NR 4 for n ≤ 8
     - MR 32 / NR 6-7 for n = 16-64
     - MR 16 / NR 12-15 for n ≥ 80
   - Best tile per shape: 134.1 ms per pass, against 145.6 ms for the best single tile.
2. **Paper shapes (#71).**
   - 1T: 32x7 (16x15 for B2) still wins among our 19 tiles.
   - 8T: 16x8 wins 8 of 10 shapes.
3. **Packing (#83, #85).** Keep the vector packer.
   - YOLO: scalar ÷ vector median 0.929 (1T) and 0.945 (8T), down to 0.58 at n = 4.
   - BERT/Square: within the 3% board/day gap, except B3 at 1T (1.227), where the vector packer is slow (#36).
   - Slowdown ≈ pack share (set by n) × extra pack cost (set by k and m).

## Metric correction (affects #83 and #85)

**What was wrong.** The #83 and #85 scripts read `KerGOPs_Mean` and called it "GOPs by mean time". That column is the average of each rep's GOPs. The house metric is 2mnk ÷ mean time per call, `KerGOPs_Harmonic`. #61, #63 and #71 use the house metric.

**What the deck does.** It recomputes every packing number from the same runs with `KerGOPs_Harmonic` (`make_data.py`).

**How much the numbers move:**

| | #83 / #85 report | true mean time |
|---|---|---|
| YOLO s/v median, 1T / 8T | 0.930 / 0.946 | 0.929 / 0.945 |
| YOLO best tile changes, 1T / 8T | 14 / 17 | 13 / 20 |
| BERT/Square s/v median, 8T | 0.987 | 0.992 |
| #85 model within 0.02, 1T / 8T | 25 / 25 of 46 | 24 / 20 of 46 |
| #85 8T, 1200×16×256 | 0.944 | 0.982 (inside the launch range; 0.94 by median rep) |
| #85 8T, 1200×256×256 16x4 | 1.42 | 1.46 |
| #83 YOLO Y34, 8T | 0.921 | 0.818 (0.932 by median rep) |
| #83 B3, 1T | 1.194 (called noise) | 1.227, steady: vector best 32x7 7.01, scalar 16x15 8.60 |

- **1T:** ratios move by up to 0.02 (YOLO) and 0.03 (B3).
- **8T:** a few slow reps per launch hit the mean time. Some ratios move by up to 0.10, and some gaps fall inside the launch range.
- **Still open:** the #83 and #85 reports themselves are not corrected yet.

## Essential test (not run)

- **The test:** our tile family inside Héctor's own GEMM (#82 §1). It is needed before "use these tiles" applies to his code.
- **When to skip it:** if he meant best tiles in any parallel GEMM.
