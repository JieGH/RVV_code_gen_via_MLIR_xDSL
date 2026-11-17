# Issue #83: preliminary figures (17 Nov 2026, ~15:40)

Vector against scalar packing, from the launches finished so far. **Not final:** scalar has 1 launch (paper 1T) or 2 launches (paper 8T, YOLO 8T) of 3; YOLO 1T not run yet. The final report replaces these.

All numbers: best tile per shape, GOPs by mean time, averaged over finished launches; uninstrumented build, reference protocol. Vector: #61, #63, #71 logs (3 launches). Paper vector ran on `jbpi`, paper scalar on `jbpi2`; YOLO both on `jbpi`.

- `fig1_paper_vector_vs_scalar.png`: BERT (B1-B5) and Square (S1-S5), 1T and 8T, head to head.
- `fig2_ratio_vs_reuse_n.png`: scalar/vector against n (uses of each packed A element), all 8T shapes.
- `fig3_yolo8t_per_layer.png`: the 50 YOLO layers at 8T, head to head.
- `vector_vs_scalar_tables.md`: the tables behind them (also GOPs by median rep).
