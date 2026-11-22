# datasets/: GEMM shapes for the bench

One shape per line: `label M N K`. A config names datasets by path relative to this folder (`["bert_b4.dat"]`, `["experiments/issue85_sweep.dat"]`). A config with no dataset list runs every `*.dat` at the top of this folder only, not the subfolders.

| Where | What |
|---|---|
| top level | shared workloads: BERT, Square, YOLO, paper and correctness sets |
| `experiments/` | inputs made for one issue, named `issueNN_*.dat`; used by `../experiments/issueNN/` or one config |
| `unused/` | shapes no config, script or doc names (issue #146); kept, not deleted |

## Renamed in issue #146

Old issue files, reports and logs still use the old names.

| Old name | New name | Holds |
|---|---|---|
| `issue2_bert_16x15.dat` | `bert_b2_b3.dat` | BERT B2, B3 (tile 16x15) |
| `issue2_bert_32x6.dat` | `bert_b1_b5.dat` | BERT B1, B5 (tile 32x6) |
| `issue2_square_bert_32x7.dat` | `square_bert_b4.dat` | Square S1-S5 and BERT B4 (tile 32x7) |
| `issue26_b4.dat` | `bert_b4.dat` | BERT B4 alone |
| `issue46_ref_b3.dat` | `b3_controls.dat` | B3 with controls B2, B3_K256, R24 |
| `issueNN_*.dat` (16 files) | `experiments/issueNN_*.dat` | one-issue inputs |
| `gnn`, `googlenet*`, `gpt2_large`, `resnet50*`, `vgg16*`, `square_small`, `tiny_test`, `trail` (`.dat`) | `unused/` | not named anywhere |
