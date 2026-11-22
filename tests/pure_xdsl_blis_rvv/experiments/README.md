# experiments/: past investigations

One folder per past issue: its run script, configs, report and tests, kept so its results can be reproduced. You don't need these for normal use. The write-up of each is in `.scratch/issues/` and on GitHub; [the issue index](../../../.scratch/issues/README.md) maps each folder to its write-up, which has a different (local) number. Inputs made for one experiment are in `../datasets/experiments/issueNN_*.dat` (issue #146).

Run a script from the repo root; most take `--help` and `--dry-run`:

```bash
./tests/pure_xdsl_blis_rvv/experiments/issue48/run_issue48.sh --dry-run
```

| Folder | Question |
|---|---|
| `issue34/` | Why B4 is slower than the square shapes on the same tile |
| `issue35/` | How a smaller `nc` affects B3 and the small-M layers |
| `issue36/` | Why the vector packer is slow on B3 |
| `issue37/` | 32x6 against 32x7 for B4 and S1-S5 |
| `issue41/` | Why B3 falls into a slow state (L2 counters per rep) |
| `issue43/` | Clearing pack-buffer lines with `cbo.zero` before packing |
| `issue46/` | Whether the Reference Design's B3 also has two speeds |
| `issue48/` | Sweep of microkernel tiles on B3, packing and compute apart |
| `issue49/` | Where B's read streams land in memory (B3_K256) |
| `issue50/` | Prefetcher counters at the tiles that flip |
| `issue51/` | How small M must be: M sweep at fixed tiles |
| `issue53/` | Scalar against vector packer on the #48 shapes |
| `issue54/` | Which B-packer read order sets off B3's slow state |
| `issue55/` | B-packer variants in C intrinsics (`--pack-b`) and a pack-only loop |
| `issue63/` | 8-thread Tile Search on YOLO, against Héctor's GEMMLib |
| `issue83/` | YOLO and paper sweeps with the scalar packer |
| `issue85/` | When vector packing pays off: sweep of m, n, k, tile, threads |
| `issue94/` | YOLO GEMM at 1 and 8 threads: vector, scalar and Héctor's |
| `issue95/` | Why our YOLO GEMM is slower than Héctor's |
| `issue112/` | Mixed precision f16f16f32: correctness and hand-over kernels |
