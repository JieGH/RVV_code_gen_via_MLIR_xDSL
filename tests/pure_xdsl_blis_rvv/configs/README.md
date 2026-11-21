# configs/: what to generate and run

Pass one to the bench with `--config` (path relative to `tests/pure_xdsl_blis_rvv/`):

```bash
./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh $CONN --no-nc-overrides --cpu 4 \
  --config configs/correctness/config_correctness.json --tmin 0.2
```

Without `--config` the bench runs `../config.json`. The row format (data type, method, tile, datasets, pinned or search): the [repo README](../../../README.md#choosing-what-to-generate).

| Folder | Use | Main configs |
|---|---|---|
| `correctness/` | the scalar reference check is on; small shapes; run with `--tmin 0.2`; every row must say `PASS` | `config_correctness.json` (fp32, 3 tiles), `config_issue112_mixed_correctness.json` (f16f16f32, with `--scalar-pack`), `config_feature_*.json` (DOT, int8, search mode: used by `../runners/run_feature_baseline.sh`), `config_yolo*_correctness*.json`, `config_paper_correctness.json` |
| `tile_search/` | one pinned row per candidate tile; read with `../readers/search_report.py` | `config_yolo_search.json`, `config_paper_search.json`, `config_yolo53_search_{1t,8t}.json` |
| `benchmark/` | fixed tiles, timing only | `config_pack_swap*.json` (`run_pack_swap.sh`), `config_parallel_gemm.json` (`run_parallel_gemm.sh`), `config_yolo53_*.json` |
