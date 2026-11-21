- `correctness/`: oracle-on correctness passes on small shapes. `config_feature_*.json` are the extra data types and search mode of `../runners/run_feature_baseline.sh`.
- `tile_search/`: one pinned row per Candidate Tile, read by `../readers/search_report.py`.
- `benchmark/`: fixed-tile timing runs.

`config.json` at the top level is the default of `compile_macro_bench.sh`.
