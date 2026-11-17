"""Issue #79 part B4: where do our slow 8-thread reps come from?

For each shape (one tile), splits reps into "typical" (<= 1.5x the median) and "slow" (> 1.5x),
and reports how many are slow, how much slower, and whether they come in runs (consecutive reps).
Usage: rep_tails.py <run dir> <tile, e.g. MR16_NR8> [shape ...]
"""
import csv
import gzip
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

run, tile = Path(sys.argv[1]), sys.argv[2]
want = set(sys.argv[3:])
f = next((run / f"fp32_AXPY_FP32_Pack_vlen256_{tile.replace('_', '')}").glob("reps_*.gz"))
reps: dict[str, list[float]] = defaultdict(list)
for r in csv.DictReader(gzip.open(f, "rt")):
    if not want or r["Layer"] in want:
        reps[r["Layer"]].append(float(r["Seconds"]))

print("| shape | reps | median ms | mean ms | mean/median | slow reps (>1.5x median) | slow rep median ms "
      "| longest run of slow reps | share of time in slow reps |")
print("|---|---|---|---|---|---|---|---|---|")
for s, v in sorted(reps.items(), key=lambda kv: int(kv[0][1:])):
    md = st.median(v)
    slow = [x > 1.5 * md for x in v]
    run_len = best = 0
    for b in slow:
        run_len = run_len + 1 if b else 0
        best = max(best, run_len)
    sv = [x for x, b in zip(v, slow) if b]
    share = sum(sv) / sum(v)
    print(f"| {s} | {len(v)} | {md * 1e3:.3f} | {st.mean(v) * 1e3:.3f} | {st.mean(v) / md:.2f} | "
          f"{len(sv)} ({len(sv) / len(v):.1%}) | {st.median(sv) * 1e3 if sv else float('nan'):.3f} | {best} | {share:.1%} |")
