"""Build index.html, a standalone interactive view of the 19-20 Sep 2026 overnight runs.

Reads every logs/run_2026-09-19_23* and logs/run_2026-09-20_* folder (raw logs of the
reference-protocol build), computes GOPs = 2mnk / RVV_Time_ms per layer and run, embeds
the rows in template.html and writes index.html. Open index.html in a browser; it needs
no server and loads nothing from the network.

The scalar-pack sweep (run_scalar_pack_sweep.sh, issue 26) is embedded too: every run listed in
the runs.txt of the newest logs/sweep_scalar_* folder (or the one given with --sweep), so
the page can show the scalar packer next to the vector packer, layer by layer. A sweep that
is still running gives a partial set; run this again when more runs have finished.

The reference-design sweep (run_ref_sweep.sh, issue 29) is embedded as well: every result dir listed in
the runs.txt of the newest logs/sweep_ref_* folder (or the one given with --ref-sweep). The reference
alone, its own packers and kernels, so the page can show it next to our two packers.

Run from anywhere: python3 build_report.py [--sweep DIR] [--ref-sweep DIR]
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(LOGS), "ref_baseline"))
from compare_ref_sweep import load_ref_runs  # noqa: E402  (reads a refbase_* dir the way the comparison does)
# The overnight sweep, by run stamp. Later runs (the daytime ones, the vector-pack experiments) are
# not part of it, and would otherwise match the same glob.
OVERNIGHT = ("2026-09-19_230000", "2026-09-20_070000")
RUN_GLOBS = ["run_2026-09-19_23*", "run_2026-09-20_0[0-6]*"]

rows = []
for pattern in RUN_GLOBS:
    for d in sorted(glob.glob(os.path.join(LOGS, pattern))):
        prov = open(os.path.join(d, "provenance.txt")).read()
        board = re.search(r"board\s*:\s*\S+@(\w+)", prov).group(1)
        raw = glob.glob(os.path.join(d, "all_raw_*refproto_no_nc_override.txt"))
        if not raw:
            continue
        for _, r in pd.read_csv(raw[0]).iterrows():
            flops = 2.0 * r.M * r.N * r.K
            rows.append({
                "run": os.path.basename(d)[4:], "board": board, "case": r.Layer,
                "m": int(r.M), "n": int(r.N), "k": int(r.K),
                "cfg": r.Variant.replace("MR", "").replace("_NR", "x"),
                "gops_mean_time": flops / (r.RVV_Time_ms * 1e6), "median": r.KerGOPs_Median,
                "std": r.KerGOPs_Std, "mean": r.KerGOPs_Mean, "min": r.KerGOPs_Min,
                "p5": r.KerGOPs_P5, "p95": r.KerGOPs_P95, "max": r.KerGOPs_Max,
                "iqr": r.KerGOPs_IQR, "reps": int(r.N_Reps),
                "cv": r.KerGOPs_Std / r.KerGOPs_Mean * 100 if r.KerGOPs_Mean else 0.0,
            })
# ---- the scalar-pack sweep
ap = argparse.ArgumentParser()
ap.add_argument("--sweep", help="sweep_scalar_* folder (default: the newest)")
ap.add_argument("--ref-sweep", help="sweep_ref_* folder (default: the newest)")
args = ap.parse_args()
sweeps = sorted(glob.glob(os.path.join(LOGS, "sweep_scalar_*")))
sweep_dir = args.sweep or (sweeps[-1] if sweeps else None)
sweep_rows, sweep_meta = [], {"planned": 0, "done": 0, "note": ""}
if sweep_dir and os.path.isfile(os.path.join(sweep_dir, "runs.txt")):
    plan = os.path.join(sweep_dir, "plan.txt")
    m = re.search(r"(\d+) runs", open(plan).read()) if os.path.isfile(plan) else None
    sweep_meta["planned"] = int(m.group(1)) if m else 0
    for line in open(os.path.join(sweep_dir, "runs.txt")).read().splitlines():
        board, name = line.split("\t")
        raw = glob.glob(os.path.join(LOGS, name, "all_raw_*refproto_scalarpack_no_nc_override.txt"))
        if not raw:
            continue
        sweep_meta["done"] += 1
        for _, r in pd.read_csv(raw[0]).iterrows():
            sweep_rows.append({"run": name[4:], "board": board, "case": r.Layer,
                               "gops_mean_time": 2.0 * r.M * r.N * r.K / (r.RVV_Time_ms * 1e6), "median": r.KerGOPs_Median,
                               "p5": r.KerGOPs_P5, "p95": r.KerGOPs_P95, "max": r.KerGOPs_Max, "reps": int(r.N_Reps)})
sweep_rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in sweep_rows]

rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]

# ---- the reference-design sweep. Per layer the reference has one row per kernel it built: the exact
# one where it ships it (B1, B5), otherwise the two even-NR neighbours. `near` marks the one the chart
# draws: the exact kernel, else the lower-NR neighbour (the higher one needs more than 32 vector
# registers and spills, so it is not the close one).
ref_sweeps = sorted(glob.glob(os.path.join(LOGS, "sweep_ref_*")))
ref_dir = args.ref_sweep or (ref_sweeps[-1] if ref_sweeps else None)
ref_rows, ref_meta = [], {"planned": 0, "done": 0}
if ref_dir and os.path.isfile(os.path.join(ref_dir, "runs.txt")):
    plan = os.path.join(ref_dir, "plan.txt")
    m = re.search(r"(\d+) runs", open(plan).read()) if os.path.isfile(plan) else None
    ref_meta["planned"] = int(m.group(1)) if m else 0
    per_run = []
    for line in open(os.path.join(ref_dir, "runs.txt")).read().splitlines():
        board, name = line.split("\t")
        per_run.append((board, name, load_ref_runs([Path(LOGS) / name])))
        ref_meta["done"] += 1
    def near_built(layer):
        cands = {row.built: row.kind for _, _, rr in per_run for (lay, _), row in rr.items() if lay == layer}
        exact = [b for b, k in cands.items() if k == "exact"]
        return exact[0] if exact else min(cands, key=lambda b: int(b.split("x")[1]))
    near = {lay: near_built(lay) for lay in {lay for _, _, rr in per_run for (lay, _) in rr}}
    for board, name, rr in per_run:
        for (lay, built), row in rr.items():
            for g in row.gflops:
                ref_rows.append({"run": name[len("refbase_"):], "board": board, "case": lay, "built": built,
                                 "kind": row.kind, "near": built == near[lay], "gops": round(g, 4)})

body = (open(os.path.join(HERE, "template.html")).read()
        .replace("/*__DATA__*/[]", json.dumps(rows, separators=(",", ":")))
        .replace("/*__SWEEP__*/[]", json.dumps(sweep_rows, separators=(",", ":")))
        .replace('/*__SWEEP_META__*/{planned:0, done:0, note:""}', json.dumps(sweep_meta, separators=(",", ":")))
        .replace("/*__REFSWEEP__*/[]", json.dumps(ref_rows, separators=(",", ":")))
        .replace("/*__REFSWEEP_META__*/{planned:0, done:0}", json.dumps(ref_meta, separators=(",", ":"))))
page = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n' + body
open(os.path.join(HERE, "index.html"), "w").write(page)
print(f"{len(rows)} rows from {len({r['run'] for r in rows})} overnight runs, "
      f"{len(sweep_rows)} rows from {sweep_meta['done']} of {sweep_meta['planned']} scalar-pack sweep runs, "
      f"{len(ref_rows)} rows from {ref_meta['done']} of {ref_meta['planned']} reference sweep runs -> index.html")
