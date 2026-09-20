"""Build index.html, a standalone interactive view of the 19-20 Sep 2026 overnight runs.

Reads every logs/run_2026-09-19_23* and logs/run_2026-09-20_* folder (raw logs of the
reference-protocol build), computes GOPs = 2mnk / RVV_Time_ms per layer and run, embeds
the rows in template.html and writes index.html. Open index.html in a browser; it needs
no server and loads nothing from the network.

Run from anywhere: python3 build_report.py
"""
import glob, json, os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.dirname(HERE)
RUN_GLOBS = ["run_2026-09-19_23*", "run_2026-09-20_*"]

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
rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]

body = open(os.path.join(HERE, "template.html")).read().replace("/*__DATA__*/[]", json.dumps(rows, separators=(",", ":")))
page = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n' + body
open(os.path.join(HERE, "index.html"), "w").write(page)
print(f"{len(rows)} rows from {len({r['run'] for r in rows})} runs -> index.html")
