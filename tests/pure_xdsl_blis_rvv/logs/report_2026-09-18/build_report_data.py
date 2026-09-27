import pandas as pd
import json, os

BASE = "/home/jlei/Documents/ws_MLIR/repo_xdsl_microkernels/RVV_code_gen_via_MLIR_xDSL/tests/pure_xdsl_blis_rvv/logs"
RUNS = [
    "run_2026-09-18_004053","run_2026-09-18_014128","run_2026-09-18_024151",
    "run_2026-09-18_034155","run_2026-09-18_044110",
]
RUN_LABEL = {"run_2026-09-18_004053":"00:01","run_2026-09-18_014128":"01:01",
             "run_2026-09-18_024151":"02:01","run_2026-09-18_034155":"03:01",
             "run_2026-09-18_044110":"04:01"}

raw_frames, perf_frames = [], []
for run in RUNS:
    r = pd.read_csv(os.path.join(BASE, run, "all_raw_fp32_AXPY_FP32_Pack.txt"))
    r["Run"] = run
    raw_frames.append(r)
    p = pd.read_csv(os.path.join(BASE, run, "all_perf_fp32_AXPY_FP32_Pack.txt"))
    p["Run"] = run
    perf_frames.append(p)
raw = pd.concat(raw_frames, ignore_index=True)
perf = pd.concat(perf_frames, ignore_index=True)

def short_layer(p):
    if "MR16NR15" in p: return "MR16xNR15"
    if "MR32NR6" in p: return "MR32xNR6"
    if "MR32NR7" in p: return "MR32xNR7"
    return p
raw["LayerShort"] = raw["Project"].apply(short_layer)
perf["LayerShort"] = perf["Project"].apply(short_layer)

raw["CV_pct"] = raw["KerGOPs_Std"] / raw["KerGOPs_Median"] * 100.0
raw["mx_md"] = raw["KerGOPs_Max"] / raw["KerGOPs_Median"]
raw["Vector_Pack_Gain_pct"] = (raw["Scalar_Pack_Time_ms"] - raw["Vector_Pack_Time_ms"]) / raw["Scalar_Pack_Time_ms"] * 100.0
raw["Shape"] = raw["M"].astype(str) + "x" + raw["N"].astype(str) + "x" + raw["K"].astype(str)

# ---- Reference design ----
# S1-S5 fixed 2026-09-18 (see GitHub issue #10): the previous 8.5/8.7/8.9/9.0/9.0
# figures had no traceable source (comment claimed "supplied directly by the
# user" but the user does not recall giving them, and no file in the repo
# backs them up). Restored to the archived flat values, corroborated by three
# independent sources: ext/square_comparison.txt, ext/output/retro_europar_plot.txt,
# and the xdsl_gflops array in ext/plot_square.py. This is an interim fix —
# a fresh baseline run on jbpi2 via tests/pure_xdsl_blis_rvv/ref_baseline/ is
# still pending and may supersede these again.
# B1-B5 openblas/ratio retained from the old ext/*_comparison.txt pending
# updated OpenBLAS numbers (unused in the report chart, which dropped the
# OpenBLAS series).
REF = {
    # Layer(case) -> {xdsl_ref, openblas, ratio}   matched by shape
    "B1": {"m":1024,"n":384,"k":1024, "xdsl_ref":8.0, "openblas":6.00, "ratio":1.333},
    "B2": {"m":384,"n":384,"k":64,    "xdsl_ref":12.5,"openblas":5.10, "ratio":2.451},
    "B3": {"m":64,"n":384,"k":384,    "xdsl_ref":8.8, "openblas":7.30, "ratio":1.205},
    "B4": {"m":4096,"n":384,"k":1024, "xdsl_ref":8.5, "openblas":5.90, "ratio":1.441},
    "B5": {"m":1024,"n":384,"k":4096, "xdsl_ref":7.5, "openblas":5.90, "ratio":1.271},
    "S1": {"m":1000,"n":1000,"k":1000,"xdsl_ref":8.32,"openblas":6.10, "ratio":1.393},
    "S2": {"m":2000,"n":2000,"k":2000,"xdsl_ref":8.53,"openblas":6.80, "ratio":1.279},
    "S3": {"m":3000,"n":3000,"k":3000,"xdsl_ref":8.51,"openblas":6.80, "ratio":1.309},
    "S4": {"m":4000,"n":4000,"k":4000,"xdsl_ref":8.31,"openblas":6.90, "ratio":1.304},
    "S5": {"m":5000,"n":5000,"k":5000,"xdsl_ref":8.36,"openblas":7.00, "ratio":1.286},
}

# ---- Per-case aggregate across the 5 runs (performance) ----
perf_cases = []
for (layer_short, layer), g in raw.groupby(["LayerShort","Layer"]):
    g = g.sort_values("Run")
    vals = g["KerGOPs_Median"].tolist()
    mean = sum(vals)/len(vals)
    perf_cases.append({
        "LayerShort": layer_short,
        "Case": layer,
        "Shape": g["Shape"].iloc[0],
        "M": int(g["M"].iloc[0]), "N": int(g["N"].iloc[0]), "K": int(g["K"].iloc[0]),
        "KerGOPs_runs": [round(v,3) for v in vals],
        "KerGOPs_mean": round(mean,3),
        "KerGOPs_min": round(min(vals),3),
        "KerGOPs_max": round(max(vals),3),
        "KerGOPs_std": round(g["KerGOPs_Median"].std(),3),
        "CV_pct_mean": round(g["CV_pct"].mean(),3),
        "CV_pct_max": round(g["CV_pct"].max(),3),
        "mx_md_mean": round(g["mx_md"].mean(),3),
        "RVV_Pack_GBs_mean": round(g["RVV_Pack_GBs"].mean(),3),
        "RVV_Pack_GBs_runs": [round(v,3) for v in g["RVV_Pack_GBs"].tolist()],
        "Scl_Pack_GBs_mean": round(g["Scl_Pack_GBs"].mean(),3),
        "Scl_Pack_GBs_runs": [round(v,3) for v in g["Scl_Pack_GBs"].tolist()],
        "Vector_Pack_Gain_pct_mean": round(g["Vector_Pack_Gain_pct"].mean(),3),
        "Vector_Pack_Gain_pct_runs": [round(v,3) for v in g["Vector_Pack_Gain_pct"].tolist()],
        "Vector_Pack_ms_mean": round(g["Vector_Pack_Time_ms"].mean(),4),
        "Scalar_Pack_ms_mean": round(g["Scalar_Pack_Time_ms"].mean(),4),
        "Compute_ms_mean": round(g["Compute_Time_ms"].mean(),4),
        # Pack-inclusive throughput (issue #10, Finding 1): KerGOPs above only
        # times the microkernel calls and excludes packing, while every
        # reference number includes packing. e2e is the metric that's
        # actually comparable to a reference figure.
        "e2e_GOPs_mean": round(
            2 * int(g["M"].iloc[0]) * int(g["N"].iloc[0]) * int(g["K"].iloc[0])
            / ((g["Vector_Pack_Time_ms"].mean() + g["Compute_Time_ms"].mean()) * 1e6),
            3,
        ),
        "ref": REF.get(layer),
    })

# ---- Per-run overall averages (still same single machine, jbpi2) ----
run_overview = []
for run in RUNS:
    g = raw[raw["Run"]==run]
    run_overview.append({
        "Run": run, "Label": RUN_LABEL[run], "Machine": "jbpi2",
        "KerGOPs_mean": round(g["KerGOPs_Median"].mean(),3),
        "CV_pct_mean": round(g["CV_pct"].mean(),3),
        "Vector_Pack_Gain_pct_mean": round(g["Vector_Pack_Gain_pct"].mean(),3),
        "RVV_Pack_GBs_mean": round(g["RVV_Pack_GBs"].mean(),3),
        "Scl_Pack_GBs_mean": round(g["Scl_Pack_GBs"].mean(),3),
    })

# ---- Hardware perf counters, averaged per case across 5 runs, per region ----
regions = ["Vector_Pack","Scalar_Pack","Compute"]
fields = ["IPC","L1D_Read_Miss_Pct","L1D_Write_Miss_Pct","Stall_BE_Pct","Stall_FE_Pct"]
perf_cases_hw = []
for (layer_short, layer), g in perf.groupby(["LayerShort","Layer"]):
    row = {"LayerShort": layer_short, "Case": layer, "Counters_Valid_all": bool(g["Counters_Valid"].all())}
    for reg in regions:
        for f in fields:
            col = f"{reg}_{f}"
            row[col] = round(g[col].mean(),3) if col in g.columns else None
    row["Compute_FLOPs_per_Cycle_mean"] = round(g["Compute_FLOPs_per_Cycle"].mean(),3)
    row["Vector_Pack_Cycles_CV_Pct_mean"] = round(g["Vector_Pack_Cycles_CV_Pct"].mean(),3)
    row["Compute_Cycles_CV_Pct_mean"] = round(g["Compute_Cycles_CV_Pct"].mean(),3)
    perf_cases_hw.append(row)

# groupby(["LayerShort","Layer"]) sorts by layer name first, which scrambles case
# order (B2,B3,B1,B5,B4,S1..S5). Re-sort into the natural B1-B5, S1-S5 order.
CASE_ORDER = ["B1","B2","B3","B4","B5","S1","S2","S3","S4","S5"]
perf_cases.sort(key=lambda c: CASE_ORDER.index(c["Case"]))
perf_cases_hw.sort(key=lambda c: CASE_ORDER.index(c["Case"]))

out = {
    "cases": perf_cases,
    "runs": run_overview,
    "hw": perf_cases_hw,
    "run_labels": [RUN_LABEL[r] for r in RUNS],
}

with open(os.path.join(os.path.dirname(__file__), "bench_data.json"),"w") as f:
    json.dump(out, f)

print("cases:", len(perf_cases))
print("hw rows:", len(perf_cases_hw))
print(json.dumps(perf_cases[:2], indent=2))
print(json.dumps(perf_cases_hw[:2], indent=2))
