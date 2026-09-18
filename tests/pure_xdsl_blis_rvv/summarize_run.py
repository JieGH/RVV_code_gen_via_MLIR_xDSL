"""
Turns a run's merged `all_raw_<dtype>_<method>.txt` CSV (written by
`compile_macro_bench.sh`'s `[4/4] Fetch Logs` step, see docs/metrics.md) into a
per-case Markdown comparison table written to `<run_dir>/comparison.md`. When the
matching `all_perf_<dtype>_<method>.txt` exists alongside it, hardware perf-counter
columns (IPC, L1D miss %, stall %) for all three regions (Vector Pack, Scalar Pack,
Compute) are joined in by (Project, Layer, M, N, K, Variant); otherwise those columns
read "N/A".

Usage: python3 summarize_run.py <run_dir>
"""

import csv
import glob
import os
import re
import sys

VARIANT_RE = re.compile(r"MR(\d+)_NR(\d+)")


def _short_variant(name: str) -> str:
    m = VARIANT_RE.match(name)
    return f"{m.group(1)}x{m.group(2)}" if m else name


def _pct_of_total(part_ms: float, compute_ms: float) -> str:
    total = part_ms + compute_ms
    return f"{100.0 * part_ms / total:.1f}%" if total > 0 else "N/A"


def _gain_pct(scalar_ms: float, vector_ms: float) -> str:
    if scalar_ms <= 0:
        return "N/A"
    gain = 100.0 * (scalar_ms - vector_ms) / scalar_ms
    return f"+{gain:.1f}%" if gain >= 0 else f"{gain:.1f}%"


PERF_REGIONS = ["Vector_Pack", "Scalar_Pack", "Compute"]
PERF_METRICS = [
    ("IPC", "IPC", 2, ""),
    ("L1D_Read_Miss_Pct", "L1D Read Miss %", 1, "%"),
    ("L1D_Write_Miss_Pct", "L1D Write Miss %", 1, "%"),
    ("Stall_BE_Pct", "Stall BE %", 1, "%"),
]


def _display_region(region: str) -> str:
    return region.replace("_", " ")  # "Vector_Pack" -> "Vector Pack"


def _load_perf_index(perf_csv_path: str) -> dict[tuple[str, ...], dict[str, str]]:
    index: dict[tuple[str, ...], dict[str, str]] = {}
    with open(perf_csv_path) as f:
        for r in csv.DictReader(f):
            key = (r.get("Project", ""), r["Layer"], r["M"], r["N"], r["K"], r["Variant"])
            index[key] = r
    return index


def _perf_field(perf_row: dict[str, str] | None, field: str, decimals: int, suffix: str) -> str:
    if perf_row is None or perf_row.get("Counters_Valid") != "1":
        return "N/A"
    try:
        v = float(perf_row[field])
    except (KeyError, ValueError):
        return "N/A"
    if v != v:  # NaN: a single unsupported event blanks the whole counter group
        return "N/A"
    return f"{v:.{decimals}f}{suffix}"


def _perf_columns(perf_row: dict[str, str] | None) -> dict[str, str]:
    if perf_row is None:
        counters_valid = "N/A"
    else:
        counters_valid = "yes" if perf_row.get("Counters_Valid") == "1" else "no"

    cols: dict[str, str] = {"Counters Valid": counters_valid}
    for region in PERF_REGIONS:
        for field_suffix, label, decimals, unit in PERF_METRICS:
            cols[f"{_display_region(region)} {label}"] = _perf_field(
                perf_row, f"{region}_{field_suffix}", decimals, unit
            )
    cols["Compute FLOPs/Cycle"] = _perf_field(perf_row, "Compute_FLOPs_per_Cycle", 2, "")
    return cols


def _jitter_bracket(r: dict[str, str]) -> str:
    # Mirrors driver_benchmark.cpp's console summary formula/precision exactly:
    # min/p5/p95/max/iqr/s at 1 decimal, CV% as an integer, mx/md at 2 decimals.
    median = float(r["KerGOPs_Median"])
    std = float(r["KerGOPs_Std"])
    max_ = float(r["KerGOPs_Max"])
    cv_pct = (std / median) * 100.0 if median > 0 else 0.0
    max_md = (max_ / median) if median > 0 else 0.0
    return (
        f"[min:{float(r['KerGOPs_Min']):.1f}, p5:{float(r['KerGOPs_P5']):.1f}, "
        f"p95:{float(r['KerGOPs_P95']):.1f}, max:{max_:.1f}, "
        f"iqr:{float(r['KerGOPs_IQR']):.1f}, s:{std:.1f}, CV:{cv_pct:.0f}%, "
        f"mx/md:{max_md:.2f}]"
    )


def build_comparison_rows(
    raw_csv_path: str, perf_csv_path: str | None = None
) -> list[dict[str, str]]:
    perf_index = _load_perf_index(perf_csv_path) if perf_csv_path else None
    rows: list[dict[str, str]] = []
    with open(raw_csv_path) as f:
        for r in csv.DictReader(f):
            m, n, k = int(r["M"]), int(r["N"]), int(r["K"])
            scalar_pack_ms = float(r["Scalar_Pack_Time_ms"])
            vector_pack_ms = float(r["Vector_Pack_Time_ms"])
            compute_ms = float(r["Compute_Time_ms"])
            ker_gops = float(r["KerGOPs_Median"])
            ref_ms = float(r["Ref_Time_ms"])
            status = r["Status"]

            if ref_ms > 0:
                ref_gops = 2.0 * m * n * k / (ref_ms / 1000.0) / 1e9
                ref_gops_str = f"{ref_gops:.2f}"
                delta_str = f"{ker_gops / ref_gops:.1f}×"
                info = status
            else:
                ref_gops_str = "N/A"
                delta_str = "N/A"
                info = f"{status} (skip_ref_run)"

            # Key order must match _load_perf_index's (Project, Layer, M, N, K, Variant).
            perf_key = (r.get("Project", ""), r["Layer"], r["M"], r["N"], r["K"], r["Variant"])
            perf_row = perf_index.get(perf_key) if perf_index is not None else None

            row = {
                "Case": r["Layer"],
                "Shape (m×n×k)": f"{m}×{n}×{k}",
                "MRxNR": _short_variant(r["Variant"]),
                "Scalar Pack ms": f"{scalar_pack_ms:.2f}",
                "Vector Pack ms": f"{vector_pack_ms:.2f}",
                "Compute ms": f"{compute_ms:.2f}",
                "Scalar Pack % of total": _pct_of_total(scalar_pack_ms, compute_ms),
                "Vector Pack % of total": _pct_of_total(vector_pack_ms, compute_ms),
                "KerGOPs (this run)": f"{ker_gops:.2f}",
                "KerGOPs Jitter [min, p5, p95, max, iqr, s, CV%, mx/md]": _jitter_bracket(r),
                "Ref GOPs": ref_gops_str,
                "Δ vs ref": delta_str,
                "Vector Pack Gain %": _gain_pct(scalar_pack_ms, vector_pack_ms),
                **_perf_columns(perf_row),
                "additional info": info,
            }
            rows.append(row)
    return rows


def write_markdown_table(rows: list[dict[str, str]], out_path: str) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_run.py <run_dir>", file=sys.stderr)
        return 1
    run_dir = sys.argv[1]
    raw_files = sorted(glob.glob(os.path.join(run_dir, "all_raw_*.txt")))
    if not raw_files:
        print(f"no all_raw_*.txt found under {run_dir}", file=sys.stderr)
        return 1

    all_rows: list[dict[str, str]] = []
    for raw_csv in raw_files:
        perf_csv = raw_csv.replace("all_raw_", "all_perf_")
        all_rows.extend(
            build_comparison_rows(raw_csv, perf_csv if os.path.exists(perf_csv) else None)
        )

    out_path = os.path.join(run_dir, "comparison.md")
    write_markdown_table(all_rows, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
