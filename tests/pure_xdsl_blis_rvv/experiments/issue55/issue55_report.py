#!/usr/bin/env python3
"""
Issues #54 / #55: read `run_issue55.sh` series, the B packer read-order variants.

  1. GEMM arms, per board, tile and shape, one row per variant: GOPs by mean time (average over launches, with the
     lowest and highest launch) and by median rep, the share of slow reps, the fast start, the CV, the launches and the
     blocking (kc). Uninstrumented build, reference protocol (--tmin 10), core 4, --no-nc-overrides.
  2. Pack-only loop (`--pack-only`), per board, tile, shape and variant: median pack time per rep, the mean pack time
     of the fast reps and of the slow reps (the split of reps_report.py on the per-rep times, rep 0 left out), and the
     share of slow reps.

Only runs with status ok count (a runs.tsv without the status column, from before #55, counts exit 0). Runs marked
discarded-load or failed are left out and named. The per-rep files are read ONE AT A TIME: a file is one tile, and
tiles are never merged (the #49 lesson). A row seen in fewer than 3 launches, a discarded run, and a run folder made
under another user's load (provenance.txt) are flagged. More than one series folder can be given: launches are pooled.

Usage: issue55_report.py <logs/issue55_<stamp> series folder> [...] [--out FILE]
"""

from __future__ import annotations

import csv
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

from reps_report import (  # noqa: E402
    MIN_LAUNCHES,
    CaseStats,
    case_stats,
    conditions_lines,
    read_reps,
    reps_files,
    run_conditions,
    slow_reps,
)

VARIANT_ORDER = ["row", "row-intr", "scalar", "col-g1", "col-g2", "col-g4", "col-gall", "col-g1-m4"]
LAYER_ORDER = ["B3", "B3_K256", "B2", "R24"]
BUILD_LINE = ("Build: uninstrumented (INSTRUMENTED=0); protocol: reference, --tmin 10; core 4; --no-nc-overrides "
              "(each case's nc is the cache model's)")


def _vkey(v: str) -> tuple[int, str]:
    return (VARIANT_ORDER.index(v) if v in VARIANT_ORDER else len(VARIANT_ORDER), v)


def _lkey(layer: str) -> tuple[int, str]:
    return (LAYER_ORDER.index(layer) if layer in LAYER_ORDER else len(LAYER_ORDER), layer)


def _tile(variant: str) -> tuple[int, int]:
    m = re.fullmatch(r"MR(\d+)_NR(\d+)", variant)
    if not m:
        raise ValueError(f"cannot read a tile from variant {variant!r}")
    return int(m.group(1)), int(m.group(2))


def _tsv(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if not lines:
        return []
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"), strict=False)) for ln in lines[1:] if ln.strip()]


def _status(row: dict[str, str]) -> str:
    """ok / failed / discarded-load; a series from before the status column: exit 0 is ok."""
    if "status" in row:
        return row["status"]
    return "ok" if row.get("exit") == "0" else "failed"


@dataclass
class Run:
    board: str
    launch: int
    arm: str
    folder: Path
    status: str


def read_runs(series: Path) -> list[Run]:
    out: list[Run] = []
    for row in _tsv(series / "runs.tsv"):
        folder = row["result_dir"]
        out.append(Run(row["board"], int(row["launch"]), row["arm"],
                       series.parent / Path(folder).name if folder != "-" else series, _status(row)))
    return out


@dataclass
class Sample:
    board: str
    launch: int
    variant: str
    layer: str
    nc: int
    tile: tuple[int, int]
    kc: int
    stats: CaseStats
    red: bool  # another user's job was on the board (provenance.txt)


def collect(series: list[Path]) -> tuple[list[Sample], list[Run]]:
    samples: list[Sample] = []
    excluded: list[Run] = []
    for sd in series:
        for run in read_runs(sd):
            if run.status != "ok":
                excluded.append(run)
                continue
            red = run_conditions(run.folder).other_users not in ("none", "unknown", "unavailable")
            for f in reps_files(run.folder):  # one file at a time: a file is one tile
                for (layer, nc), case in read_reps(f).items():
                    samples.append(Sample(run.board, run.launch, run.arm, layer, nc, _tile(case.variant), case.kc,
                                          case_stats(case.ops, case.seconds), red))
    return samples, excluded


@dataclass
class Row:
    board: str
    tile: tuple[int, int]
    layer: str
    variant: str
    kc: int
    launches: int
    gops_mean_avg: float
    gops_mean_lo: float
    gops_mean_hi: float
    gops_median: float
    slow_share: float
    fast_start_s: float
    cv_pct: float
    red: int  # samples from a run made under another user's load


def rows_of(samples: list[Sample]) -> list[Row]:
    per: dict[tuple[str, tuple[int, int], str, str], list[Sample]] = {}
    for s in samples:
        per.setdefault((s.board, s.tile, s.layer, s.variant), []).append(s)
    rows: list[Row] = []
    for (board, tile, layer, variant), ss in per.items():
        means = [s.stats.gops_mean_time for s in ss]
        rows.append(Row(board, tile, layer, variant, ss[0].kc, len(ss), statistics.fmean(means), min(means), max(means),
                        statistics.fmean(s.stats.gops_median for s in ss),
                        statistics.fmean(s.stats.slow_share for s in ss),
                        statistics.fmean(s.stats.fast_start_s for s in ss),
                        statistics.fmean(s.stats.cv_pct for s in ss), sum(s.red for s in ss)))
    rows.sort(key=lambda r: (r.board, r.tile, _lkey(r.layer), _vkey(r.variant)))
    return rows


# ---- the pack-only loop's output ----

@dataclass
class PackFile:
    header: dict[str, str]
    seconds: list[float]
    blocks: list[str]
    check: str


def read_pack_only(path: Path) -> PackFile:
    """The loop's output: `# key: value` header lines, `# block:` lines, `Rep,Pack_Seconds` then one row per rep,
    and a final `# check:` line."""
    header: dict[str, str] = {}
    blocks: list[str] = []
    data: list[str] = []
    check = "missing"
    for ln in path.read_text().splitlines():
        if ln.startswith("#"):
            key, _, value = ln[1:].partition(":")
            if key.strip() == "block":
                blocks.append(value.strip())
            elif key.strip() == "check":
                check = value.strip()
            else:
                header[key.strip()] = value.strip()
        elif ln.strip():
            data.append(ln)
    rows = list(csv.DictReader(data))
    return PackFile(header, [float(r["Pack_Seconds"]) for r in rows], blocks, check)


@dataclass
class PackStats:
    n: int
    median_ms: float
    fast_ms: float
    slow_ms: float  # nan: no slow reps
    slow_share: float


def pack_stats(seconds: list[float]) -> PackStats:
    """The fast/slow split of reps_report.py on per-rep pack times: a rep's rate is 1/seconds (so a slow rep is a long
    one), rep 0 left out of the split. A thin wrapper; reps_report is unchanged."""
    mask = slow_reps(1e9, seconds)
    warm = list(zip(seconds, mask, strict=True))[1:] if len(seconds) > 1 else list(zip(seconds, mask, strict=True))
    fast = [s for s, m in warm if not m]
    slow = [s for s, m in warm if m]
    return PackStats(len(seconds), 1e3 * statistics.median(seconds),
                     1e3 * statistics.fmean(fast) if fast else float("nan"),
                     1e3 * statistics.fmean(slow) if slow else float("nan"),
                     len(slow) / len(warm))


@dataclass
class PackRow:
    board: str
    tile: str
    shape: str
    variant: str
    launches: int
    median_ms: float
    fast_ms: float
    slow_ms: float
    slow_share: float
    kc: str
    bad_check: int


def pack_rows(series: list[Path]) -> tuple[list[PackRow], list[str]]:
    per: dict[tuple[str, str, str, str], list[tuple[PackStats, PackFile]]] = {}
    excluded: list[str] = []
    for sd in series:
        tsv = sd / "packonly.tsv"
        if not tsv.is_file():
            continue
        for row in _tsv(tsv):
            if _status(row) != "ok":
                excluded.append(f"{row['board']} L{row['launch']} pack-only {row['variant']} {row['shape']} {row['tile']}: {_status(row)}")
                continue
            pf = read_pack_only(sd / row["file"])
            per.setdefault((row["board"], row["tile"], row["shape"], row["variant"]), []).append((pack_stats(pf.seconds), pf))
    rows: list[PackRow] = []
    for (board, tile, shape, variant), items in per.items():
        sts = [s for s, _ in items]
        fast = [s.fast_ms for s in sts if s.fast_ms == s.fast_ms]
        slow = [s.slow_ms for s in sts if s.slow_ms == s.slow_ms]
        blocking = items[0][1].header.get("blocking", "")
        kc = blocking.split("kc=")[-1] if "kc=" in blocking else "?"
        rows.append(PackRow(board, tile, shape, variant, len(sts), statistics.fmean(s.median_ms for s in sts),
                            statistics.fmean(fast) if fast else float("nan"),
                            statistics.fmean(slow) if slow else float("nan"),
                            statistics.fmean(s.slow_share for s in sts), kc,
                            sum(pf.check != "ok" for _, pf in items)))
    rows.sort(key=lambda r: (r.board, r.tile, _lkey(r.shape), _vkey(r.variant)))
    return rows, excluded


# ---- formatting ----

def _ms(x: float) -> str:
    return f"{x:.4f}" if x == x else "-"


def _flags(r: Row, discarded: dict[tuple[str, str], int]) -> str:
    out: list[str] = []
    if r.launches < MIN_LAUNCHES:
        out.append(f"only {r.launches} launch(es)")
    d = discarded.get((r.board, r.variant), 0)
    if d:
        out.append(f"{d} run(s) of this arm discarded")
    if r.red:
        out.append(f"RED NOTE: {r.red} sample(s) under another user's load")
    return "; ".join(out)


def format_speed(rows: list[Row], discarded: dict[tuple[str, str], int]) -> str:
    out = [
        "| variant | kc | launches | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % | fast start (s) | CV % | flags |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r.variant} | {r.kc} | {r.launches} | {r.gops_mean_avg:.2f} ({r.gops_mean_lo:.2f}-{r.gops_mean_hi:.2f}) | "
            f"{r.gops_median:.2f} | {100 * r.slow_share:.0f} | {r.fast_start_s:.2f} | {r.cv_pct:.1f} | {_flags(r, discarded)} |"
        )
    return "\n".join(out)


def format_pack(rows: list[PackRow]) -> str:
    out = [
        "| variant | kc | launches | median pack (ms) | fast reps: mean (ms) | slow reps: mean (ms) | slow reps % | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        flags = [f"only {r.launches} launch(es)"] if r.launches < MIN_LAUNCHES else []
        if r.bad_check:
            flags.append(f"CHECK FAILED in {r.bad_check} run(s)")
        out.append(f"| {r.variant} | {r.kc} | {r.launches} | {_ms(r.median_ms)} | {_ms(r.fast_ms)} | "
                   f"{_ms(r.slow_ms) if r.slow_share else 'no slow reps'} | {100 * r.slow_share:.0f} | {'; '.join(flags)} |")
    return "\n".join(out)


def report(series: list[Path]) -> str:
    samples, excluded = collect(series)
    folders = list(dict.fromkeys(r.folder for sd in series for r in read_runs(sd) if r.status == "ok" and r.folder != sd))
    rows = rows_of(samples)
    discarded: dict[tuple[str, str], int] = {}
    for r in excluded:
        if r.status == "discarded-load":
            discarded[(r.board, r.arm)] = discarded.get((r.board, r.arm), 0) + 1
    out = [f"# Issues #54/#55 series {', '.join(sd.name for sd in series)}", "", BUILD_LINE, "", "## Run conditions", "",
           *conditions_lines(folders), ""]
    if excluded:
        out += ["Runs left out (status is not ok): " + "; ".join(
            f"{r.board} L{r.launch} {r.arm} ({r.status})" for r in excluded), ""]
    out += ["## 1. GEMM arms per tile and shape", ""]
    groups: dict[tuple[str, tuple[int, int], str], list[Row]] = {}
    for r in rows:
        groups.setdefault((r.board, r.tile, r.layer), []).append(r)
    for (board, tile, layer), rs in groups.items():
        out += [f"### {board}, {tile[0]}x{tile[1]}, {layer}", "", format_speed(rs, discarded), ""]
    prow, pexcl = pack_rows(series)
    if prow or pexcl:
        out += ["## 2. Pack-only loop (no compute; 1 clock read per rep; rep 0 left out of the split)", ""]
        if pexcl:
            out += ["Pack-only runs left out: " + "; ".join(pexcl), ""]
        pg: dict[tuple[str, str, str], list[PackRow]] = {}
        for p in prow:
            pg.setdefault((p.board, p.tile, p.shape), []).append(p)
        for (board, tile, shape), ps in pg.items():
            out += [f"### {board}, {tile}, {shape}", "", format_pack(ps), ""]
    few = [f"{r.board} {r.tile[0]}x{r.tile[1]} {r.layer} {r.variant}" for r in rows if r.launches < MIN_LAUNCHES]
    if few:
        out += [f"WARNING: {len(few)} row(s) seen in fewer than {MIN_LAUNCHES} launches: not a result under issue #39's "
                f"house rules (one launch is not a result): {', '.join(few[:6])}" + (" ..." if len(few) > 6 else ""), ""]
    if any(r.status == "discarded-load" for r in excluded):
        n = sum(r.status == "discarded-load" for r in excluded)
        out += [f"RED NOTE: {n} run(s) were discarded because the board was busy after the run; they are left out above.", ""]
    return "\n".join(out)


def main(argv: list[str]) -> int:
    out_file: Path | None = None
    args: list[str] = []
    it = iter(argv)
    for a in it:
        if a == "--out":
            out_file = Path(next(it, ""))
        else:
            args.append(a)
    if not args:
        print("Usage: issue55_report.py <logs/issue55_<stamp> series folder> [...] [--out FILE]", file=sys.stderr)
        return 2
    text = report([Path(a) for a in args])
    if out_file is not None:
        out_file.write_text(text)
        print(f"wrote {out_file}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
