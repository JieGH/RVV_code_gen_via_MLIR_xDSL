#!/usr/bin/env python3
"""
Issue #47: the slow state across every shape, from the per-rep logs already committed.

B3 runs at two speeds (#36, #41); nobody had looked at the other shapes. This reads every
`logs/run_*/*/reps_*.txt[.gz]` with reps_report.py's reader and puts four tables side by side:

  1. Inventory: one row per per-rep file (run folder, board, build, protocol, core, packer, nc
     override), and whether it counts. Issue #39's rules drop a core-0 run, a run whose core is not
     known (no provenance.txt) or not pinned; the instrumented build and the warm-up protocol are
     kept out of the speed tables (their timers cost 11-17%: counts only).
  2. Slow state per case: per (board, packer, tile, shape, nc), the slow-rep share, fast start,
     clumping and CV over its launches, with the lowest and highest launch. A case seen in fewer
     than 3 launches is "no result" (house rule 2).
  3. Time, reps or bytes: for each launch that has a slow state, the reps, the seconds and the
     bytes packed before its first slow rep; which of the three varies least across shapes.
  4. Shape features: M, N, K, nc, block sizes, leading dimensions and powers of two beside the
     slow-rep share.

A launch has a slow STATE when at least 5% of its reps are slow (reps_report.py's two-means split,
rep 0 left out): one or two slow reps in a hundred are an interrupt, not a state.

Bytes packed per rep, from the loop nest of generate_pure_blis_project.py (jc over N by nc, pc over K
by kc, pack the B block kc x nc, ic over M by mc, pack the A block mc x kc): every element of B goes
through the packer once, and every element of A goes through it once per jc block:

    bytes_per_rep = 4 * (K*N + M*K*ceil(N/nc))            (fp32; the driver's own `pack_bytes` count)

Usage: slow_state_survey.py [logs dir (default: logs/)] [--out FILE] [--until STAMP]
(issue #47 read the logs up to --until 2026-09-28_235959)
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reps_report import MIN_LAUNCHES, CaseStats, case_stats, packer_of, read_reps, reps_files, run_conditions  # noqa: E402

TWO_SPEED_SHARE = 0.05  # a launch is in a slow state when at least this share of its reps are slow
ELEM_BYTES = 4  # fp32
L2_BYTES = 512 * 1024  # the X60's L2, shared by cores 4-7


def bytes_per_rep(m: int, n: int, k: int, nc: int) -> int:
    """Bytes the packers write in one call of the GEMM (see the module docstring)."""
    return ELEM_BYTES * (k * n + m * k * math.ceil(n / nc))


def _pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


@dataclass
class InventoryRow:
    run: str
    board: str
    build: str
    protocol: str
    core: str
    packer: str
    nc_override: bool
    tiles: str
    layers: list[str]
    used: str  # "yes", or "no: <why>"


class CaseKey(NamedTuple):
    """What makes two launches the same case: never pooled across boards, packers, tiles, shapes or nc."""
    board: str
    packer: str
    tile: str
    layer: str
    nc: int
    m: int
    n: int
    k: int


@dataclass
class Launch:
    run: str
    board: str
    packer: str
    tile: str
    layer: str
    nc: int
    m: int
    n: int
    k: int
    mc: int
    kc: int
    seconds: list[float]
    ops: float

    @property
    def key(self) -> CaseKey:
        return CaseKey(self.board, self.packer, self.tile, self.layer, self.nc, self.m, self.n, self.k)

    def stats(self) -> CaseStats:
        return case_stats(self.ops, self.seconds)

    @property
    def two_speed(self) -> bool:
        return self.stats().slow_share >= TWO_SPEED_SHARE


def _counts_for_speed(board_core: str, build: str, protocol: str) -> str:
    if board_core == "unknown":
        return "no: core unknown"
    if not board_core.isdigit():
        return "no: not pinned"
    if board_core == "0":
        return "no: core 0"
    if not build.startswith("UNINSTRUMENTED"):
        return "no: instrumented build (counts only)"
    if not protocol.startswith("REFERENCE"):
        return "no: warm-up protocol"
    return "yes"


def _tile(variant: str) -> str:
    """`MR16_NR15` -> `16x15`."""
    try:
        mr, nr = variant.split("_")
        return f"{int(mr.removeprefix('MR'))}x{int(nr.removeprefix('NR'))}"
    except ValueError:
        return variant


def collect(logs: Path, until: str | None = None) -> tuple[list[Launch], list[InventoryRow]]:
    """Every per-rep file under logs/run_*: the launches that count for speed, and one inventory
    row for each file, counted or not. `until` (a run folder stamp, `2026-09-28_235959`) leaves
    out later runs, so an issue's survey can be repeated on the logs it read."""
    launches: list[Launch] = []
    inventory: list[InventoryRow] = []
    for run_dir in sorted(p for p in logs.glob("run_*") if p.is_dir()):
        if until is not None and run_dir.name.removeprefix("run_") > until:
            continue
        cond = run_conditions(run_dir)
        for f in reps_files(run_dir):
            cases = read_reps(f)
            used = _counts_for_speed(cond.core, cond.build, cond.protocol)
            board = cond.board.split("@")[-1]
            packer = packer_of(f)
            inventory.append(InventoryRow(
                run_dir.name, board, cond.build.split(" ")[0].lower(), cond.protocol.split(" ")[0].lower(),
                cond.core, packer, "_no_nc_override" in f.name,
                ",".join(sorted({_tile(c.variant) for c in cases.values()})),
                [layer for layer, _ in cases], used))
            if used != "yes":
                continue
            for (layer, nc), c in cases.items():
                launches.append(Launch(run_dir.name, board, packer, _tile(c.variant), layer, nc,
                                       c.m, c.n, c.k, c.mc, c.kc, c.seconds, c.ops))
    return launches, inventory


def _by_case(launches: list[Launch]) -> dict[CaseKey, list[Launch]]:
    out: dict[CaseKey, list[Launch]] = {}
    for ln in launches:
        out.setdefault(ln.key, []).append(ln)
    return dict(sorted(out.items(), key=lambda kv: (kv[0].board, kv[0].layer, kv[0].packer, kv[0].nc, kv[0].tile)))


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


@dataclass
class SlowRow:
    board: str
    packer: str
    tile: str
    layer: str
    nc: int
    m: int
    n: int
    k: int
    launches: int
    two_speed_launches: int
    reps_avg: float
    gops_avg: float
    gops_lo: float
    gops_hi: float
    gops_median_avg: float
    slow_share_avg: float
    slow_share_lo: float
    slow_share_hi: float
    fast_start_s_avg: float
    fast_start_reps_avg: float
    clumping_avg: float
    cv_avg: float


@dataclass
class SlowTable:
    rows: list[SlowRow] = field(default_factory=list[SlowRow])
    no_result: list[SlowRow] = field(default_factory=list[SlowRow])


def slow_table(launches: list[Launch]) -> SlowTable:
    table = SlowTable()
    for key, group in _by_case(launches).items():
        sts = [ln.stats() for ln in group]
        shares = [st.slow_share for st in sts]
        state = [st for st in sts if st.slow_share >= TWO_SPEED_SHARE]
        clumps = [st.clumping for st in state if st.clumping == st.clumping]
        gops = [st.gops_mean_time for st in sts]
        row = SlowRow(
            key.board, key.packer, key.tile, key.layer, key.nc, key.m, key.n, key.k, len(group), len(state),
            _mean([float(st.n) for st in sts]), _mean(gops), min(gops), max(gops),
            _mean([st.gops_median for st in sts]), _mean(shares), min(shares), max(shares),
            _mean([st.fast_start_s for st in state]), _mean([float(st.fast_start_reps) for st in state]),
            _mean(clumps), _mean([st.cv_pct for st in sts]))
        (table.rows if len(group) >= MIN_LAUNCHES else table.no_result).append(row)
    return table


@dataclass
class SwitchRow:
    board: str
    packer: str
    tile: str
    layer: str
    nc: int
    launches: int
    two_speed_launches: int
    reps: float
    seconds: float
    bytes: float
    bytes_per_rep: int
    cv: dict[str, float]  # within the case, over its slow-state launches, in %
    reps_min: float = 0.0
    reps_median: float = 0.0
    reps_max: float = 0.0
    seconds_min: float = 0.0
    seconds_median: float = 0.0
    seconds_max: float = 0.0


@dataclass
class ContextSummary:
    board: str
    packer: str
    cases: int
    across: dict[str, float]
    closest: str | None


@dataclass
class SwitchTable:
    rows: list[SwitchRow]
    contexts: list[ContextSummary]
    across: dict[str, float]
    closest: str | None


def _cv(xs: list[float]) -> float:
    """Spread as a share of the mean, in %: population standard deviation over the mean."""
    if len(xs) < 2 or statistics.fmean(xs) == 0:
        return 0.0 if len(xs) >= 1 else float("nan")
    return statistics.pstdev(xs) / statistics.fmean(xs) * 100


QUANTITIES = ("reps", "seconds", "bytes")


def _closest(rows: list[SwitchRow]) -> tuple[dict[str, float], str | None]:
    """The spread of each quantity's per-case mean, over cases whose launches mostly switch (2 or
    more slow-state launches), and the one that varies least; None with fewer than 2 such cases."""
    usable = [r for r in rows if r.two_speed_launches >= 2]
    if len(usable) < 2:
        return {q: float("nan") for q in QUANTITIES}, None
    across = {q: _cv([getattr(r, q) for r in usable]) for q in QUANTITIES}
    return across, min(QUANTITIES, key=lambda q: across[q])  # ties: reps, seconds, bytes in that order


def switch_table(launches: list[Launch]) -> SwitchTable:
    rows: list[SwitchRow] = []
    for key, group in _by_case(launches).items():
        if len(group) < MIN_LAUNCHES:
            continue
        state = [ln for ln in group if ln.two_speed]
        if not state:
            continue
        per = bytes_per_rep(key.m, key.n, key.k, key.nc)
        reps = [float(ln.stats().fast_start_reps) for ln in state]
        secs = [ln.stats().fast_start_s for ln in state]
        byts = [r * per for r in reps]
        rows.append(SwitchRow(key.board, key.packer, key.tile, key.layer, key.nc, len(group), len(state),
                              _mean(reps), _mean(secs), _mean(byts), per,
                              {"reps": _cv(reps), "seconds": _cv(secs), "bytes": _cv(byts)},
                              min(reps), statistics.median(reps), max(reps), min(secs), statistics.median(secs), max(secs)))
    contexts: list[ContextSummary] = []
    for board, packer in dict.fromkeys((r.board, r.packer) for r in rows):
        here = [r for r in rows if (r.board, r.packer) == (board, packer)]
        across, closest = _closest(here)
        contexts.append(ContextSummary(board, packer, sum(1 for r in here if r.two_speed_launches >= 2), across, closest))
    across, closest = _closest(rows)
    return SwitchTable(rows, contexts, across, closest)


@dataclass
class FeatureRow:
    board: str
    packer: str
    tile: str
    layer: str
    m: int
    n: int
    k: int
    nc: int
    mc: int
    kc: int
    b_block_bytes: int
    a_block_bytes: int
    pack_bytes_per_rep: int
    pack_bytes_per_flop: float
    jc_blocks: int
    pc_blocks: int
    ic_blocks: int
    lda: int
    ldb: int
    ldc: int
    slow_share: float

    @property
    def lda_pow2(self) -> bool:
        return _pow2(self.lda)

    @property
    def ldb_pow2(self) -> bool:
        return _pow2(self.ldb)

    @property
    def ldc_pow2(self) -> bool:
        return _pow2(self.ldc)

    @property
    def b_block_pow2(self) -> bool:
        return _pow2(self.b_block_bytes)

    @property
    def a_block_pow2(self) -> bool:
        return _pow2(self.a_block_bytes)


def feature_table(launches: list[Launch]) -> list[FeatureRow]:
    """One row per case with 3 or more launches, slowest first. Column-major, as the driver runs it:
    lda = m, ldb = k, ldc = m. The packed B block is kc x nc, the A block mc x kc (both cut at the
    matrix edge)."""
    first: dict[CaseKey, Launch] = {}
    for ln in launches:
        first.setdefault(ln.key, ln)
    out: list[FeatureRow] = []
    for row in slow_table(launches).rows:
        ln = first[CaseKey(row.board, row.packer, row.tile, row.layer, row.nc, row.m, row.n, row.k)]
        out.append(FeatureRow(
            row.board, row.packer, row.tile, row.layer, row.m, row.n, row.k, row.nc, ln.mc, ln.kc,
            min(ln.kc, ln.k) * min(row.nc, ln.n) * ELEM_BYTES, min(ln.mc, ln.m) * min(ln.kc, ln.k) * ELEM_BYTES,
            bytes_per_rep(ln.m, ln.n, ln.k, row.nc), bytes_per_rep(ln.m, ln.n, ln.k, row.nc) / ln.ops,
            math.ceil(ln.n / row.nc), math.ceil(ln.k / ln.kc), math.ceil(ln.m / ln.mc),
            ln.m, ln.k, ln.m, row.slow_share_avg))
    return sorted(out, key=lambda f: -f.slow_share)


# ---- markdown ---------------------------------------------------------------------------------
def _f(x: float, d: int = 2) -> str:
    return "-" if x != x else f"{x:.{d}f}"


def format_inventory(inv: list[InventoryRow]) -> str:
    summary: dict[tuple[str, str, str, str, str, bool, str], list[InventoryRow]] = {}
    for r in inv:
        summary.setdefault((r.board, r.build, r.protocol, r.core, r.packer, r.nc_override, r.used), []).append(r)
    out = ["| board | build | protocol | core | packer | nc override | counts? | per-rep files | layers seen |",
           "|---|---|---|---|---|---|---|---|---|"]
    for (board, build, proto, core, packer, nco, used), rows in sorted(summary.items()):
        layers = sorted({layer for r in rows for layer in r.layers})
        out.append(f"| {board} | {build} | {proto} | {core} | {packer} | {'ignored' if nco else 'shipped'} | {used} | "
                   f"{len(rows)} | {', '.join(layers)} |")
    out += ["", "One row per per-rep file:", "",
            "| run folder | board | build | protocol | core | packer | tile | counts? |", "|---|---|---|---|---|---|---|---|"]
    for r in inv:
        out.append(f"| {r.run} | {r.board} | {r.build} | {r.protocol} | {r.core} | {r.packer}{' (nc ignored)' if r.nc_override else ''} | {r.tiles} | {r.used} |")
    return "\n".join(out)


def format_slow_table(t: SlowTable) -> str:
    out = ["| board | packer | tile | case | nc | launches (with a slow state) | GOPs by mean time (avg, lo-hi) | GOPs by median rep | slow reps % (avg, lo-hi) | fast start s (avg) | fast start reps (avg) | clumping | CV % |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.rows:
        out.append(f"| {r.board} | {r.packer} | {r.tile} | {r.layer} ({r.m}x{r.n}x{r.k}) | {r.nc} | {r.launches} ({r.two_speed_launches}) | "
                   f"{_f(r.gops_avg)} ({_f(r.gops_lo)}-{_f(r.gops_hi)}) | {_f(r.gops_median_avg)} | "
                   f"{100 * r.slow_share_avg:.0f} ({100 * r.slow_share_lo:.0f}-{100 * r.slow_share_hi:.0f}) | "
                   f"{_f(r.fast_start_s_avg)} | {_f(r.fast_start_reps_avg, 0)} | {_f(r.clumping_avg)} | {_f(r.cv_avg, 1)} |")
    if t.no_result:
        out += ["", f"No result (fewer than {MIN_LAUNCHES} clean launches, issue #39 rule 2): " +
                "; ".join(f"{r.board} {r.packer} {r.tile} {r.layer} nc={r.nc} ({r.launches})" for r in t.no_result)]
    return "\n".join(out)


def format_switch_table(t: SwitchTable) -> str:
    out = ["| board | packer | tile | case | nc | launches (slow state) | reps before (min / median / max) | seconds before (min / median / max) | MiB packed before (mean) | packed MiB per rep | CV % reps / s / bytes |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t.rows:
        out.append(f"| {r.board} | {r.packer} | {r.tile} | {r.layer} | {r.nc} | {r.launches} ({r.two_speed_launches}) | "
                   f"{r.reps_min:.0f} / {r.reps_median:.0f} / {r.reps_max:.0f} | "
                   f"{r.seconds_min:.3f} / {r.seconds_median:.3f} / {r.seconds_max:.3f} | {r.bytes / 2**20:.0f} | {r.bytes_per_rep / 2**20:.2f} | "
                   f"{r.cv['reps']:.0f} / {r.cv['seconds']:.0f} / {r.cv['bytes']:.0f} |")
    out += ["", "Spread across shapes of each case's mean switch point (CV %, cases with 2 or more slow-state launches; lower = closer to constant):", "",
            "| board | packer | cases | reps | seconds | bytes packed | closest to constant |", "|---|---|---|---|---|---|---|"]
    for c in t.contexts:
        out.append(f"| {c.board} | {c.packer} | {c.cases} | {_f(c.across['reps'], 0)} | {_f(c.across['seconds'], 0)} | {_f(c.across['bytes'], 0)} | {c.closest or 'not enough cases'} |")
    return "\n".join(out)


def format_feature_table(rows: list[FeatureRow]) -> str:
    def yn(b: bool) -> str:
        return "yes" if b else "no"

    out = ["| slow reps % | board | packer | tile | case | M | N | K | nc | mc | kc | jc x pc x ic blocks | B block KiB (>L2?) | A block KiB | packed MiB/rep | packed B per kFLOP | lda=M 2^k | ldb=K 2^k | B block 2^k | A block 2^k |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in rows:
        out.append(f"| {100 * f.slow_share:.0f} | {f.board} | {f.packer} | {f.tile} | {f.layer} | {f.m} | {f.n} | {f.k} | {f.nc} | {f.mc} | {f.kc} | "
                   f"{f.jc_blocks}x{f.pc_blocks}x{f.ic_blocks} | "
                   f"{f.b_block_bytes / 1024:.0f} ({yn(f.b_block_bytes > L2_BYTES)}) | {f.a_block_bytes / 1024:.0f} | {f.pack_bytes_per_rep / 2**20:.2f} | "
                   f"{1000 * f.pack_bytes_per_flop:.1f} | "
                   f"{yn(f.lda_pow2)} | {yn(f.ldb_pow2)} | {yn(f.b_block_pow2)} | {yn(f.a_block_pow2)} |")
    return "\n".join(out)


def report(logs: Path, until: str | None = None) -> str:
    launches, inventory = collect(logs, until)
    table = slow_table(launches)
    return "\n".join([
        f"# Slow state across every shape (issue #47): {len(launches)} case-launches from {len({i.run for i in inventory})} run folders"
        + (f" up to {until}" if until else ""),
        "",
        f"Slow state = at least {100 * TWO_SPEED_SHARE:.0f}% of a launch's reps slow (reps_report.py's split). "
        f"Speeds from uninstrumented reference-protocol logs on a pinned core other than 0 only.",
        "",
        "## 1. Inventory", "", format_inventory(inventory), "",
        "## 2. Slow state per case", "", format_slow_table(table), "",
        "## 3. Time, reps or bytes", "",
        "Bytes packed per rep = 4 * (K*N + M*K*ceil(N/nc)) (fp32).", "", format_switch_table(switch_table(launches)), "",
        "## 4. Shape features", "", format_feature_table(feature_table(launches)), ""])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="?", type=Path, default=Path(__file__).resolve().parent / "logs")
    ap.add_argument("--out", type=Path, help="also write the report here")
    ap.add_argument("--until", help="leave out run folders after this stamp, e.g. 2026-09-28_235959 (the logs issue #47 read)")
    args = ap.parse_args(argv)
    text = report(args.logs, args.until)
    print(text, end="")
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
