"""
The raw-log reader (issue #117): the one place that knows how the driver names a run's merged
CSV and how to read it. Reports ask this module for "the log of this variant"; none of them
spells a suffix.

The driver builds the name in `templates/driver_benchmark.cpp` (`log_suffix`), in this order:

    _noinstr  _refproto  _scalarpack  _zeroline  _padldb<N>  _packb-<name>  _no_nc_override

`compile_macro_bench.sh` merges a run's per-project logs into `all_raw_<dtype>_<method><suffix>.txt`.
An instrumented build has no `_noinstr` and no `_refproto`. If the driver gets a new part, add it
here, in the driver's position, and nowhere else.
"""

import csv
from pathlib import Path

# What an instrumented run with the old protocol wrote; kept so old committed runs stay readable.
INSTRUMENTED_NAME = "all_raw_fp32_AXPY_FP32_Pack.txt"
SCALAR_PACK_PART = "_scalarpack"
ZERO_LINE_PART = "_zeroline"
NO_NC_OVERRIDE_PART = "_no_nc_override"
# How an explicit --raw-name of a run without the config.json nc overrides ends.
NO_NC_OVERRIDE_SUFFIX = "_no_nc_override.txt"


def raw_glob(
    *,
    instrumented: bool = False,
    refproto: bool = True,
    scalar_pack: bool = False,
    zero_line: bool = False,
    pad_ldb: int = 0,
    pack_b: str = "",
    no_nc_override: bool = False,
) -> str:
    """The glob that matches the merged raw log of one variant, parts in the driver's order."""
    suffix = (
        ("" if instrumented else "_noinstr")
        + ("_refproto" if refproto else "")
        + (SCALAR_PACK_PART if scalar_pack else "")
        + (ZERO_LINE_PART if zero_line else "")
        + (f"_padldb{pad_ldb}" if pad_ldb > 0 else "")
        + (f"_packb-{pack_b}" if pack_b else "")
        + (NO_NC_OVERRIDE_PART if no_nc_override else "")
    )
    return f"all_raw_*{suffix}.txt"


def any_raw(run_dir: Path) -> list[Path]:
    """Every merged raw log of a run, whatever its variant."""
    return sorted(Path(run_dir).glob("all_raw_*.txt"))


def find_raw(run_dir: Path, **variant: object) -> list[Path]:
    """The merged raw logs of one variant in a run. Empty when the run has none."""
    return sorted(Path(run_dir).glob(raw_glob(**variant)))  # type: ignore[arg-type]


def require_raw(run_dir: Path, **variant: object) -> list[Path]:
    """Like find_raw, but a missing log stops the report and names what the run does hold."""
    found = find_raw(run_dir, **variant)
    if not found:
        held = ", ".join(p.name for p in any_raw(run_dir)) or "no all_raw_*.txt"
        raise SystemExit(f"{run_dir}: no {raw_glob(**variant)} to read (it holds: {held})")  # type: ignore[arg-type]
    return found


def resolve_raw(run_dir: Path, name: str | None = None) -> Path | None:
    """The log `compare_ref_baseline.py` reads for a run. An explicit file name wins. Without one,
    only the old instrumented name is read: a Reference Protocol log has an extra column and its own
    name, so it is compared on request and never merged into a default run's numbers (issue #27)."""
    path = Path(run_dir) / (name or INSTRUMENTED_NAME)
    return path if path.is_file() else None


def read_rows(path: Path) -> list[dict[str, str]]:
    """The rows of a raw log, by column name."""
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def read_tsv(path: Path) -> list[dict[str, str]]:
    """A series' `runs.tsv` (or any header-first TSV), by column name. Blank lines are skipped."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        return []
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]
