"""Issue #146: write .scratch/issues/README.md, one row per issue, linking the places its files live.

Issue files use a local number (0051-...md); everything else uses the GitHub number (#85). The bridge is
each file's `GitHub: .../issues/N` line; files that share a local number share that GitHub number.

    python .scratch/issues/make_index.py           # rewrite README.md
    python .scratch/issues/make_index.py --check   # exit 1 if README.md is stale
    python .scratch/issues/make_index.py --stdout  # print, write nothing
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

ISSUES = Path(__file__).resolve().parent
REPO = ISSUES.parents[1]
PROJECT = REPO / "tests" / "pure_xdsl_blis_rvv"
REPORTS = REPO / ".scratch" / "reports"
OUT = ISSUES / "README.md"

GITHUB = re.compile(r"^GitHub: https://github\.com/\S+/issues/(\d+)", re.M)
# Report names: "issue-85", "issue-54-brief.tex", "issues-31-34-brief" (a range), "issues-35-36-39-brief" (a list).
# Issue numbers have at most 3 digits, so a date in a name ("issue-85-2026-11-18") is not read as issues.
REPORT = re.compile(r"^issues?-(\d{1,3}(?:-\d{1,3})*)(?=[-.]|$)")
EXPERIMENT = re.compile(r"^issue(\d{1,3})$")
DATASET = re.compile(r"^issue(\d{1,3})_")


def _report_numbers(spec: str) -> list[int]:
    nums = [int(n) for n in spec.split("-")]
    return list(range(nums[0], nums[1] + 1)) if len(nums) == 2 and nums[1] > nums[0] else nums


def _title(text: str) -> str:
    first = text.splitlines()[0] if text else ""
    return first.lstrip("# ").replace("|", "/").strip()


def build() -> str:
    groups: dict[str, list[Path]] = defaultdict(list)  # local number -> its files
    for f in sorted(ISSUES.glob("[0-9][0-9][0-9][0-9]-*")):
        groups[f.name[:4]].append(f)

    rows: dict[int, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    local_only: list[str] = []
    for num, files in groups.items():
        texts = {f: f.read_text(errors="replace") for f in files if f.suffix == ".md"}
        links = ", ".join(f"[{f.name}]({f.name})" for f in files)
        linked = [(f, int(gh)) for f, t in texts.items() for gh in GITHUB.findall(t)]
        if not linked:
            local_only.append(f"| local {num} | {_title(next(iter(texts.values()), ''))} | {links} |")
            continue
        for f, gh in linked:
            rows[gh]["title"].append(_title(texts[f]))
            rows[gh]["files"].append(links)

    for d in sorted((PROJECT / "experiments").iterdir()):
        if d.is_dir() and (m := EXPERIMENT.match(d.name)):
            rows[int(m.group(1))]["exp"].append(
                f"[experiments/{d.name}/](../../tests/pure_xdsl_blis_rvv/experiments/{d.name}/)")
    for f in sorted((PROJECT / "datasets" / "experiments").glob("*.dat")):
        if m := DATASET.match(f.name):
            rows[int(m.group(1))]["data"].append(f.name)
    unmatched: list[str] = []
    for r in sorted(REPORTS.iterdir()):
        slash = "/" if r.is_dir() else ""
        link = f"[reports/{r.name}{slash}](../reports/{r.name}{slash})"
        if m := REPORT.match(r.name):
            for n in _report_numbers(m.group(1)):
                rows[n]["reports"].append(link)
        else:
            unmatched.append(link)

    lines = [
        "# Issue index",
        "",
        "Written by `make_index.py` (issue #146); don't edit by hand. Rerun it after adding an issue file,",
        "experiment folder, one-issue dataset or report: `python .scratch/issues/make_index.py`.",
        "",
        "Issue files use a local number; everything else uses the GitHub number.",
        "",
        "| GitHub | Title | Issue file(s) | Experiment | One-issue datasets | Reports |",
        "|---|---|---|---|---|---|",
    ]
    for gh in sorted(rows):
        r = rows[gh]
        lines.append(" ".join([
            f"| #{gh} |", (r["title"] or [""])[0], "|", ", ".join(r["files"]), "|", ", ".join(r["exp"]), "|",
            ", ".join(r["data"]), "|", ", ".join(r["reports"]), "|",
        ]))
    lines += ["", "## Local only (no `GitHub:` line)", "", "| Local | Title | Files |", "|---|---|---|", *local_only]
    lines += ["", "## Reports not tied to one issue number", "", *(f"- {u}" for u in unmatched), ""]
    return "\n".join(lines)


def main() -> int:
    text = build()
    if "--stdout" in sys.argv:
        print(text, end="")
    elif "--check" in sys.argv:
        if not OUT.is_file() or OUT.read_text() != text:
            print(f"{OUT.relative_to(REPO)} is stale; run: python .scratch/issues/make_index.py")
            return 1
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
