"""
Issue #146: `.scratch/issues/README.md` maps each GitHub issue to its local issue file(s), experiment
folder, one-issue datasets and report files. `.scratch/issues/make_index.py` writes it from the tree.

Seam: the script's output and its `--check` exit code. No board needed.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SCRIPT = REPO / ".scratch" / "issues" / "make_index.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO, capture_output=True, text=True)


def test_committed_index_is_up_to_date():
    result = _run("--check")
    assert result.returncode == 0, "index is stale; run: python .scratch/issues/make_index.py\n" + result.stdout


def test_issue_85_row_links_all_its_places():
    row = next(line for line in _run("--stdout").stdout.splitlines() if line.startswith("| #85 "))
    assert "0051-when-does-vector-packing-pay-off.md" in row  # local number 0051 is GitHub #85
    assert "experiments/issue85/" in row
    assert "issue85_sweep.dat" in row
    assert "reports/issue-85/" in row


def test_a_file_with_no_github_line_is_listed_as_local_only():
    out = _run("--stdout").stdout
    assert "| local 0001 " in out


def test_a_two_number_report_name_covers_the_whole_range():
    # "issues-31-34-brief" covers #31 to #34; "issues-35-36-39-brief" only those three.
    out = _run("--stdout").stdout.splitlines()
    row = lambda n: next(line for line in out if line.startswith(f"| #{n} "))  # noqa: E731
    assert "issues-31-34-brief.tex" in row(32)
    assert "issues-35-36-39-brief.tex" not in row(37)
