"""
Issue #39: the house rules for a benchmark result, as tests.

1. Never time on core 0: every script refuses it (test_board_pin.py); here, no documented
   command asks for it without --allow-core-0.
2. One launch is not a result: the per-rep log is on by default (test_ref_protocol.py) and
   reps_report.py warns below 3 launches (test_reps_report.py).
3. Every quoted number comes from a committed log: here, every run folder an issue file, a
   report or a doc cites is committed. Except the per-rep logs (reps_*) and pid files: those
   stay on the dev machine and are never committed (issue #143).
4. Name build, protocol and core: provenance.txt and reps_report.py (test_reps_report.py).

Seam: the repo's own files and `git ls-files`. No board needed.
"""

import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO_ROOT = HERE.parents[1]

# Where results are quoted.
QUOTING_FILES = sorted(
    [*REPO_ROOT.glob(".scratch/issues/*.md"), *REPO_ROOT.glob(".scratch/reports/*.tex"),
     *REPO_ROOT.glob("docs/**/*.md"), REPO_ROOT / "README.md",
     *HERE.glob("*.md"), *HERE.glob("notes/*.md"), *HERE.glob("ref_baseline/*.md")]
)
# Where commands are documented for others to copy.
COMMAND_DOCS = [REPO_ROOT / "README.md", REPO_ROOT / "docs" / "agents" / "execution.md",
                REPO_ROOT / "docs" / "metrics.md", HERE / "ref_baseline" / "README.md"]

RUN_FOLDER = re.compile(r"\b(?:run|refbase)_\d{4}-\d{2}-\d{2}_\d{6}\b")


def _committed_run_folders() -> set[str]:
    listed = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    return set(RUN_FOLDER.findall(listed))


def test_every_cited_run_folder_is_committed():
    committed = _committed_run_folders()
    missing = sorted(
        f"{path.relative_to(REPO_ROOT)}: {name}"
        for path in QUOTING_FILES if path.is_file()
        for name in set(RUN_FOLDER.findall(path.read_text(errors="replace")))
        if name not in committed
    )
    assert not missing, "cited but not committed (issue #39, rule 3):\n" + "\n".join(missing)


PRIVATE_FILE = re.compile(r"(?:^|/)reps_[^/]*\.txt(?:\.gz)?$|^tests/pure_xdsl_blis_rvv/logs/[^/]*\.pid$")


def test_no_per_rep_log_or_pid_file_is_committed():
    # Issue #143: per-rep logs (reps_*, reps_ref_*) are 82% of the repo and stay on the dev machine.
    listed = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    tracked = [p for p in listed if PRIVATE_FILE.search(p)]
    assert not tracked, f"{len(tracked)} private files tracked (issue #143), e.g.:\n" + "\n".join(tracked[:5])


@pytest.mark.parametrize("path", [
    "tests/pure_xdsl_blis_rvv/logs/run_2099-01-01_000000/p/reps_fp32_x.txt.gz",
    "tests/pure_xdsl_blis_rvv/logs/refbase_2099-01-01_000000/reps_ref_32x7_B3.txt.gz",
    ".scratch/reports/issue-999/runs/run_2099-01-01_000000/reps_fp32_x.txt",
    "tests/pure_xdsl_blis_rvv/logs/new_series.pid",
])
def test_new_per_rep_logs_and_pid_files_are_ignored(path: str):
    ignored = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO_ROOT)
    assert ignored.returncode == 0, f"git would pick up {path} (issue #143)"


@pytest.mark.parametrize("doc", COMMAND_DOCS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_documented_command_pins_to_core_0_without_saying_so(doc: Path):
    bad = [
        line.strip() for line in doc.read_text().splitlines()
        if re.search(r"--cpu\s+0\b", line) and "--allow-core-0" not in line
    ]
    assert not bad, f"{doc.name} tells readers to run on core 0 (issue #39, rule 1):\n" + "\n".join(bad)
