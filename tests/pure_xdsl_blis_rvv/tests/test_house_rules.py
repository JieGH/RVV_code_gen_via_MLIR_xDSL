"""
Issue #39: the house rules for a benchmark result, as tests.

1. Never time on core 0: every script refuses it (test_board_pin.py); here, no documented
   command asks for it without --allow-core-0.
2. One launch is not a result: the per-rep log is on by default (test_ref_protocol.py) and
   reps_report.py warns below 3 launches (test_reps_report.py).
3. Every quoted number comes from a committed log: here, every run folder an issue file, a
   report or a doc cites is committed.
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


@pytest.mark.parametrize("doc", COMMAND_DOCS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_documented_command_pins_to_core_0_without_saying_so(doc: Path):
    bad = [
        line.strip() for line in doc.read_text().splitlines()
        if re.search(r"--cpu\s+0\b", line) and "--allow-core-0" not in line
    ]
    assert not bad, f"{doc.name} tells readers to run on core 0 (issue #39, rule 1):\n" + "\n".join(bad)
