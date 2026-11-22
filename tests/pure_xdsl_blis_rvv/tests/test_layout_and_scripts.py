"""
Issue #137: the folder layout holds, no path in code or docs is broken, and every runner and reader
starts from where it lives. No board is touched.

- Top level: only the main workflow (generator, compile_macro_bench.sh, board_pin.sh, config.json).
- Paths: every `tests/pure_xdsl_blis_rvv/<file>` in code and docs, and every `${HERE}/<file>` in a
  shell script (HERE resolved from the script's own `HERE=` line), names a file that exists.
- Scripts: each runner and reader answers `--help` from the repo root; each runner that can plan
  offline finishes `--dry-run`.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]  # the project folder
REPO = HERE.parents[1]
needs_ref = pytest.mark.skipif(not (REPO.parent / "ext" / "RVV_ukernels_benchmark").is_dir(),
                            reason="needs ../ext/RVV_ukernels_benchmark (README: Reference Design)")  # issue #149

TOP_LEVEL = {
    "generate_pure_blis_project.py", "pack_builder.py", "config.py", "cache_config.py",
    "compile_macro_bench.sh", "board_pin.sh", "config.json", "README.md", "__init__.py",
}
RUNNERS = sorted((HERE / "runners").glob("*.sh"))
PY_SCRIPTS = sorted([*(HERE / "readers").glob("*.py"), *(HERE / "runners").glob("*.py")])
SOURCED_ONLY = {"hector_yolo_dump.sh", "cron_bench_run.sh", "gather_sweep_report.sh"}  # no --help: sourced, or cron/wait
DRY_RUN = {  # runner -> extra args it needs to plan offline (run_hector_gemm.sh needs his clone: test_run_hector_gemm.py)
    "run_yolo_sweep.sh": [], "run_paper_sweep.sh": ["--issue", "0"], "run_parallel_gemm.sh": [],
    "run_ref_sweep.sh": [], "run_scalar_pack_sweep.sh": [], "run_pack_swap.sh": [], "run_head_to_head.sh": [],
}
SKIP_DIRS = {"logs", "notes", "pj_gen", "__pycache__", "generated"}  # history and generated output keep old paths


def _text_files() -> list[Path]:
    own = [f for f in HERE.rglob("*") if f.is_file() and f.suffix in {".sh", ".py", ".md", ".json"}
           and not SKIP_DIRS & set(f.relative_to(HERE).parts)]
    return [*own, REPO / "README.md", REPO / "CONTEXT.md", *REPO.glob("docs/**/*.md")]


def _here_of(script: Path, text: str) -> Path | None:
    m = re.search(r'^HERE="\$\(cd "\$\(dirname "(?:\$\{BASH_SOURCE\[0\]\}|\$0)"\)((?:/\.\.)*)" && pwd\)"', text, re.M)
    if not m:
        return None
    base = script.parent
    for _ in range(m[1].count("/..")):
        base = base.parent
    return base


def test_top_level_holds_only_the_main_workflow():
    files = {f.name for f in HERE.iterdir() if f.is_file() and not f.name.startswith(".")}
    assert files == TOP_LEVEL, f"unexpected: {files - TOP_LEVEL}, missing: {TOP_LEVEL - files}"


CONFIGS = sorted([HERE / "config.json", *(HERE / "configs").rglob("*.json"), *(HERE / "experiments").rglob("*.json")])


@pytest.mark.parametrize("cfg", CONFIGS, ids=lambda p: str(p.relative_to(HERE)))
def test_every_config_names_datasets_that_exist(cfg: Path):
    from config import BLISProjectConfig

    missing = [d for c in BLISProjectConfig.from_json(str(cfg)) for d in (c.datasets or [])
               if not (HERE / "datasets" / d).is_file()]
    assert not missing, f"{cfg.name} names datasets that don't exist: {missing}"


def test_top_level_datasets_hold_only_shared_workloads():
    # Issue #146: one-issue inputs live in datasets/experiments/, unused ones in datasets/unused/.
    one_issue = sorted(p.name for p in (HERE / "datasets").glob("issue*.dat"))
    assert not one_issue, f"one-issue datasets at the top of datasets/: {one_issue}"
    assert (HERE / "datasets" / "README.md").is_file(), "datasets/README.md (old -> new names) is missing"


def test_no_broken_paths_in_code_or_docs():
    full = re.compile(r"tests/pure_xdsl_blis_rvv/([\w./-]+\.(?:sh|py|json|md|dat|tsv))")
    here = re.compile(r"\$\{HERE\}/([\w./-]+\.(?:sh|py|json|dat|tsv))")
    broken = []
    for f in _text_files():
        text = f.read_text(errors="replace")
        broken += [f"{f.relative_to(REPO)}: {m[0]}" for m in full.finditer(text) if not (HERE / m[1]).exists()]
        base = _here_of(f, text) if f.suffix == ".sh" else None
        if base:
            broken += [f"{f.relative_to(REPO)}: {m[0]}" for m in here.finditer(text) if not (base / m[1]).exists()]
    assert not broken, "\n".join(broken)


def test_here_resolver_on_a_known_runner():
    script = HERE / "runners" / "run_yolo_sweep.sh"
    assert _here_of(script, script.read_text()) == HERE


@pytest.mark.parametrize("script", RUNNERS, ids=lambda p: p.name)
def test_runner_parses(script: Path):
    assert subprocess.run(["bash", "-n", str(script)], capture_output=True).returncode == 0


@pytest.mark.parametrize("script", [r for r in RUNNERS if r.name not in SOURCED_ONLY], ids=lambda p: p.name)
def test_runner_help(script: Path):
    done = subprocess.run(["bash", str(script), "--help"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and script.name in done.stdout, done.stdout + done.stderr


@pytest.mark.parametrize("script", PY_SCRIPTS, ids=lambda p: p.name)
def test_python_script_help(script: Path):
    if script.name == "summarize_run.py":  # no argparse: no arguments prints its usage and exits 1
        done = subprocess.run([sys.executable, str(script)], cwd=REPO, capture_output=True, text=True, timeout=120)
        assert done.returncode == 1 and "usage: summarize_run.py" in done.stderr, done.stderr
        return
    done = subprocess.run([sys.executable, str(script), "--help"], cwd=REPO, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs_ref) if n == "run_head_to_head.sh" else n for n in sorted(DRY_RUN)])
def test_runner_dry_run(name: str):
    done = subprocess.run(["bash", str(HERE / "runners" / name), *DRY_RUN[name], "--dry-run"],
                          cwd=REPO, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and "--dry-run:" in done.stdout, done.stdout[-2000:] + done.stderr[-2000:]


def test_runners_and_readers_have_a_readme_entry():
    for folder in ("runners", "readers"):
        readme = (HERE / folder / "README.md").read_text()
        missing = [f.name for f in (HERE / folder).iterdir() if f.suffix in {".sh", ".py"} and f"`{f.name}" not in readme]
        assert not missing, f"{folder}/README.md does not list {missing}"
