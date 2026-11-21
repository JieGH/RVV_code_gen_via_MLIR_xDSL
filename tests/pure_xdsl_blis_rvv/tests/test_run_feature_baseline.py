"""Issue #137: run_feature_baseline.sh, the board correctness run of every user-facing option. No board here:
its plan covers every data type and packer the bench offers, the bench accepts every case, and its configs generate."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
SCRIPT = HERE / "runners" / "run_feature_baseline.sh"
sys.path.insert(0, str(Path(__file__).resolve().parent))  # P/tests, for sibling test modules

from test_features import _generate  # noqa: E402


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), *args], cwd=REPO, capture_output=True, text=True, timeout=600)


def test_cases_cover_every_option():
    names = _run("--list").stdout.split()
    text = SCRIPT.read_text()
    bench = (HERE / "compile_macro_bench.sh").read_text()
    pack_b = re.search(r'^PACK_B_NAMES="([^"]+)"', bench, re.M)[1].split()
    assert [n for n in pack_b if n != "row" and f"pack-b-{n}" not in names] == []  # row is the default case
    for flag in ("--scalar-pack", "--zero-pack-lines", "--pad-ldb", "--threads"):
        assert flag in text
    for dtype in ("fp32-axpy", "fp32-dot", "f16f16f32", "int8"):
        assert any(n.startswith(dtype) for n in names), dtype
    assert "search-mode" in names


def test_bench_accepts_every_case():
    done = _run("--dry-run")
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr
    assert done.stdout.count("  accepted") == len(_run("--list").stdout.split())


def test_only_runs_a_subset():
    done = _run("--only", "int8,search-mode", "--dry-run")
    assert done.returncode == 0 and done.stdout.count("  accepted") == 2


@pytest.mark.parametrize("config", ["config_feature_dot.json", "config_feature_int8.json", "config_feature_search.json"])
def test_feature_config_generates(tmp_path: Path, config: str):
    rows = re.sub(r"//.*", "", (HERE / "configs" / "correctness" / config).read_text()).strip()
    _generate(tmp_path, rows.strip()[1:-1])


def _count_pass(run: Path) -> tuple[int, int]:
    fn = re.search(r"^count_pass\(\) \{.*?^\}", SCRIPT.read_text(), re.M | re.S)[0]
    out = subprocess.run(["bash", "-c", f'{fn}\ncount_pass "$1"', "_", str(run)], capture_output=True, text=True).stdout.split()
    return int(out[0]), int(out[1])


def test_verdict_reads_the_status_column_by_name(tmp_path: Path):
    # The merged all_raw log puts Project first, so Status is not the per-project CSV's 6th column.
    header = "Project,Layer,M,N,K,Variant,Status,KerGOPs_Harmonic\n"
    (tmp_path / "all_raw_fp32_x.txt").write_text(
        header + "p,C_full,96,210,128,MR16_NR15,PASS,12.0\np,C_edges,123,57,300,MR16_NR15,FAIL,8.4\n")
    (tmp_path / "all_raw_fp32_y.txt").write_text(header + "p,C_tiny,12,5,17,MR32_NR6,PASS,1.1\n")
    assert _count_pass(tmp_path) == (2, 3)


def test_verdict_on_a_real_run():
    run = HERE / "logs" / "run_2026-11-21_184415"  # the fp32 AXPY case of the first feature baseline (issue #137)
    assert _count_pass(run) == (9, 9)
