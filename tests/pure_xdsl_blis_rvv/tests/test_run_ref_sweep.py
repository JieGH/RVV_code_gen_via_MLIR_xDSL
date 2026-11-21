"""
Issue #29: `run_ref_sweep.sh` runs the reference design N times and says which runs it can trust.

The reference runner is replaced by a stub (`SWEEP_REF`) and the logs go to a temp dir
(`SWEEP_LOGS`), so no board is touched. What is checked is the loop: which settings each run is
given, that a run which exits non-zero or leaves a row out is named as failed and left out of
`runs.txt`, that the sweep goes on after it, and that the exit code says something failed.
`run_ref_baseline.sh` itself only warns when one of its builds fails and exits 0, which is why the
sweep checks the results.
"""

import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SWEEP = HERE / "run_ref_sweep.sh"

# Two layers on one build, so "a row missing" is one line of the CSV, not a whole file.
STUB = r"""#!/bin/bash
# Stand-in for run_ref_baseline.sh. The Nth call (counted in $SWEEP_LOGS/.calls) does what
# $STUB_BEHAVIOUR says for it: ok, exit1, short (leaves layer 2's row out) or nan (writes nan for it).
echo "$@" >> "${SWEEP_LOGS}/args.txt"
n=$(( $(cat "${SWEEP_LOGS}/.calls" 2>/dev/null || echo 0) + 1 ))
echo "${n}" > "${SWEEP_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
sleep 1.1   # result dirs are named to the second
d="${SWEEP_LOGS}/refbase_$(date +%Y-%m-%d_%H%M%S)"
mkdir -p "${d}"
printf '# case\tnumid\tM\tN\tK\tpMR\tpNR\tbMR\tbNR\tkind\tMC\tNC\tKC\ttag\n' > "${d}/run_plan.tsv"
printf 'B1\t1\t8\t8\t8\t32\t6\t32\t6\texact\t8\t8\t8\t32x6\n' >> "${d}/run_plan.tsv"
printf 'B2\t2\t8\t8\t8\t32\t6\t32\t6\texact\t8\t8\t8\t32x6\n' >> "${d}/run_plan.tsv"
printf '#l;m;n;k;Gflops\n1;8;8;8;7.00e+00\n' > "${d}/result_32x6_macc.csv"
[ "${what}" = short ] || printf '2;8;8;8;%s\n' "$([ "${what}" = nan ] && echo nan || echo 7.00e+00)" >> "${d}/result_32x6_macc.csv"
exit 0
"""


def run_sweep(tmp_path: Path, behaviour: str, *extra: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    stub = tmp_path / "stub_ref.sh"
    stub.write_text(STUB)
    stub.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    env = {**os.environ, "SWEEP_LOGS": str(logs), "SWEEP_REF": str(stub), "STUB_BEHAVIOUR": behaviour}
    done = subprocess.run(
        [str(SWEEP), "--boards", "a,b", "--runs", "2", "--interval", "0", "--no-compare", *extra],
        env=env, capture_output=True, text=True, timeout=60, check=False,
    )
    return done, logs


def sweep_dir(logs: Path) -> Path:
    return sorted(logs.glob("sweep_ref_*"))[-1]


def test_a_clean_sweep_lists_every_run_and_exits_zero(tmp_path: Path):
    done, logs = run_sweep(tmp_path, "ok,ok,ok,ok")

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Sweep finished: 4 of 4 runs ok." in done.stdout
    boards = [line.split("\t")[0] for line in (sweep_dir(logs) / "runs.txt").read_text().splitlines()]
    assert boards == ["a", "b", "a", "b"]  # the boards take turns


def test_every_run_gets_the_reference_configuration(tmp_path: Path):
    _, logs = run_sweep(tmp_path, "ok,ok,ok,ok")

    first = (logs / "args.txt").read_text().splitlines()[0]
    for flag in ("--jump", "--opt LDX", "--variants 2", "--bracket", "--cpu 4", "--tmin 10", "--no-compare", "--riscv-ip a"):
        assert flag in first, f"{flag!r} missing from: {first}"


def test_a_run_that_exits_non_zero_is_named_and_the_sweep_goes_on(tmp_path: Path):
    done, logs = run_sweep(tmp_path, "ok,exit1,ok,ok")

    assert done.returncode == 1
    assert "Sweep finished: 3 of 4 runs ok." in done.stdout
    assert "failed: run 2 on b (exit 1)" in done.stdout
    assert len((sweep_dir(logs) / "runs.txt").read_text().splitlines()) == 3


def test_a_run_that_leaves_a_row_out_is_failed_even_though_it_exited_zero(tmp_path: Path):
    done, logs = run_sweep(tmp_path, "ok,short,ok,ok")

    assert done.returncode == 1
    assert "no result for 32x6/2" in done.stdout
    assert "Sweep finished: 3 of 4 runs ok." in done.stdout
    assert len((sweep_dir(logs) / "runs.txt").read_text().splitlines()) == 3


def test_the_settings_can_be_changed_and_are_passed_through(tmp_path: Path):
    _, logs = run_sweep(tmp_path, "ok,ok,ok,ok", "--opt", "OPT", "--variants", "0", "--tmin", "3", "--cpu", "4")

    first = (logs / "args.txt").read_text().splitlines()[0]
    assert "--opt OPT" in first and "--variants 0" in first and "--tmin 3" in first and "--cpu 4" in first


@pytest.mark.parametrize(
    "bad",
    [["--runs", "0"], ["--interval", "x"], ["--start-at", "25:00"], ["--opt", "FOO"], ["--variants", "3"],
     ["--cpu", "1", "--no-pin"], ["--nonsense"]],
)
def test_bad_flags_are_refused_before_anything_runs(tmp_path: Path, bad: list[str]):
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "SWEEP_LOGS": str(logs)}

    done = subprocess.run([str(SWEEP), *bad], env=env, capture_output=True, text=True, timeout=30, check=False)

    assert done.returncode != 0
    assert not list(logs.glob("sweep_ref_*"))


def test_the_dry_run_prints_the_schedule_and_stops(tmp_path: Path):
    stub = tmp_path / "stub_ref.sh"
    stub.write_text("#!/bin/bash\necho \"STUB $*\"\n")
    stub.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "SWEEP_LOGS": str(logs), "SWEEP_REF": str(stub)}

    done = subprocess.run([str(SWEEP), "--dry-run"], env=env, capture_output=True, text=True, timeout=30, check=False)

    assert done.returncode == 0
    assert "Reference sweep: 16 runs (8 per board on jbpi,jbpi2), one every 30 min" in done.stdout
    assert "STUB" in done.stdout and "--dry-run" in done.stdout
    assert not list(logs.glob("sweep_ref_*"))


def test_a_run_whose_gflops_is_not_a_number_is_failed(tmp_path: Path):
    done, logs = run_sweep(tmp_path, "ok,nan,ok,ok")

    assert done.returncode == 1
    assert "no result for 32x6/2" in done.stdout
    assert len((sweep_dir(logs) / "runs.txt").read_text().splitlines()) == 3


def test_core_0_reaches_each_run_only_when_allowed_on_purpose(tmp_path: Path):
    # Issue #39: never core 0, unless --allow-core-0 says it is to reproduce an old run.
    _, logs = run_sweep(tmp_path, "ok,ok,ok,ok", "--cpu", "0", "--allow-core-0")
    first = (logs / "args.txt").read_text().splitlines()[0]
    assert "--cpu 0" in first and "--allow-core-0" in first
