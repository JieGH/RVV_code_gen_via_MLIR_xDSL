"""
Issue #46: `run_issue46.sh` runs ours, the reference unchanged and the reference with a per-rep record
in launches whose order changes, and says which runs it can trust.

Both runners are replaced by stubs (`ISSUE46_OURS`, `ISSUE46_REF`), the board's process list by a
canned one (`ISSUE46_PS_CMD`), and the logs go to a temp dir, so no board is touched. What is checked
is the loop: the order of the runs in each launch and the boards taking turns, the settings each run
is given, that a run which exits non-zero or leaves its result out is named as failed and the series
goes on, and that another user's heavy job is written down as a RED NOTE without stopping anything.
"""

import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SERIES = HERE / "run_issue46.sh"

STUB_OURS = r"""#!/bin/bash
echo "OURS $*" >> "${ISSUE46_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE46_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE46_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
sleep 1.1
d="${ISSUE46_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"; mkdir -p "${d}/proj"
[ "${what}" = empty ] || : > "${d}/proj/reps_x.txt.gz"
"""

STUB_REF = r"""#!/bin/bash
echo "REF $*" >> "${ISSUE46_LOGS}/calls.txt"
n=$(( $(cat "${ISSUE46_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE46_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
sleep 1.1
d="${ISSUE46_LOGS}/refbase_$(date +%Y-%m-%d_%H%M%S)"; mkdir -p "${d}"
printf '#l;m;n;k;Gflops\n3;64;384;384;%s\n' "$([ "${what}" = nan ] && echo nan || echo 7.00e+00)" > "${d}/result_16x14_macc.csv"
case " $* " in *" --per-rep "*) [ "${what}" = empty ] || : > "${d}/reps_ref_16x14_macc.txt.gz" ;; esac
[ "${what}" = mismatch ] && echo "  MISMATCH: driver says 5.0 GFLOPS"
exit 0
"""


def run_series(tmp_path: Path, behaviour: str, *extra: str, ps: str = "") -> tuple[subprocess.CompletedProcess[str], Path]:
    for name, body in (("ours.sh", STUB_OURS), ("ref.sh", STUB_REF)):
        (tmp_path / name).write_text(body)
        (tmp_path / name).chmod(0o755)
    (tmp_path / "ps.sh").write_text(f"#!/bin/bash\ncat <<'EOF'\n{ps}EOF\n")
    (tmp_path / "ps.sh").chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    # Folders that sort after every stamped one, as the real logs/ has (refbase_issue33_2026-09-25): they must not be taken for a result.
    (logs / "refbase_issue33_2026-09-25").mkdir(exist_ok=True)
    (logs / "run_zz_old").mkdir(exist_ok=True)
    env = {**os.environ, "ISSUE46_LOGS": str(logs), "ISSUE46_OURS": str(tmp_path / "ours.sh"),
           "ISSUE46_REF": str(tmp_path / "ref.sh"), "ISSUE46_PS_CMD": str(tmp_path / "ps.sh"),
           "STUB_BEHAVIOUR": behaviour}
    done = subprocess.run([str(SERIES), "--boards", "a,b", "--launches", "2", *extra],
                          env=env, capture_output=True, text=True, timeout=120, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue46_*"))[-1]


def steps(logs: Path) -> list[str]:
    """`<board> <launch>: <step>` for every call, from runs.tsv."""
    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    return [f"{r[0]}{r[1]}:{r[3]}" for r in rows]


ALL_OK = ",".join(["ok"] * 12)


def test_the_boards_take_turns_and_the_order_of_the_runs_changes_between_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "Series finished: 12 of 12 runs ok." in done.stdout
    assert steps(logs) == [
        "a1:ours", "a1:ref", "a1:ref-per-rep", "b1:ours", "b1:ref", "b1:ref-per-rep",
        "a2:ref-per-rep", "a2:ref", "a2:ours", "b2:ref-per-rep", "b2:ref", "b2:ours",
    ]


def test_the_result_folder_is_the_stamped_one_not_an_older_folder_that_sorts_after_it(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    rows = [ln.split("\t") for ln in (series_dir(logs) / "runs.tsv").read_text().splitlines()[1:]]
    assert all(r[5] == "0" for r in rows)
    assert all(r[4].startswith(("run_2", "refbase_2")) for r in rows), rows


def test_only_the_per_rep_run_gets_the_patch_and_every_run_the_same_pin_and_time(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK)

    calls = (logs / "calls.txt").read_text().splitlines()
    ours, ref, per_rep = calls[0], calls[1], calls[2]
    assert ours.startswith("OURS") and "--config config_issue46_ref_b3.json" in ours and "--no-nc-overrides" in ours
    assert "--per-rep" not in ref and "--per-rep" in per_rep
    for line in (ours, ref, per_rep):
        assert "--cpu 4" in line and "--tmin 10" in line and "--riscv-ip a" in line
    for line in (ref, per_rep):
        assert "--opt LDX" in line and "--variants 2" in line and "--bracket" in line and "cases_issue46.tsv" in line


def test_a_run_that_exits_non_zero_is_named_and_the_series_goes_on(tmp_path: Path):
    done, logs = run_series(tmp_path, "ok,exit1," + ",".join(["ok"] * 10))

    assert done.returncode == 1
    assert "Series finished: 11 of 12 runs ok." in done.stdout
    assert "failed: launch 1 a ref (exit 1)" in done.stdout
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


@pytest.mark.parametrize("what", ["empty", "nan"])
def test_a_run_that_leaves_its_result_out_is_failed_though_it_exited_zero(tmp_path: Path, what: str):
    # ours/per-rep with no reps file, or the plain reference with a nan GFLOPS
    behaviour = {"empty": "empty,ok,ok", "nan": "ok,nan,ok"}[what] + "," + ",".join(["ok"] * 9)
    done, _ = run_series(tmp_path, behaviour)

    assert done.returncode == 1
    assert "Series finished: 11 of 12 runs ok." in done.stdout


def test_a_per_rep_conversion_that_disagrees_with_the_driver_is_failed(tmp_path: Path):
    done, _ = run_series(tmp_path, "ok,ok,mismatch," + ",".join(["ok"] * 9))

    assert done.returncode == 1
    assert "launch 1 a ref-per-rep" in done.stdout


PS_BUSY = "psiwins        92.0   0 01:21:45 spmv\njlei           20.0   1 00:00:00 bash\nroot            9.5   4 00:00:01 sshd\n"


def test_another_users_heavy_job_is_a_red_note_and_does_not_stop_the_series(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, ps=PS_BUSY)

    assert done.returncode == 0                                   # the runs went ahead
    assert "Series finished: 12 of 12 runs ok." in done.stdout
    assert "RED NOTE" in done.stdout and "psiwins spmv 92.0% on core 0" in done.stdout
    warned = (series_dir(logs) / "load_warnings.txt").read_text()
    assert "a launch 1 (before)" in warned and "b launch 2 (after)" in warned
    assert (series_dir(logs) / "load_a_L1_before.txt").read_text().startswith("psiwins")


def test_a_quiet_board_writes_no_warning(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, ps="jlei 20.0 1 00:00:00 bash\nroot 9.5 4 00:00:01 sshd\n")

    assert "RED NOTE" not in done.stdout
    assert not (series_dir(logs) / "load_warnings.txt").exists()


@pytest.mark.parametrize(
    "bad",
    [["--launches", "0"], ["--start-at", "25:00"], ["--opt", "FOO"], ["--cpu", "1", "--no-pin"], ["--cpu", "0"], ["--nonsense"]],
)
def test_bad_flags_are_refused_before_anything_runs(tmp_path: Path, bad: list[str]):
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "ISSUE46_LOGS": str(logs)}

    done = subprocess.run([str(SERIES), *bad], env=env, capture_output=True, text=True, timeout=30, check=False)

    assert done.returncode != 0
    assert not list(logs.glob("issue46_*"))


def test_the_dry_run_prints_the_plan_and_stops(tmp_path: Path):
    stub = tmp_path / "stub.sh"
    stub.write_text('#!/bin/bash\necho "STUB $*"\n')
    stub.chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "ISSUE46_LOGS": str(logs), "ISSUE46_REF": str(stub), "ISSUE46_OURS": str(stub)}

    done = subprocess.run([str(SERIES), "--dry-run"], env=env, capture_output=True, text=True, timeout=30, check=False)

    assert done.returncode == 0
    assert "4 launches on each of jbpi,jbpi2" in done.stdout and "Launch 2   : R P O" in done.stdout
    assert "--per-rep --dry-run" in done.stdout
    assert not list(logs.glob("issue46_*"))
