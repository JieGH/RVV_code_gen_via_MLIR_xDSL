"""
Issue #55: `run_issue55.sh` runs the B packer variants (--pack-b) on the three tiles of #54 as launches of arms, rotating the
arms between launches, letting the boards take turns, and refusing to time a board that is not idle.

No board is touched: the bench script is a stub (`ISSUE55_RUN`), the board's `top -bn2 -d1` is canned (`BOARD_TOP_CMD`),
the pack-only board work is a stub (`ISSUE55_PO`), the logs go to a temp dir. What is checked: the order of arms and boards,
the flags each arm is given, the lock, the idle gate (busy then idle waits and then runs; never idle stops the series
without running; a busy reading after a run discards it and queues it once more; root and our own processes never block;
the SECOND `top` sample decides), the pack-only part, and that --dry-run makes no board call.
"""

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
SERIES = HERE / "experiments/issue55/run_issue55.sh"

CONFIG = "experiments/issue55/config_issue55_tiles.json"

# Leaves one per-rep file per row of the config it is given, unless told not to. Also tries the lock: it is run inside the
# series' flock, so a second flock on the same file must find it held.
STUB_RUN = r"""#!/bin/bash
echo "RUN $*" >> "${ISSUE55_LOGS}/calls.txt"
flock -n "${ISSUE55_LOGS}/.cron_bench.lock" true && echo free >> "${ISSUE55_LOGS}/lock_seen.txt" || echo held >> "${ISSUE55_LOGS}/lock_seen.txt"
n=$(( $(cat "${ISSUE55_LOGS}/.calls" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE55_LOGS}/.calls"
what=$(echo "${STUB_BEHAVIOUR}" | cut -d, -f"${n}")
[ "${what}" = exit1 ] && exit 1
cfg=""; while [ $# -gt 0 ]; do [ "$1" = --config ] && cfg="$2"; shift; done
rows=$(grep -c '^[[:space:]]*\["fp32"' "${STUB_CONFIG_DIR}/${cfg}")
sleep 1.1
d="${ISSUE55_LOGS}/run_$(date +%Y-%m-%d_%H%M%S)"
for ((i = 0; i < rows; i++)); do mkdir -p "${d}/proj${i}"; : > "${d}/proj${i}/reps_x.txt.gz"; done
exit 0
"""

# Canned `top -bn2 -d1` outputs. Each has a first sample (what top shows since boot: must be ignored) and a second one.
FIRST = ("top - 17:46:59 up 80 days,  1:02,  2 users,  load average: 2.00, 2.00, 2.00\n"
         "%Cpu(s): 60.0 us,  0.5 sy,  0.0 ni, 39.5 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st\n\n"
         "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\n"
         "   4000 psiwins   20   0    1000    100     10 R  92.0   0.1   1:00.00 spmv\n\n")


def top(idle: str, procs: list[tuple[str, str, str]]) -> str:
    rows = "".join(f"{1000 + i:7d} {u:<9} 20   0    1000    100     10 S {c:>5}   0.1   0:01.00 {name}\n"
                   for i, (u, c, name) in enumerate(procs))
    return (FIRST + "top - 17:47:00 up 80 days,  1:02,  2 users,  load average: 2.00, 2.00, 2.00\n"
            f"%Cpu(s):  1.0 us,  0.5 sy,  0.0 ni, {idle} id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st\n\n"
            "    PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\n" + rows)


CANNED = {
    "idle": top("98.5", [("root", "11.7", "bluetoothd"), ("jlei", "3.0", "top")]),
    "busy": top("40.0", [("psiwins", "92.0", "spmv"), ("root", "1.0", "kworker")]),
    "rootonly": top("90.0", [("root", "95.0", "bluetoothd")]),
    "lowidle": top("50.0", [("root", "1.0", "kworker")]),
    "oursbusy": top("88.0", [("jlei", "99.0", "make"), ("jlei", "50.0", "cc1plus")]),
    "justunder": top("98.0", [("psiwins", "19.9", "x")]),
    "justat": top("98.0", [("psiwins", "20.0", "x")]),
    "garbage": "ssh: connect to host jbpi port 22: Connection refused\n",
}

STUB_TOP = r"""#!/bin/bash
echo "TOP $1" >> "${ISSUE55_LOGS}/top_calls.txt"
n=$(( $(cat "${ISSUE55_LOGS}/.top_n" 2>/dev/null || echo 0) + 1 )); echo "${n}" > "${ISSUE55_LOGS}/.top_n"
IFS=, read -ra seq <<< "${TOP_SEQ}"
i=$(( n - 1 )); [ "${i}" -ge "${#seq[@]}" ] && i=$(( ${#seq[@]} - 1 ))
cat "${STUB_CANNED}/${seq[$i]}.txt"
"""

STUB_PO = r"""#!/bin/bash
echo "PO $*" >> "${ISSUE55_LOGS}/po_calls.txt"
[ "$1" = prepare ] && exit "${PO_PREPARE_RC:-0}"
# run <board> M N K MR NR variant reps cpu
echo "# board: $2"; echo "# variant: $8"; echo "# shape: M=$3 N=$4 K=$5"; echo "# tile: MR=$6 NR=$7"
echo "# blocking: mc=64 nc=384 kc=128"; echo "Rep,Pack_Seconds"; echo "0,0.001"; echo "1,0.001"; echo "# check: ok"
"""


def run_series(tmp_path: Path, behaviour: str, *extra: str, top_seq: str = "idle", po_rc: str = "0",
               boards: str = "a,b", launches: str = "2", gate: bool = True) -> tuple[subprocess.CompletedProcess[str], Path]:
    for name, body in (("run.sh", STUB_RUN), ("top.sh", STUB_TOP), ("po.sh", STUB_PO)):
        (tmp_path / name).write_text(body)
        (tmp_path / name).chmod(0o755)
    canned = tmp_path / "canned"
    canned.mkdir(exist_ok=True)
    for k, v in CANNED.items():
        (canned / f"{k}.txt").write_text(v)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "run_zz_old").mkdir(exist_ok=True)  # sorts after every stamped folder: must not be taken for a result
    env = {**os.environ, "ISSUE55_LOGS": str(logs), "ISSUE55_RUN": str(tmp_path / "run.sh"),
           "BOARD_TOP_CMD": str(tmp_path / "top.sh"), "ISSUE55_PO": str(tmp_path / "po.sh"),
           "STUB_BEHAVIOUR": behaviour, "STUB_CONFIG_DIR": str(HERE), "STUB_CANNED": str(canned), "TOP_SEQ": top_seq,
           "PO_PREPARE_RC": po_rc}
    fast = ["--idle-gap", "0", "--idle-poll", "1", "--idle-max-wait", "3"] if gate else []
    done = subprocess.run([str(SERIES), "--boards", boards, "--launches", launches, *fast, *extra],
                          env=env, capture_output=True, text=True, timeout=300, check=False)
    return done, logs


def series_dir(logs: Path) -> Path:
    return sorted(logs.glob("issue55_*"))[-1]


def tsv(logs: Path, name: str = "runs.tsv") -> list[dict[str, str]]:
    lines = (series_dir(logs) / name).read_text().splitlines()
    cols = lines[0].split("\t")
    return [dict(zip(cols, ln.split("\t"), strict=True)) for ln in lines[1:]]


def steps(logs: Path) -> list[str]:
    """`<board><launch>:<arm>` for every run, from runs.tsv."""
    return [f"{r['board']}{r['launch']}:{r['arm']}" for r in tsv(logs)]


ALL_OK = ",".join(["ok"] * 40)


def test_the_boards_take_turns_and_the_arms_rotate_left_between_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,row-intr,scalar")

    assert done.returncode == 0, done.stdout + done.stderr
    assert steps(logs) == ["a1:row", "a1:row-intr", "a1:scalar", "b1:row", "b1:row-intr", "b1:scalar",
                           "a2:row-intr", "a2:scalar", "a2:row", "b2:row-intr", "b2:scalar", "b2:row"]
    assert all(r["status"] == "ok" and r["exit"] == "0" for r in tsv(logs))


def test_each_arm_is_first_once_in_three_launches(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "col-g1,col-g2,col-g4", "--launches", "3", boards="a")

    assert done.returncode == 0, done.stdout + done.stderr
    order = [s.split(":")[1] for s in steps(logs)]
    assert [order[0], order[3], order[6]] == ["col-g1", "col-g2", "col-g4"]


def test_every_arm_gets_the_config_its_packer_flag_and_the_same_uninstrumented_pin(tmp_path: Path):
    arms = "row,row-intr,scalar,col-g1,col-g2,col-g4,col-gall,col-g1-m4"
    _, logs = run_series(tmp_path, ALL_OK, "--variants", arms, boards="a", launches="1")

    calls = dict(zip(arms.split(","), (logs / "calls.txt").read_text().splitlines(), strict=True))
    for arm, line in calls.items():
        for flag in (f"--config {CONFIG}", "--no-nc-overrides", "--cpu 4", "--tmin 10", "--riscv-ip a"):
            assert flag in line, (arm, line)
        assert "--instr" not in line and "--raw-events" not in line
    assert "--pack-b" not in calls["row"] and "--scalar-pack" not in calls["row"]  # row = the shipped packer, no flag
    assert "--scalar-pack" in calls["scalar"] and "--pack-b" not in calls["scalar"]
    for v in ("row-intr", "col-g1", "col-g2", "col-g4", "col-gall", "col-g1-m4"):
        assert f"--pack-b {v}" in calls[v].replace("  ", " ") and "--scalar-pack" not in calls[v]
        assert calls[v].endswith(f"--pack-b {v}")


def test_a_workspace_is_passed_on_to_every_run_and_left_out_by_default(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, logs = run_series(tmp_path / "a", ALL_OK, "--workspace", "ws_b", boards="a", launches="1")
    _, plain = run_series(tmp_path / "b", ALL_OK, boards="a", launches="1")

    assert all("--riscv-workspace ws_b" in ln for ln in (logs / "calls.txt").read_text().splitlines())
    assert all("--riscv-workspace" not in ln for ln in (plain / "calls.txt").read_text().splitlines())


def test_the_run_happens_inside_the_shared_lock(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK, boards="a", launches="1")

    seen = (logs / "lock_seen.txt").read_text().split()
    assert seen and set(seen) == {"held"}  # the stub could not take .cron_bench.lock: the series holds it


def test_core_0_and_unknown_variants_and_bad_numbers_are_refused(tmp_path: Path):
    done, _ = run_series(tmp_path, ALL_OK, "--cpu", "0")
    assert done.returncode != 0 and "core 0" in (done.stdout + done.stderr)
    done, _ = run_series(tmp_path, ALL_OK, "--variants", "row,col-g3")
    assert done.returncode == 1 and "--variants takes" in done.stderr
    done, _ = run_series(tmp_path, ALL_OK, "--idle-gap", "x")
    assert done.returncode == 1 and "whole numbers" in done.stderr
    done, _ = run_series(tmp_path, ALL_OK, "--variants", "row", "--pack-only")
    assert done.returncode == 1 and "needs at least one variant other than row" in done.stderr


def test_a_run_that_exits_non_zero_is_named_and_the_series_goes_on(tmp_path: Path):
    done, logs = run_series(tmp_path, "ok,exit1," + ALL_OK, "--variants", "row,row-intr", launches="1", boards="a,b")

    assert done.returncode == 1
    assert "failed: launch 1 a row-intr (exit 1)" in done.stdout
    assert [r["status"] for r in tsv(logs)] == ["ok", "failed", "ok", "ok"]
    assert "FAIL(1)" in (series_dir(logs) / "runs.tsv").read_text()


def test_the_result_folder_is_the_stamped_one_not_an_older_folder_that_sorts_after_it(tmp_path: Path):
    _, logs = run_series(tmp_path, ALL_OK, boards="a", launches="1")

    assert all(r["result_dir"].startswith("run_2") for r in tsv(logs))


# ---- the idle gate ----

def test_a_busy_board_is_waited_for_and_the_run_goes_ahead_when_it_is_idle(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="busy,idle")

    assert done.returncode == 0, done.stdout + done.stderr
    waits = (series_dir(logs) / "idle_waits.txt").read_text()
    assert "busy: other user busy: psiwins spmv 92.0%" in waits and "waiting 1 s" in waits
    assert [r["status"] for r in tsv(logs)] == ["ok"]
    readings = (series_dir(logs) / "idle_readings.txt").read_text().splitlines()
    assert [ln.split("\t")[-1] for ln in readings] == ["BUSY", "idle", "idle", "idle"]  # first pre, two pre, one post
    assert "psiwins" in (series_dir(logs) / "load_warnings.txt").read_text()
    assert any(p.name.startswith("idle_a_L1_1_row_pre_") for p in series_dir(logs).iterdir())  # every reading kept


def test_both_readings_of_the_gate_must_pass(tmp_path: Path):
    # first pre-reading idle, second (after the gap) busy: not idle, wait, then two good ones.
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="idle,busy,idle")

    assert done.returncode == 0, done.stdout + done.stderr
    readings = [ln.split("\t")[-1] for ln in (series_dir(logs) / "idle_readings.txt").read_text().splitlines()]
    assert readings == ["idle", "BUSY", "idle", "idle", "idle"]
    assert (series_dir(logs) / "idle_waits.txt").exists()


def test_a_board_that_never_goes_idle_stops_the_series_without_running_anything(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,row-intr", launches="1", top_seq="busy")

    assert done.returncode == 2, done.stdout + done.stderr
    assert "STOPPING: the board never went idle" in done.stdout
    assert not (logs / "calls.txt").exists()  # no run was started under load
    waits = (series_dir(logs) / "idle_waits.txt").read_text()
    assert "giving up" in waits and waits.count("waiting 1 s") >= 1
    assert len(tsv(logs)) == 0


def test_a_busy_reading_after_a_run_discards_it_and_it_is_queued_once_more_at_the_end_of_the_launch(tmp_path: Path):
    # row: pre idle, pre idle, post busy -> discarded; row-intr: idle x3; then row again: idle x3.
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,row-intr", boards="a", launches="1",
                            top_seq="idle,idle,busy,idle,idle,idle,idle,idle,idle")

    assert done.returncode == 0, done.stdout + done.stderr
    rows = tsv(logs)
    assert [(r["arm"], r["status"]) for r in rows] == [("row", "discarded-load"), ("row-intr", "ok"), ("row", "ok")]
    assert len((logs / "calls.txt").read_text().splitlines()) == 3
    assert "discarded-load: launch 1 a row" in done.stdout


def test_a_run_discarded_twice_is_not_queued_a_third_time(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row", boards="a", launches="1",
                            top_seq="idle,idle,busy,idle,idle,busy,idle")

    assert [r["status"] for r in tsv(logs)] == ["discarded-load", "discarded-load"]
    assert len((logs / "calls.txt").read_text().splitlines()) == 2
    assert done.returncode == 0  # a discarded run is not a failed one; the report names it


def test_roots_processes_and_our_own_never_block_but_root_is_recorded(tmp_path: Path):
    lines: dict[str, str] = {}
    for seq in ("rootonly", "oursbusy"):
        sub = tmp_path / seq
        sub.mkdir()
        done, logs = run_series(sub, ALL_OK, "--variants", "row", boards="a", launches="1", top_seq=seq)
        assert done.returncode == 0, done.stdout + done.stderr
        assert not (series_dir(logs) / "idle_waits.txt").exists()
        assert not (series_dir(logs) / "load_warnings.txt").exists()
        assert [r["status"] for r in tsv(logs)] == ["ok"]
        lines[seq] = (series_dir(logs) / "idle_readings.txt").read_text().splitlines()[0]
    assert "root=root bluetoothd 95.0%" in lines["rootonly"] and "other_users=none" in lines["rootonly"]
    assert "other_users=none" in lines["oursbusy"]


def test_the_threshold_is_twenty_percent_of_a_core_and_overall_idle_is_85(tmp_path: Path):
    (tmp_path / "under").mkdir()
    done, logs = run_series(tmp_path / "under", ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="justunder")
    assert done.returncode == 0 and [r["status"] for r in tsv(logs)] == ["ok"]
    (tmp_path / "at").mkdir()
    done, logs = run_series(tmp_path / "at", ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="justat")
    assert done.returncode == 2 and not (logs / "calls.txt").exists()  # exactly at 20% is busy
    (tmp_path / "low").mkdir()
    done, logs = run_series(tmp_path / "low", ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="lowidle")
    assert done.returncode == 2 and "overall idle 50.0% is under 85%" in (series_dir(logs) / "idle_waits.txt").read_text()
    (tmp_path / "flag").mkdir()
    done, logs = run_series(tmp_path / "flag", ALL_OK, "--variants", "row", "--idle-min", "40", boards="a", launches="1", top_seq="lowidle")
    assert done.returncode == 0  # the threshold is a flag


def test_the_second_top_sample_decides_not_the_first(tmp_path: Path):
    # FIRST (in every canned output) shows psiwins at 92% and 39.5 idle; the second sample is idle.
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="idle")

    assert done.returncode == 0 and not (series_dir(logs) / "idle_waits.txt").exists()
    kept = next(series_dir(logs).glob("idle_a_L1_1_row_pre_1.txt")).read_text()
    assert "spmv" not in kept and "bluetoothd" in kept  # the kept sample is the second one


def test_an_unreadable_board_is_not_idle(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row", boards="a", launches="1", top_seq="garbage")

    assert done.returncode == 2 and not (logs / "calls.txt").exists()
    assert "no reading" in (series_dir(logs) / "idle_waits.txt").read_text()


# ---- the pack-only part ----

def test_pack_only_runs_after_the_gemm_arms_for_every_variant_but_row_and_saves_each_output(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,col-g1,scalar", "--pack-only", "--po-reps", "5",
                            boards="a", launches="2")

    assert done.returncode == 0, done.stdout + done.stderr
    po = (logs / "po_calls.txt").read_text().splitlines()
    assert [p for p in po if p.startswith("PO prepare")] == ["PO prepare a", "PO prepare a"]  # once per launch per board
    runs = [p.split() for p in po if p.startswith("PO run")]
    assert len(runs) == 2 * 2 * 2 * 2  # launches x variants (not row) x shapes x tiles
    assert all(r[2] == "a" and r[9] == "5" and r[10] == "4" and r[8] != "row" for r in runs)
    first = [r[3:8] for r in runs[:4]]
    assert first == [["64", "384", "384", "16", "8"], ["64", "384", "384", "16", "15"],
                     ["64", "384", "256", "16", "8"], ["64", "384", "256", "16", "15"]]
    assert [r[8] for r in runs[::4]] == ["col-g1", "scalar", "scalar", "col-g1"]  # rotates left per launch
    rows = tsv(logs, "packonly.tsv")
    assert len(rows) == 16 and all(r["status"] == "ok" and r["exit"] == "0" for r in rows)
    assert (series_dir(logs) / rows[0]["file"]).read_text().startswith("# board: a")
    # the GEMM arms of a launch all come before its pack-only runs
    calls = (logs / "po_calls.txt").read_text().splitlines()
    assert calls[0] == "PO prepare a" and len((logs / "calls.txt").read_text().splitlines()) == 6


def test_pack_only_runs_also_pass_the_idle_gate_and_a_discarded_one_is_queued_once_more(tmp_path: Path):
    # every run takes 3 readings (2 before, 1 after): row 1-3, col-g1 4-6, the first pack-only run 7-9, its post reading (9) busy.
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,col-g1", "--pack-only", "--po-reps", "3", boards="a", launches="1",
                            top_seq="idle,idle,idle,idle,idle,idle,idle,idle,busy,idle")

    rows = tsv(logs, "packonly.tsv")
    statuses = [r["status"] for r in rows]
    assert statuses.count("discarded-load") == 1 and statuses.count("ok") == 4 and len(rows) == 5
    assert rows[-1]["file"].endswith("_retry.txt") and rows[-1]["status"] == "ok"  # re-queued at the end, own file
    assert "discarded-load: launch 1 a pack-only" in done.stdout


def test_a_failed_pack_only_build_is_named_and_the_launch_goes_on(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--variants", "row,col-g1", "--pack-only", boards="a", launches="1", po_rc="1")

    assert done.returncode == 1
    assert "failed: launch 1 a pack-only build (exit 1)" in done.stdout
    assert len((logs / "calls.txt").read_text().splitlines()) == 2  # the GEMM arms ran
    assert tsv(logs, "packonly.tsv") == []


# ---- dry run and plan ----

def test_the_dry_run_prints_the_plan_and_makes_no_board_call(tmp_path: Path):
    done, logs = run_series(tmp_path, ALL_OK, "--dry-run", "--variants", "row,row-intr,scalar,col-gall,col-g1,col-g1-m4", "--pack-only",
                            boards="jbpi,jbpi2", launches="3", gate=False)

    out = done.stdout
    assert done.returncode == 0, out + done.stderr
    assert "Launch 1   : row row-intr scalar col-gall col-g1 col-g1-m4" in out
    assert "Launch 2   : row-intr scalar col-gall col-g1 col-g1-m4 row" in out
    assert "Pack-only launch 1: row-intr scalar col-gall col-g1 col-g1-m4" in out
    assert "Board time per board (estimate" in out and "2.4 min per tile per arm" in out and "1 min per variant per case" in out
    assert "(36 GEMM runs, 120 pack-only runs)" in out
    assert "--pack-b col-g1-m4" in out and "--scalar-pack" in out
    assert not list(logs.glob("issue55_*"))  # no series folder
    assert not (logs / "top_calls.txt").exists() and not (logs / "po_calls.txt").exists()  # nothing asked of a board


def test_the_script_never_runs_a_board_command_in_the_dry_run_even_with_the_real_hooks_unset(tmp_path: Path):
    # no hooks at all: with a blocker ssh/scp/rsync/sshpass first on PATH, any board call would print BLOCKED.
    blocker = tmp_path / "bin"
    blocker.mkdir()
    for name in ("ssh", "scp", "rsync", "sshpass"):
        (blocker / name).write_text(f"#!/bin/bash\necho 'BLOCKED: {name}' >&2\nexit 255\n")
        (blocker / name).chmod(0o755)
    logs = tmp_path / "logs"
    logs.mkdir()
    env = {**os.environ, "PATH": f"{blocker}:{os.environ['PATH']}", "ISSUE55_LOGS": str(logs)}
    done = subprocess.run([str(SERIES), "--dry-run", "--variants", "row,col-g1", "--pack-only"], env=env, capture_output=True, text=True, timeout=120)

    assert done.returncode == 0, done.stdout + done.stderr
    assert "BLOCKED" not in done.stdout + done.stderr


# ---- the config ----

def test_the_config_holds_the_three_pinned_tiles_on_the_b3_dataset():
    rows = [ln.strip() for ln in (HERE / CONFIG).read_text().splitlines() if ln.strip().startswith('["fp32"')]
    assert len(rows) == 3
    tiles = [r.split("[", 2)[2].split("]")[0] for r in rows]
    assert tiles == ["16, 15", "16, 8", "16, 6"]
    assert all('"AXPY_FP32_Pack"' in r and '"issue46_ref_b3.dat"' in r for r in rows)
    assert "#54" in (HERE / CONFIG).read_text() and "#55" in (HERE / CONFIG).read_text()
    # the same columns as the #50 sweep's rows
    ref = [ln.strip() for ln in (HERE / "experiments/issue50/config_issue50_sweep.json").read_text().splitlines() if ln.strip().startswith('["fp32"')][0]
    assert [c.strip() for c in ref.rstrip(",]").split(",")][-3:] == [c.strip() for c in rows[0].rstrip(",]").split(",")][-3:]


def test_the_config_parses_with_the_projects_own_config_reader():
    sys.path.insert(0, str(HERE))
    from config import BLISProjectConfig

    cfgs = BLISProjectConfig.from_json(str(HERE / CONFIG))
    assert len(cfgs) == 3
    assert [(c.datatype, c.method) for c in cfgs] == [("fp32", "AXPY_FP32_Pack")] * 3
    assert [c.mr_nr_family for c in cfgs] == [[(16, 15)], [(16, 8)], [(16, 6)]] and all(c.is_pinned for c in cfgs)
    assert all(c.datasets == ["issue46_ref_b3.dat"] for c in cfgs)
