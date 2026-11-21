"""
Issue #116: the board access module in `board_pin.sh`. Before, each run script carried its own
copy of how it reaches a board, who else is on it and when it is idle. These tests cross the
module's interface once, for every caller. No board is touched: `ssh`, `sshpass` and `rsync`
are stubs that print their arguments, and `ps` / `top` on the board are the two shared hooks
`BOARD_PS_CMD` and `BOARD_TOP_CMD`.
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
BOARD_PIN = HERE / "board_pin.sh"

PS_BUSY = (
    "jlei                 99.0  4 01:00 gemm\n"
    "root                 50.0  1 05:00 kworker\n"
    "alice                80.0  5 00:10 gemm.x\n"
    "bob                   2.0  6 00:10 vim\n"
)
PS_QUIET = "jlei  99.0  4 01:00 gemm\nroot  50.0  1 05:00 kworker\n"


def top(idle: str, procs: list[tuple[str, str, str]]) -> str:
    rows = "".join(f"{i + 1:5d} {u:<8} 20 0 1 1 1 R {cpu:>5} 0.1 0:01.00 {cmd}\n" for i, (u, cpu, cmd) in enumerate(procs))
    sample = (f"top - 10:00:00 up 1 day\n%Cpu(s):  1.0 us,  0.5 sy,  0.0 ni, {idle} id,  0.0 wa\n\n"
              "  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\n" + rows)
    return "top - 09:59:59 up 1 day\n%Cpu(s): 60.0 us, 39.5 id\n\n" + sample


def run(tmp_path: Path, snippet: str, **env: str) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("ssh", "sshpass", "rsync"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/bash\necho "{name} $*"\n')
        stub.chmod(0o755)
    base = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "RISCV_USER": "jlei", "RISCV_PORT": "22",
            "REMOTE_HOST": "fej", "REMOTE_PORT": "3322", "REMOTE_PASS": "pw", "SERIES": str(tmp_path),
            "IDLE_GAP": "0", "IDLE_POLL": "1", "IDLE_MAX_WAIT": "0", "IDLE_MIN": "85", "IDLE_PROC_MAX": "20"}
    return subprocess.run(["bash", "-c", f'source "{BOARD_PIN}"; {snippet}'], capture_output=True, text=True,
                          timeout=60, env={**base, **env})


def hook(tmp_path: Path, name: str, body: str, rc: int = 0) -> str:
    p = tmp_path / name
    p.write_text(f"#!/bin/bash\ncat <<'EOF'\n{body}EOF\nexit {rc}\n")
    p.chmod(0o755)
    return str(p)


def test_board_ssh_goes_straight_to_the_board_with_direct(tmp_path: Path):
    out = run(tmp_path, 'MODE_ARG=--direct; board_ssh jbpi uname -a').stdout
    assert out.strip() == "ssh -p 22 -o StrictHostKeyChecking=no -o ConnectTimeout=15 jlei@jbpi uname -a"


def test_board_ssh_goes_through_the_jump_host_by_default(tmp_path: Path):
    out = run(tmp_path, 'MODE_ARG=--jump; board_ssh jbpi uname -a').stdout
    assert out.startswith("sshpass -p pw ssh -p 3322 ")
    assert "jlei@fej ssh -p 22 -o StrictHostKeyChecking=no -o ConnectTimeout=15 jbpi uname -a" in out


def test_board_rsync_writes_the_target_host_for_each_route(tmp_path: Path):
    direct = run(tmp_path, 'MODE_ARG=--direct; board_rsync jbpi src/ @T@:dst/').stdout
    jump = run(tmp_path, 'MODE_ARG=--jump; board_rsync jbpi src/ @T@:dst/').stdout
    assert direct.strip() == "rsync -av -e ssh -p 22 -o StrictHostKeyChecking=no src/ jlei@jbpi:dst/"
    assert jump.strip() == "sshpass -p pw rsync -av -e ssh -p 3322 -o StrictHostKeyChecking=no src/ jlei@fej:dst/"


def test_board_ps_calls_the_hook_with_the_board_name(tmp_path: Path):
    h = hook(tmp_path, "ps.sh", PS_BUSY)
    proc = run(tmp_path, "board_ps jbpi2", BOARD_PS_CMD=h)
    assert proc.stdout == PS_BUSY


def test_board_busy_names_another_users_job_but_not_root_us_or_idle_ones(tmp_path: Path):
    h = hook(tmp_path, "ps.sh", PS_BUSY)
    out = run(tmp_path, 'board_busy jbpi "$SERIES/ps.txt"', BOARD_PS_CMD=h).stdout
    assert "alice gemm.x 80.0% on core 5" in out
    assert "root" not in out and "bob" not in out and "jlei" not in out


def test_board_busy_is_empty_on_a_quiet_board(tmp_path: Path):
    h = hook(tmp_path, "ps.sh", PS_QUIET)
    assert run(tmp_path, 'board_busy jbpi "$SERIES/ps.txt"', BOARD_PS_CMD=h).stdout.strip() == ""


def test_a_board_that_cannot_be_read_counts_as_busy(tmp_path: Path):
    h = hook(tmp_path, "ps.sh", "", rc=1)
    out = run(tmp_path, 'board_busy jbpi "$SERIES/ps.txt"', BOARD_PS_CMD=h).stdout
    assert "cannot read the board's processes" in out


def test_top_summary_reads_idle_and_splits_other_users_from_root(tmp_path: Path):
    f = tmp_path / "top.txt"
    f.write_text(top("97.0", [("alice", "55.0", "gemm"), ("root", "30.0", "kworker"), ("jlei", "90.0", "mine"),
                               ("bob", "3.0", "vim")]))
    out = run(tmp_path, f'board_top_summary "{f}"').stdout.strip()
    idle, heavy, root = out.split("|")
    assert idle == "97.0"
    assert heavy.startswith("alice gemm 55.0%") and "mine" not in heavy and "vim" not in heavy
    assert root.startswith("root kworker 30.0%")


def test_idle_gate_passes_on_an_idle_board_and_records_both_readings(tmp_path: Path):
    h = hook(tmp_path, "top.sh", top("97.0", []))
    proc = run(tmp_path, 'board_idle_gate jbpi L1', BOARD_TOP_CMD=h)
    assert proc.returncode == 0, proc.stderr
    lines = (tmp_path / "idle_readings.txt").read_text().splitlines()
    assert len(lines) == 2 and all(line.endswith("idle") for line in lines)


def test_idle_gate_gives_up_on_a_busy_board_and_says_why(tmp_path: Path):
    h = hook(tmp_path, "top.sh", top("97.0", [("alice", "80.0", "gemm.x")]))
    proc = run(tmp_path, 'board_idle_gate jbpi L1', BOARD_TOP_CMD=h)
    assert proc.returncode == 1
    assert "still busy" in (tmp_path / "idle_waits.txt").read_text()
    assert "alice" in (tmp_path / "load_warnings.txt").read_text()


def test_idle_gate_refuses_a_board_that_is_not_idle_enough_even_with_no_other_user(tmp_path: Path):
    h = hook(tmp_path, "top.sh", top("40.0", []))
    proc = run(tmp_path, 'board_idle_gate jbpi L1', BOARD_TOP_CMD=h)
    assert proc.returncode == 1
    assert "under 85%" in (tmp_path / "idle_waits.txt").read_text()


def test_board_ssh_hook_gets_the_board_and_command_and_board_ps_goes_through_it(tmp_path: Path):
    h = tmp_path / "ssh.sh"
    h.write_text('#!/bin/bash\necho "hook $1 $2"\n')
    h.chmod(0o755)
    assert run(tmp_path, "board_ssh jbpi 'uname -a'", BOARD_SSH_CMD=str(h)).stdout == "hook jbpi uname -a\n"
    assert run(tmp_path, "board_ps jbpi2", BOARD_SSH_CMD=str(h)).stdout.startswith("hook jbpi2 ps -eo")


def test_board_ssh_tty_switch_adds_t_on_both_hops_of_the_jump_route_only(tmp_path: Path):
    out = run(tmp_path, 'MODE_ARG=--jump; board_ssh jbpi true', BOARD_SSH_TTY="1").stdout
    assert "-o ConnectTimeout=15 -t jlei@fej ssh -p 22 -o StrictHostKeyChecking=no -o ConnectTimeout=15 -t jbpi true" in out
    out = run(tmp_path, 'MODE_ARG=--direct; board_ssh jbpi true', BOARD_SSH_TTY="1").stdout
    assert " -t " not in out
