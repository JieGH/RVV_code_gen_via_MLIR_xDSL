"""
Issue #21: both harnesses time on one pinned core, record the conditions they ran under,
and the comparison says whether the two runs were made under the same ones.

Until now neither side pinned anything, our `run_*` dirs came from a cron job in the small
hours and the reference baseline was taken a day later, on a shared board in a different
thermal state. That is an unknown-sized variable under every percentage in the comparison.

Seams, none of which needs the board:
  * `board_pin.sh`, the shell helpers both scripts source: the `taskset` prefix, and the
    provenance snippet the scripts pipe to the board (run here on the host, where `taskset`
    and `gcc` exist too);
  * the run overview of `compile_macro_bench.sh --dry-run`, and the argument refusals of
    both scripts, which happen before either touches the board;
  * `compare_ref_baseline.conditions()`, which reads the two provenance files and the two
    timestamps and says whether the pair is sound.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
REPO_ROOT = HERE.parents[1]
BOARD_PIN = HERE / "board_pin.sh"
BENCH_SCRIPT = HERE / "compile_macro_bench.sh"
REF_SCRIPT = HERE / "ref_baseline" / "run_ref_baseline.sh"
H2H_SCRIPT = HERE / "run_head_to_head.sh"

sys.path.append(str(HERE / "ref_baseline"))

from compare_ref_baseline import conditions  # noqa: E402


def _bash(snippet: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{BOARD_PIN}"; {snippet}'],
        capture_output=True, text=True, timeout=30,
    )


def _default_cpu() -> str:
    proc = _bash('printf %s "$BENCH_CPU_DEFAULT"')
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _script(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )


# ---------------------------------------------------------------------------
# board_pin.sh
# ---------------------------------------------------------------------------
def test_the_default_core_is_a_number_and_not_cpu0():
    # The board routes every device interrupt to CPU0 (/proc/interrupts), so a timed
    # run pinned there would be timed among the interrupts.
    assert _default_cpu().isdigit() and _default_cpu() != "0"


def test_a_pin_becomes_a_taskset_prefix_and_no_pin_becomes_nothing():
    assert _bash("pin_prefix 5").stdout == "taskset -c 5 "
    assert _bash('pin_prefix ""').stdout == ""


@pytest.mark.parametrize("cpu", ["4", "12", "0"])
def test_a_core_number_is_accepted(cpu: str):
    assert _bash(f"pin_cpu_valid {cpu}").returncode == 0


@pytest.mark.parametrize("cpu", ["", "x", "-1", "2,3", "1-3", "4x"])
def test_anything_but_one_core_number_is_refused(cpu: str):
    assert _bash(f'pin_cpu_valid "{cpu}"').returncode != 0


def _provenance(cpu: str) -> str:
    snippet = _bash(f'board_provenance "{cpu}"')
    assert snippet.returncode == 0, snippet.stderr
    ran = subprocess.run(
        ["sh", "-s"], input=snippet.stdout, capture_output=True, text=True, timeout=30
    )
    assert ran.returncode == 0, ran.stderr
    return ran.stdout


def _field(text: str, key: str) -> str:
    found = re.search(rf"^  {re.escape(key)}\s*: (.*)$", text, re.MULTILINE)
    assert found is not None, f"no '{key}' line in:\n{text}"
    return found.group(1)


def test_provenance_records_the_conditions_the_issue_names():
    text = _provenance("0")
    assert _field(text, "cpu pin") == "0"
    assert _field(text, "governor")  # a value, or a saying that the board does not say
    assert _field(text, "load average")
    assert "processor" in text  # the /proc/cpuinfo snapshot


def test_a_probe_the_board_cannot_answer_says_so_instead_of_leaving_a_blank(tmp_path: Path):
    # A board without a compiler on PATH (or without cpufreq) must not leave an empty field
    # that reads as "recorded, and empty".
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("sh", "hostname", "uname", "head", "cat", "sort", "uniq", "tr", "date", "taskset"):
        found = shutil.which(tool)
        assert found is not None, tool
        (bin_dir / tool).symlink_to(found)
    snippet = _bash('board_provenance ""').stdout
    ran = subprocess.run(
        [str(bin_dir / "sh"), "-s"], input=snippet, capture_output=True, text=True,
        timeout=30, env={"PATH": str(bin_dir)},
    )
    assert ran.returncode == 0, ran.stderr
    assert _field(ran.stdout, "compiler (gcc)") == "unavailable"
    assert _field(ran.stdout, "compiler (g++)") == "unavailable"


def test_provenance_records_both_compilers_the_bench_builds_with():
    # The driver is built with g++ and model_level.c with gcc (templates/Makefile).
    text = _provenance("0")
    assert "compiler (gcc)" in text and "compiler (g++)" in text


def test_provenance_shows_the_core_the_run_was_really_confined_to():
    # Not what was asked for but what the kernel says: `taskset` run under the pin.
    text = _provenance("0")
    assert re.search(r"affinity list: 0\b", _field(text, "affinity"))


def test_an_unpinned_run_says_so_in_its_provenance():
    assert _field(_provenance(""), "cpu pin") == "none"


# ---------------------------------------------------------------------------
# compile_macro_bench.sh and run_ref_baseline.sh
# ---------------------------------------------------------------------------
def test_the_overview_says_which_core_a_default_run_is_pinned_to():
    proc = _script(BENCH_SCRIPT, "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    line = next(ln for ln in proc.stdout.splitlines() if ln.lstrip().startswith("Pinning"))
    assert f"taskset -c {_default_cpu()}" in line and "[default]" in line


def test_the_overview_follows_cpu_and_no_pin():
    line = next(
        ln for ln in _script(BENCH_SCRIPT, "--dry-run", "--cpu", "6").stdout.splitlines()
        if ln.lstrip().startswith("Pinning")
    )
    assert "taskset -c 6" in line and "[--cpu]" in line

    line = next(
        ln for ln in _script(BENCH_SCRIPT, "--dry-run", "--no-pin").stdout.splitlines()
        if ln.lstrip().startswith("Pinning")
    )
    assert "NOT pinned" in line and "[--no-pin]" in line


@pytest.mark.parametrize("script", [BENCH_SCRIPT, REF_SCRIPT])
@pytest.mark.parametrize("flags", [["--cpu", "x"], ["--cpu", "2,3"], ["--cpu", "1", "--no-pin"]])
def test_a_bad_pin_is_refused_before_anything_runs(script: Path, flags: list[str]):
    proc = _script(script, "--dry-run", *flags)
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "--cpu" in out
    assert "Dependency Check" not in out and "Generating datasets" not in out


# ---------------------------------------------------------------------------
# run_head_to_head.sh: refusals only; the rest of it is the two scripts above
# ---------------------------------------------------------------------------
def test_head_to_head_refuses_a_bad_pin_before_running_either_side():
    proc = _script(H2H_SCRIPT, "--cpu", "x")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "--cpu" in out and "Run overview" not in out


def test_head_to_head_refuses_an_unknown_flag():
    proc = _script(H2H_SCRIPT, "--frobnicate")
    assert proc.returncode != 0
    assert "Unknown parameter: --frobnicate" in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Issue #39, house rule 1: never time on core 0 (it takes every device interrupt and runs
# 5-10% slower). Every script refuses --cpu 0 while parsing its arguments, unless
# --allow-core-0 says it is on purpose (to reproduce an old core-0 run).
# ---------------------------------------------------------------------------
PACK_SWAP_SCRIPT = HERE / "run_pack_swap.sh"
SCALAR_SWEEP_SCRIPT = HERE / "run_scalar_pack_sweep.sh"
REF_SWEEP_SCRIPT = HERE / "run_ref_sweep.sh"
DIRECT_CHECK_SCRIPT = HERE / "ref_baseline" / "run_direct_check.sh"
EVERY_SCRIPT = [
    BENCH_SCRIPT, REF_SCRIPT, H2H_SCRIPT, PACK_SWAP_SCRIPT,
    SCALAR_SWEEP_SCRIPT, REF_SWEEP_SCRIPT, DIRECT_CHECK_SCRIPT,
]


def test_pin_check_refuses_core_0_and_says_how_to_allow_it():
    proc = _bash("pin_check true false 0")
    assert proc.returncode != 0
    assert "core 0" in proc.stdout and "--allow-core-0" in proc.stdout
    assert _bash("pin_check true false 0 true").returncode == 0
    assert _bash("pin_check true false 4").returncode == 0


@pytest.mark.parametrize("script", EVERY_SCRIPT, ids=lambda p: p.name)
def test_every_script_refuses_core_0_before_anything_runs(script: Path):
    proc = _script(script, "--dry-run", "--cpu", "0")
    assert proc.returncode != 0
    out = proc.stdout + proc.stderr
    assert "core 0" in out and "--allow-core-0" in out
    assert "Run overview" not in out and "Dependency Check" not in out


@pytest.mark.parametrize("script", EVERY_SCRIPT, ids=lambda p: p.name)
def test_every_script_takes_core_0_when_told_it_is_on_purpose(script: Path):
    proc = _script(script, "--dry-run", "--cpu", "0", "--allow-core-0")
    assert "core 0 takes every device interrupt" not in proc.stdout + proc.stderr
    assert "Unknown parameter" not in proc.stdout + proc.stderr


@pytest.mark.parametrize("script", [H2H_SCRIPT, PACK_SWAP_SCRIPT], ids=lambda p: p.name)
def test_a_wrapper_passes_allow_core_0_on_to_the_scripts_it_runs(script: Path):
    proc = _script(script, "--dry-run", "--cpu", "0", "--allow-core-0")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "taskset -c 0" in proc.stdout or "cpu 0" in proc.stdout


@pytest.mark.parametrize("script", [PACK_SWAP_SCRIPT, SCALAR_SWEEP_SCRIPT, REF_SWEEP_SCRIPT], ids=lambda p: p.name)
def test_no_script_defaults_to_core_0(script: Path):
    text = script.read_text()
    assert "CPU_ARGS=(--cpu 0)" not in text
    assert "CPU_ARGS=(--cpu 4)" in text


def test_the_cron_wrapper_does_not_pin_to_core_0():
    code = [ln for ln in (HERE / "cron_bench_run.sh").read_text().splitlines() if not ln.lstrip().startswith("#")]
    assert not any("--cpu 0" in ln for ln in code)


# ---------------------------------------------------------------------------
# compare_ref_baseline.conditions()
# ---------------------------------------------------------------------------
def _run_dir(root: Path, name: str, pin: str | None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    if pin is not None:
        (d / "provenance.txt").write_text(f"header\n  cpu pin        : {pin}\n  governor       : performance\n")
    return d


def _warnings(lines: list[str]) -> list[str]:
    return [ln for ln in lines if "WARNING" in ln]


def test_a_pair_run_back_to_back_on_the_same_core_is_sound(tmp_path: Path):
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", "4")
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "4")
    lines = conditions(ref, [ours])
    assert _warnings(lines) == [], lines
    assert any("apart" in ln for ln in lines) and any("cpu 4" in ln for ln in lines)


def test_a_pair_made_a_day_apart_is_flagged(tmp_path: Path):
    # The very pair issue #21 was written about: 09-19 18:54 against 09-18 16:47.
    ours = _run_dir(tmp_path, "run_2026-09-19_185453", "4")
    ref = _run_dir(tmp_path, "refbase_2026-09-18_164740", "4")
    lines = conditions(ref, [ours])
    warned = _warnings(lines)
    assert len(warned) == 1 and "26.1 h apart" in warned[0], lines


def test_the_gap_that_counts_is_the_worst_of_the_runs_averaged(tmp_path: Path):
    near = _run_dir(tmp_path, "run_2026-09-19_101500", "4")
    far = _run_dir(tmp_path, "run_2026-09-18_004053", "4")
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "4")
    assert any("apart" in w for w in _warnings(conditions(ref, [near, far])))


def test_different_cores_are_flagged(tmp_path: Path):
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", "4")
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "5")
    warned = _warnings(conditions(ref, [ours]))
    assert len(warned) == 1 and "different cores" in warned[0], warned


def test_an_unpinned_side_is_flagged(tmp_path: Path):
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", "none")
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "4")
    warned = _warnings(conditions(ref, [ours]))
    assert len(warned) == 1 and "not pinned" in warned[0], warned


def test_a_run_from_before_provenance_existed_is_flagged_not_guessed(tmp_path: Path):
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", None)
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "4")
    warned = _warnings(conditions(ref, [ours]))
    assert len(warned) == 1 and "pinning unknown" in warned[0], warned


def test_two_unpinned_sides_are_flagged_as_unpinned_not_as_matching(tmp_path: Path):
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", "none")
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", "none")
    warned = _warnings(conditions(ref, [ours]))
    assert warned and all("not pinned" in w for w in warned), warned



@pytest.mark.parametrize(
    "cpu, words", [("0", "cpu 0"), ("", "unpinned")]
)
def test_the_comparison_reads_what_the_provenance_script_really_writes(
    tmp_path: Path, cpu: str, words: str
):
    # The other tests hand-write the `cpu pin` line; this one closes the loop, so the shell
    # format and the parser cannot drift apart unnoticed.
    ours = _run_dir(tmp_path, "run_2026-09-19_101500", None)
    ref = _run_dir(tmp_path, "refbase_2026-09-19_104000", None)
    for d in (ours, ref):
        (d / "provenance.txt").write_text(_provenance(cpu))
    pinning = next(ln for ln in conditions(ref, [ours]) if ln.startswith("Pinning"))
    assert pinning == f"Pinning  : ours {words}; reference {words}"
