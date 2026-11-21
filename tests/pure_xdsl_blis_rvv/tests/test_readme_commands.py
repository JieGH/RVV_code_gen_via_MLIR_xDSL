"""The board commands in the repo README ("Run a model", "Feature examples") still work: the bench accepts
each one (--dry-run, no board) and the two model configs generate. The board runs themselves: see the README."""

import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # P/tests, for sibling test modules

from test_features import _generate  # noqa: E402

CONN = "--jump --riscv-user u --riscv-ip jbpi --riscv-port 22"


def _readme_commands() -> list[str]:
    """Every `$B` / `$CHECK` line of the README's two board sections, with its variables filled in."""
    text = REPO / "README.md"
    body = text.read_text().split("## Run a model", 1)[1].split("## Essential commands", 1)[0]
    env = {"B": "./tests/pure_xdsl_blis_rvv/compile_macro_bench.sh", "CONN": CONN}
    cmds = []
    for line in re.findall(r"```bash\n(.*?)```", body, re.S)[0:2]:
        for raw in line.splitlines():
            raw = raw.split("#", 1)[0].strip()
            if m := re.fullmatch(r'(\w+)="?(.*?)"?', raw):
                env[m[1]] = re.sub(r"\$(\w+)", lambda v: env[v[1]], m[2])
            elif raw.startswith("$"):
                cmds.append(re.sub(r"\$(\w+)", lambda v: env[v[1]], raw))
    return cmds


COMMANDS = _readme_commands()


def test_readme_lists_the_model_and_feature_commands():
    assert len([c for c in COMMANDS if "configs/benchmark/" in c]) == 2
    assert len(COMMANDS) >= 13


@pytest.mark.parametrize("cmd", COMMANDS)
def test_bench_accepts_readme_command(cmd: str):
    done = subprocess.run(f"{cmd} --dry-run", shell=True, cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0 and "--dry-run: stopping here" in done.stdout, done.stdout[-2000:] + done.stderr


@pytest.mark.parametrize("config", ["config_bert.json", "config_square.json"])
def test_model_config_generates(tmp_path: Path, config: str):
    rows = re.sub(r"//.*", "", (HERE / "configs" / "benchmark" / config).read_text()).strip()
    _generate(tmp_path, rows[1:-1])
