"""
Regression test for issue #12: `modelLevel/archs/bananapi.h` (what our binaries
actually build with, via `model_level.h`'s default `#include`) and
`cache_config.py`'s `ARCH_CONFIGS["bananapi"]` (the Python port used by
`ref_baseline/gen_ref_datasets.py` to predict blocking) drifted apart once before --
`WL2` went 8 -> 16 in the header only, silently changing MC/NC on 8 of 10 comparison
shapes while the predictor kept assuming the old value. Guards against that
recurring for any of the NL/CL/WL constants.

`amd_zen`/`arm_carmel` are not covered: they are unused elsewhere in this repo and
already have pre-existing, unrelated drift from their own headers -- out of scope
for this issue.
"""

import re
from pathlib import Path

from cache_config import ARCH_CONFIGS

HERE = Path(__file__).resolve().parent
BANANAPI_HEADER = HERE / "modelLevel" / "archs" / "bananapi.h"

_DEFINE_RE = re.compile(r"^\s*#define\s+(NL1|CL1|WL1|NL2|CL2|WL2|NL3|CL3|WL3)\s+(\d+)")


def _active_constants(header: Path) -> dict[str, int]:
    """Last (i.e. active/uncommented) #define wins, matching the C preprocessor."""
    constants: dict[str, int] = {}
    for line in header.read_text().splitlines():
        if line.strip().startswith("//"):
            continue
        m = _DEFINE_RE.match(line)
        if m:
            constants[m.group(1)] = int(m.group(2))
    return constants


def test_cache_config_matches_bananapi_header():
    header_constants = _active_constants(BANANAPI_HEADER)
    assert ARCH_CONFIGS["bananapi"] == header_constants
