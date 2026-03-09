from collections.abc import Callable

from xdsl.ir import Dialect

# from . import emitc_ext, emitc_ptr, hello, mydialect, riscv_bpi, rvv


def get_all_dialects() -> dict[str, Callable[[], Dialect]]:
    """Returns all available dialects."""

    # Add your dialects here to be discovered by `xdsl-opt`
    return {}
