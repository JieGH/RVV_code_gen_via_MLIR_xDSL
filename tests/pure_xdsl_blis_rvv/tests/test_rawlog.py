"""
Issue #117: the raw-log reader. One module knows how the driver names a run's merged CSV; the
reports ask it for a variant instead of spelling a suffix. Fixture logs are made in tmp_path, one
per variant the driver can write.
"""

from pathlib import Path

import pytest

import rawlog

HERE = Path(__file__).resolve().parents[1]
DRIVER = HERE / "templates" / "driver_benchmark.cpp"

HEADER = "Layer,Variant,M,N,K,Status\n"


def make_run(tmp_path: Path, *names: str) -> Path:
    for n in names:
        (tmp_path / n).write_text(HEADER + "L1,MR8_NR8,8,8,8,PASS\n")
    return tmp_path


def test_parts_come_in_the_drivers_order():
    got = rawlog.raw_glob(scalar_pack=True, zero_line=True, pad_ldb=4, pack_b="x", no_nc_override=True)
    assert got == "all_raw_*_noinstr_refproto_scalarpack_zeroline_padldb4_packb-x_no_nc_override.txt"


def test_the_order_matches_the_driver_source():
    src = DRIVER.read_text()
    start = src.index("const string log_suffix")
    parts = ["_refproto", "_scalarpack", "_zeroline", "_padldb", "_packb-", "_no_nc_override"]
    at = [src.index(f'"{p}', start) for p in parts]
    assert at == sorted(at), "the driver builds its log suffix in a different order than rawlog.raw_glob"
    probe = rawlog.raw_glob(scalar_pack=True, zero_line=True, pad_ldb=1, pack_b="a", no_nc_override=True)
    assert [probe.index(p) for p in parts] == sorted(probe.index(p) for p in parts)


def test_an_instrumented_run_has_neither_noinstr_nor_refproto():
    assert rawlog.raw_glob(instrumented=True, refproto=False) == "all_raw_*.txt"


def test_find_raw_never_mixes_variants(tmp_path: Path):
    run = make_run(
        tmp_path,
        "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_no_nc_override.txt",
        "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_scalarpack_no_nc_override.txt",
        "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_scalarpack.txt",
    )
    vec = rawlog.find_raw(run, no_nc_override=True)
    sca = rawlog.find_raw(run, scalar_pack=True, no_nc_override=True)
    assert [p.name for p in vec] == ["all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_no_nc_override.txt"]
    assert [p.name for p in sca] == ["all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_scalarpack_no_nc_override.txt"]


def test_a_missing_variant_stops_and_names_what_the_run_holds(tmp_path: Path):
    run = make_run(tmp_path, "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto_scalarpack.txt")
    with pytest.raises(SystemExit) as err:
        rawlog.require_raw(run, no_nc_override=True)
    msg = str(err.value)
    assert "no_nc_override" in msg and "scalarpack.txt" in msg


def test_resolve_raw_never_reads_a_refproto_log_by_default(tmp_path: Path):
    """Issue #27: a Reference Protocol log is read only when named."""
    refproto = "all_raw_fp32_AXPY_FP32_Pack_noinstr_refproto.txt"
    run = make_run(tmp_path, refproto)
    assert rawlog.resolve_raw(run) is None
    assert rawlog.resolve_raw(run, refproto).name == refproto
    make_run(tmp_path, rawlog.INSTRUMENTED_NAME)
    assert rawlog.resolve_raw(run).name == rawlog.INSTRUMENTED_NAME
    assert rawlog.resolve_raw(run, "nope.txt") is None


def test_resolve_raw_is_none_for_a_run_without_a_log(tmp_path: Path):
    assert rawlog.resolve_raw(tmp_path) is None


def test_read_rows_gives_columns_by_name(tmp_path: Path):
    run = make_run(tmp_path, "all_raw_x_noinstr_refproto.txt")
    rows = rawlog.read_rows(run / "all_raw_x_noinstr_refproto.txt")
    assert rows == [{"Layer": "L1", "Variant": "MR8_NR8", "M": "8", "N": "8", "K": "8", "Status": "PASS"}]


def test_read_tsv_reads_a_series_by_header_and_skips_blank_lines(tmp_path: Path):
    f = tmp_path / "runs.tsv"
    f.write_text("launch\tthreads\texit\n1\t1\t0\n\n2\t8\tFAIL(x)\n")
    assert rawlog.read_tsv(f) == [
        {"launch": "1", "threads": "1", "exit": "0"},
        {"launch": "2", "threads": "8", "exit": "FAIL(x)"},
    ]
    f.write_text("")
    assert rawlog.read_tsv(f) == []
