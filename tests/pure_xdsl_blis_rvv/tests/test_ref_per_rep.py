"""
Issue #46: per-rep times of the reference design, from a patched STAGED copy of `test_gemm.c`.

The reference driver keeps only the total time of its `while (time <= tmin)` loop. `ref_per_rep.py`
patches a staged copy (never `ext/`) to stamp the clock after every call into a pre-sized array and
write it out after the loop, then converts that file to the layout `reps_report.py` reads.

What is checked here: the patch lands once at each anchor, keeps the timed loop as it was (the
same two clock reads and the same exit test), refuses `ext/` and a second patch, compiles as C
when the reference tree is next to the repository; and the converter turns the stamps into per-rep
seconds, maps the driver's numeric layer id back to our label, and cross-checks the total against
the GFLOPS the driver itself wrote.
"""

import gzip
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.append(os.path.join(HERE, "ref_baseline"))
sys.path.append(str(HERE))

import ref_per_rep as rpr  # noqa: E402
from reps_report import case_stats, read_reps  # noqa: E402

REF_SRC = HERE.parents[2] / "ext" / "RVV_ukernels_benchmark" / "gemm_blis_family" / "src"

# The five anchors of test_gemm.c, in the order they occur, with what surrounds them.
MINI = """\
#include <string.h>
int main(int argc, char *argv[]) {
  FILE *fd_csv;
  fd_csv = fopen(argv[argc - 1], "w");
  fprintf(fd_csv, "#l;m;n;k;Gflops\\n");
  for (;;) {
	  time  = 0.0;
	  t1    = dclock();
	  nreps = 0;
	  while ( time <= tmin ) {
	    nreps++;
	    gemm_blis_B3A2C0( 1 );
	    t2   = dclock();
	    time = ( t2 > t1 ? t2 - t1 : 0.0 );
	  }
	  time = time/nreps;
  }
  return 0;
}
"""


def test_the_patch_adds_a_stamp_after_the_clock_read_and_leaves_the_loop_alone():
    out = rpr.patch_test_gemm(MINI)

    assert out.index("t2   = dclock();") < out.index("rep_stamp[nreps - 1] = t2;") < out.index("time = ( t2 > t1")
    for line in (
        "t1    = dclock();",
        "while ( time <= tmin ) {",
        "gemm_blis_B3A2C0( 1 );",
        "time = ( t2 > t1 ? t2 - t1 : 0.0 );",
    ):
        assert out.count(line) == 1, line  # nothing of the timed loop was moved or copied


def test_the_patch_takes_nothing_from_the_heap_so_the_drivers_buffers_keep_their_addresses():
    # v1 malloc'd the array and v2 kept a second stdio stream: both put something on the heap before the
    # driver allocates A, B, C, Ac, Bc (aligned to 32 bytes only), and the control run then disagreed with
    # the unchanged driver by up to 30% on one case (issue #46). The array is static (.bss) and the file is
    # written with open(2)/write(2) from a static buffer: no malloc, no FILE.
    out = rpr.patch_test_gemm(MINI)

    assert "static double rep_stamp[REP_CAP];" in out and "static char   rep_row[" in out
    for heap in ("malloc(", "calloc(", "fopen(reps_name", "FILE   *fd_reps", "fprintf(fd_reps"):
        assert heap not in out, heap
    assert "open(reps_name" in out and "write(fd_reps" in out and "#include <fcntl.h>" in out


def test_the_file_is_opened_next_to_the_result_csv_and_written_after_the_loop():
    out = rpr.patch_test_gemm(MINI)

    assert "open(reps_name" in out and '".reps"' in out
    assert out.index("time = time/nreps;") < out.index('"%d,%zu')  # rows are formatted after the timed loop
    assert "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds" in out


def test_the_driver_names_each_case_by_its_layer_id_and_the_built_tile():
    out = rpr.patch_test_gemm(MINI)

    assert "cnn_enable ? testConf->cnn[cnn_i].layer : 0" in out
    assert "MR%d_NR%d" in out


@pytest.mark.parametrize("missing", ["t2   = dclock();", "time = time/nreps;", "FILE *fd_csv;", '"#l;m;n;k;Gflops\\n");', "#include <string.h>"])
def test_an_anchor_that_is_not_there_stops_the_patch(missing: str):
    with pytest.raises(rpr.PatchError, match="anchor"):
        rpr.patch_test_gemm(MINI.replace(missing, "/* gone */"))


def test_an_anchor_that_occurs_twice_stops_the_patch():
    with pytest.raises(rpr.PatchError, match="anchor"):
        rpr.patch_test_gemm(MINI + "\n" + MINI)


def test_a_patched_file_is_not_patched_again():
    with pytest.raises(rpr.PatchError, match="already"):
        rpr.patch_test_gemm(rpr.patch_test_gemm(MINI))


def test_the_command_line_refuses_a_path_under_ext(tmp_path: Path):
    ext = tmp_path / "ext" / "src"
    ext.mkdir(parents=True)
    f = ext / "test_gemm.c"
    f.write_text(MINI)

    done = subprocess.run([sys.executable, str(HERE / "ref_baseline" / "ref_per_rep.py"), "patch", str(f)],
                          capture_output=True, text=True, check=False)

    assert done.returncode == 1 and "REFUSING" in done.stderr
    assert f.read_text() == MINI


def test_the_command_line_patches_a_staged_copy_in_place(tmp_path: Path):
    f = tmp_path / "stage" / "test_gemm.c"
    f.parent.mkdir()
    f.write_text(MINI)

    done = subprocess.run([sys.executable, str(HERE / "ref_baseline" / "ref_per_rep.py"), "patch", str(f)],
                          capture_output=True, text=True, check=False)

    assert done.returncode == 0, done.stderr
    assert "rep_stamp" in f.read_text()


def _compiler_errors(stage: Path) -> list[str]:
    """`gcc -fsyntax-only` errors on the staged test_gemm.c, without line numbers. On a host the
    kernel headers (ukrFunction and friends) are missing, so a clean pass is not the test: the
    patch must add no error of its own. The cross-compile of run_ref_baseline.sh's pre-flight
    is the full check."""
    done = subprocess.run(
        ["gcc", "-fsyntax-only", "-DFP32", "-DFAMILY_EXO", "-DMR=16", "-DNR=14", "-DKR=1",
         "-I", str(HERE / "ref_baseline" / "blis_shim"), "-I", str(stage), str(stage / "test_gemm.c")],
        capture_output=True, text=True, check=False,
    )
    return sorted(re.sub(r":\d+:\d+", "", ln).replace(str(stage), "") for ln in done.stderr.splitlines() if " error: " in ln)


@pytest.mark.skipif(not (REF_SRC / "test_gemm.c").is_file() or shutil.which("gcc") is None,
                    reason="the reference tree or a C compiler is not here")
def test_the_real_test_gemm_c_gets_no_new_compiler_error_from_the_patch(tmp_path: Path):
    stage = tmp_path / "src"
    shutil.copytree(REF_SRC, stage)
    before = _compiler_errors(stage)
    (stage / "test_gemm.c").write_text(rpr.patch_test_gemm((stage / "test_gemm.c").read_text()))

    assert _compiler_errors(stage) == before


# ---- converter ---------------------------------------------------------------------------------

CASES = "# case\tnumid\tM\tN\tK\tMR\tNR\nB2\t2\t384\t384\t64\t16\t15\nB3\t3\t64\t384\t384\t16\t15\n"

# The driver's file: the clock stamps' differences, one row per call, layer as its numeric id.
RAW = (
    "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds\n"
    "3,64,384,384,MR16_NR14,64,384,128,0,0.020000000\n"
    "3,64,384,384,MR16_NR14,64,384,128,1,0.010000000\n"
    "3,64,384,384,MR16_NR14,64,384,128,2,0.010000000\n"
)


def test_the_converter_maps_the_layer_id_to_its_label_and_keeps_our_columns(tmp_path: Path):
    cases = tmp_path / "cases.tsv"
    cases.write_text(CASES)

    text = rpr.convert_reps(RAW, rpr.load_labels(cases))

    lines = text.splitlines()
    assert lines[0] == "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles"
    assert lines[1] == "B3,64,384,384,MR16_NR14,64,384,128,0,0.020000000,nan,nan,nan,nan"


def test_a_converted_file_is_read_by_reps_report_unchanged(tmp_path: Path):
    cases = tmp_path / "cases.tsv"
    cases.write_text(CASES)
    f = tmp_path / "reps_ref_16x14_macc.txt.gz"
    f.write_bytes(gzip.compress(rpr.convert_reps(RAW, rpr.load_labels(cases)).encode()))

    got = read_reps(f)

    assert list(got) == [("B3", 384)]
    assert got[("B3", 384)].seconds == [0.02, 0.01, 0.01]
    assert case_stats(got[("B3", 384)].ops, got[("B3", 384)].seconds).n == 3


def test_an_unknown_layer_id_is_an_error_not_a_silent_label(tmp_path: Path):
    cases = tmp_path / "cases.tsv"
    cases.write_text(CASES)

    with pytest.raises(rpr.PatchError, match="layer id 9"):
        rpr.convert_reps(RAW.replace("3,64,384,384", "9,64,384,384"), rpr.load_labels(cases))


def test_the_total_is_checked_against_the_gflops_the_driver_wrote():
    seconds = [0.02, 0.01, 0.01]
    ops = 2.0 * 64 * 384 * 384
    mean_gflops = len(seconds) * ops / (sum(seconds) * 1e9)

    assert rpr.total_matches_driver(seconds, ops, f"{mean_gflops:.2e}")
    assert not rpr.total_matches_driver(seconds, ops, f"{mean_gflops * 1.2:.2e}")


def test_a_rep_count_over_the_array_is_reported_as_truncated():
    # The driver writes REP_CAP rows at most; a file with that many rows may have lost the rest.
    assert rpr.REP_CAP >= 100_000
    assert rpr.rows_may_be_truncated(rpr.REP_CAP) and not rpr.rows_may_be_truncated(rpr.REP_CAP - 1)
