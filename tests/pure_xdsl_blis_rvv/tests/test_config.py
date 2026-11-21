from config import BLISProjectConfig


def _pinned_config(mr_nr_family):
    return BLISProjectConfig(
        datatype="fp32",
        method="AXPY_FP32_Pack",
        mr_nr_family=mr_nr_family,
        vlen=256,
        arch="jbpi2",
        datasets=["some.dat"],
        skip_ref_run=True,
        iterations=1000,
        is_pinned=True,
    )


def test_out_dir_disambiguates_pinned_mr_nr_variants():
    # Issue #2: multiple pinned MR/NR rows sharing datatype/method/vlen must not
    # collide on the same pj_gen/ output directory, or the generator silently
    # overwrites earlier variants with the last one processed.
    dir_a = _pinned_config([(32, 7)]).out_dir()
    dir_b = _pinned_config([(32, 6)]).out_dir()
    dir_c = _pinned_config([(16, 15)]).out_dir()

    assert dir_a != dir_b
    assert dir_b != dir_c
    assert dir_a != dir_c


def test_out_dir_unpinned_has_no_mr_nr_suffix():
    # Sweep-mode (unpinned) configs generate a full macrokernel pool, not a
    # single (mr, nr) build, so the directory name stays as before.
    config = _pinned_config([(8, 8), (16, 16)])
    config.is_pinned = False

    assert config.out_dir() == "pj_gen/fp32_AXPY_FP32_Pack_vlen256"


def test_out_dir_pinned_includes_mr_nr():
    config = _pinned_config([(32, 7)])

    assert config.out_dir() == "pj_gen/fp32_AXPY_FP32_Pack_vlen256_MR32NR7"
