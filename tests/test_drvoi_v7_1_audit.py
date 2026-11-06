from benchmarks.export_drvoi_v7_1_development_audit import t_interval


def test_small_sample_t_interval_uses_seed_as_independent_unit():
    mean, interval = t_interval([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mean == 3.0
    assert interval[0] < mean < interval[1]
    assert round(interval[0], 6) == 1.036757
