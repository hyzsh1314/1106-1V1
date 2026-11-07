import pytest

from benchmarks.train_torch_wvr_mods_rhppo import intrinsic_beta_schedule


def test_intrinsic_beta_holds_then_cosine_anneals():
    values = [intrinsic_beta_schedule(0.05, 0.005, update, 11, 0.4)
              for update in range(1, 12)]
    assert values[:5] == pytest.approx([0.05] * 5)
    assert values[-1] == pytest.approx(0.005)
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_intrinsic_beta_single_update_uses_initial_value():
    assert intrinsic_beta_schedule(0.05, 0.005, 1, 1, 0.4) == 0.05


def test_intrinsic_beta_rejects_invalid_hold_fraction():
    with pytest.raises(ValueError):
        intrinsic_beta_schedule(0.05, 0.005, 1, 10, 1.1)
