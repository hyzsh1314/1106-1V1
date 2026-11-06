import numpy as np

from envs.JSBSim.utils.utils import (
    LLA2NEU,
    LLA2NEU_reference,
    NEU2LLA,
    NEU2LLA_reference,
    get_wgs84_local_frame,
)


def test_cached_wgs84_frame_matches_pymap3d_in_wvr_envelope():
    rng = np.random.default_rng(20261102)
    origin = (120.0, 60.0, 0.0)
    for _ in range(1000):
        lon = origin[0] + rng.uniform(-0.5, 0.5)
        lat = origin[1] + rng.uniform(-0.5, 0.5)
        alt = rng.uniform(-100.0, 20000.0)
        actual = LLA2NEU(lon, lat, alt, *origin)
        expected = LLA2NEU_reference(lon, lat, alt, *origin)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_cached_wgs84_inverse_matches_pymap3d_in_wvr_envelope():
    rng = np.random.default_rng(20261103)
    origin = (120.0, 60.0, 0.0)
    for _ in range(1000):
        north, east = rng.uniform(-60000.0, 60000.0, size=2)
        up = rng.uniform(-100.0, 20000.0)
        actual = NEU2LLA(north, east, up, *origin)
        expected = NEU2LLA_reference(north, east, up, *origin)
        np.testing.assert_allclose(actual[:2], expected[:2], rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(actual[2], expected[2], rtol=0.0, atol=1e-5)


def test_cached_wgs84_round_trip_and_frame_cache():
    origin = (120.0, 60.0, 0.0)
    frame = get_wgs84_local_frame(*origin)
    assert frame is get_wgs84_local_frame(*origin)
    lla = np.asarray([119.97, 60.04, 7250.0])
    restored = frame.neu_to_lla(*frame.lla_to_neu(*lla))
    np.testing.assert_allclose(restored[:2], lla[:2], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(restored[2], lla[2], rtol=0.0, atol=1e-6)

