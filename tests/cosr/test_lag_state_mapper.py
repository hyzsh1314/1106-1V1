import math

import numpy as np

from ac_cosr_wvr.env.lag_state_mapper import LAGStateMapper, ReloadTolerance


class _Simulator(object):
    uid = "mapper-test"
    def get_position(self):
        return np.asarray([100.0, -50.0, 3000.0])

    def get_rpy(self):
        return np.asarray([0.0, 0.0, math.pi / 2.0])

    def get_velocity(self):
        return np.asarray([200.0, 0.0, 0.0])

    def get_property_value(self, property):
        return {"velocities/p-rad_sec": 0.0, "velocities/q-rad_sec": 0.0,
                "velocities/r-rad_sec": 0.0}[property.name_jsbsim]


def test_lag_state_mapper_converts_and_verifies_reloaded_state():
    mapper = LAGStateMapper(120.0, 60.0, 0.0, ReloadTolerance(1.0, 1.0, .01, 1.0, .01, .01, .01, .01, .01))
    state = {"north_m": 100.0, "east_m": -50.0, "altitude_m": 3000.0, "speed_mps": 200.0,
             "heading_rad": math.pi / 2.0, "roll_rad": 0.0, "pitch_rad": 0.0,
             "p_rad_sec": 0.0, "q_rad_sec": 0.0, "r_rad_sec": 0.0}
    lag = mapper.to_lag(state)
    assert lag["ic_h_sl_ft"] > 9800.0 and lag["ic_u_fps"] > 650.0
    assert lag["ic_psi_true_deg"] == 90.0
    assert mapper.validate_reload(_Simulator(), state)["errors"]["position_m"] == 0.0
