from dataclasses import replace

import numpy as np

from ac_cosr_wvr.env.scenario_sampler import ScenarioSampler


def tail_chase_spec(sampler, episode_index=0, swapped=False):
    base = sampler.sample(episode_index, 0, "validation")
    red = {"north_m": -1500.0, "east_m": 0.0, "altitude_m": 5000.0,
           "speed_mps": 220.0, "heading_rad": 0.0}
    blue = {"north_m": 0.0, "east_m": 0.0, "altitude_m": 5000.0,
            "speed_mps": 220.0, "heading_rad": 0.0}
    red.update(sampler.neutral_state); blue.update(sampler.neutral_state)
    result = replace(base, episode_id=base.episode_id + "-tail", red_init=red, blue_init=blue,
                     candidate_side="red", side_swap=False)
    return ScenarioSampler.swapped(result) if swapped else result


def run_to_terminal(env, spec, steps=6):
    env.reset(spec)
    result = None
    actions = np.asarray([[0., 0., 0., .5], [0., 0., 0., .5]], dtype=np.float32)
    for _ in range(steps):
        result = env.step_physical(actions)
        if result["terminated"] or result["truncated"]:
            break
    return result
