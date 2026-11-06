"""Print one deterministic generic PBGR launch/rejection/resolution flow."""

from __future__ import annotations

import numpy as np

from envs.JSBSim.envs.multiplecombat_env import MultipleCombatEnv


def main():
    env = MultipleCombatEnv("2v2/ShootMissile/PBGR_V22_PLATFORM_CORRECT")
    try:
        env.seed(20261103)
        env.set_initial_geometry_reset_index(0)
        env.reset()
        masks = env.get_v22_weapon_action_masks()
        print("initial_target_masks=", masks["target_masks"].astype(int).tolist())
        print("initial_fire_masks=", masks["fire_masks"].astype(int).tolist())

        fire = np.asarray([[1, 2, 1, 0, 1]] * 4, dtype=np.int64)
        hold = np.asarray([[1, 2, 1, 0, 0]] * 4, dtype=np.int64)
        first = env.step(fire)
        print("step1_events=", first[4]["weapon_events"])
        print("ammo_after_launch=", {
            uid: aircraft.num_left_missiles
            for uid, aircraft in env.agents.items()
        })
        print("active_missiles=", sorted(env._tempsims))

        second = env.step(fire)
        print("step2_events=", second[4]["weapon_events"])
        print("step2_reasons=", {
            uid: state["reason"]
            for uid, state in second[4]["weapon_legality"].items()
        })

        result = second
        for _ in range(200):
            result = env.step(hold)
            events = [
                event for event in result[4]["weapon_events"]
                if event["event_type"] in (
                    "hit", "redundant_hit", "miss", "invalidated"
                )
            ]
            if events:
                print(
                    "resolved_at_step=", result[4]["current_step"],
                    "events=", events,
                )
            if np.asarray(result[3]).all():
                print("episode_done_at=", result[4]["current_step"])
                break
        print("final_status=", {
            uid: ("alive" if aircraft.is_alive else "dead")
            for uid, aircraft in env.agents.items()
        })
        print("total_weapon_events=", result[4]["weapon_event_count"])
        print("weapon_event_counts=", result[4]["weapon_event_counts"])
        print(
            "decision_transition_steps=",
            result[4]["weapon_decision_step"],
            result[4]["weapon_transition_step"],
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()

