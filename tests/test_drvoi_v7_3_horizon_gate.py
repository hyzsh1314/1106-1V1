from benchmarks.validate_drvoi_v7_3_return_horizon import decide


def matrix():
    return {name: {str(value): False for value in (3, 12, 30, 120, "full")}
            for name in ("registered_v7_1", "full_observation",
                         "mechanistic_oracle")}


def test_no_oracle_horizon_is_target_no_go():
    assert decide(matrix(), 30) == ("RETURN_TARGET_NO_GO", None, False)


def test_only_short_oracle_horizon_is_myopic_no_go():
    value = matrix()
    value["mechanistic_oracle"]["12"] = True
    assert decide(value, 30) == ("MYOPIC_ONLY_NO_GO", None, False)


def test_oracle_without_full_observation_requires_representation_work():
    value = matrix()
    value["mechanistic_oracle"]["30"] = True
    assert decide(value, 30) == (
        "REPRESENTATION_PREDICTOR_REQUIRED", None, False)


def test_largest_joint_passing_horizon_is_selected():
    value = matrix()
    for horizon in ("30", "120"):
        value["mechanistic_oracle"][horizon] = True
        value["full_observation"][horizon] = True
    assert decide(value, 30) == ("HORIZON_FOUND", 120, True)


def test_full_episode_dominates_finite_horizons_when_it_passes():
    value = matrix()
    value["mechanistic_oracle"]["full"] = True
    value["full_observation"]["full"] = True
    assert decide(value, 30) == ("HORIZON_FOUND", "full", True)
