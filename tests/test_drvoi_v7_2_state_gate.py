from benchmarks.validate_drvoi_v7_2_state_sufficiency import decide


def gates(passed):
    return {"only": {"pass": passed}}


REPRESENTATIONS = {
    "registered_v7_1": {"deployable": True},
    "full_observation": {"deployable": True},
    "future_belief": {"deployable": False},
    "mechanistic_oracle": {"deployable": False},
}


def test_oracle_failure_stops_state_target():
    result = {key: gates(False) for key in REPRESENTATIONS}
    assert decide(result, REPRESENTATIONS) == ("STATE_TARGET_NO_GO", False)


def test_oracle_only_requires_new_predictor():
    result = {key: gates(False) for key in REPRESENTATIONS}
    result["mechanistic_oracle"] = gates(True)
    assert decide(result, REPRESENTATIONS) == ("NEW_PREDICTOR_REQUIRED", False)


def test_deployable_and_oracle_pass_authorizes_response_development():
    result = {key: gates(False) for key in REPRESENTATIONS}
    result["mechanistic_oracle"] = gates(True)
    result["full_observation"] = gates(True)
    assert decide(result, REPRESENTATIONS) == (
        "DEPLOYABLE_REPRESENTATION_FOUND", True)
