from benchmarks.collect_drvoi_v7_1_confirmation_data import (
    validate_confirmation_payload,
)


def test_confirmation_payload_requires_both_calibration_and_test():
    report = {"b0_seed": 85, "contains_confirmation_test": True}
    episodes = [{"split": "calibration"}, {"split": "test"}]
    validate_confirmation_payload(report, episodes)
    try:
        validate_confirmation_payload(report, episodes[:1])
    except RuntimeError as error:
        assert "wrong splits" in str(error)
    else:
        raise AssertionError("incomplete confirmation splits were accepted")
