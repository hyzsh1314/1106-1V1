from benchmarks.summarize_drvoi_v7_gates import summarize


def test_only_all_prerequisite_gates_authorize_short_screen():
    report = summarize(
        {"all_gates_pass": True}, {"ready": True}, {"pass": True})
    assert report["short_screen_100k_to_500k_authorized"] is True
    assert report["v7_long_training_authorized"] is False


def test_failed_gate_keeps_every_training_stage_blocked():
    report = summarize(
        {"all_gates_pass": True}, {"ready": False}, {"pass": True})
    assert report["short_screen_100k_to_500k_authorized"] is False
    assert report["v7_long_training_authorized"] is False
