from benchmarks.validate_drvoi_v7_1_b0_robustness import group_pass


def test_b0_robustness_requires_four_of_five_positive_directions():
    result = {
        "mean_delta_nll_per_dimension": 0.01,
        "delta_nll_by_collection_seed": {
            "1": 1, "2": 1, "3": 1, "4": 1, "5": -1},
        "permutation_gain_per_dimension": 0.02,
        "permutation_gain_by_collection_seed": {
            "1": 1, "2": 1, "3": 1, "4": 1, "5": -1},
        "training": {
            "initial_function_max_difference": 0.0,
            "gradient_fault_over_1000": False,
        },
    }
    assert group_pass(result)
    result["delta_nll_by_collection_seed"]["4"] = -1
    assert not group_pass(result)
