from ac_cosr_wvr.env.scenario_sampler import ScenarioSampler
from ac_cosr_wvr.env.outcome import OutcomeResolver, SafetyVector
from .conftest import smoke_sampler


def test_side_swap_exchanges_initial_states_and_result_signs():
    scenario = smoke_sampler(1, 2).sample(0, 3, "validation")
    swapped = ScenarioSampler.swapped(scenario)
    assert swapped.red_init == scenario.blue_init
    assert swapped.blue_init == scenario.red_init
    resolver = OutcomeResolver()
    red, blue = SafetyVector(), SafetyVector()
    assert resolver.resolve(red, blue, red_gun_kill=True).result_red == 1
    assert resolver.resolve(red, blue, blue_gun_kill=True).result_red == -1
