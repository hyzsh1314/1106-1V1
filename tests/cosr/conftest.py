from pathlib import Path
import yaml

from ac_cosr_wvr.env.scenario_sampler import ScenarioSampler


def smoke_sampler(seed=1, split_code=1):
    root = Path(__file__).resolve().parents[2]
    return ScenarioSampler.from_config_path(root / "configs" / "cosr" / "smoke_autogun.yaml", seed, split_code)


def smoke_config():
    root = Path(__file__).resolve().parents[2]
    with (root / "configs" / "cosr" / "smoke_autogun.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
