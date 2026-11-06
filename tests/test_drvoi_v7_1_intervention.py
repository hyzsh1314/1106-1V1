import torch

from benchmarks.diagnose_drvoi_v7_1_paired_intervention import build_probe_actions
from benchmarks.drvoi_v7_1.history import NEUTRAL_ACTION


def test_probe_actions_change_exactly_one_declared_head_per_lane():
    actions, heads = build_probe_actions(24, "cpu")
    neutral = torch.tensor(NEUTRAL_ACTION).expand_as(actions)
    changed = actions != neutral
    assert torch.equal(changed.sum(-1), torch.ones(24, dtype=torch.long))
    assert torch.equal(changed.nonzero()[:, 1], heads)
