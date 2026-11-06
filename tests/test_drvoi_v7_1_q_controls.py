import torch

from benchmarks.drvoi_v7_1.q_controls import (
    irrelevant_perturbation_report,
    normalized_argmax_regret,
    pairwise_ranking_accuracy,
    structured_flip_report,
)


def test_pairwise_accuracy_and_normalized_regret_have_action_semantics():
    truth = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    perfect = truth.clone()
    wrong = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert torch.equal(pairwise_ranking_accuracy(perfect, truth), torch.ones(2))
    assert bool((normalized_argmax_regret(perfect, truth) == 0).all())
    assert bool((normalized_argmax_regret(wrong, truth) > 0).all())


def test_structured_flip_and_irrelevant_controls_are_distinct():
    before = torch.tensor([[2.0, 1.0], [3.0, 1.0]])
    after = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    flip = structured_flip_report(
        before[:1], after[:1], torch.tensor([0]), torch.tensor([1]))
    irrelevant = irrelevant_perturbation_report(
        before[1:], after[1:], torch.tensor([0]))
    assert flip["all_pass"]
    assert irrelevant["all_pass"]
