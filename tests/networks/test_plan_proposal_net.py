import pytest
import torch

from hulc2.models.plan_encoders.plan_proposal_net import PlanProposalNetwork
from hulc2.utils.distributions import Distribution, State


@pytest.fixture(scope="module")
def plan_proposal_net():
    plan_prop_net = PlanProposalNetwork(
        perceptual_features=73,
        plan_features=1024,
        latent_goal_features=32,
        activation_function="ReLU",
        hidden_size=2048,
        dist=Distribution(**{"dist": "discrete", "category_size": 32, "class_size": 32}),
    )
    return plan_prop_net


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs", [(2, 64, 32, 9), (8, 64, 32, 9)]
)
def test_expected_input_shape(
    plan_proposal_net: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
) -> None:
    example_emb = torch.zeros(batch_size, visual_features + n_state_obs)
    example_goal = torch.zeros(batch_size, latent_goal_features)
    _ = plan_proposal_net(example_emb, example_goal)


@pytest.mark.parametrize("visual_features,, latent_goal_features, n_state_obs", [(32, 32, 9), (64, 32, 5)])
def test_input_too_small(
    plan_proposal_net: torch.nn.Module, visual_features: int, latent_goal_features: int, n_state_obs: int
) -> None:
    example_emb = torch.zeros(1, visual_features + n_state_obs)
    example_goal = torch.zeros(1, latent_goal_features)
    with pytest.raises(RuntimeError):
        _ = plan_proposal_net(example_emb, example_goal)


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs", [(2, 64, 32, 9), (32, 64, 32, 9)]
)
def test_expected_output_shape(
    plan_proposal_net: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
) -> None:
    example_emb = torch.zeros(batch_size, visual_features + n_state_obs)
    example_goal = torch.zeros(batch_size, latent_goal_features)
    output = plan_proposal_net(example_emb, example_goal)
    # assert isinstance(output, State)
    assert output.logit.shape == (batch_size, 1024)
