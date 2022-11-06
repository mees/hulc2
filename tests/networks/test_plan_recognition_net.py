import pytest
import torch
from torch.distributions import Distribution

from hulc2.models.plan_encoders.plan_recognition_net import PlanRecognitionBiLSTMNetwork
from hulc2.utils.distributions import Distribution as my_dist


@pytest.fixture(scope="module")
def plan_recognition_net():
    dist = my_dist(dist="discrete", category_size=32, class_size=32)
    plan_recog_net = PlanRecognitionBiLSTMNetwork(
        in_features=73, plan_features=256, action_space=8, birnn_dropout_p=0, dist=my_dist
    )
    return plan_recog_net


@pytest.mark.parametrize(
    "batch_size, seq_len, visual_features, n_state_obs, action_space", [(1, 32, 64, 9, 8), (8, 16, 64, 9, 8)]
)
def test_expected_input_shape(
    plan_recognition_net: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    visual_features: int,
    n_state_obs: int,
    action_space: int,
) -> None:
    example_emb = torch.zeros(batch_size, seq_len, visual_features + n_state_obs)
    _ = plan_recognition_net(example_emb)


@pytest.mark.parametrize("seq_len, visual_features, n_state_obs, action_space", [(16, 32, 9, 8), (32, 64, 5, 8)])
def test_input_too_small(
    plan_recognition_net: torch.nn.Module, seq_len: int, visual_features: int, n_state_obs: int, action_space: int
) -> None:
    example_emb = torch.zeros(1, seq_len, visual_features + n_state_obs)
    with pytest.raises(RuntimeError):
        _ = plan_recognition_net(example_emb)


@pytest.mark.parametrize(
    "batch_size, seq_len, visual_features, n_state_obs, action_space", [(1, 32, 64, 9, 8), (32, 32, 64, 9, 8)]
)
def test_expected_output_shape(
    plan_recognition_net: torch.nn.Module,
    batch_size: int,
    seq_len: int,
    visual_features: int,
    n_state_obs: int,
    action_space: int,
) -> None:
    example_emb = torch.zeros(batch_size, seq_len, visual_features + n_state_obs)
    output = plan_recognition_net(example_emb)
    assert isinstance(output, Distribution)
    assert output.batch_shape == (batch_size,)
    assert output.event_shape == (32, 32)
