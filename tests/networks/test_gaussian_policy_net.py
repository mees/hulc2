import pytest
import torch

from hulc2.models.decoders.gaussian_policy_network import GaussianPolicyNetwork

SEQ_LEN = 32
N_MIXTURES = 10
OUT_FEATURES = 9


@pytest.fixture(scope="module")
def gaussian_policy_decoder():
    gaussian_policy_net = GaussianPolicyNetwork(
        perceptual_features=73,
        latent_goal_features=32,
        plan_features=256,
        hidden_size=2048,
        out_features=9,
        n_mixtures=N_MIXTURES,
        log_scale_min=-7.0,
        log_scale_max=7.0,
    )
    return gaussian_policy_net


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs, plan_features",
    [(1, 64, 32, 9, 256), (32, 64, 32, 9, 256)],
)
def test_expected_input_shape(
    gaussian_policy_decoder: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(batch_size, plan_features)
    perceptual_emb = torch.zeros((batch_size, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(batch_size, latent_goal_features)
    _ = gaussian_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize(
    "visual_features, latent_goal_features, n_state_obs, plan_features", [(32, 32, 9, 256), (64, 16, 8, 256)]
)
def test_input_too_small(
    gaussian_policy_decoder: torch.nn.Module,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(plan_features)
    perceptual_emb = torch.zeros((1, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(latent_goal_features)
    with pytest.raises(RuntimeError):
        _ = gaussian_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize("visual_features, latent_goal_features, n_state_obs, plan_features", [(64, 32, 9, 256)])
def test_input_no_batch_dim(
    gaussian_policy_decoder: torch.nn.Module,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(plan_features)
    perceptual_emb = torch.zeros((SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(latent_goal_features)
    with pytest.raises(RuntimeError):
        _ = gaussian_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs, plan_features",
    [(1, 64, 32, 9, 256), (32, 64, 32, 9, 256)],
)
def test_expected_output_shape(
    gaussian_policy_decoder: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(batch_size, plan_features)
    perceptual_emb = torch.zeros((batch_size, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(batch_size, latent_goal_features)
    outputs = gaussian_policy_decoder(latent_plan, perceptual_emb, latent_goal)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert outputs[0].shape[0] == batch_size
    assert outputs[0].shape[1] == SEQ_LEN
    assert outputs[0].shape[2] == N_MIXTURES
    assert outputs[1].shape[0] == batch_size
    assert outputs[1].shape[1] == SEQ_LEN
    assert outputs[1].shape[2] == N_MIXTURES
    assert outputs[1].shape[3] == OUT_FEATURES
    assert outputs[1].shape == outputs[2].shape
