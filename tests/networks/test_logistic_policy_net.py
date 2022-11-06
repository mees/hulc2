import pytest
import torch

from hulc2.models.decoders.logistic_policy_network import LogisticPolicyNetwork

SEQ_LEN = 32
N_MIXTURES = 10
OUT_FEATURES = 9
ACT_MAX = [1, 1, 1, 1, 1, 1, 1, 1, 1]
ACT_MIN = [-1, -1, -1, -1, -1, -1, -1, -1, -1]


@pytest.fixture(scope="module")
def logistic_policy_decoder():
    logistic_policy_net = LogisticPolicyNetwork(
        perceptual_features=73,
        plan_features=256,
        latent_goal_features=32,
        n_mixtures=N_MIXTURES,
        hidden_size=2048,
        out_features=OUT_FEATURES,
        log_scale_min=-7.0,
        act_max_bound=ACT_MAX,
        act_min_bound=ACT_MIN,
        policy_rnn_dropout_p=0.0,
        num_classes=256,
        load_action_bounds=False,
        dataset_dir="",
    )
    return logistic_policy_net


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs, plan_features",
    [(1, 64, 32, 9, 256), (32, 64, 32, 9, 256)],
)
def test_expected_input_shape(
    logistic_policy_decoder: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(batch_size, plan_features)
    perceptual_emb = torch.zeros((batch_size, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(batch_size, latent_goal_features)
    _ = logistic_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize(
    "visual_features, latent_goal_features, n_state_obs, plan_features", [(32, 32, 9, 256), (64, 32, 8, 256)]
)
def test_input_too_small(
    logistic_policy_decoder: torch.nn.Module,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(plan_features)
    perceptual_emb = torch.zeros((1, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(latent_goal_features)
    with pytest.raises(RuntimeError):
        _ = logistic_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize("visual_features, latent_goal_features, n_state_obs, plan_features", [(64, 32, 9, 256)])
def test_input_no_batch_dim(
    logistic_policy_decoder: torch.nn.Module,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(plan_features)
    perceptual_emb = torch.zeros((SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(latent_goal_features)
    with pytest.raises(RuntimeError):
        _ = logistic_policy_decoder(latent_plan, perceptual_emb, latent_goal)


@pytest.mark.parametrize(
    "batch_size, visual_features, latent_goal_features, n_state_obs, plan_features",
    [(1, 64, 32, 9, 256), (32, 32, 64, 9, 256)],
)
def test_expected_output_shape(
    logistic_policy_decoder: torch.nn.Module,
    batch_size: int,
    visual_features: int,
    latent_goal_features: int,
    n_state_obs: int,
    plan_features: int,
) -> None:
    latent_plan = torch.zeros(batch_size, plan_features)
    perceptual_emb = torch.zeros((batch_size, SEQ_LEN, visual_features + n_state_obs))
    latent_goal = torch.zeros(batch_size, latent_goal_features)
    outputs = logistic_policy_decoder(latent_plan, perceptual_emb, latent_goal)
    assert isinstance(outputs, tuple)
    assert len(outputs) == 3
    assert outputs[0].shape[0] == batch_size
    assert outputs[0].shape[1] == SEQ_LEN
    assert outputs[0].shape[2] == OUT_FEATURES
    assert outputs[0].shape[3] == N_MIXTURES
    assert outputs[0].shape == outputs[1].shape
