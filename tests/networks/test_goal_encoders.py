import pytest
import torch

from hulc2.models.encoders.goal_encoders import LanguageGoalEncoder, VisualGoalEncoder


@pytest.fixture(scope="module")
def language_goal_encoder():
    language_goal_encoder_net = LanguageGoalEncoder(
        language_features=512,
        hidden_size=2048,
        latent_goal_features=32,
        word_dropout_p=0,
        l2_normalize_goal_embeddings=False,
        activation_function="ReLU",
    )
    return language_goal_encoder_net


@pytest.fixture(scope="module")
def visual_goal_encoder():
    visual_goal_encoder_net = VisualGoalEncoder(
        in_features=73,
        hidden_size=2048,
        latent_goal_features=32,
        activation_function="ReLU",
        l2_normalize_goal_embeddings=False,
    )
    return visual_goal_encoder_net


@pytest.mark.parametrize("batch_size, proprio_features", [(2, 73), (8, 73)])
def test_expected_input_shape_vis(visual_goal_encoder: torch.nn.Module, batch_size: int, proprio_features: int) -> None:
    example_input_array = torch.zeros(batch_size, proprio_features)
    _ = visual_goal_encoder(example_input_array)


@pytest.mark.parametrize("batch_size, language_features", [(2, 512), (8, 512)])
def test_expected_input_shape_lang(
    language_goal_encoder: torch.nn.Module, batch_size: int, language_features: int
) -> None:
    example_input_array = torch.zeros(batch_size, language_features)
    _ = language_goal_encoder(example_input_array)


@pytest.mark.parametrize("visual_features", [9, 16, 32])
def test_input_too_small_vis(visual_goal_encoder: torch.nn.Module, visual_features: int) -> None:
    example_input_array = torch.zeros(1, visual_features)
    with pytest.raises(RuntimeError):
        _ = visual_goal_encoder(example_input_array)


@pytest.mark.parametrize("language_features", [32, 64, 128])
def test_input_too_small_lang(language_goal_encoder: torch.nn.Module, language_features: int) -> None:
    example_input_array = torch.zeros(1, language_features)
    with pytest.raises(RuntimeError):
        _ = language_goal_encoder(example_input_array)


@pytest.mark.parametrize("batch_size, proprio_features", [(2, 73), (32, 73)])
def test_expected_output_shape_vis(
    visual_goal_encoder: torch.nn.Module, batch_size: int, proprio_features: int
) -> None:
    example_input_array = torch.zeros(batch_size, proprio_features)
    outputs = visual_goal_encoder(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 32


@pytest.mark.parametrize("batch_size, language_features", [(2, 512), (32, 512)])
def test_expected_output_shape_lang(
    language_goal_encoder: torch.nn.Module, batch_size: int, language_features: int
) -> None:
    example_input_array = torch.zeros(batch_size, language_features)
    outputs = language_goal_encoder(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 32
