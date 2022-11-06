import pytest
import torch

from hulc2.models.encoders.language_network import SBert


@pytest.fixture(scope="module")
def language_encoder():
    language_encoder_net = SBert("mpnet")
    return language_encoder_net


# @pytest.mark.slow
@pytest.mark.parametrize(
    "input_str", ["fetch the red cup", "fetch the yellow banana, open the drawer and place it there"]
)
def test_expected_input_shape(language_encoder: torch.nn.Module, input_str: str) -> None:
    _ = language_encoder(input_str)


@pytest.mark.parametrize(
    "input_str", ["fetch the red cup", "fetch the yellow banana, open the drawer and place it there"]
)
def test_expected_output_shape(language_encoder: torch.nn.Module, input_str: str) -> None:
    outputs = language_encoder(input_str)
    assert torch.is_tensor(outputs)
    assert outputs.shape[0] == 768
    assert outputs.shape[1] == 1
