import pytest
import torch

from hulc2.models.decoders.state_decoder import StateDecoder


@pytest.fixture(scope="module")
def state_decoder():
    state_decoder_net = StateDecoder(visual_features=64, n_state_obs=9)
    return state_decoder_net


@pytest.mark.parametrize("batch_size, window_size, visual_features", [(1, 16, 64), (8, 32, 64)])
def test_expected_input_shape(
    state_decoder: torch.nn.Module, batch_size: int, window_size: int, visual_features: int
) -> None:
    example_input_array = torch.zeros(batch_size, window_size, visual_features)
    _ = state_decoder(example_input_array)


@pytest.mark.parametrize("visual_features", [9, 16, 32])
def test_input_too_small(state_decoder: torch.nn.Module, visual_features: int) -> None:
    example_input_array = torch.zeros(1, visual_features)
    with pytest.raises(RuntimeError):
        _ = state_decoder(example_input_array)


@pytest.mark.parametrize("batch_size, window_size, visual_features", [(1, 16, 64), (32, 32, 64)])
def test_expected_output_shape(
    state_decoder: torch.nn.Module, batch_size: int, window_size: int, visual_features: int
) -> None:
    example_input_array = torch.zeros(batch_size, window_size, visual_features)
    outputs = state_decoder(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == window_size
    assert outputs.shape[2] == 9
