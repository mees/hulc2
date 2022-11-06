import pytest
import torch

from hulc2.models.perceptual_encoders.tactile_encoder import TactileEncoder


@pytest.fixture(scope="module")
def tactile_encoder():
    tactile_encoder_net = TactileEncoder(visual_features=64)
    return tactile_encoder_net


@pytest.mark.parametrize("batch_size, img_h, img_w", [(1, 64, 64), (8, 160, 120), (1, 224, 224)])
def test_expected_input_shape(tactile_encoder: torch.nn.Module, batch_size: int, img_h: int, img_w: int) -> None:
    example_input_array = torch.zeros(batch_size, 6, img_h, img_w)
    _ = tactile_encoder(example_input_array)


@pytest.mark.parametrize("img_height, img_width", [(32, 9), (64, 5), (64, 32)])
def test_input_too_small(tactile_encoder: torch.nn.Module, img_height: int, img_width: int) -> None:
    example_input_array = torch.zeros(1, 6, img_width, img_width)
    with pytest.raises(ValueError):
        _ = tactile_encoder(example_input_array)


@pytest.mark.parametrize("batch_size, img_h, img_w", [(2, 64, 64), (8, 160, 120), (16, 224, 224)])
def test_expected_output_shape(tactile_encoder: torch.nn.Module, batch_size: int, img_h: int, img_w: int) -> None:
    example_input_array = torch.zeros(batch_size, 6, img_h, img_w)
    outputs = tactile_encoder(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 64
