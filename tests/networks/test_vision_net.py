import pytest
import torch

from hulc2.models.perceptual_encoders.vision_network import VisionNetwork


@pytest.fixture(scope="module")
def vision_network():
    vis_net = VisionNetwork(
        input_width=200,
        input_height=200,
        activation_function="ReLU",
        dropout_vis_fc=0,
        l2_normalize_output=False,
        visual_features=64,
        num_c=3,
    )
    return vis_net


@pytest.mark.parametrize("batch_size, img_h, img_w", [(2, 200, 200), (8, 200, 200)])
def test_expected_input_shape(vision_network: torch.nn.Module, batch_size: int, img_h: int, img_w: int) -> None:
    example_input_array = torch.zeros(batch_size, 3, img_h, img_w)
    _ = vision_network(example_input_array)


@pytest.mark.parametrize("img_height, img_width", [(64, 64), (100, 100)])
def test_input_too_small(vision_network: torch.nn.Module, img_height: int, img_width: int) -> None:
    example_input_array = torch.zeros(1, 3, img_height, img_width)
    with pytest.raises(RuntimeError):
        _ = vision_network(example_input_array)


@pytest.mark.parametrize("img_height, img_width", [(200, 200), (300, 300)])
def test_input_no_batch_dim(vision_network: torch.nn.Module, img_height: int, img_width: int) -> None:
    example_input_array = torch.zeros(3, img_height, img_width)
    with pytest.raises(RuntimeError):
        _ = vision_network(example_input_array)


@pytest.mark.parametrize("batch_size, img_h, img_w", [(2, 200, 200), (8, 200, 200)])
def test_expected_output_shape(vision_network: torch.nn.Module, batch_size: int, img_h: int, img_w: int) -> None:
    example_input_array = torch.zeros(batch_size, 3, img_h, img_w)
    outputs = vision_network(example_input_array)
    assert outputs.shape[0] == batch_size
    assert outputs.shape[1] == 64
