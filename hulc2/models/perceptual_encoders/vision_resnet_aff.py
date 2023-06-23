import numpy as np
from segmentation_models_pytorch.encoders import get_encoder
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class VisionResnetAff(nn.Module):
    def __init__(self, visual_features: int, input_shape: list, depth: int = 3, freeze_backbone: bool = True):
        super(VisionResnetAff, self).__init__()
        # Load pre-trained resnet-18
        self.net = get_encoder("resnet18", in_channels=input_shape[-1], depth=depth, weights="imagenet")
        # Remove the last fc layer, and rebuild
        for param in self.net.parameters():
            param.requires_grad = False
        if freeze_backbone:
            for param in self.net.layer4.parameters():
                param.requires_grad = True

        out_shape = self.calc_img_enc_size(list(input_shape))
        self.fc1 = nn.Linear(np.prod(out_shape), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, visual_features)

    def calc_img_enc_size(self, input_shape):
        test_tensor = torch.zeros(input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.unsqueeze(0)
        shape = self.net(test_tensor)[-1].shape[1:]
        return shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)[-1]  # batch, 128, n, n
        # Add fc layer for final prediction
        x = torch.flatten(x, start_dim=1)  # batch, n*n*128
        output = F.relu(self.fc1(x))  # batch, 512
        output = F.relu(self.fc2(output))  # batch, 256
        output = self.fc3(output)  # batch, 64
        return output
