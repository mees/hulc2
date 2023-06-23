import torch
from torch import nn
import torch.nn.functional as F

from r3m import load_r3m


class VisionR3M(nn.Module):
    def __init__(
        self, device: torch.device, visual_features: int, resnet_model: str = "resnet18", freeze_backbone: bool = True
    ):
        super(VisionR3M, self).__init__()
        # Load pre-trained R3M resnet-18
        self.r3m = load_r3m(resnet_model, device).module
        # set all grads to false
        for param in self.r3m.parameters():
            param.requires_grad = False
        if not freeze_backbone:
            # finetune last layer
            for param in self.r3m.convnet.layer4.parameters():
                param.requires_grad = True
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.r3m(x)  # batch, 512, 1, 1
        # Add fc layer for final prediction
        x = torch.flatten(x, start_dim=1)  # batch, 512
        output = F.relu(self.fc1(x))  # batch, 256
        output = self.fc2(output)  # batch, 64
        return output
