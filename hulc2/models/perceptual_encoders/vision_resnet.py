import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class VisionResnet(nn.Module):
    def __init__(self, visual_features: int, freeze_backbone: bool = True):
        super(VisionResnet, self).__init__()
        # Load pre-trained resnet-18
        net = models.resnet18(pretrained=True)
        # Remove the last fc layer, and rebuild
        modules = list(net.children())[:-1]
        for param in net.parameters():
            param.requires_grad = False

        # Only finetune last layer
        if not freeze_backbone:
            for param in net.layer4.parameters():
                param.requires_grad = True
        self.net = nn.Sequential(*modules)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)  # batch, 512, 1, 1
        # Add fc layer for final prediction
        x = torch.flatten(x, start_dim=1)  # batch, 512
        output = F.relu(self.fc1(x))  # batch, 256
        output = self.fc2(output)  # batch, 64
        return output
