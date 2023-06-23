import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import DistilBertModel, DistilBertTokenizer

from hulc2.affordance.models.core import fusion
from hulc2.affordance.models.core.resnet import ConvBlock, IdentityBlock
from hulc2.affordance.models.core.unet import Up
from hulc2.affordance.models.visual_lang_encoders.base_lingunet import BaseLingunet


class RN50LingUNet(BaseLingunet):
    """ImageNet RN50 & Bert with U-Net skip connections"""

    def __init__(self, input_shape, output_dim, cfg, device):
        super(RN50LingUNet, self).__init__()
        self.output_dim = output_dim
        self.input_dim = 2048
        self.batchnorm = self.cfg["batchnorm"]
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        self.freeze_backbone = True
        self.device = device
        self._load_vision_fcn()
        self._load_lang_enc()
        self._build_decoder()

    def _load_vision_fcn(self):
        resnet50 = models.resnet50(pretrained=self.pretrained)
        modules = list(resnet50.children())[:-2]

        self.stem = nn.Sequential(*modules[:4])
        self.layer1 = modules[4]
        self.layer2 = modules[5]
        self.layer3 = modules[6]
        self.layer4 = modules[7]

    def _load_lang_enc(self):
        self.lang_fuser1 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 2)
        self.lang_fuser2 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 4)
        self.lang_fuser3 = fusion.names[self.lang_fusion_type](input_dim=self.input_dim // 8)

        self.proj_input_dim = 1024
        self.lang_proj1 = nn.Linear(self.proj_input_dim, 1024)
        self.lang_proj2 = nn.Linear(self.proj_input_dim, 512)
        self.lang_proj3 = nn.Linear(self.proj_input_dim, 256)

    def _build_decoder(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 1024, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(True)
        )
        self.up1 = Up(2048, 1024 // self.up_factor, self.bilinear)
        self.up2 = Up(1024, 512 // self.up_factor, self.bilinear)
        self.up3 = Up(512, 256 // self.up_factor, self.bilinear)

        self.layer1 = nn.Sequential(
            ConvBlock(128, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(64, [64, 64, 64], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer2 = nn.Sequential(
            ConvBlock(64, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(32, [32, 32, 32], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.layer3 = nn.Sequential(
            ConvBlock(32, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            IdentityBlock(16, [16, 16, 16], kernel_size=3, stride=1, batchnorm=self.batchnorm),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

        self.conv2 = nn.Sequential(nn.Conv2d(16, self.output_dim, kernel_size=1))

    def resnet50(self, x):
        im = []
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            im.append(x)
        return x, im

    def encode_image(self, img):
        with torch.requires_grad(not self.freeze_backbone):
            img_encoding, img_im = self.resnet50(img)
        return img_encoding, img_im

    def forward(self, x, l):
        in_type = x.dtype
        in_shape = x.shape
        x = x[:, :3]  # select RGB
        x, im = self.encode_image(x)
        x = x.to(in_type)

        # encode language
        l_enc, l_emb, l_mask = self.encode_text(l)
        l_input = l_enc.to(dtype=x.dtype)

        # encode image
        assert x.shape[1] == self.input_dim
        x = self.conv1(x)

        x = self.lang_fuser1(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj1)
        x = self.up1(x, im[-2])

        x = self.lang_fuser2(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj2)
        x = self.up2(x, im[-3])

        x = self.lang_fuser3(x, l_input, x2_mask=l_mask, x2_proj=self.lang_proj3)
        x = self.up3(x, im[-4])

        for layer in [self.layer1, self.layer2, self.layer3, self.conv2]:
            x = layer(x)

        x = F.interpolate(x, size=(in_shape[-2], in_shape[-1]), mode="bilinear")
        return x
