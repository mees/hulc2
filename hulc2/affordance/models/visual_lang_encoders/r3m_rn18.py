from r3m import load_r3m
import torch
import torch.nn as nn

from hulc2.affordance.models.core.unet_decoder import UnetLangFusionDecoder
from hulc2.affordance.models.core import fusion
from hulc2.affordance.models.visual_lang_encoders.base_lingunet import BaseLingunet


class R3M(BaseLingunet):
    """ R3M RN 18 & SBert with U-Net skip connections """

    def __init__(self, input_shape, output_dim, cfg):
        super(R3M, self).__init__(input_shape, output_dim, cfg)
        self.output_dim = output_dim
        self.input_dim = 512
        self.lang_embed_dim = 1024
        self.batchnorm = self.cfg['batchnorm']
        self.bilinear = True
        self.up_factor = 2 if self.bilinear else 1
        _encoder_name = "resnet18"
        self.freeze_backbone = cfg.freeze_encoder.aff
        self._load_vision(_encoder_name)
        self._build_decoder(cfg.unet_cfg.decoder_channels)

    def _load_vision(self, resnet_model):
        self.r3m = load_r3m(resnet_model, device="cuda").module
        modules = list(list(self.r3m.children())[3].children())

        self.stem = nn.Sequential(*modules[:4])
        self.layer1 = modules[4]
        self.layer2 = modules[5]
        self.layer3 = modules[6]
        self.layer4 = modules[7]

        # Fix encoder weights. Only train decoder
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in layer.parameters():
                param.requires_grad = False
        if not self.freeze_backbone:
            for param in self.layer4.parameters():
                param.requires_grad = True

    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.unsqueeze(0).to(self.r3m.device)
        shape = self.r3m_resnet18(test_tensor)[0].shape[1:]
        return shape

    def _build_decoder(self, decoder_channels):
        # decoder_channels = (256, 128, 64, 64, 32)
        decoder_channels = (512, 256, 128, 64, 32)        
        self.decoder = UnetLangFusionDecoder(
            fusion_module=fusion.names[self.lang_fusion_type],
            lang_embed_dim=self.lang_embed_dim,
            encoder_channels=(3, 64, 64, 128, 256, 512),
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels))

        kernel_size = 3
        n_classes = 1
        self.segmentation_head = nn.Conv2d(decoder_channels[-1],
                                           n_classes,
                                           kernel_size=kernel_size,
                                           padding=kernel_size // 2)

    def r3m_resnet18(self, x):
        im = []
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)
            im.append(x)
        return x, im

    def forward(self, x, text_enc):
        input = x[:,:3]  # select RGB
        #input 32, 3, 224, 224
        x, im = self.r3m_resnet18(input)  # 32, 512, 7, 7

        # encode language
        l_enc, l_emb, l_mask = text_enc
        l_input = l_enc.to(dtype=x.dtype)  # [32, 1024]

        encoder_feat = [input, *im]
        decoder_feat = self.decoder(l_input, *encoder_feat)
        aff_out = self.segmentation_head(decoder_feat)

        info = {"decoder_out": [decoder_feat],
                "hidden_layers": encoder_feat,
                "affordance": aff_out,
                "text_enc": l_input}
        return aff_out, info