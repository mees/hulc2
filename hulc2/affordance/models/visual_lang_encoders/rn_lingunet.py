import torch

from hulc2.affordance.models.core import fusion
import segmentation_models_pytorch as smp
from hulc2.affordance.models.core.unet_decoder import UnetLangFusionDecoder
from hulc2.affordance.models.visual_lang_encoders.base_lingunet import BaseLingunet


class RNLingunet(BaseLingunet):
    """Resnet 18 with U-Net skip connections and [] language encoder"""

    def __init__(self, input_shape, output_dim, cfg, device=0):
        super(RNLingunet, self).__init__(input_shape, output_dim, cfg)
        self.in_channels = input_shape[-1]
        self.n_classes = output_dim
        self.lang_embed_dim = 1024
        self.freeze_backbone = cfg.freeze_encoder.aff

        _encoder_name = "resnet18" if "encoder_name" not in cfg else cfg.encoder_name
        self.unet = self._build_model(cfg.unet_cfg.decoder_channels,
                                      self.in_channels,
                                      _encoder_name)

    def calc_img_enc_size(self):
        test_tensor = torch.zeros(self.input_shape).permute(2, 0, 1)
        test_tensor = test_tensor.unsqueeze(0)
        shape = self.unet.encoder(test_tensor)[-1].shape[1:]
        return shape

    def _build_model(self, decoder_channels, in_channels, encoder_name):
        # encoder_depth Should be equal to number of layers in decoder
        unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=self.n_classes,
            encoder_depth=len(decoder_channels),
            decoder_channels=tuple(decoder_channels),
            activation=None,
        )

        self.decoder = UnetLangFusionDecoder(
            fusion_module = fusion.names[self.lang_fusion_type],
            lang_embed_dim = self.lang_embed_dim,
            encoder_channels=unet.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels))

        self.decoder_channels = decoder_channels
        # Fix encoder weights. Only train decoder
        for param in unet.encoder.parameters():
            param.requires_grad = False
        if not self.freeze_backbone:
            for param in unet.encoder.layer4.parameters():
                param.requires_grad = True
        return unet

    def forward(self, x, text_enc, softmax=True):
        # in_type = x.dtype
        # in_shape = x.shape
        x = x[:,:3]  # select RGB
        encoder_feat = self.unet.encoder(x)  # List of all encoder outputs

        # Language encoding
        l_enc, l_emb, l_mask  = text_enc
        l_input = l_enc.to(dtype=x.dtype)

        # Decoder
        # encode image
        decoder_feat = self.decoder(l_input, *encoder_feat)
        aff_out = self.unet.segmentation_head(decoder_feat)

        info = {"decoder_out": [decoder_feat],
                "hidden_layers": encoder_feat,
                "affordance": aff_out,
                "text_enc": l_input}
        return aff_out, info