import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            lang_fusion=None,
            lang_embed_dim=1024,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.lang_fuser = lang_fusion
        self.lang_proj = nn.Linear(lang_embed_dim, in_channels)

    def forward(self, x, l_input=None, skip=None, out_dim=None):
        if self.lang_fuser is not None and l_input is not None:
            x = self.lang_fuser(x, l_input, x2_proj=self.lang_proj)
        # Upscaling
        if skip is not None:
            scale_factor = skip.shape[-1] // x.shape[-1]
        elif out_dim:
            scale_factor = out_dim[-1] // x.shape[-1]
        else:
            scale_factor = 2
        x = F.interpolate(x, scale_factor=scale_factor, mode="nearest")

        # Double conv
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetLangFusionDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            fusion_module,
            lang_embed_dim,
            n_blocks=5,
            use_batchnorm=True,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch,
                         lang_fusion=fusion_module(),
                         lang_embed_dim=lang_embed_dim,
                         **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels[:3], skip_channels[:3], out_channels[:3])
        ]
        blocks.extend([
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels[3:], skip_channels[3:], out_channels[3:])
        ])
        self.blocks = nn.ModuleList(blocks)


    def forward(self, l_input, *features):
        out_dim = features[0].shape
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]
        
        for i, decoder_block in enumerate(self.blocks):
            if i < len(skips):
                x = decoder_block(x, l_input, skips[i])
            else:
                x = decoder_block(x, l_input, None, out_dim)


        return x


class PredictionHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling, activation)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)
