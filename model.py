import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from eva3d_deepfashion import VoxelHuman as EVA3D_DEEPFASHION_MODEL
from fused_act import fused_leaky_relu


class MappingLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, activation=None, is_last=False):
        super().__init__()
        if is_last:
            weight_std = 0.25
        else:
            weight_std = 1

        self.weight = nn.Parameter(weight_std * nn.init.kaiming_normal_(
            torch.empty(out_dim, in_dim), a=0.2, mode='fan_in', nonlinearity='leaky_relu'))

        if bias:
            self.bias = nn.Parameter(nn.init.uniform_(torch.empty(
                out_dim), a=-np.sqrt(1/in_dim), b=np.sqrt(1/in_dim)))
        else:
            self.bias = None

        self.activation = activation

    def forward(self, input):
        if self.activation != None:
            out = F.linear(input, self.weight)
            out = fused_leaky_relu(out, self.bias, scale=1)
        else:
            out = F.linear(input, self.weight, bias=self.bias)

        return out


class VoxelHumanGenerator(nn.Module):
    def __init__(self, model_opt, renderer_opt, blur_kernel=[1, 3, 3, 1], ema=False, full_pipeline=True, voxhuman_name=None):
        super().__init__()
        self.size = model_opt.size
        self.style_dim = model_opt.style_dim
        self.num_layers = 1
        self.train_renderer = not model_opt.freeze_renderer
        self.full_pipeline = full_pipeline
        model_opt.feature_encoder_in_channels = renderer_opt.width
        self.model_opt = model_opt
        self.voxhuman_name = voxhuman_name

        if ema or 'is_test' in model_opt.keys():
            self.is_train = False
        else:
            self.is_train = True

        # volume renderer mapping_network
        layers = []
        for i in range(3):
            layers.append(
                MappingLinear(self.style_dim, self.style_dim,
                              activation="fused_lrelu")
            )

        self.style = nn.Sequential(*layers)

        # volume renderer
        thumb_im_size = model_opt.renderer_spatial_output_dim
        smpl_cfgs = {
            'model_folder': model_opt.smpl_model_folder,
            'model_type': 'smpl',
            'gender': model_opt.smpl_gender,
            'num_betas': 10
        }
        VoxHuman_Class = EVA3D_DEEPFASHION_MODEL
        self.renderer = VoxHuman_Class(renderer_opt, smpl_cfgs, out_im_res=tuple(
            model_opt.renderer_spatial_output_dim), style_dim=self.style_dim)

        if self.full_pipeline:
            raise NotImplementedError

    def mean_latent(self, n_latent, device):
        latent_in = torch.randn(n_latent, self.style_dim, device=device)
        renderer_latent = self.style(latent_in)
        renderer_latent_mean = renderer_latent.mean(0, keepdim=True)
        if self.full_pipeline:
            decoder_latent_mean = self.decoder.mean_latent(renderer_latent)
        else:
            decoder_latent_mean = None

        return [renderer_latent_mean, decoder_latent_mean]

    def styles_and_noise_forward(self, styles, inject_index=None, truncation=1,
                                 truncation_latent=None, input_is_latent=False):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent[0] + truncation *
                    (style - truncation_latent[0])
                )

            styles = style_t

        return styles
