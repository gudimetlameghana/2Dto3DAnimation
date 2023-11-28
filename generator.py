import torch
import numpy as np
from torch import nn
from torch.nn import functional

from eva3d_deepfashion import VoxelHuman
from fused_act import fused_leaky_relu


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.activation = "fused_lrelu"

        self.bias = nn.Parameter(
            nn.init.uniform_(
                torch.empty(out_dim),
                a=-np.sqrt(1/in_dim),
                b=np.sqrt(1/in_dim)
            )
        )

        self.weight = nn.Parameter(
            nn.init.kaiming_normal_(torch.empty(out_dim, in_dim), a=0.2)
        )

    def forward(self, input):
        next = fused_leaky_relu(
            functional.linear(input, self.weight),
            self.bias,
            scale=1
        )
        return next


class Generator(nn.Module):
    def __init__(self, model_hp, renderer_hp, smpl_hp, ema=False):
        super().__init__()

        self.is_train = not ema
        self.size = model_hp.size
        self.style_dim = model_hp.style_dim

        self.style = nn.Sequential(
            LinearLayer(self.style_dim, self.style_dim),
            LinearLayer(self.style_dim, self.style_dim),
            LinearLayer(self.style_dim, self.style_dim)
        )

        self.renderer = VoxelHuman(
            renderer_hp,
            smpl_hp,
            style_dim=self.style_dim
        )

    def mean_latent(self, inference_hp):
        trunc_mean = inference_hp.truncation_mean

        renderer_latent_mean = self.style(torch.randn(
            trunc_mean, self.style_dim, device='cuda')).mean(0, keepdim=True)

        decoder_latent_mean = None
        return [renderer_latent_mean, decoder_latent_mean]

    def styles_and_noise_forward(self, styles, inference_hp, trunc_latent):
        styles = [self.style(s) for s in styles]
        trunc_ratio = inference_hp.truncation_ratio

        if trunc_ratio < 1:
            styles = list(map(
                lambda style: trunc_latent[0] +
                trunc_ratio * (style - trunc_latent[0]),
                styles
            ))

        return styles
