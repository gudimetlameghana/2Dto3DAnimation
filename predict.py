import os
import torch
import trimesh

from utils import extract_mesh_with_marching_cubes


def predict3d(g_ema, inference_params, dataset):
    for name, p in g_ema.named_parameters():
        p.requires_grad = False
    g_ema.renderer.is_train = False
    g_ema.renderer.perturb = 0

    style_dim = inference_params.style_dim
    truncation_mean = inference_params.truncation_mean
    truncation_ratio = inference_params.truncation_ratio

    trans, beta, theta = dataset.sample_smpl_param(1, 'cuda', val=False)
    mean_latent = g_ema.mean_latent(truncation_mean, 'cuda')

    styles = torch.randn(1, style_dim, device='cuda')
    styles = g_ema.styles_and_noise_forward(
        styles[:1], truncation=truncation_ratio, truncation_latent=mean_latent)

    sdf = g_ema.renderer.marching_cube_posed(
        styles[0], beta, theta, resolution=500, size=1.4).detach()

    mesh, _, _ = extract_mesh_with_marching_cubes(sdf)
    mesh = trimesh.smoothing.filter_humphrey(mesh, beta=0.2, iterations=5)

    output_path = '/content/2Dto3DAnimation/evaluations'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mesh_path = output_path + '/mesh.obj'
    with open(mesh_path, 'w') as f:
        mesh.export(f, file_type='obj')
