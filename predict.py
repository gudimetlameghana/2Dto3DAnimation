import os
import torch
import trimesh

from utils import extract_mesh_with_marching_cubes


def predict3d(g_ema, model_hp, dataset, inference_hp):
    for name, p in g_ema.named_parameters():
        p.requires_grad = False

    g_ema.renderer.is_train = False
    g_ema.renderer.perturb = 0

    style_dim = model_hp.style_dim

    _, beta, theta = dataset.sample_smpl_param()
    mean_latent = g_ema.mean_latent(inference_hp)

    styles = torch.randn(1, style_dim, device='cuda')
    styles = g_ema.styles_and_noise_forward(
        styles[:1],
        inference_hp,
        mean_latent
    )

    sdf = g_ema.renderer.marching_cube_posed(
        styles[0],
        beta,
        theta,
        resolution=500,
        size=1.4
    ).detach()

    mesh, _, _ = extract_mesh_with_marching_cubes(sdf)
    mesh = trimesh.smoothing.filter_humphrey(mesh, beta=0.2, iterations=5)

    output_path = inference_hp.path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mesh_path = output_path + inference_hp.file_name
    with open(mesh_path, 'w') as f:
        mesh.export(f, file_type='obj')
