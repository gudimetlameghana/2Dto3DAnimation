from munch import *


HyperParameters = Munch({
    'dataset': Munch({
        'dataset_path': './data/DeepFashion',
        'random_flip': False,
        'gaussian_weighted_sampler': False,
        'sampler_std': 15
    }),
    'experiment': Munch({
        'config': None,
        'expname': '512x256_deepfashion',
        'ckpt': '420000',
        'continue_training': False
    }),
    'training': Munch({
        'checkpoints_dir': './checkpoint',
        'iter': 300000,
        'batch': 1,
        'chunk': 1,
        'val_n_sample': 8,
        'd_reg_every': 16,
        'g_reg_every': 4,
        'local_rank': 0,
        'mixing': 0.9,
        'lr': 0.002,
        'r1': 300,
        'eikonal_lambda': 0.5,
        'min_surf_lambda': 1.5,
        'min_surf_beta': 100.0,
        'path_regularize': 2,
        'path_batch_shrink': 2,
        'wandb': False,
        'small_aug': False,
        'adjust_gamma': False,
        'gamma_lb': 20,
        'glr': 2e-05,
        'dlr': 0.0002,
        'deltasdf': True,
        'fid_path': ''
    }),
    'inference': Munch({
        'results_dir': './evaluations',
        'truncation_ratio': 0.5,
        'truncation_mean': 10000,
        'identities': 1,
        'num_views_per_id': 1,
        'no_surface_renderings': False,
        'fixed_camera_angles': False,
        'azim_video': False,
        'size': [256, 128],
        'camera': Munch({
            'uniform': False,
            'azim': 0.3,
            'elev': 0.15,
            'fov': 6,
            'dist_radius': 0.12
        }),
        'renderer_output_size': [512, 256],
        'style_dim': 128,
        'project_noise': False,
        'return_xyz': False
    }),
    'model': Munch({
        'size': [256, 128],
        'style_dim': 128,
        'channel_multiplier': 2,
        'n_mlp': 8,
        'lr_mapping': 0.01,
        'renderer_spatial_output_dim': [512, 256],
        'project_noise': False,
        'smpl_model_folder': 'smpl_models',
        'smpl_gender': 'neutral',
        'voxhuman_name': 'eva3d_deepfashion',
        'is_test': True,
        'freeze_renderer': False
    }),
    'camera': Munch({
        'uniform': False,
        'azim': 0.3,
        'elev': 0.15,
        'fov': 6,
        'dist_radius': 0.12
    }),
    'rendering': Munch({
        'depth': 5,
        'width': 128,
        'no_sdf': False,
        'no_z_normalize': False,
        'static_viewdirs': True,
        'is_aist': False,
        'N_samples': 28,
        'no_offset_sampling': False,
        'perturb': 0,
        'raw_noise_std': 0.0,
        'force_background': True,
        'return_xyz': False,
        'return_sdf': False,
        'stepsize': 0.02,
        'white_bg': True,
        'input_ch_views': 3,
        'render_video': False,
        'no_features_output': True,
        'offset_sampling': True
    })})
