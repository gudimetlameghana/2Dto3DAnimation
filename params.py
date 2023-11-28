from munch import *


HyperParameters = Munch({
    'dataset': Munch({
        'file_list_path': '/content/2Dto3DAnimation/data/DeepFashion/train_list.txt', # Path of the file list
        'path': './data/DeepFashion', # Path of the DeepFashion dataset
        'random_flip': False,
        'gaussian_weighted_sampler': False,
        'sampler_std': 15
    }),
    'inference': Munch({
        'file_name': 'mesh.obj', # File name of the output object
        'path': '/content/2Dto3DAnimation/evaluations', # Path of the inference outputs
        'truncation_mean': 10000, # Vectors count to calculate mean
        'truncation_ratio': 0.5, # Diversity-Quality tradeoff. Higher ratio will generate more diverse results
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
        'path': 'models/eva3d.pt', # Path of the pretrained model
        'size': [256, 128], # Size of the generator nn
        'style_dim': 128,  # Input dimensions for the generator nn
        'channel_multiplier': 2,
        'n_mlp': 8,
        'lr_mapping': 0.01,
        'project_noise': False,
        'voxhuman_name': 'eva3d_deepfashion',
        'is_test': True,
        'freeze_renderer': False
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
    }),
    'smpl': Munch({
        'gender': 'neutral', # Gender of the smpl model
        'model_folder': 'smpl_models', # Path of the smpl models
        'model_type': 'smpl', # Type of the model
        'num_betas': 10 # Number of betas to build the layer
    })
})
