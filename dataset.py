import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

from smpl_utils import init_smpl


def flip_theta(theta):
    thetas_flip = theta.clone().view(24, 3)
    # reflect horizontally
    thetas_flip[:, 1] = -1 * thetas_flip[:, 1]
    thetas_flip[:, 2] = -1 * thetas_flip[:, 2]
    # change left-right parts
    theta_pairs = [
        [22, 23], [20, 21], [18, 19], [16, 17], [
            13, 14], [1, 2], [4, 5], [7, 8], [10, 11]
    ]
    for pair in theta_pairs:
        thetas_flip[pair[0], :], thetas_flip[pair[1], :] = \
            thetas_flip[pair[1], :], thetas_flip[pair[0], :].clone()

    return thetas_flip


class DeepFashionDataset(Dataset):
    def __init__(self, path, transform, resolution=(256, 128), nerf_resolution=(128, 64), flist=None,
                 white_bg=False,
                 random_flip=False,
                 exclude_list=None,
                 gaussian_weighted_sampler=False, sampler_std=0.3 / np.pi * 180):
        self.path = path
        self.transform = transform
        self.resolution = (resolution[1], resolution[0])
        self.nerf_resolution = (nerf_resolution[1], nerf_resolution[0])
        self.flist = flist
        self.white_bg = white_bg
        self.random_flip = random_flip
        self.gaussian_weighted_sampler = gaussian_weighted_sampler

        smpl_cfgs = {
            'model_folder': 'smpl_models',
            'model_type': 'smpl',
            'gender': 'neutral',
            'num_betas': 10
        }
        self.smpl_model = init_smpl(
            model_folder=smpl_cfgs['model_folder'],
            model_type=smpl_cfgs['model_type'],
            gender=smpl_cfgs['gender'],
            num_betas=smpl_cfgs['num_betas'],
            device='cpu'
        )
        self.parents = self.smpl_model.parents.cpu().numpy()

        self.flist = flist

        if flist is not None:
            with open(flist, 'r') as f:
                lines = f.readlines()
            self.image_names = [
                '.'.join(l.strip().split('.')[:-1]) for l in lines]
            if exclude_list is not None:
                self.image_names = [
                    l for i, l in enumerate(self.image_names) if i not in exclude_list
                ]
        else:
            lines = os.listdir(os.path.join(path, 'images'))
            self.image_names = [
                '.'.join(l.strip().split('.')[:-1]) for l in lines]

        self.image_names = sorted(self.image_names)
        self.image_path = [os.path.join(
            path, 'images', n + '.png') for n in self.image_names]
        self.seg_path = [os.path.join(path, 'segm', n + '_segm.png')
                         for n in self.image_names]
        with open(os.path.join(path, 'smpl.pkl'), 'rb') as f:
            self.smpl_dict = pickle.load(f)
        self.smpl_list = []
        valid_list = []
        for i, p in enumerate(self.image_names):
            tmp = self.read_smpl(self.smpl_dict[p])
            if tmp is None:
                continue
            self.smpl_list.append(tmp)
            valid_list.append(i)
        print("image num: {}".format(len(self.image_names)))
        print("smpl num: {}".format(len(self.smpl_list)))

        self.image_names = [n for i, n in enumerate(
            self.image_names) if i in valid_list]
        self.image_path = [n for i, n in enumerate(
            self.image_path) if i in valid_list]
        self.seg_path = [n for i, n in enumerate(
            self.seg_path) if i in valid_list]

    def read_smpl(self, data):
        try:
            data_dict = {
                'camera_rotation': torch.from_numpy(data['camera_rotation'].reshape(1, 3, 3)),
                'trans': torch.from_numpy(data['camera_translation'].reshape(1, 3)),
                'beta': torch.from_numpy(data['betas'].reshape(1, 10)),
                'theta': torch.from_numpy(np.concatenate([data['global_orient'], data['body_pose']], -1).reshape(1, 72))
            }
            for k, v in data_dict.items():
                if torch.any(torch.isnan(v)):
                    return None
        except:
            return None
        return data_dict

    def __len__(self):
        return len(self.image_names)

    def sample_smpl_param(self, batch_size, device, val=False, idx=None):
        _rand_ind = random.sample(range(self.__len__()), batch_size)
        _rand_ind_beta = random.sample(range(self.__len__()), batch_size)
        rand_ind = _rand_ind
        rand_ind_beta = _rand_ind_beta
        trans_list = []
        beta_list = []
        theta_list = []
        for i, i_beta in zip(rand_ind, rand_ind_beta):
            print(f"image_name: {self.image_names[i]}")
            if self.random_flip and random.random() >= 0.5 and (idx is None):
                need_flip = True
            else:
                need_flip = False

            trans = self.smpl_list[i]['trans']
            if need_flip:
                trans[0, 0] *= -1
            trans_list.append(trans)

            beta = self.smpl_list[i]['beta']
            beta_list.append(beta)

            theta = self.smpl_list[i]['theta']
            if need_flip:
                theta = flip_theta(theta).view(1, 72)
            theta_list.append(theta)
        return torch.cat(trans_list, 0).to(device), \
            torch.cat(beta_list, 0).to(device), \
            torch.cat(theta_list, 0).to(device)
