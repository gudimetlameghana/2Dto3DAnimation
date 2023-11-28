import os
import random
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class DeepFashionDataset(Dataset):
    def __init__(self, dataset_hp):
        self.flist = dataset_hp.file_list_path
        self.path = dataset_hp.path

        with open(self.flist, 'r') as f:
            lines = f.readlines()

        self.image_names = sorted([
            '.'.join(l.strip().split('.')[:-1]) for l in lines
        ])
        self.image_path = [
            os.path.join(self.path, 'images', n + '.png') for n in self.image_names
        ]

        with open(os.path.join(self.path, 'smpl.pkl'), 'rb') as f:
            self.smpl_hp = pickle.load(f)

        self.validate_images()

    def validate_images(self):
        self.smpl_list = []
        valid_list = []

        for i, p in enumerate(self.image_names):
            tmp = self.read_smpl(self.smpl_hp[p])
            if tmp is None:
                continue

            self.smpl_list.append(tmp)
            valid_list.append(i)

        self.image_names = [
            n for i, n in enumerate(self.image_names) if i in valid_list
        ]
        self.image_path = [
            n for i, n in enumerate(self.image_path) if i in valid_list
        ]

    def read_smpl(self, data):
        try:
            data_dict = {
                'beta': torch.from_numpy(data['betas'].reshape(1, 10)),
                'camera_rotation': torch.from_numpy(data['camera_rotation'].reshape(1, 3, 3)),
                'theta': torch.from_numpy(np.concatenate([data['global_orient'], data['body_pose']], -1).reshape(1, 72)),
                'trans': torch.from_numpy(data['camera_translation'].reshape(1, 3))
            }

            for k, v in data_dict.items():
                if torch.any(torch.isnan(v)):
                    return None
        except:
            return None

        return data_dict

    def sample_smpl_param(self):
        images_count = len(self.image_names)

        rand_ind = random.sample(range(images_count), 1)
        rand_ind_beta = random.sample(range(images_count), 1)

        i, i_beta = rand_ind[0], rand_ind_beta[0]
        print(f"image_name: {self.image_names[i]}")

        trans = self.smpl_list[i]['trans']
        beta = self.smpl_list[i]['beta']
        theta = self.smpl_list[i]['theta']

        return trans.to('cuda'), beta.to('cuda'), theta.to('cuda')
