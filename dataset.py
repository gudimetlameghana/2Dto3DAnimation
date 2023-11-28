import os
import random
import pickle
import torch
import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
from torchvision import transforms, utils
from smplx.lbs import transform_mat
from smpl_utils import init_smpl, get_J, get_shape_pose, batch_rodrigues, get_J_batch_cpu

def flip_theta(theta):
    thetas_flip = theta.clone().view(24, 3)
    # reflect horizontally
    thetas_flip[:, 1] = -1 * thetas_flip[:, 1]
    thetas_flip[:, 2] = -1 * thetas_flip[:, 2]
    # change left-right parts
    theta_pairs = [
        [22, 23], [20, 21], [18, 19], [16, 17], [13, 14], [1, 2], [4, 5], [7, 8], [10, 11]
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
            model_folder = smpl_cfgs['model_folder'],
            model_type = smpl_cfgs['model_type'],
            gender = smpl_cfgs['gender'],
            num_betas = smpl_cfgs['num_betas'],
            device = 'cpu'
        )
        self.parents = self.smpl_model.parents.cpu().numpy()

        self.flist = flist

        if flist is not None:
            with open(flist, 'r') as f:
                lines = f.readlines()
            self.image_names = ['.'.join(l.strip().split('.')[:-1]) for l in lines]
            if exclude_list is not None:
                self.image_names = [
                    l for i, l in enumerate(self.image_names) if i not in exclude_list
                ]
        else:
            lines = os.listdir(os.path.join(path, 'images'))
            self.image_names = ['.'.join(l.strip().split('.')[:-1]) for l in lines]

        self.image_names = sorted(self.image_names)
        self.image_path = [os.path.join(path, 'images', n + '.png') for n in self.image_names]
        self.seg_path = [os.path.join(path, 'segm', n + '_segm.png') for n in self.image_names]
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

        self.image_names = [n for i, n in enumerate(self.image_names) if i in valid_list]
        self.image_path = [n for i, n in enumerate(self.image_path) if i in valid_list]
        self.seg_path = [n for i, n in enumerate(self.seg_path) if i in valid_list]

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

    # def get_smpl_param(self, i, need_flip):
    #     trans = self.smpl_list[i]['trans']
    #     if need_flip:
    #         trans[0, 0] *= -1        
    #     beta = self.smpl_list[i]['beta']
    #     theta = self.smpl_list[i]['theta']
    #     if need_flip:
    #         theta = flip_theta(theta).view(1, 72)
    #     return trans.reshape(-1), beta.reshape(-1), theta.reshape(-1)

    # def get_camera_extrinsics(self, batch_size, device, val=False):
    #     def norm_np_arr(arr):
    #         return arr / np.linalg.norm(arr)

    #     def lookat(eye, at, up):
    #         zaxis = norm_np_arr(eye - at)
    #         xaxis = norm_np_arr(np.cross(up, zaxis))
    #         yaxis = np.cross(zaxis, xaxis)
    #         _viewMatrix = np.array([
    #             [xaxis[0], yaxis[0], zaxis[0], eye[0]],
    #             [xaxis[1], yaxis[1], zaxis[1], eye[1]],
    #             [xaxis[2], yaxis[2], zaxis[2], eye[2]],
    #             [0       , 0       , 0       , 1     ]
    #         ])
    #         return _viewMatrix

    #     def fix_eye(phi, theta):
    #         camera_distance = 13
    #         # phi = -np.pi / 2
    #         # theta = np.pi / 2
    #         return np.array([
    #             camera_distance * np.sin(theta) * np.cos(phi),
    #             camera_distance * np.sin(theta) * np.sin(phi),
    #             camera_distance * np.cos(theta)
    #         ])

    #     def random_eye_normal():
    #         camera_distance = np.random.uniform(1.8, 2.5)
    #         phi = np.random.uniform(0, 2 * np.pi)
    #         theta = np.random.normal(0, np.pi / 4)
    #         if theta > np.pi / 2 or theta < -np.pi / 2:
    #             is_front = 0
    #         else:
    #             is_front = 1
    #         return np.array([
    #             camera_distance * np.sin(theta) * np.cos(phi),
    #             camera_distance * np.sin(theta) * np.sin(phi),
    #             camera_distance * np.cos(theta)
    #         ]), theta, phi, is_front

    #     def random_eye(is_front=None, distance=None, theta_std=None):
    #         camera_distance = np.random.uniform(1, 2) if distance is None else distance
    #         phi = np.random.uniform(0, 2 * np.pi)
    #         if theta_std == None:
    #             theta_std = np.pi / 6
    #         theta = np.random.normal(0, theta_std)
    #         theta = np.clip(theta, -np.pi / 2, np.pi / 2)
    #         is_front = np.random.choice(2) if is_front is None else is_front
    #         if is_front == 0:
    #             theta += np.pi
    #         return np.array([
    #             camera_distance * np.sin(theta) * np.cos(phi),
    #             camera_distance * np.sin(theta) * np.sin(phi),
    #             camera_distance * np.cos(theta)
    #         ]), theta, phi, is_front
        
    #     if self.nerf_resolution[1] == 128:
    #         focal = 5000 / 8
    #     elif self.nerf_resolution[1] == 256:
    #         focal = 5000 / 4
    #     elif self.nerf_resolution[1] == 512:
    #         focal = 5000 / 2
    #     elif self.nerf_resolution[1] == 1024:
    #         focal = 5000
    #     else:
    #         raise NotImplementedError

    #     if val:
    #         extrinsic_list = []
    #         for i in range(batch_size):
    #             eye = fix_eye(np.pi * 2 * i / batch_size, np.pi / 2).astype(np.float32)
    #             at = np.zeros([3]).astype(np.float32)
    #             extrinsics = torch.from_numpy(lookat(eye, at, np.array([0, 0, 1])).astype(np.float32))
    #             extrinsic_list.append(extrinsics.view(1, 4, 4))
    #         default_extrinsics = torch.cat(extrinsic_list, 0).to(device)
    #         default_focal = torch.Tensor([focal]).view(1, 1).repeat(batch_size, 1).to(device)
    #     else:
    #         if 'SHHQ' in self.flist:
    #           focal *= 2
    #         default_extrinsics = torch.eye(4)
    #         default_extrinsics = default_extrinsics.view(1, 4, 4).repeat(batch_size, 1, 1).to(device)
    #         default_focal = torch.Tensor([focal]).view(1, 1).repeat(batch_size, 1).to(device)

    #     return default_extrinsics, default_focal

    # def batch_rigid_transform(self, rot_mats, init_J):
    #     joints = torch.from_numpy(init_J.reshape(-1, 24, 3, 1))
    #     parents = self.parents

    #     rel_joints = joints.clone()
    #     rel_joints[:, 1:] -= joints[:, parents[1:]]

    #     transforms_mat = transform_mat(
    #         rot_mats.reshape(-1, 3, 3),
    #         rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    #     transform_chain = [transforms_mat[:, 0]]
    #     for i in range(1, parents.shape[0]):
    #         curr_res = torch.matmul(transform_chain[parents[i]],
    #                                 transforms_mat[:, i])
    #         transform_chain.append(curr_res)

    #     transforms = torch.stack(transform_chain, dim=1)

    #     posed_joints = transforms[:, :, :3, 3]

    #     joints_homogen = F.pad(joints, [0, 0, 0, 1])

    #     rel_transforms = transforms - F.pad(
    #         torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    #     return posed_joints, rel_transforms

    # def calculate_rotation_weighted_sample(self, std=None):
    #     all_theta = []
    #     for smpl_data in self.smpl_list:
    #         all_theta.append(
    #             smpl_data['theta'].numpy().reshape(1, 72)
    #         )
    #     all_theta = np.concatenate(all_theta, 0)
    #     batch_size = all_theta.shape[0]
    #     all_theta = torch.from_numpy(all_theta)
    #     all_theta_rot = batch_rodrigues(all_theta.reshape(-1, 3)).reshape(batch_size, 24, 3, 3)

    #     init_J = get_J_batch_cpu(torch.zeros(batch_size, 10), self.smpl_model)
    #     _, rel_transforms = self.batch_rigid_transform(all_theta_rot, init_J)

    #     head_rot = rel_transforms[:, 15, :3, :3].cpu().numpy().reshape(batch_size, 9)

    #     head_euler = []
    #     for i in range(batch_size):
    #         r = R.from_matrix(head_rot[i].reshape(3, 3))
    #         head_euler.append(r.as_euler('zyx', degrees=True).reshape(1, 3))
    #     head_euler = np.concatenate(head_euler, 0)
    #     center_euler = head_euler
    #     n_clusters = head_euler.shape[0]
    #     r = R.from_euler('zyx', center_euler, degrees=True)
    #     v = [1, 0, 0]
    #     centers = []
    #     for i in range(n_clusters):
    #         centers.append(r[i].apply(v))
    #     centers = np.array(centers)
    #     angles = np.arctan2(centers[:, 0], centers[:, 2]) + np.pi
    #     angles = angles.reshape(-1)
    #     frac_num = 360
    #     frac = 2 * np.pi / frac_num
    #     num_list = []
    #     index_list = []
    #     for i in range(frac_num):
    #         index = np.logical_and(angles >= i * frac, angles < (i + 1) * frac)
    #         num_list.append(index.sum())
    #         index_list.append(index)
    #     weights = np.zeros_like(angles).reshape(-1)
    #     all_samples = sum(num_list)
    #     pdf_list = []
    #     for i, index, num in zip(range(len(index_list)), index_list, num_list):
    #         if i < 90:
    #             shift_i = 360 + i - 90
    #         else:
    #             shift_i = i - 90
    #         pdf = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((shift_i - 180)/std)**2 / 2)
    #         if shift_i < 180-20 or shift_i > 180+20:
    #             pdf = max(pdf, num / all_samples)
    #         pdf_list.append(pdf)
    #         if num > 0:
    #             weights[index] = pdf * 1000 / num
    #     return weights

    # def save_for_fid(self, base_folder):
    #     assert not self.random_flip
    #     assert self.white_bg
    #     if not os.path.exists(base_folder):
    #         os.makedirs(base_folder)
    #     for i in tqdm(range(self.__len__())):
    #         _, img, _, _, _ = self.__getitem__(i)
    #         img = ((img.permute(1,2,0) + 1)/2) * 255
    #         img = Image.fromarray(img.numpy().astype(np.uint8))
    #         img.save(os.path.join(base_folder, f"{str(i).zfill(7)}.png"))

    # def __getitem__(self, index):
    #     try:
    #         _img = Image.open(self.image_path[index]).convert("RGB")
    #         _seg = Image.open(self.seg_path[index]).convert("RGB")
    #     except:
    #         return self.__getitem__((index + 1) % self.__len__())

    #     img = np.asarray(_img.resize(self.resolution, Image.LANCZOS))
    #     thumb_img = np.asarray(_img.resize(self.nerf_resolution, Image.LANCZOS))
    #     seg = np.asarray(_seg.resize(self.resolution, Image.HAMMING))
    #     thumb_seg = np.asarray(_seg.resize(self.nerf_resolution, Image.HAMMING))
        
    #     mask = seg.sum(-1) == 0
    #     thumb_mask = thumb_seg.sum(-1) == 0

    #     if self.white_bg:
    #         img[mask, :] = 255
    #         thumb_img[thumb_mask, :] = 255
    #     else:
    #         img[mask, :] = 0
    #         thumb_img[thumb_mask, :] = 0

    #     img = self.transform(img)
    #     thumb_img = self.transform(thumb_img)

    #     if self.random_flip and random.random() >= 0.5:
    #         img = torch.flip(img, [-1])
    #         thumb_img = torch.flip(thumb_img, [-1])
    #         need_flip = True
    #     else:
    #         need_flip = False
    
    #     trans, beta, theta = self.get_smpl_param(index, need_flip)

    #     return img, thumb_img, trans, beta, theta
