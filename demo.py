import torch
from munch import *
from torchvision import transforms

from EVA3D.dataset import DeepFashionDataset
from EVA3D.model import VoxelHumanGenerator as Generator
from params import HyperParameters
from predict import predict3d


if __name__ == "__main__":
    opt = HyperParameters

    eva3d = torch.load('models/eva3d.pt',
                       map_location=lambda storage, loc: storage)

    g_ema = Generator(opt.model, opt.rendering, ema=True,
                      full_pipeline=False).to("cuda")
    g_ema.load_state_dict(eva3d["g_ema"])

    # move inside dataset.py
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    ])
    file_list = '/content/2Dto3DAnimation/data/DeepFashion/train_list.txt'

    dataset = DeepFashionDataset(opt.dataset.dataset_path, transform, opt.model.size,
                                 opt.model.renderer_spatial_output_dim, file_list)

    predict3d(g_ema, opt.inference, dataset)
