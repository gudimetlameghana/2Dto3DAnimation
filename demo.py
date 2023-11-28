import torch
from munch import *
from torchvision import transforms

from dataset import DeepFashionDataset
from generator import Generator
from params import HyperParameters as hp
from predict import predict3d


if __name__ == "__main__":
    dataset_hp = hp.dataset
    inference_hp = hp.inference
    model_hp = hp.model
    renderer_hp = hp.rendering
    smpl_hp = hp.smpl

    eva3d = torch.load(model_hp.path, map_location=lambda storage, loc: storage)

    g_ema = Generator(model_hp, renderer_hp, smpl_hp, ema=True).to("cuda")
    g_ema.load_state_dict(eva3d["g_ema"])

    dataset = DeepFashionDataset(dataset_hp)

    predict3d(g_ema, model_hp, dataset, inference_hp)
