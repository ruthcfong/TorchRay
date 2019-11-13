import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.gradient import gradient, gradient_sum
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.guided_backprop import guided_backprop, guided_backprop_sum
from torchray.attribution.linear_approx import linear_approx
from torchray.attribution.meta_saliency import meta_saliency
from torchray.attribution.norm_grad import norm_grad, norm_grad_selective
from torchray.attribution.rise import rise

from torchray.benchmark.datasets import get_dataset
from torchray.benchmark.models import get_model, get_transform
from torchray.benchmark.common import compute_similarity

from torchray.utils import imsc, imarraysc, accuracy

from tqdm import tqdm


dataset = 'imagenet'
subset = 'val'

saliency_funcs = {
    'contrastive_excitation_backprop': contrastive_excitation_backprop,
    'deconvnet': deconvnet,
    'excitation_backprop': excitation_backprop,
    'grad_cam': grad_cam,
    'gradient': gradient,
    'gradient_sum': gradient_sum,
    'guided_backprop': guided_backprop,
    'guided_backprop_sum': guided_backprop_sum,
    'linear_approx': linear_approx,
    'norm_grad': norm_grad,
    'norm_grad_selective': norm_grad_selective,
}


def meta_correlation(arch="resnet50",
                     method="gradient",
                     saliency_layer="",
                     lr=1e-4,
                     batch_size=256,
                     num_workers=4,
                     plot=False,
                     gpu=None):
    # Set device.
    if isinstance(gpu, np.int):
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    # Get model.
    model = get_model(arch=arch,
                      dataset=dataset,
                      convert_to_fully_convolutional=False)
    model.eval()
    model = model.to(device)

    softmax = nn.Softmax(dim=1)

    # Get data transformation.
    transform = get_transform(dataset=dataset, size=(224, 224))

    # Get dataset and data loader.
    dset = get_dataset(dataset, subset, transform=transform)
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Get saliency method.
    saliency_func = saliency_funcs[method]

    kwargs = {
        "resize": True,
        "saliency_layer": saliency_layer,
    }

    if plot:
        f, ax = plt.subplots(1, 4, figsize=(4*4, 4))
    if False:
        f, ax = plt.subplots(1, 1)

    corrs = []
    scores = []
    class_ids = []
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        saliency_map = saliency_func(model, x, y, **kwargs)
        meta_saliency_map = meta_saliency(saliency_func,
                                          model,
                                          x,
                                          y,
                                          lr=lr,
                                          **kwargs)
        out = softmax(model(x))
        score = torch.gather(out, 1, y.unsqueeze(1)).cpu().data.numpy().squeeze()

        if plot:
            plt.axes(ax[0])
            imarraysc(x)
            plt.axes(ax[1])
            imarraysc(saliency_map)
            plt.axes(ax[2])
            imarraysc(meta_saliency_map)
            plt.axes(ax[3])
            imarraysc(torch.abs(saliency_map - meta_saliency_map))
            plt.pause(1)

        corr, _ = compute_similarity(saliency_map, meta_saliency_map)
        corrs.extend(list(corr))
        scores.extend(list(score))
        class_ids.extend(list(y.cpu().data.numpy()))
        if False and i % 10 == 0:
            ax.plot(corrs, scores, '.')
            plt.pause(1)

    out_dir = "./data/meta_correlation"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    prefix = f"{arch}_{method}_{saliency_layer}_{lr:.4f}"
    np.savetxt(os.path.join(out_dir, f"{prefix}_corrs.txt"),
               np.array(corrs),
               delimiter="\n")
    np.savetxt(os.path.join(out_dir, f"{arch}_scores.txt"),
               np.array(scores),
               delimiter="\n")
    np.savetxt(os.path.join(out_dir, f"{arch}_class_ids.txt"),
               np.array(scores),
               delimter="\n",
               fmt="%d")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--method", choices=saliency_funcs.keys(), default="gradient")
    parser.add_argument("--saliency_layer", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument("--gpu", type=int, default=None)

    args = parser.parse_args()

    meta_correlation(arch=args.arch,
                     method=args.method,
                     saliency_layer=args.saliency_layer,
                     lr=args.lr,
                     batch_size=args.batch_size,
                     plot=args.plot,
                     gpu=args.gpu)
