import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

from torchray.attribution.norm_grad import norm_grad, norm_grad_proper

from torchray.benchmark.datasets import get_dataset
from torchray.benchmark.models import get_model, get_transform
from torchray.benchmark.common import compute_similarity

from torchray.utils import imsc, imarraysc, accuracy

from tqdm import tqdm


dataset = 'imagenet'
subset = 'val'

def correlate(arch="resnet50",
              saliency_layer="",
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

    # Set optional arguments.
    kwargs = {
        "resize": True,
    }

    if plot:
        f, ax = plt.subplots(1, 4, figsize=(4*4, 4))

    corrs = []
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        saliency_approx = norm_grad(model,
                                    x,
                                    y,
                                    saliency_layer=saliency_layer,
                                    **kwargs)
        saliency_proper = norm_grad_proper(model,
                                           x,
                                           y,
                                           saliency_layer=saliency_layer,
                                           **kwargs)

        if plot:
            plt.axes(ax[0])
            imarraysc(x)
            plt.axes(ax[1])
            imarraysc(saliency_approx)
            plt.axes(ax[2])
            imarraysc(saliency_proper)
            plt.axes(ax[3])
            imarraysc(torch.abs(saliency_approx - saliency_proper))
            plt.pause(1)

        corr, _ = compute_similarity(saliency_approx, saliency_proper)
        if plot:
            print(corr)
        corrs.extend(list(corr))

    out_dir = "./data/norm_grad_virtual_identity_correlation"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    corrs = np.array(corrs)
    prefix = f"{arch}_{saliency_layer}"
    np.savetxt(os.path.join(out_dir, f"{prefix}_corrs.txt"),
               corrs,
               delimiter="\n",
               fmt="%.4f")

    res = [arch, saliency_layer, str(np.mean(corrs)), str(np.std(corrs)), str(len(corrs))]
    np.savetxt(os.path.join(out_dir, f"{prefix}_corrs_summary.csv"),
               res,
               delimiter=",",
               fmt="%s")
    print(",".join(res))



if __name__ == '__main__':
    archs = ["vgg16", "resnet50"]

    layers = {
        'vgg16':
            ['features.28',
             'features.21',
             'features.14',
             'features.7',
             'features.2',
             ],
        'resnet50':
            ['layer4.0.conv1',
             'layer3.0.conv1',
             'layer2.0.conv1',
             'layer1.0.conv1',
             ],
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--saliency_layer", type=str, default="layer4.0.conv1")
    parser.add_argument("--batch_size", type=int, default=9)
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=None)

    args = parser.parse_args()

    if False:
        for a in archs:
            for l in layers[a]:
                correlate(arch=args.arch,
                          saliency_layer=args.saliency_layer,
                          batch_size=args.batch_size,
                          plot=args.plot,
                          num_workers=args.workers,
                          gpu=args.gpu)
    else:
        correlate(arch=args.arch,
                  saliency_layer=args.saliency_layer,
                  batch_size=args.batch_size,
                  plot=args.plot,
                  num_workers=args.workers,
                  gpu=args.gpu)
