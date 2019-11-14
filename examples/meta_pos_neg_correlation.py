import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

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


def get_subset(dset, num_per_class=5):
    idx = {}
    for i, (_, class_id) in enumerate(dset.samples):
        if class_id not in idx:
            idx[class_id] = [i]
        else:
            idx[class_id].append(i)

    subset_indices = []
    for class_id in idx.keys():
        subset_indices.extend(idx[class_id][:num_per_class])

    subset_indices = sorted(subset_indices)

    return Subset(dset, subset_indices)


def meta_correlation(arch="resnet50",
                     method="gradient",
                     saliency_layer="",
                     lr=0.0025,
                     batch_size=1,
                     num_per_class=None,
                     num_workers=4,
                     plot=False,
                     verbose=False,
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
    if num_per_class is not None:
        dset = get_subset(dset, num_per_class=num_per_class)
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

    corrs = []
    scores = []
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        pos_saliency_map = meta_saliency(saliency_func,
                                          model,
                                          x,
                                          y,
                                          lr=lr,
                                          **kwargs)
        neg_saliency_map = meta_saliency(saliency_func,
                                          model,
                                          x,
                                          y,
                                          lr=lr,
                                          negate_loss=True,
                                          **kwargs)
        out = softmax(model(x))
        score = torch.gather(out, 1, y.unsqueeze(1)).cpu().data.numpy().squeeze()

        if plot:
            plt.axes(ax[0])
            imarraysc(x)
            plt.axes(ax[1])
            imarraysc(pos_saliency_map)
            plt.axes(ax[2])
            imarraysc(neg_saliency_map)
            plt.axes(ax[3])
            imarraysc(torch.abs(pos_saliency_map - neg_saliency_map))
            plt.pause(1)

        corr, _ = compute_similarity(pos_saliency_map, neg_saliency_map)
        if plot or verbose:
            print(corr)

        if batch_size == 1:
            corrs.append(corr)
            scores.append(float(score))
        else:
            corrs.extend(list(corr))
            scores.extend(list(score))

    out_dir = "./data/meta_correlation_pos_neg"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    prefix = f"{arch}_{method}_{saliency_layer}_lr_{lr:.4f}_bs_{batch_size}_nc_{num_per_class}"
    np.savetxt(os.path.join(out_dir, f"{prefix}_pos_neg_corrs.txt"),
               np.array(corrs),
               delimiter="\n",
               fmt="%.4f")

    np.savetxt(os.path.join(out_dir, f"{prefix}_order_scores.txt"),
               np.array(scores),
               delimiter="\n",
               fmt="%.4f")

    res = ",".join([arch,
                    method,
                    saliency_layer,
                    str(lr),
                    str(len(corrs)),
                    str(np.mean(corrs)),
                    str(np.std(corrs)),
                    str(np.mean(scores)),
                    str(np.std(scores))])

    with open(os.path.join(out_dir, f"{prefix}_order_corrs_summary.csv"), "w") as f:
        f.write(res + "\n")

    print(res)

    if num_per_class is not None:
        samples = np.array(dset.dataset.samples)[dset.indices]
    else:
        samples = np.array(dset.samples)
    np.savetxt(os.path.join(out_dir, f"{prefix}_pos_neg_corrs_imdb.txt"),
               samples,
               delimiter="\n",
               fmt="%s,%s")


if __name__ == '__main__':
    archs = ["vgg16", "resnet50"]

    layers = {
        'vgg16':
            ['features.29',
             'features.22',
             'features.15',
             'features.8',
             'features.3',
             ''],
        'resnet50':
            ['layer4',
             'layer3',
             'layer2',
             'layer1',
             ''
             ],
    }

    methods = [
        'norm_grad',
        'norm_grad_selective',
        'gradient',
        'gradient_sum',
        'linear_approx',
        'contrastive_excitation_backprop',
        'deconvnet',
        'excitation_backprop',
        'grad_cam',
        'guided_backprop',
        'guided_backprop_sum',
    ]

    lrs = [
        1e-2,
        5e-3,
        1e-3,
        5e-4,
        1e-4,
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--method", choices=saliency_funcs.keys(), default="norm_grad")
    parser.add_argument("--saliency_layer", type=str, default="layer4")
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--plot', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_per_class", type=int, default=None)

    args = parser.parse_args()

    if False:
        for a in archs:
            for m in methods:
                for l in layers:
                    for lr in lrs:
                        meta_correlation(arch=args.arch,
                                         method=args.method,
                                         saliency_layer=args.saliency_layer,
                                         lr=args.lr,
                                         batch_size=args.batch_size,
                                         num_per_class=args.num_per_class,
                                         plot=args.plot,
                                         verbose=args.verbose,
                                         num_workers=args.workers,
                                         gpu=args.gpu)
    else:
        meta_correlation(arch=args.arch,
                         method=args.method,
                         saliency_layer=args.saliency_layer,
                         lr=args.lr,
                         batch_size=args.batch_size,
                         num_per_class=args.num_per_class,
                         plot=args.plot,
                         verbose=args.verbose,
                         num_workers=args.workers,
                         gpu=args.gpu)
