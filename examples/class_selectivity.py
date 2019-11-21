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


CHECKPOINT_DIR = "/scratch/local/ssd/ruthfong/models/resnet50_checkpoints"
dataset = 'imagenet'

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


def get_subset(dset, num_per_class=1):
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
                     subset="val",
                     epoch=None,
                     method="norm_grad",
                     use_meta=False,
                     saliency_layer="layer4",
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

    # Load weights from checkpoint.
    if epoch is not None:
        assert arch == "resnet50"
        model = nn.DataParallel(model)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch:02d}.pth.tar")
        print(f"Loading from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict["state_dict"])
        model = model.module

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

    if method == "contrastive_excitation_backprop":
        if arch == 'vgg16':
            contrast_layer = 'classifier.4'  # relu7
        elif self.experiment.arch == 'resnet50':
            contrast_layer = 'avgpool'  # pool before fc layer
        else:
            assert False
        kwargs["contrast_layer"] = contrast_layer

    if plot:
        f, ax = plt.subplots(1, 4, figsize=(4*4, 4))

    corrs = []
    scores = []
    for i, (x, y) in enumerate(tqdm(loader)):
        x = x.to(device)
        y = y.to(device)
        out = softmax(model(x))

        min_class_idx = out.argmin(1)
        max_class_idx = out.argmax(1)
        if use_meta:
            saliency_map_min = meta_saliency(saliency_func,
                                             model,
                                             x,
                                             min_class_idx,
                                             lr=lr,
                                             **kwargs)
            saliency_map_max = meta_saliency(saliency_func,
                                             model,
                                             x,
                                             max_class_idx,
                                             lr=lr,
                                             **kwargs)
        else:
            saliency_map_min = saliency_func(model,
                                             x,
                                             min_class_idx,
                                             **kwargs)
            saliency_map_max = saliency_func(model,
                                             x,
                                             max_class_idx,
                                             **kwargs)
        score = torch.gather(out, 1, y.unsqueeze(1)).cpu().data.numpy().squeeze()

        if plot:
            plt.axes(ax[0])
            imarraysc(x)
            plt.axes(ax[1])
            imarraysc(saliency_map)
            plt.axes(ax[2])
            imarraysc(saliency_order_0)
            plt.axes(ax[3])
            imarraysc(torch.abs(saliency_map_min - saliency_min_max))
            plt.pause(1)

        corr, _ = compute_similarity(saliency_map_min, saliency_map_max)

        if plot or verbose:
            print(corr)

        if batch_size == 1:
            corrs.append(corr)
            scores.append(float(score))
        else:
            corrs.extend(list(corr))
            scores.extend(list(score))

    out_dir = "./data/class_selectivity"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    prefix = f"{arch}_{method}_meta_{use_meta}_{saliency_layer}_{subset}_e_{epoch}_lr_{lr:.4f}_bs_{batch_size}_nc_{num_per_class}"

    np.savetxt(os.path.join(out_dir, f"{prefix}_corrs.txt"),
               np.array(corrs),
               delimiter="\n",
               fmt="%.4f")

    np.savetxt(os.path.join(out_dir, f"{prefix}_scores.txt"),
               np.array(scores),
               delimiter="\n",
               fmt="%.4f")

    corrs = np.array(corrs)

    res = ",".join([arch,
                    method,
                    str(use_meta),
                    saliency_layer,
                    subset,
                    str(epoch),
                    str(lr),
                    str(len(corrs)),
                    str(np.mean(corrs[np.isfinite(corrs)])),
                    str(np.std(corrs[np.isfinite(corrs)])),
                    str(np.mean(scores)),
                    str(np.std(scores))])

    with open(os.path.join(out_dir, f"{prefix}_summary.csv"), "w") as f:
        f.write(res + "\n")

    print(res)

    if num_per_class is not None:
        samples = np.array(dset.dataset.samples)[dset.indices]
    else:
        samples = np.array(dset.samples)

    np.savetxt(os.path.join(out_dir, f"{prefix}_imdb.txt"),
               samples,
               delimiter="\n",
               fmt="%s,%s")


if __name__ == '__main__':
    archs = ["vgg16", "resnet50"]
    arches = ["vgg16"]

    layers = {
        'vgg16':
            ['',
             #'features.3',
             #'features.8',
             #'features.15',
             'features.22',
             'features.29'],
        'resnet50':
            ['',
             'layer1',
             'layer2',
             'layer3',
             'layer4',
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

    methods = ["grad_cam"]

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
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--subset",
                        choices=["train", "val"],
                        type=str,
                        default="val")
    parser.add_argument("--use_meta", action="store_true", default=False)

    args = parser.parse_args()

    if True:
        for a in archs:
            for l in layers[a]:
                for m in methods:
                    meta_correlation(arch=a,
                                     method=m,
                                     saliency_layer=l,
                                     lr=args.lr,
                                     batch_size=args.batch_size,
                                     num_per_class=args.num_per_class,
                                     plot=args.plot,
                                     verbose=args.verbose,
                                     num_workers=args.workers,
                                     epoch=args.epoch,
                                     subset=args.subset,
                                     use_meta=args.use_meta,
                                     gpu=args.gpu)
    elif False:
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
                         epoch=args.epoch,
                         subset=args.subset,
                         use_meta=args.use_meta,
                         gpu=args.gpu)
