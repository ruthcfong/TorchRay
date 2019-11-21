# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
Example script for evaluating the Pointing Game benchmark (see
:mod:`torch.benchmark.pointing_game`).
"""
import argparse
import os

from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader, Subset

from torchray.attribution.common import get_pointing_gradient
from torchray.attribution.deconvnet import deconvnet
from torchray.attribution.excitation_backprop import contrastive_excitation_backprop
from torchray.attribution.excitation_backprop import excitation_backprop
from torchray.attribution.excitation_backprop import update_resnet
from torchray.attribution.grad_cam import grad_cam
from torchray.attribution.gradient import gradient
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.linear_approx import linear_approx
from torchray.attribution.norm_grad import norm_grad, norm_grad_selective
from torchray.attribution.rise import rise
from torchray.attribution.weighted_saliency import weighted_saliency
from torchray.benchmark.datasets import get_dataset
from torchray.benchmark.models import get_model, get_transform
from torchray.benchmark.sanity_check import sanity_check
from torchray.benchmark.sanity_check import SanityCheckBenchmark
from torchray.benchmark.sanity_check import SIMILARITY_METRICS
from torchray.utils import imsc, get_device, xmkdir
import torchray.attribution.extremal_perturbation as elp

# v2: use weights from model finetuned on pascal
# series = 'attribution_benchmarks_v2'
series = 'attribution_benchmarks_accuracy'
series_dir = os.path.join('data', series)
log = 0
seed = 0
# chunk = None #range(1)
save = True

datasets = [
    'imagenet',
]

archs = [
    'vgg16',
    'resnet50'
]

methods = [
    # 'center',
    'contrastive_excitation_backprop',
    # 'deconvnet',
    'excitation_backprop',
    'grad_cam',
    'gradient',
    'gradient_sum',
    'guided_backprop',
    'guided_backprop_sum',
    'linear_approx',
    'norm_grad',
    'norm_grad_selective',
    # 'rise',
    # 'extremal_perturbation'
]

backprop_based = [
    'contrastive_excitation_backprop',
    'deconvnet',
    'excitation_backprop',
    'grad_cam',
    'gradient',
    'gradient_sum',
    'guided_backprop',
    'guided_backprop_sum',
    'linear_approx',
    'norm_grad',
    'norm_grad_selective',
]

perturbation_based = [
    'rise',
    'extremal_perturbation'
]

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

weights = {
    'pointing_accuracy': {
        'vgg16': {
            'contrastive_excitation_backprop': {
                'features.3': 0.75997,
                'features.8': 0.75707,
                'features.15': 0.75653,
                'features.22': 0.7774,
                'features.29': 0.82117,
             },
            'excitation_backprop': {
                'features.3': 0.73782,
                'features.8': 0.73756,
                'features.15': 0.73681,
                'features.22': 0.7532,
                'features.29': 0.77578,
             },
            'grad_cam': {
                'features.3': 0.59072,
                'features.8': 0.5835,
                'features.15': 0.55421,
                'features.22': 0.58151,
                'features.29': 0.86642,
             },
            'gradient': {
                'features.3': 0.76415,
                'features.8': 0.78116,
                'features.15': 0.80649,
                'features.22': 0.82091,
                'features.29': 0.39528,
             },
            'guided_backprop': {
                'features.3': 0.75383,
                'features.8': 0.72592,
                'features.15': 0.70552,
                'features.22': 0.73107,
                'features.29': 0.25494,
             },
            'linear_approx': {
                'features.3': 0.81499,
                'features.8': 0.83012,
                'features.15': 0.8311,
                'features.22': 0.84617,
                'features.29': 0.86486,
             },
            'norm_grad': {
                'features.3': 0.78859,
                'features.8': 0.8101,
                'features.15': 0.81867,
                'features.22': 0.81925,
                'features.29': 0.77689,
             },
            'norm_grad_selective': {
                'features.3': 0.81393,
                'features.8': 0.81372,
                'features.15': 0.81942,
                'features.22': 0.84845,
                'features.29': 0.86022,
             },
        },
        'resnet50': {
            'contrastive_excitation_backprop': {
                'layer1': 0.88073,
                'layer2': 0.89327,
                'layer3': 0.90722,
                'layer4': 0.89735,
            },
            'excitation_backprop': {
                'layer1': 0.82129,
                'layer2': 0.83304,
                'layer3': 0.84524,
                'layer4': 0.84555,
            },
            'grad_cam': {
                'layer1': 0.54451,
                'layer2': 0.52873,
                'layer3': 0.66292,
                'layer4': 0.90372,
            },
            'gradient': {
                'layer1': 0.78599,
                'layer2': 0.83924,
                'layer3': 0.82786,
                'layer4': 0.18019,
            },
            'guided_backprop': {  #
                'layer1': 0.76874,
                'layer2': 0.77696,
                'layer3': 0.82317,
                'layer4': 0.18019,
            },
            'linear_approx': {
                'layer1': 0.83281,
                'layer2': 0.85528,
                'layer3': 0.85595,
                'layer4': 0.90241,
            },
            'norm_grad': {
                'layer1': 0.81853,
                'layer2': 0.84045,
                'layer3': 0.84613,
                'layer4': 0.78765,
            },
            'norm_grad_selective': {
                'layer1': 0.8227,
                'layer2': 0.86304,
                'layer3': 0.86891,
                'layer4': 0.87439,
            },
        },
    },
    'activation': {
        'vgg16': {
            'features.3': 0.2119383764047884,
            'features.8': 0.3225606074029479,
            'features.15': 0.3907056723433652,
            'features.22': 0.0674916783223005,
            'features.29': 0.0073036655265981,
        },
        'resnet50': {
            'layer1': 0.1698746686351632,
            'layer2': 0.1950209363640008,
            'layer3': 0.1930607604361503,
            'layer4': 0.4420436345646859,
        }
    },
    'activation_in': {
        'vgg16': {
            'features.3': 0.1533106022599574,
            'features.8': 0.2196877151286012,
            'features.15': 0.3078518139348388,
            'features.22': 0.1812688130049373,
            'features.29': 0.1378810556716654,
        },
        'resnet50': {
            'layer1': 0.0876279093517559,
            'layer2': 0.0712395003421224,
            'layer3': 0.0493272299610808,
            'layer4': 0.7918053603450410,
        }
    },
    'accuracy': {
        'resnet50': {
            'layer1': 0.07792207792207792,
            'layer2': 0.15584415584415584,
            'layer3': 0.3181818181818182,
            'layer4': 0.44805194805194803,
        },
        'vgg16': {
            'features.3': 0.0326937514514653,
            'features.8': 0.1252633500153707,
            'features.15': 0.2119629074318453,
            'features.22': 0.3040917640919230,
            'features.29': 0.3259882270093957,
        }
    }
}

accumulation = [
    'sum',
    'product'
]

modules_to_run = {
    'test': {
        'vgg16': ['features.29'],
        'resnet50': ['layer4'],
    },
    'short': {
        'vgg16': [
            '',
            'features.3',
            'features.8',
            'features.15',
            'features.22',
            'features.29',
            'classifier.2',
            'classifier.4',
            'classifier.6',
        ],
        'resnet50': [
            'maxpool',
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            'fc',
        ],
    },
    'long': {
        'vgg16': [
            '',
            'features.1',
            'features.3',
            'features.6',
            'features.8',
            'features.11',
            'features.13',
            'features.15',
            'features.18',
            'features.20',
            'features.22',
            'features.25',
            'features.27',
            'features.29',
            'classifier.2',
            'classifier.4',
            'classifier.6',
        ],
        'resnet50': [
            ''
            'maxpool',
            'layer1.0',
            'layer1.1',
            'layer1.2',
            'layer2.0',
            'layer2.1',
            'layer2.2',
            'layer2.3',
            'layer3.0',
            'layer3.1',
            'layer3.2',
            'layer3.3',
            'layer3.4',
            'layer4.0'
            'layer4.1',
            'layer4.2',
            'fc',
        ],
    },
}

from torchray.attribution.linear_approx import gradient_to_linear_approx_saliency

def gradient_sum(*args, **kwargs):
    return gradient(*args, gradient_to_saliency=gradient_to_linear_approx_saliency, **kwargs)

def guided_backprop_sum(*args, **kwargs):
    return guided_backprop(*args, gradient_to_saliency=gradient_to_linear_approx_saliency, **kwargs)

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


class ProcessingError(Exception):
    def __init__(self, executor, experiment, model, image, label, class_id):
        super().__init__(f"Error processing {str(label):20s}")
        self.executor = executor
        self.experiment = experiment
        self.model = model
        self.image = image
        self.label = label
        self.class_id = class_id


def _saliency_to_point(saliency):
    assert len(saliency.shape) == 4
    w = saliency.shape[3]
    point = torch.argmax(
        saliency.view(len(saliency), -1),
        dim=1,
        keepdim=True
    )
    return torch.cat((point % w, point // w), dim=1)


class ExperimentExecutor():

    def __init__(self, experiment, chunk=None, debug=0, log=0, seed=seed, device=None):
        self.experiment = experiment
        self.device = device 
        self.model = None
        self.data = None
        self.loader = None
        self.sanity_check = None
        self.debug = debug
        self.log = log
        self.seed = seed
        self.similarity_metrics = SIMILARITY_METRICS

        if self.experiment.saliency_layer is None:
            if self.experiment.weights_strategy is None:
                if self.experiment.arch == 'vgg16':
                    if self.experiment.method in ['grad_cam', 'norm_grad', 'norm_grad_selective']:
                        self.saliency_layer = 'features.29'  # relu before pool5
                    elif 'excitation_backprop' in self.experiment.method:
                        self.saliency_layer = 'features.23'  # pool4
                    else:
                        self.saliency_layer = ''  # input
                elif self.experiment.arch == 'resnet50':
                    if self.experiment.method in ['grad_cam', 'norm_grad', 'norm_grad_selective']:
                        self.saliency_layer = 'layer4'
                    elif 'excitation_backprop' in self.experiment.method:
                        self.saliency_layer = 'layer3'  # res4a
                    else:
                        self.saliency_layer = ''  # input
                else:
                    assert False
            else:
                self.saliency_layer = None
        else:
            self.saliency_layer = self.experiment.saliency_layer

        if self.experiment.arch == 'vgg16':
            self.contrast_layer = 'classifier.4'  # relu7
        elif self.experiment.arch == 'resnet50':
            self.contrast_layer = 'avgpool'  # pool before fc layer
        else:
            assert False

        if self.experiment.dataset == 'voc_2007':
            subset = 'test'
        elif self.experiment.dataset == 'coco':
            subset = 'val2014'
        elif self.experiment.dataset == 'imagenet':
            subset = 'val'
        else:
            assert False

        # Load the model.
        input_size = (224,224)
        transform = get_transform(size=input_size,
                                  dataset=self.experiment.dataset)

        self.data = get_dataset(name=self.experiment.dataset,
                                subset=subset,
                                transform=transform,
                                download=False,
                                limiter=None)

        # Get subset of data. This is used for debugging and for
        # splitting computation on a cluster.
        if chunk is None:
            chunk = self.experiment.chunk

        if isinstance(chunk, dict):
            dataset_filter = chunk
            chunk = []
            if 'image_name' in dataset_filter:
                for i, name in enumerate(self.data.images):
                    if dataset_filter['image_name'] in name:
                        chunk.append(i)

            print(f"Filter selected {len(chunk)} image(s).")

         # Limit the chunk to the actual size of the dataset.
        if chunk is not None:
            chunk = list(set(range(len(self.data))).intersection(set(chunk)))

        # Extract the data subset.
        chunk = Subset(self.data, chunk) if chunk is not None else self.data

        # Get a data loader for the subset of data just selected.
        self.loader = DataLoader(chunk,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=self.data.collate)

        self.sanity_check = SanityCheckBenchmark(self.experiment.modules_to_run,
                                                 similarity_metrics=self.similarity_metrics)

        self.data_iterator = iter(self.loader)

    def _lazy_init(self):
        if self.device is not None and isinstance(self.device, torch.device):
            return

        if self.log:
            from torchray.benchmark.logging import mongo_connect
            self.db = mongo_connect(self.experiment.series)

        if self.device is not None:
            assert isinstance(self.device, int)
            self.device = torch.device(f'cuda:{self.device}')
        else:
            self.device = get_device()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.model = get_model(
            arch=self.experiment.arch,
            dataset=self.experiment.dataset,
            convert_to_fully_convolutional=False,
        )

        # Some methods require patching the models further for
        # optimal performance.
        if self.experiment.arch == 'resnet50':
            if self.experiment.method in perturbation_based:
                pass
            elif self.experiment.method in backprop_based:
                # Patch all back-prop based methods.
                self.model.avgpool = torch.nn.AvgPool2d((7, 7), stride=1)
            else:
                assert False

            if 'excitation_backprop' in self.experiment.method:
                # Replace skip connection with EltwiseSum.
                self.model = update_resnet(self.model, debug=True)

        # Change model to eval modself.
        self.model.eval()

        # Move model to device.
        self.model.to(self.device)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader.dataset)

    def __next__(self):
        self._lazy_init()
        x, y = next(self.data_iterator)
        torch.manual_seed(self.seed)

        if self.log:
            from torchray.benchmark.logging import mongo_load, mongo_save, \
                data_from_mongo, data_to_mongo

        try:
            assert len(x) == 1
            x = x.to(self.device)
            if self.experiment.dataset == "imagenet":
                class_ids = y
            else:
                class_ids = self.data.as_class_ids(y[0])

            results = {}

            for class_id in class_ids:
                results[class_id] = {}
                info = {}

                # Try to recover this result from the log.
                if self.log > 0:
                    image_name = self.data.as_image_name(y[0])
                    data = mongo_load(
                        self.db,
                        self.experiment.name,
                        f"{image_name}-{class_id}",
                    )
                    if data is not None:
                        data = data_from_mongo(data)
                        # TODO(ruthfong): update.
                        results['pointing'][class_id] = data['pointing']
                        results['pointing_difficult'][class_id] = data['pointing_difficult']
                        if self.debug:
                            print(f'{image_name}-{class_id} loaded from log')
                        continue

                if x.grad is not None:
                    x.grad.data.zero_()

                if self.experiment.method in backprop_based:
                    saliency_func = saliency_funcs[self.experiment.method]
                    args = [self.model, x, class_id]
                    kwargs = {
                        'saliency_layer': self.saliency_layer,
                        'resize': True,
                        'get_backward_gradient': get_pointing_gradient,
                    }
                    if self.experiment.method == 'contrastive_excitation_backprop':
                        kwargs['contrast_layer'] = self.contrast_layer

                    res = sanity_check(x,
                                       class_id,
                                       saliency_func,
                                       arch=self.experiment.arch,
                                       randomization_type=self.experiment.randomization_type,
                                       should_update_resnet=(self.experiment.arch == "resnet50"
                                                             and "excitation_backprop" in self.experiment.method),
                                       modules_to_run=self.experiment.modules_to_run,
                                       **kwargs,
                                       )

                    results[class_id]['similarity_stats'] = res['similarity_stats']
                    results[class_id]['modules_to_run'] = res['modules_to_run']
                    info['saliency'] = res['saliency_maps'].cpu()

                    if False:
                        if self.experiment.weights_strategy is None:
                            kwargs['smooth'] = 0.02 if self.saliency_layer == '' else 0.0
                            saliency = saliency_func(
                                *args,
                                saliency_layer=self.saliency_layer,
                                **kwargs,
                            )
                        else:
                            if self.experiment.weights_strategy == 'pointing_accuracy':
                                lw = weights[self.experiment.weights_strategy][self.experiment.arch][self.experiment.method]
                            elif self.experiment.weights_strategy in ['accuracy', 'activation']:
                                lw = weights[self.experiment.weights_strategy][self.experiment.arch]
                            else:
                                lw = weights['activation'][self.experiment.arch]

                            saliency = weighted_saliency(
                                saliency_func,
                                lw,
                                *args,
                                weights_strategy=self.experiment.weights_strategy,
                                accumulation=self.experiment.accumulation,
                                **kwargs,
                            )

                elif False and self.experiment.method in perturbation_based:
                    assert self.weighting_strategy is None

                    if self.experiment.method == "rise":
                        # For RISE, compute saliency map for all classes.
                        if rise_saliency is None:
                            rise_saliency = rise(self.model, x, resize=image_size, seed=self.seed)
                        saliency = rise_saliency[:, class_id, :, :].unsqueeze(1)
                        point = _saliency_to_point(saliency)
                        info['saliency'] = saliency.cpu()

                    elif self.experiment.method == "extremal_perturbation":

                        if self.experiment.dataset == 'voc_2007':
                            areas = [0.025, 0.05, 0.1, 0.2]
                        else:
                            areas = [0.018, 0.025, 0.05, 0.1]

                        if self.experiment.boom:
                            raise RuntimeError("BOOM!")

                        mask, energy = elp.extremal_perturbation(
                            self.model, x, class_id,
                            areas=areas,
                            num_levels=8,
                            step=7,
                            sigma=7 * 3,
                            max_iter=800,
                            debug=self.debug > 0,
                            jitter=True,
                            smooth=0.09,
                            resize=image_size,
                            perturbation='blur',
                            reward_func=elp.simple_reward,
                            variant=elp.PRESERVE_VARIANT,
                        )

                        saliency = mask.sum(dim=0, keepdim=True)
                        point = _saliency_to_point(saliency)

                        info = {
                            'saliency': saliency.cpu(),
                            'mask': mask,
                            'areas': areas,
                            'energy': energy
                        }
                    else:
                        assert False
                else:
                    assert False

                if False:
                    plt.figure()
                    plt.subplot(1, 2, 1)
                    imsc(saliency[0])
                    plt.plot(point[0, 0], point[0, 1], 'ro')
                    plt.subplot(1, 2, 2)
                    imsc(x[0])
                    plt.pause(0)

                if self.log > 0:
                    image_name = self.data.as_image_name(y[0])
                    mongo_save(
                        self.db,
                        self.experiment.name,
                        f"{image_name}-{class_id}",
                        data_to_mongo({
                            'image_name': image_name,
                            'class_id': class_id,
                            'similarity_stats': results[class_id]['similarity_stats'],
                        })
                    )

                if self.log > 1:
                    mongo_save(
                        self.db,
                        str(self.experiment.name) + "-details",
                        f"{image_name}-{class_id}",
                        data_to_mongo(info)
                    )

            return results

        except Exception as ex:
            raise ProcessingError(
                self, self.experiment, self.model, x, y, class_id) from ex

    def aggregate(self, results):
        for class_id, res in results.items():
            self.sanity_check.aggregate(res)

    def run(self, save=save):
        all_results = []
        for itr, results in enumerate(self):
            all_results.append(results)
            self.aggregate(results)
            if itr % max(len(self) // 20, 1) == 0 or itr == len(self) - 1:
                print("[{}/{}]".format(itr + 1, len(self)))
                print(self)

        print("[final result]")
        print(self)

        self.experiment.pointing = self.pointing.accuracy
        self.experiment.pointing_difficult = self.pointing_difficult.accuracy
        if save:
            self.experiment.save()

        return all_results

    def __str__(self):
        return (
            f"{self.experiment.method} "
            f"{self.saliency_layer} "
            #f"{self.experiment.weights_strategy} "
            #f"{self.experiment.accumulation} "
            f"{self.experiment.arch} "
            f"{self.experiment.dataset}\n"
            f"{self.sanity_check}\n"
        )


class Experiment():
    def __init__(self,
                 series,
                 method,
                 arch,
                 dataset,
                 modules_to_run,
                 randomization_type='cascade',
                 saliency_layer=None,
                 # weights_strategy=None,
                 # accumulation=None,
                 root='',
                 chunk=None,
                 boom=False):
        self.series = series
        self.root = root
        self.method = method
        self.arch = arch
        self.dataset = dataset
        self.chunk = chunk
        self.boom = boom
        self.modules_to_run = modules_to_run
        self.randomization_type = randomization_type
        self.saliency_layer = saliency_layer
        if False:
            self.weights_strategy = weights_strategy
            self.accumulation = accumulation
            if self.weights_strategy is not None:
                assert self.accumulation is not None
                assert self.saliency_layer is None
        self.sanity_check = SanityCheckBenchmark(modules_to_run=self.modules_to_run)

    def __str__(self):
        # TODO(ruthfong): Update.
        if True: # or self.weights_strategy is None:
            return (
                f"{self.method},{self.saliency_layer},{self.randomization_type},"
                f"{self.arch},{self.dataset}\n"
                f"{self.sanity_check}"
            )
        else:
            return (
                f"{self.method},{self.weights_strategy},{self.accumulation},"
                f"{self.arch},{self.dataset}\n"
                f"{self.sanity_check}"
            )


    @property
    def name(self):
        # TODO(ruthfong): Update.
        return f"{self.method}-{self.saliency_layer}-{self.arch}-{self.dataset}"

    @property
    def path(self):
        return os.path.join(self.root, self.name + ".csv")

    def save(self):
        with open(self.path, "w") as f:
            f.write(self.__str__() + "\n")

    def load(self):
        # TODO(ruthfong): Update.
        with open(self.path, "r") as f:
            data = f.read()
        if len(data.split(",")) == 5:
            method, arch, dataset, pointing, pointing_difficult = data.split(",")
        elif len(data.split(",")) == 6:
            method, saliency_layer, arch, dataset, pointing, pointing_difficult = data.split(",")
            assert self.saliency_layer == saliency_layer
        elif len(data.split(",")) == 7:
            method, weights_strategy, accumulation, arch, dataset, pointing, pointing_difficult = data.split(",")
            assert self.weights_strategy == weights_strategy
            assert self.accumulation == accumulation

        assert self.method == method
        assert self.arch == arch
        assert self.dataset == dataset
        self.pointing = float(pointing)
        self.pointing_difficult = float(pointing_difficult)

    def done(self):
        return os.path.exists(self.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--datasets', nargs='*', default=datasets)
    parser.add_argument('--archs', nargs='*', default=archs)
    parser.add_argument('--methods', nargs='*', default=methods)
    parser.add_argument('--module_list', choices=['long', 'short', 'test'], default='short')
    parser.add_argument('--randomization_type',
                        choices=['cascade', 'independent'],
                        default='cascade')
    parser.add_argument('--chunk', type=int, default=None)
    # parser.add_argument('--weights_strategy', default=None)
    parser.add_argument('--layers', nargs='*', default=None)
    # parser.add_argument('--accumulation', nargs='*', default=accumulation)
    args = parser.parse_args()

    experiments = []
    xmkdir(series_dir)

    if args.chunk is None:
        chunk = None
    else:
        chunk = range(args.chunk)
    provided_layers = args.layers is not None
    for d in args.datasets:
        for a in args.archs:
            for m in args.methods:
                if not provided_layers:
                    args.layers = layers[a]
                else:
                    assert len(args.archs) == 1
                for l in args.layers:
                    experiments.append(
                        Experiment(series=series,
                                   method=m,
                                   arch=a,
                                   dataset=d,
                                   modules_to_run=modules_to_run[args.module_list][a],
                                   randomization_type=args.randomization_type,
                                   saliency_layer=l,
                                   # weights_strategy=None,
                                   # accumulation=None,
                                   chunk=chunk,
                                   root=series_dir))

    for e in experiments:
        if e.done():
            e.load()
            continue
        ExperimentExecutor(e, log=log, device=args.device).run()