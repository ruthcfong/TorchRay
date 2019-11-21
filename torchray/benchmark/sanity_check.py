r"""
This module implements the *sanity check* model parameter randomization test
of [SANITY]_ for checking if a saliency method is sensitive to model
parameters. The simplest interface is given by the :func:`sanity_check`
function:

.. literalinclude:: ../examples/sanity_check.py
    :language: python
    :linenos:

Warning:
    :func:`sanity_check` only randomizes torch.nn.Module objects with
    learnable weights (i.e., module has weight and bias attributes).

References:

    .. [SANITY] Zeiler and Fergus,
                *Sanity Checks for Saliency Maps*,
                NeurIPS 2018.
"""
import inspect
from inspect import signature

import numpy as np
from scipy.stats import spearmanr

import torch

from torchvision import models

from torchray.attribution.common import get_module
from torchray.attribution.excitation_backprop import update_resnet
from torchray.utils import get_device, normalize, MomentsMeter

__all__ = ["module_types_with_weights", "sanity_check"]


SPEARMAN_NO_ABS = "spearman-no_abs"
SPEARMAN_ABS = "spearman_abs"


SIMILARITY_METRICS = [
    SPEARMAN_NO_ABS,
    SPEARMAN_ABS,
]


def get_modules_with_weights(model):
    """Return list of module names that have learnable weights.

    Args:
        model (:class:`torch.nn.Module`): model in which to search for
            module names.

    Returns:
        list: list of module names that have learnable weights.
    """
    modules_with_weights = []
    for module_name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module, "bias"):
            modules_with_weights.append(module_name)
    return modules_with_weights


def partially_randomize_model(model,
                              randomized_model,
                              randomize_blob,
                              randomization_type="cascade",
                              verbose=False):
    """Partially randomize the weights of a model using either cascading or
       independent randomization (i.e., cascading randomization randomizes in
       reverse all learnable weights of a model to :attr:`randomize_blob`
       inclusive; independent randomization only randomizes the
       learnable weights in :attr:`randomize_blob`).

    Args:
        model (:class:`torch.nn.Module`): pre-trained model.
        randomized_model (:class:`torch.nn.Module`): randomly initialized
            model with same architecture as model.
        randomize_blob (str): name of module to randomize to.
        randomization_type (str): type of randomization to use;
            choose from ["cascade", "independent"].
        verbose (bool): If True, print layers that are randomized.

    Warning:
        :func:`partially_randomize_model` changes :attr:`model`.

    Returns:
        :class:`torch.nn.Module`: model with partially randomized weights.
    """
    assert type(model) == type(randomized_model)
    assert randomization_type in ["cascade", "independent"]
    assert randomize_blob is not None

    if randomization_type == "cascade":
        modules_gen = model.named_modules()
    elif randomization_type == "independent":
        randomize_module = get_module(model, randomize_blob)
        assert randomize_module is not None
        modules_gen = randomize_module.named_modules()
    else:
        assert False

    def get_composite_name(module_name,
                           randomize_blob,
                           randomization_type):
        if randomization_type == "cascade":
            return module_name
        elif randomization_type == "independent":
            if module_name == "":
                return randomize_blob
            return "{}.{}".format(randomize_blob, module_name)
        else:
            assert False

    for module_name, module in reversed(list(modules_gen)):
        if hasattr(module, "weight") and hasattr(module, "bias"):
            composite_name = get_composite_name(module_name,
                                                randomize_blob,
                                                randomization_type)
            if verbose:
                print("Randomizing {} ({})".format(composite_name,
                                                   type(module)))
            assert hasattr(module, "weight")
            assert hasattr(module, "bias")
            random_module = get_module(randomized_model, composite_name)
            try:
                assert random_module is not None
            except:
                 import pdb; pdb.set_trace();
            if module.weight is not None:
                module.weight.data[...] = random_module.weight.data[...]
            if module.bias is not None:
                module.bias.data[...] = random_module.bias.data[...]
        if module_name == randomize_blob:
            if verbose:
                print("Stopping at {}".format(composite_name,
                                              type(module)))
            break

    return model


def get_model(arch="vgg16",
              randomize_blob=None,
              randomization_type="cascade",
              should_update_resnet=False):
    """Returns model that has been initialized with partial randomization
       to :attr:`randomize_blob` (inclusive).

    Args:
        arch (str): name of architecture.
        randomize_blob (str): name of module to randomize to.
        randomization_type (str): type of randomization to use;
            choose from ["cascade", "independent"].
        should_update_resnet (bool): If True and arch == "resnet50", replace
            skip connections with EltwiseSum module.

    Returns:
        :class:`torch.nn.Module`: model that has been initialized with
            cascading randomization to :attr:`randomize_blob`.
    """
    # Get pre-trained model.
    model = models.__dict__[arch](pretrained=True)

    if randomize_blob is not None:
        # Get randomly initialized model.
        randomized_model = models.__dict__[arch](pretrained=False)

        # Get partially randomized model.
        model = partially_randomize_model(model,
                                          randomized_model,
                                          randomize_blob,
                                          randomization_type=randomization_type)

    # Update resnet model if necessary.
    if should_update_resnet and arch == "resnet50":
        model = update_resnet(model, debug=True)

    model.eval()

    return model


def sanity_check(x,
                 y,
                 saliency_func,
                 arch="vgg16",
                 randomization_type="cascade",
                 should_update_resnet=False,
                 modules_to_run=None,
                 verbose=False,
                 **kwargs):
    """Run model parameter randomization test.

    Args:
        x (:class:`torch.Tensor`): input tensor.
        y (int): target class.
        saliency_func (function): saliency method function with parameters
            structured as follows: :func:`saliency_func(model, x, y)`.
        arch (str): name of architecture.
        randomization_type (str): type of randomization to use;
            choose from ["cascade", "independent"] (default: "cascade").
        should_update_resnet (bool): If True and arch == "resnet50", replace
            skip connections with EltwiseSum module (default: False).
        modules_to_run (list): list of strs with names of modules
            at which to randomize (default: None).
        verbose (bool): If True, print layers that are randomized
            (default: False).

    Returns:
        tuple of :class:`torch.Tensor` and list: 4D tensor with saliency maps
            computed by randomizing every learnable modules and list of
            learnable layer names.
    """
    # TODO(ruthfong): Add batch support.
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 1
    assert isinstance(y, int)

    device = get_device()

    if modules_to_run is None:
        model = get_model(arch,
                          should_update_resnet=should_update_resnet)
        modules_with_weights = get_modules_with_weights(model)
        modules_with_weights.append(None)
        modules_with_weights = modules_with_weights[::-1]
        modules_to_run = modules_with_weights
    else:
        modules_to_run.insert(0, None)

    cum_saliency = []
    for i, randomize_blob in enumerate(modules_to_run):
        if verbose:
            print("[{}/{}] {}".format(i+1,
                                      len(modules_to_run),
                                      randomize_blob))
        model = get_model(arch,
                          should_update_resnet=should_update_resnet)

        randomized_model = get_model(arch,
                                     randomize_blob="",
                                     randomization_type="cascade",
                                     should_update_resnet=should_update_resnet)
        if randomize_blob is None:
            partial_model = model
        else:
            partial_model = partially_randomize_model(model,
                                                      randomized_model,
                                                      randomize_blob,
                                                      randomization_type=randomization_type,
                                                      verbose=verbose)
        partial_model.to(device)
        x = x.to(device)

        if randomize_blob is not None:
            category_id = np.argmax(partial_model(x).cpu().data.numpy()[0])
        else:
            category_id = y

        saliency = saliency_func(partial_model, x, category_id, **kwargs)
        norm_saliency = normalize(saliency, a_min=-1, a_max=1)

        cum_saliency.append(norm_saliency)

    similarity_stats = {sim_metric:[] for sim_metric in SIMILARITY_METRICS}

    bsl_saliency = cum_saliency[0]
    for i, curr_saliency in enumerate(cum_saliency[1:]):
        for similarity_metric in similarity_stats.keys():
            if "spearman" in similarity_metric:
                if "no_abs" in similarity_metric:
                    _, p = spearmanr(curr_saliency.reshape(-1).cpu().data.numpy(),
                                     bsl_saliency.reshape(-1).cpu().data.numpy())
                else:
                    _, p = spearmanr(torch.abs(curr_saliency.reshape(-1)).cpu().data.numpy(),
                                     torch.abs(bsl_saliency.reshape(-1)).cpu().data.numpy())
                import pdb; pdb.set_trace();
            # TODO(ruthfong): add other similarity metrics.
            else:
                assert False
            similarity_stats[similarity_metric].append(p)
    cum_saliency = torch.cat(cum_saliency)

    return {
        "similarity_stats": similarity_stats,
        "saliency_maps": cum_saliency,
        "modules_to_run": modules_to_run[1:],
    }


class SanityCheckBenchmark:
    def __init__(self,
                 modules_to_run,
                 similarity_metrics=SIMILARITY_METRICS):
        self.modules_to_run = modules_to_run
        self.similarity_metrics = similarity_metrics
        self.running_metrics = {}
        for sim_metric in self.similarity_metrics:
            self.running_metrics[sim_metric] = {m: MomentsMeter() for m in
                                                self.modules_to_run}

    def aggregate(self, results):
        assert results["modules_to_run"] == self.modules_to_run
        assert(set(results["similarity_stats"].keys())
               == set(self.similarity_metrics))
        for sim_metric in self.similarity_metrics:
            for i, m in enumerate(self.modules_to_run):
                res = results["similarity_stats"][sim_metric][i]
                self.running_metrics[sim_metric][m].update(res)

    @property
    def metrics(self):
        return self.running_metrics

    def __str__(self):
        out = "\t" + "\t".join(self.modules_to_run)
        for sim_metric in self.similarity_metrics:
            res = [str(self.running_metrics[sim_metric][m]) for m in
                   self.modules_to_run]
            out += f"\n{sim_metric}\t" + "\t".join(res)
        return out


