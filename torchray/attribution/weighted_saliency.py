import numpy as np
import torch


def normalize(x):
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def weighted_saliency(saliency_func,
                      layer_weights,
                      *args,
                      accumulation="sum",
                      **kwargs):
    if not isinstance(layer_weights, dict):
        raise TypeError(f"layer_weights not a dict; it's a {type(layer_weights)}")

    norm_term = np.sum(list(layer_weights.values()))

    cum_saliency_map = 0
    for saliency_layer, layer_weight in layer_weights.items():
        saliency_map = saliency_func(*args,
                                     saliency_layer=saliency_layer,
                                     resize=True,
                                     **kwargs)
        norm_saliency_map = normalize(saliency_map)
        norm_layer_weight = layer_weight / norm_term
        if accumulation == "sum":
            cum_saliency_map += norm_layer_weight * norm_saliency_map
        elif accumulation == "product":
            cum_saliency_map *= norm_saliency_map ** norm_layer_weight
        else:
            raise NotImplementedError(f"accumulation should be ['sum', "
                                      f"'product']. Instead, it's "
                                      f"{accumulation}.")

    return cum_saliency_map
