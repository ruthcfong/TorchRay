import numpy as np
import torch


def competitive_saliency(saliency_func,
                         model,
                         x,
                         target,
                         num_classes=None,
                         batch_size=100,
                         **kwargs):
    """Implementation of this paper:
    https://arxiv.org/pdf/1905.12152.pdf
    """

    if num_classes is None:
        y = model(x)
        assert len(y.shape) == 2 or len(y.shape) == 4
        num_classes = y.shape[1]

    target_saliency_map = saliency_func(model, x, target, **kwargs)

    assert len(x.shape) == 4
    assert x.shape[0] == 1
    _, nc, h, w = x.shape

    max_saliency_maps = None
    min_saliency_maps = None
    for start_class in range(0, num_classes, batch_size):
        end_class = min(num_classes, start_class + batch_size)
        all_target = np.array(list(range(start_class, end_class)))

        all_x = x.expand(end_class - start_class, nc, h, w)
        saliency_maps = saliency_func(model, all_x, target=all_target, **kwargs)
        if max_saliency_maps is None:
            max_saliency_maps = torch.max(saliency_maps, 0)[0].unsqueeze(0)
        else:
            max_saliency_maps = torch.max(max_saliency_maps,
                                          torch.max(saliency_maps, 0)[0].unsqueeze(0))
        if min_saliency_maps is None:
            min_saliency_maps = torch.min(saliency_maps, 0)[0].unsqueeze(0)
        else:
            min_saliency_maps = torch.min(min_saliency_maps,
                                          torch.min(saliency_maps, 0)[0].unsqueeze(0))


    zeros = torch.zeros_like(target_saliency_map)
    is_pos = target_saliency_map > 0.

    is_max = target_saliency_map > max_saliency_maps
    pos_saliency_map = torch.where(is_max, target_saliency_map, zeros)

    is_min = target_saliency_map < min_saliency_maps
    neg_saliency_map = torch.where(is_min, target_saliency_map, zeros)

    final_saliency_map = torch.where(is_pos, pos_saliency_map, neg_saliency_map)

    return final_saliency_map


