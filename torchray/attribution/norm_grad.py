

__all__ = ["norm_grad", "norm_grad_selective"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import saliency, get_module


def gradient_to_norm_grad_saliency(x):
    r"""Convert activation and gradient to a NormGrad saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    # Compute Frobenius norm of gradients.
    grad_weight = torch.norm(x.grad, 2, 1, keepdim=True)

    # Compute Frobenius norm of activations.
    act_weight = torch.norm(x, 2, 1, keepdim=True)

    saliency_map = grad_weight * act_weight

    return saliency_map


def gradient_to_norm_grad_proper_saliency(x_in,
                                          x_out,
                                          kernel_size=1,
                                          dilation=1,
                                          padding=0,
                                          stride=1):
    r"""Convert activation of an input tensor and gradient of an output tensor
    to a NormGrad saliency map.

    The tensor :attr:`x_out` must have a valid gradient ``x.grad``.

    Args:
        x_in (:class:`torch.Tensor`): activation tensor.
        x_out (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    # Compute Frobenius norm of gradients.
    grad_weight = torch.norm(x_out.grad, 2, 1, keepdim=True)

    # Compute Frobenius norm of unfolded activations.
    x_in_unfold = F.unfold(x_in,
                           kernel_size,
                           dilation=dilation,
                           padding=padding,
                           stride=stride)
    act_weight_shape = (x_out.shape[0], 1, x_out.shape[2], x_out.shape[3])
    act_weight = torch.norm(x_in_unfold,
                            2,
                            1,
                            keepdim=True).view(*act_weight_shape)

    saliency_map = grad_weight * act_weight

    return saliency_map


def gradient_to_norm_grad_selective_saliency(x):
    r"""Convert activation and gradient to a NormGrad selective saliency map.

    The tensor :attr:`x` must have a valid gradient ``x.grad``.
    The function then computes the saliency map :math:`s`: given by:

    Args:
        x (:class:`torch.Tensor`): activation tensor with a valid gradient.

    Returns:
        :class:`torch.Tensor`: saliency map.
    """
    return torch.norm(torch.clamp(x * x.grad, min=0), 2, 1,
                      keepdim=True)


def norm_grad_proper(*args,
                     saliency_layer,
                     use_input_output=True,
                     **kwargs):
    r"""NormGrad method without using the virtual identity trick.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the NormGrad method, and supports the
    same arguments and return values.
    """
    saliency_module = get_module(args[0], saliency_layer)
    assert saliency_module is not None, 'We could not find the saliency_layer'
    assert isinstance(saliency_module, torch.nn.Conv2d)

    # Set gradient_to_saliency function.
    grad_kwargs = {
        "kernel_size": saliency_module.kernel_size,
        "dilation": saliency_module.dilation,
        "padding": saliency_module.padding,
        "stride": saliency_module.stride,
    }
    gradient_to_saliency=lambda x_in, x_out: gradient_to_norm_grad_proper_saliency(x_in,
                                                                                   x_out,
                                                                                   **grad_kwargs)
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    use_input_output=use_input_output,
                    **kwargs)


def norm_grad(*args,
              saliency_layer,
              gradient_to_saliency=gradient_to_norm_grad_saliency,
              **kwargs):
    r"""NormGrad method using the virtual identity trick.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the NormGrad method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs,)


def norm_grad_selective(*args,
                        saliency_layer,
                        gradient_to_saliency=gradient_to_norm_grad_selective_saliency,
                        **kwargs):
    r"""NormGrad selective method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the NormGrad selective method, and supports
    the same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs,)
