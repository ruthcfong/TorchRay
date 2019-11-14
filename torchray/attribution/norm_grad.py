# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This module provides an implementation of the *NormGrad* method of [NORMGRAD]_
for saliency visualization. The simplest interface is given by the
:func:`norm_grad` function:

.. literalinclude:: ../examples/norm_grad.py
   :language: python
   :linenos:

Alternatively, it is possible to run the method "manually". NormGrad
is a variant of the gradient method, applied at an intermediate layer:

.. literalinclude:: ../examples/norm_grad_manual.py
   :language: python
   :linenos:

Note that the function :func:`gradient_to_norm_grad_saliency` is used to
convert activations and gradients to a saliency map.

See also :ref:`backprop` for further examples and discussion.

Theory
~~~~~~

TODO.

References:

    .. [NORMGRAD] Sylvestre-Alvise Rebuffi, Ruth Fong, Xu Ji, Hakan Bilen,
                  and Andrea Vedaldi,
                  *NormGrad: Finding the Pixels that Matter for Training,*
                  arXiv 2017,
                  `<https://arxiv.org/pdf/1910.08823.pdf>`__.
"""

__all__ = ["norm_grad", "norm_grad_selective"]

import torch
from .common import saliency


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


def gradient_to_norm_grad_proper_saliency(x_in, x_out):
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

    # Compute Frobenius norm of activations.
    act_weight = torch.norm(x_in, 2, 1, keepdim=True)

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
                     gradient_to_saliency=gradient_to_norm_grad_proper_saliency,
                     use_input_output=True,
                     **kwargs):
    r"""NormGrad method, without using the virtual identity trick.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the NormGrad method, and supports the
    same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer,
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
    r"""NormGrad method.

    The function takes the same arguments as :func:`.common.saliency`, with
    the defaults required to apply the NormGrad selective method, and supports
    the same arguments and return values.
    """
    return saliency(*args,
                    saliency_layer=saliency_layer,
                    gradient_to_saliency=gradient_to_saliency,
                    **kwargs,)
