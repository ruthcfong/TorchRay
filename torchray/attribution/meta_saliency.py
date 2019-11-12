import copy
import torch
import torch.nn.functional as F


def meta_saliency(saliency_func,
                  model,
                  x,
                  target,
                  lr=1e-4,
                  freeze_model=True,
                  negate_loss=False,
                  **kwargs):
    """Update the model with one SGD step and then apply a saliency method.

    Args:
        saliency_func (function): a saliency function
            (e.g., ``torchray.attribution.norm_grad``).
        model (:class:`torch.nn.Module`): a model.
        input (:class:`torch.Tensor`): input tensor.
        target (int or :class:`torch.Tensor`): target label(s).
        lr (float): learning rate with which to take one SGD step.
            Default: ``1e-4``.
        freeze_model (bool): If True, restore original weights to model.
            Otherwise, allow model to take one SGD step.
        negate_loss (bool): If True, negate the loss (i.e., to highlight
            the adversarial information in an image).
        kwargs: optional arguments for the :attr:`saliency_func` function.

    Returns:
        saliency_map: the output of :attr:`saliency_func`.
    """

    # Save original model weights.
    if freeze_model:
        orig_weights = copy.deepcopy(model.state_dict())

    # Disable gradients for model parameters.
    orig_requires_grad = {}
    for name, param in model.named_parameters():
        orig_requires_grad[name] = param.requires_grad
        param.requires_grad_(True)

    # Set model to eval mode.
    if model.training:
        orig_is_training = True
        model.eval()
    else:
        orig_is_training = False

    # Prepare optimizer to update model weights.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Do a forward pass.
    y = model(x)

    # Prepare target tensor.
    if not isinstance(target, torch.Tensor):
        if isinstance(target, int):
            y_target = torch.tensor([target],
                                    dtype=torch.long,
                                    device=y.device)
        else:
            y_target = torch.tensor(target, dtype=torch.long, device=y.device)

    # Handle fully-convolutional case.
    if len(y.shape) == 4:
        y = y.sum((2, 3))
    assert len(y.shape) == 2

    # Update model weights w.r.t. the cross entropy loss.
    loss = F.cross_entropy(y, y_target)

    # Negate loss to highlight opposite direction.
    if negate_loss:
        loss = -1 * loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute saliency map on model with updated weights.
    saliency_map = saliency_func(model, x, target, **kwargs)

    # Restore gradient saving for model parameters.
    for name, param in model.named_parameters():
        param.requires_grad_(orig_requires_grad[name])

    # Restore model's original mode.
    if orig_is_training:
        model.train()

    # Restore model's original weights.
    if freeze_model:
        model.load_state_dict(orig_weights)

    return saliency_map


if __name__ == '__main__':
    from torchray.attribution.norm_grad import norm_grad
    from torchray.benchmark import get_example_data, plot_example
    model, x, category_id, _ = get_example_data()

    saliency_map = meta_saliency(norm_grad, model, x, category_id, saliency_layer='features.22')
    plot_example(x, saliency_map, 'meta_norm_grad', category_id)
    import matplotlib.pyplot as plt
    plt.show()

