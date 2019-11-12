import copy
import torch
import torch.nn.functional as F


def meta_saliency(#saliency_func,
                  model,
                  x,
                  target,
                  lr=1e-4,
                  freeze_model=True,
                  **kwargs):
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

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    y = model(x)
    if len(y.shape) == 4:
        y = y.sum((2, 3))
    assert len(y.shape) == 2

    if not isinstance(target, torch.Tensor):
        if isinstance(target, int):
            y_target = torch.tensor([target], dtype=torch.long, device=y.device)
        else:
            y_target = torch.tensor(target, dtype=torch.long, device=y.device)
    loss = F.cross_entropy(y, y_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    new_weights = model.state_dict()

    # Restore gradient saving for model parameters.
    for name, param in model.named_parameters():
        param.requires_grad_(orig_requires_grad[name])

    # Restore model's original mode.
    if orig_is_training:
        model.train()

    if freeze_model:
        model.load_state_dict(orig_weights)

    return new_weights

def f2(model, x, class_id, lr=1e-4):
    base_net = copy.deepcopy(model)
    output = base_net(x)
    if len(output.shape) == 4:
        output = output.permute(0, 2, 3, 1).view(-1, output.shape[1])
    label = (class_id * torch.ones(output.shape[0])).long().cuda()
    loss = F.cross_entropy(output, label)

    gradients = torch.autograd.grad(loss, base_net.parameters())

    with torch.no_grad():
        for w, vw, g in zip(base_net.parameters(), model.parameters(),
                            gradients):
            vw.copy_(w - lr * g)

    new_weights  = model.state_dict()
    orig_weights = base_net.state_dict()

    return new_weights


if __name__ == '__main__':
    from torchray.benchmark import get_example_data
    from torchray.benchmark import get_model
    from torchvision import models
    # _, x, category_id, _ = get_example_data()
    # model = models.vgg16(pretrained=True).to(x.device)
    device = torch.device('cuda:0')
    model = get_model(dataset='voc', convert_to_fully_convolutional=True).to(device)
    x = torch.randn(1, 3, 224, 300).to(device)
    category_id = 10

    lr = 0.1
    z1 = meta_saliency(model, x, category_id, lr=lr)
    z2 = f2(model, x, category_id, lr=lr)
    for k in z1.keys():
        assert torch.abs(z1[k] - z2[k]).sum() == 0

