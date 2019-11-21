from torchray.attribution.norm_grad import norm_grad, norm_grad_proper
from torchray.benchmark import get_example_data, plot_example

if False:
    # Obtain example data.
    model, x, category_id, _ = get_example_data()
else:
    from torchray.utils import get_device
    from torchray.benchmark.models import get_transform, get_model
    from PIL import Image

    device = get_device()

    model = get_model(arch='vgg16',
                      dataset='imagenet',
                      convert_to_fully_convolutional=False)
    transform = get_transform(dataset='imagenet',
                              size=(224, 224))

    img = Image.open('./examples/dog_cat.jpeg').convert('RGB')
    x = transform(img).unsqueeze(0)

    model.to(device)
    x = x.to(device)

    # category_id = 282  # tiger cat
    category_id = 207  # golden retriever

# NormGrad backprop.
saliency_approx = norm_grad(model, x, category_id, saliency_layer='features.28')
saliency_proper = norm_grad_proper(model, x, category_id, saliency_layer='features.28')

# Plots.
import matplotlib.pyplot as plt
plot_example(x, saliency_approx, 'NormGrad approx', category_id)
plt.show()

plot_example(x, saliency_proper, 'NormGrad proper', category_id)
plt.show()
