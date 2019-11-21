from torchray.attribution.meta_saliency import meta_saliency
from torchray.attribution.norm_grad import norm_grad
from torchray.benchmark import get_example_data, plot_example

if False:
    # Obtain example data.
    model, x, category_id, _ = get_example_data()
else:
    from torchray.utils import get_device
    from torchray.benchmark.models import get_transform, get_model
    from PIL import Image

    device = get_device()

    model = get_model(arch='resnet50',
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


lr = 0.0025

# NormGrad backprop.
saliency = meta_saliency(norm_grad,
                         model,
                         x,
                         category_id,
                         saliency_layer='layer4', # 'features.29',
                         lr=lr,
                         resize=True)

# Plots.
import matplotlib.pyplot as plt
plot_example(x, saliency, 'meta NormGrad backprop', category_id)
plt.show()

saliency = meta_saliency(norm_grad,
                         model,
                         x,
                         category_id,
                         negate_loss=True,
                         saliency_layer='layer4', # 'features.29',
                         lr=lr,
                         resize=True)

# Plots.
plot_example(x, saliency, 'meta NormGrad backprop', category_id)
plt.show()
