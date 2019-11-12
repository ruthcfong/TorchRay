from torchray.attribution.meta_saliency import meta_saliency
from torchray.attribution.norm_grad import norm_grad
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

# NormGrad backprop.
saliency = meta_saliency(norm_grad,
                         model,
                         x,
                         category_id,
                         saliency_layer='features.29')

# Plots.
plot_example(x, saliency, 'meta NormGrad backprop', category_id)
