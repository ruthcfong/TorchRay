from torchray.attribution.norm_grad import norm_grad_selective
from torchray.attribution.weighted_saliency import weighted_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, category_id, _ = get_example_data()

layer_weights = {
    'features.3': 0.153,
    'features.8': 0.219,
    'features.15': 0.310,
    'features.22': 0.181,
    'features.29': 0.138,
}

# weighted NormGrad backprop.
saliency = weighted_saliency(norm_grad_selective,
                             layer_weights,
                             model,
                             x,
                             category_id)

# Plots.
plot_example(x, saliency, 'weighted NormGrad backprop', category_id)
