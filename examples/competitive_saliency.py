from torchray.attribution.linear_approx import linear_approx
from torchray.attribution.competitive_saliency import competitive_saliency
from torchray.benchmark import get_example_data, plot_example

# Obtain example data.
model, x, _,category_id = get_example_data()

saliency = competitive_saliency(linear_approx,
                                model,
                                x,
                                category_id)

# Plots.
plot_example(x, saliency, 'competitive norm grad backprop', category_id)
