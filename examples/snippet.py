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

category_id = 282  # tiger cat
category_id = 207  # golden retriever
