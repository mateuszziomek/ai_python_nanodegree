import sys
import json
import torch
import numpy as np

from torchvision import models
from torch import nn
from PIL import Image

from hyperparameters import *


def get_classifier(input_size, hidden_sizes, output_size, drop_pct=DROPOUT):
    # Add first layer
    classifier = nn.ModuleList([
        nn.Linear(input_size, hidden_sizes[0]),
        nn.ReLU(),
        nn.Dropout(drop_pct)
    ])

    # Add hidden layers
    for h1, h2 in zip(hidden_sizes[:-1], hidden_sizes[1:]):
        classifier.extend([
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(drop_pct)
        ])

    # Add output layer
    classifier.extend([
        nn.Linear(hidden_sizes[-1], output_size),
        nn.Dropout(drop_pct)
    ])

    # Add LogSoftmax logits output function
    classifier.append(nn.LogSoftmax(dim=1))

    return nn.Sequential(*classifier)


def create_model(model_name, hidden_sizes, output_size, dropout):
    # Get the model
    if model_name == "vgg":
        model = models.vgg16(pretrained=True)
        input_size = INPUT_SIZE_VGG
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = INPUT_SIZE_ALEXNET
    else:
        sys.exit("Unrecognized model architecture")

    # Freeze parameters - do not backpropagate through them
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classifier
    model.classifier = get_classifier(input_size, hidden_sizes, output_size, dropout)

    return model


def get_device(use_gpu):
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def process_image(image_path):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    # Load image from path with PIL
    im = Image.open(image_path)

    # Resize, shortest size 256px, keep aspect ratio
    width, height = im.size

    if width > height:
        im.thumbnail([10000, 256])
    else:
        im.thumbnail([256, 10000])

    # Crop to 224px
    margin_l = (im.width - 224)/2
    margin_t = (im.height - 224)/2 + 224
    margin_r = (im.width - 224)/2 + 224
    margin_b = (im.height - 224)/2

    im = im.crop((margin_l, margin_b, margin_r, margin_t))

    # Normalize: convert from 256 color values to 0-1
    np_im = np.array(im)
    np_im = np_im/255

    np_im = (np_im - MEANS)/DEVS

    # Reorder dimensions - color channel at first place
    np_im = np_im.transpose(2, 0, 1)

    return np_im


def load_mapping_file(cat_to_name_path):
    with open(cat_to_name_path, 'r') as file:
        return json.load(file)



