# todo
#   train a new network on a dataset
#   and save the model as a checkpoint

import argparse
import torch
import numpy as np

# Directories
project_dir = '/Users/mateuszziomek/IdeaProjects/ai_course/project_flowers'

data_dir = project_dir + '/assets/flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

checkpoint_dir = project_dir + '/checkpoints/checkpoint.pth'

# Define hyper-parameters

input_size = 25088
output_size = 102
hidden_sizes = [500]

drop_pct = 0.2
learn_rate = 0.001
momentum = 0.9

epochs = 7
print_every = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

means, devs = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


def get_arguments():
    parser = argparse.ArgumentParser(description='Train model')
    # Non-optional argument: data directory
    parser.add_argument('data_dir', action='store', type=str)

    # Optional arguments: save checkpoint directory, model architecture, hyper-parameters, gpu mode
    parser.add_argument('--save_dir', nargs=1, default=checkpoint_dir, help='Model checkpoint directory')
    parser.add_argument('--arch', nargs=1, default='vgg', help='Model architecture')
    parser.add_argument('--learning_rate', nargs=1, default=learn_rate, type=float, help='Learning rate value')
    parser.add_argument('--hidden_units', nargs='+', default=hidden_sizes, type=int,
                        help='List of hidden layers\' sizes')
    parser.add_argument('--epochs', nargs=1, default=epochs, type=int, help="Number of training epochs")
    parser.add_argument('--gpu', action="store_true", help="Should use GPU enhanced mode")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    get_arguments()
