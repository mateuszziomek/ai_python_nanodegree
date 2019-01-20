import sys
import argparse
import torch
import numpy as np

from torchvision import transforms, datasets, models
from torch import nn
from torch import optim

PROJECT_DIR = '/Users/mateuszziomek/IdeaProjects/ai_course/project_flowers'

DATA_DIR = PROJECT_DIR + '/assets/flowers'
TRAIN_DIR = DATA_DIR + '/train'
VALID_DIR = DATA_DIR + '/valid'
TEST_DIR = DATA_DIR + '/test'

TRAIN_LOADER = "train"
VALID_LOADER = "valid"
TEST_LOADER = "test"

CHECKPOINT_DIR = PROJECT_DIR + '/checkpoints/checkpoint.pth'

INPUT_SIZE = 25088
OUTPUT_SIZE = 102
HIDDEN_LAYERS = [500]

DROPOUT = 0.2
LEARNING_RATE = 0.001
MOMENTUM = 0.9

EPOCHS = 7
PRINT_EVERY = 30

MEANS, DEVS = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def get_arguments():
    parser = argparse.ArgumentParser(description='Train model')
    # Non-optional argument: data directory
    parser.add_argument('data_dir', action='store', type=str)

    # Optional arguments: save checkpoint directory, model architecture, hyper-parameters, gpu mode
    parser.add_argument('--save_dir', nargs=1, default=CHECKPOINT_DIR, help='Model checkpoint directory')
    parser.add_argument('--arch', nargs=1, default='vgg', help='Model architecture: vgg')
    parser.add_argument('--learning_rate', nargs=1, default=LEARNING_RATE, type=float, help='Learning rate value')
    parser.add_argument('--hidden_units', nargs='+', default=HIDDEN_LAYERS, type=int,
                        help='List of hidden layers\' sizes')
    parser.add_argument('--epochs', nargs=1, default=EPOCHS, type=int, help="Number of training epochs")
    parser.add_argument('--gpu', action="store_true", help="Should use GPU enhanced mode")
    args = parser.parse_args()
    return args


def get_data_loaders(training_dir):
    train_transforms = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, DEVS)
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, DEVS)
    ])

    train_dataset = datasets.ImageFolder(training_dir, transform=train_transforms)

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)

    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    return {
        TRAIN_LOADER: train_loader,
        TEST_LOADER: test_loader,
        VALID_LOADER: valid_loader
    }


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


def create_model_vgg16(model_name, input_size, hidden_sizes, output_size, dropout):
    if model_name != "vgg":
        sys.exit("Unrecognized model architecture")

    # Get the model
    model = models.vgg16(pretrained=True)

    # Freeze parameters - do not backpropagate through them
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classifier
    model.classifier = get_classifier(input_size, hidden_sizes, output_size, dropout)

    return model


def validate(model, loader, criterion, device):
    loss = 0
    accuracy = 0

    # Move to GPU
    model = model.to(device)

    # Set model in evaluation mode
    model.eval()

    # Turn off gradients for validation for better performance
    with torch.no_grad():
        # Do validation
        for images, labels in loader:
            # Move to GPU
            images, labels = images.to(device), labels.to(device)

            # Do forward pass
            output = model.forward(images)

            # Calculate the loss vector
            loss += criterion(output, labels).item()

            # Convert logits
            probabilities = np.exp(output).to(device)

            # Check if predictions are equal the actual labels
            is_equal = labels.data == probabilities.max(dim=1)[1]

            # Calculate model accuracy
            accuracy += is_equal.type(torch.FloatTensor).mean()

    print(
        "Valid. loss: {:.4f}".format(loss / len(loader)),
        "Valid. accuracy: {:.4f}".format(accuracy / len(loader))
    )

    # Set model back in training mode
    model.train()


def train(model, train_loader, valid_loader, criterion, optimizer, device, epochs):
    print("Model training started..")

    # Move model to GPU
    model = model.to(device)

    steps = 0

    for e in range(epochs):
        current_loss = 0

        for images, labels in train_loader:
            steps += 1

            # Move data to GPU
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()

            if steps % PRINT_EVERY == 0:
                print(
                    "Epoch: {}/{}... ".format(e + 1, epochs),
                    "Loss: {:.4f}".format(current_loss / PRINT_EVERY)
                )
                validate(model, valid_loader, criterion, device)

                current_loss = 0


def get_device(use_gpu):
    return torch.device("cuda:0" if use_gpu and torch.cuda.is_available() else "cpu")


def save_checkpoint(path, model, optimizer_state, class_to_idx, epochs, input_size, output_size, hidden_sizes, dropout):
    # Save mapping of classes to indices
    model.class_to_idx = class_to_idx

    checkpoint = {
        "state_dic": model.state_dict(),
        "class_to_idx": class_to_idx,
        "optimizer_state": optimizer_state,
        "epochs": epochs,
        "input_size": input_size,
        "output_size": output_size,
        "hidden_sizes": hidden_sizes,
        "dropout": dropout
    }

    torch.save(checkpoint, path)


def main():
    in_args = get_arguments()

    arch = in_args.arch
    hidden_sizes = in_args.hidden_units
    lr = in_args.learning_rate
    use_gpu = in_args.gpu
    train_dir = in_args.data_dir
    epochs = in_args.epochs
    checkpoint_dir = in_args.save_dir

    model = create_model_vgg16(
        arch,
        INPUT_SIZE,
        hidden_sizes,
        OUTPUT_SIZE,
        DROPOUT
    )

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=lr, momentum=MOMENTUM)
    device = get_device(use_gpu)

    data_loaders = get_data_loaders(train_dir)

    train(model, data_loaders[TRAIN_LOADER], data_loaders[VALID_LOADER], criterion, optimizer, device, epochs)

    validate(model, data_loaders[TEST_LOADER], criterion, device)

    save_checkpoint(
        checkpoint_dir,
        model,
        optimizer.state,
        data_loaders[TRAIN_LOADER].dataset.class_to_idx,
        epochs,
        INPUT_SIZE,
        OUTPUT_SIZE,
        hidden_sizes,
        DROPOUT
    )


if __name__ == '__main__':
    main()
