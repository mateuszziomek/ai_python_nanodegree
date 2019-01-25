import argparse
import torch
import numpy as np

from torchvision import transforms, datasets
from torch import optim

from hyperparameters import *
from directories import *
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser(description='Train model')
    # Non-optional argument: data directory
    parser.add_argument('data_dir', action='store', type=str)

    # Optional arguments: save checkpoint directory, model architecture, hyper-parameters, gpu mode
    parser.add_argument('--save_dir', default=CHECKPOINT_DIR, help='Model checkpoint directory')
    parser.add_argument('--arch', default='vgg16', type=str, help='Model architecture: vgg16, alexnet')
    parser.add_argument('--learning_rate', default=LEARNING_RATE, type=float, help='Learning rate value')
    parser.add_argument('--hidden_units', nargs='+', default=HIDDEN_LAYERS, type=int,
                        help='List of hidden layers\' sizes')
    parser.add_argument('--epochs', default=EPOCHS, type=int, help="Number of training epochs")
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
        transforms.Resize(256),
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


def save_checkpoint(path, model, model_name, optimizer_state, class_to_idx, epochs, output_size, hidden_sizes, dropout):
    # Save mapping of classes to indices
    model.class_to_idx = class_to_idx

    checkpoint = {
        "arch": model_name,
        "state_dic": model.state_dict(),
        "class_to_idx": class_to_idx,
        "optimizer_state": optimizer_state,
        "epochs": epochs,
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

    model = create_model(
        arch,
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
        arch,
        optimizer.state,
        data_loaders[TRAIN_LOADER].dataset.class_to_idx,
        epochs,
        OUTPUT_SIZE,
        hidden_sizes,
        DROPOUT
    )


if __name__ == '__main__':
    main()
