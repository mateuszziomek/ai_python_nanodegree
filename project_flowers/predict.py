import argparse
import torch

from directories import *
from hyperparameters import *
from utils import *


def get_arguments():
    parser = argparse.ArgumentParser(description='Train model')
    # Non-optional argument: image directory
    parser.add_argument('image_dir', action='store', type=str, help='Image file path')
    parser.add_argument('check_dir', action='store', type=str, help='Model checkpoint path')

    # Optional arguments: save checkpoint directory, model architecture, hyper-parameters, gpu mode
    parser.add_argument('--top_k', default=TOP_K, type=int, help='Top K most likely classes')
    parser.add_argument('--category_name', default=CATEGORIES, type=str, help='Mapping of categories to real names')
    parser.add_argument('--gpu', action="store_true", help="Should use GPU enhanced mode")

    args = parser.parse_args()
    return args


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)

    model = create_model(
        checkpoint['arch'],
        checkpoint["hidden_sizes"],
        checkpoint["output_size"],
        checkpoint["dropout"]
    )

    model.load_state_dict(checkpoint["state_dic"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def predict(image_path, model, topk, cat_to_name, device):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """

    image = torch.from_numpy(process_image(image_path)).type(torch.FloatTensor)
    image.unsqueeze_(0)

    with torch.no_grad():
        image = image.to(device)
        model.to(device)

        output = model.forward(image)
        probabilities = np.exp(output)

        top_probs, top_class_idx = torch.topk(probabilities, topk)

    # Convert tensors to lists
    top_class_idx = top_class_idx.numpy().tolist()[0]
    top_probs = top_probs.numpy().tolist()[0]

    # Convert keys to values and vice-versa
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # Get topk flower indices
    top_flowers_idx = [idx_to_class[i] for i in top_class_idx]

    # Get true flowers' names
    top_flowers = [cat_to_name[i] for i in top_flowers_idx]

    return top_flowers, top_probs


if __name__ == '__main__':
    in_args = get_arguments()

    image_path = in_args.image_dir
    checkpoint_path = in_args.check_dir
    top_k = in_args.top_k
    cat_to_name_path = in_args.category_name
    use_gpu = in_args.gpu

    # Get device
    device = get_device(use_gpu)

    # Load mapping file
    cat_to_name = load_mapping_file(cat_to_name_path)

    # Load the model
    model = load_checkpoint(checkpoint_path, device)

    # Get predicted probabilities and class to index map
    flowers, probs = predict(image_path, model, top_k, cat_to_name, device)

    # Print results
    print("Top {} prediction(s):".format(top_k))

    for i in range(0, len(flowers)):
        print(
            "Flower name: {}, probability: {:.0f}%".format(flowers[i], probs[i] * 100),
            sep='\n'
        )
