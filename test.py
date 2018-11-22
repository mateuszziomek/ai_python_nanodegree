import argparse
from os import listdir
from classifier import classifier

images_dir = "/Users/mateuszziomek/AI with Python/Projects/AIPND-revision/intropyproject-classify-pet-images/pet_images"


def get_input_args():
    parser = argparse.ArgumentParser(description='Provide input to process.')
    parser.add_argument('--dir', nargs=1, default=images_dir, dest='directory', help='Image Folder')
    parser.add_argument('--arch', nargs=1, default='vgg', help='CNN Model Architecture')
    parser.add_argument('--dogfile', nargs=1, default='dognames.txt', help='Text File with Dog Names')
    args = parser.parse_args()
    return args


def tag_files(image_dir):
    """"
    find the last occurrence of '_'
    trim the string to that position
    replace '_' with ' '
    lower case the string
    """
    results_dic = dict()
    files = listdir(image_dir)

    for file in files:
        tag = [file[0:file.rfind('_')].replace("_", " ").lower()]
        results_dic[file] = tag
    return results_dic


def classify_images(images_directory, results_dic, model):
    filenames = listdir(images_directory)

    for i, filename in enumerate(filenames):
        correct_label = results_dic[filename][0]

        classifier_labels_string = classifier(images_directory + '/' + filename, model)

        if correct_label in classifier_labels_string.lower():
            is_match = 1
        else:
            is_match = 0

        results_dic[filename] = [correct_label, classifier_labels_string, is_match]

    print(results_dic)

    return results_dic


if __name__ == '__main__':
    in_args = get_input_args()

    classify_images(in_args.directory, tag_files(images_dir), in_args.arch)
