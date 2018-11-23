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


def classify_images(images_dir, results_dic, model):
    filenames = listdir(images_dir)

    for i, filename in enumerate(filenames):
        correct_label = results_dic[filename][0]

        classifier_labels_string = classifier(images_dir + '/' + filename, model)

        if correct_label in classifier_labels_string.lower():
            is_match = 1
        else:
            is_match = 0

        results_dic[filename] = [correct_label, classifier_labels_string, is_match]

    return results_dic


def adjust_results4_isadog(results_dic, dogfile):
    with open(dogfile, 'r') as file:
        dog_names = [line.strip() for line in file]

    for key, value in results_dic.items():
        if value[0] in dog_names:
            value.append(1)
        else:
            value.append(0)

        is_classified_as_dog = 0
        for classifier_label in value[1].split(','):
            label = classifier_label.strip().lower()
            if label in dog_names:
                is_classified_as_dog = 1

        value.append(is_classified_as_dog)

        print(key + " " + str(value))


if __name__ == '__main__':
    in_args = get_input_args()

    results_dic = tag_files(images_dir)

    classify_images(in_args.directory, results_dic, in_args.arch)

    adjust_results4_isadog(results_dic, in_args.dogfile)
