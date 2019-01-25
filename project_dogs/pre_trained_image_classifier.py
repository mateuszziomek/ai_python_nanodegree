import argparse
from os import listdir
from project_dogs.classifier import classifier

images_dir = "/Users/mateuszziomek/AI with Python/Projects/AIPND-revision/intropyproject-classify-pet-images/pet_images"


def get_input_args():
    parser = argparse.ArgumentParser(description='Provide input to process.')
    parser.add_argument('--dir', nargs=1, default=images_dir, dest='directory', help='Image Folder')
    parser.add_argument('--arch', nargs=1, default='vgg16', help='CNN Model Architecture')
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
        # Skip file names starting with '.' character
        if file[0] != '.':
            tag = [file[0:file.rfind('_')].replace("_", " ").lower().strip()]
            results_dic[file] = tag
    return results_dic


def classify_images(images_dir, results_dic, model):
    filenames = listdir(images_dir)

    for i, filename in enumerate(filenames):
        correct_label = results_dic[filename][0].strip()

        classifier_labels_string = classifier(images_dir + '/' + filename, model)

        if correct_label in classifier_labels_string.lower().strip():
            is_match = 1
        else:
            is_match = 0

        results_dic[filename].extend([classifier_labels_string, is_match])
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


def calculates_results_stats(results_dic):
    """
    n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds
#            pct_match - percentage of correct matches
#            pct_correct_dogs - percentage of correctly classified dogs
#            pct_correct_breed - percentage of correctly classified dog breeds
#            pct_correct_notdogs - percentage of correctly classified NON-dogs
    """

    # correct label, classifier_labels, is_match, is_dog_image, is_classified_as_dog

    results_stats_dic = dict()

    n_images = len(results_dic)
    n_dogs_img = 0
    n_notdogs_img = 0
    n_match = 0
    n_correct_dogs = 0
    n_correct_notdogs = 0
    n_correct_breed = 0

    for value in results_dic.values():
        # is dog image
        if value[3] == 1:
            n_dogs_img += 1
        # is not dog image
        else:
            n_notdogs_img += 1

        # number of matches
        if value[2] == 1:
            n_match += 1

        # number of correctly classified dog images
        if value[3] == 1 and value[4] == 1:
            n_correct_dogs += 1
        # number of correctly classified NON-dog images
        elif value[3] == 0:
            n_correct_notdogs += 1

        # number of correctly classified dog breeds
        if value[2] == 1 and value[3] == 1:
            n_correct_breed += 1

    pct_match = n_match / n_images * 100

    pct_correct_dogs = n_correct_dogs / n_dogs_img * 100

    pct_correct_breed = n_correct_breed / n_dogs_img * 100

    pct_correct_notdogs = n_notdogs_img / n_correct_notdogs * 100

    results_stats_dic["n_images"] = n_images
    results_stats_dic["n_dogs_img"] = n_dogs_img
    results_stats_dic["n_notdogs_img"] = n_notdogs_img
    results_stats_dic["n_match"] = n_match
    results_stats_dic["n_correct_dogs"] = n_correct_dogs
    results_stats_dic["n_correct_notdogs"] = n_correct_notdogs
    results_stats_dic["n_correct_breed"] = n_correct_breed

    results_stats_dic["pct_match"] = pct_match
    results_stats_dic["pct_correct_dogs"] = pct_correct_dogs
    results_stats_dic["pct_correct_breed"] = pct_correct_breed
    results_stats_dic["pct_correct_notdogs"] = pct_correct_notdogs

    return results_stats_dic


def print_results(results_dic, results_stats_dic, model, print_incorrect_dogs=False, print_incorrect_breed=False):
    n_images = results_stats_dic["n_images"]
    n_correct_dogs = results_stats_dic["n_correct_dogs"]
    n_correct_notdogs = results_stats_dic["n_correct_notdogs"]
    n_correct_breed = results_stats_dic["n_correct_breed"]

    print("CNN model architecture used: {}".format(model))
    print("Number of Images: {}".format(n_images))
    print("Number of Dog Images: {}".format(results_stats_dic["n_dogs_img"]))
    print("Number of \"Not-a\" Dog Images: {}".format(results_stats_dic["n_notdogs_img"]))

    for key, value in results_stats_dic.items():
        if key[0] == 'p':
            label = key[4:].replace("_", " ").title()
            print("{}% {}".format(value, label))

    dogs_misclassified = n_correct_dogs + n_correct_notdogs != n_images

    if print_incorrect_dogs and dogs_misclassified:
        # The labels disagree on whether or not an image is of a "dog"
        print("Misclassified dogs:\n")
        for value in results_dic.values():
            if sum(value[3:]) == 1:
                print("Pet image label: {}      Classifier labels: {}".format(value[0], value[1]))

    breeds_misclassified = n_correct_dogs != n_correct_breed

    if print_incorrect_breed and breeds_misclassified:
        # The labels agree that image is of a dog, but disagree on the breed of dog
        print("\nMisclassified breeds:".upper())
        for value in results_dic.values():
            if sum(value[3:]) == 2 and value[2] == 0:
                print("Pet image label: {}      Classifier labels: {}".format(value[0], value[1]))


if __name__ == '__main__':
    in_args = get_input_args()

    results_dic = tag_files(images_dir)

    classify_images(in_args.directory, results_dic, in_args.arch)

    adjust_results4_isadog(results_dic, in_args.dogfile)

    stats = calculates_results_stats(results_dic)

    print_results(results_dic, stats, in_args.arch, True, True)
