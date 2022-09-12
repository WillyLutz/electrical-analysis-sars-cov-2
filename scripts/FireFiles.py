import os
import shutil
from glob import glob


def get_all_files(directory):
    """
    Get all the files and subdirectories within a directory recursively

    :param directory: the path of the directory to sparse
    :return: list of paths
    """
    all_paths = []
    paths = []
    subfolders = os.listdir(directory)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            if os.path.isdir(os.path.join(directory, subfolder)):
                paths = get_all_files(os.path.join(directory, subfolder))
                for p in paths:
                    all_paths.append(p)
            else:
                all_paths.append(os.path.join(directory, subfolder))

    return all_paths


def verify_dir(directory):
    """
    verify if the directory exists. If not, then create it.

    :param directory: the path of the directory to verify
    """
    isExist = os.path.exists(directory)
    if not isExist:
        os.makedirs(directory)

def split_train_test():
    base_path = "E:/HIV-AI Project/20220330 Trial Images/"
    technique = base_path + "CNN"
    train_ni_path = technique + "/Train/NI"
    train_inf_path = technique + "/Train/INF"
    test_ni_path = technique + "/Test/NI"
    test_inf_path = technique + "/Test/INF"

    basic_directories = (train_inf_path, train_ni_path, test_inf_path, test_ni_path)
    for bd in basic_directories:
        isExist = os.path.exists(bd)
        if not isExist:
            os.makedirs(bd)

    all_images = get_all_files("E:/HIV-AI Project/20220330 Trial Images/PNG")
    count_class_one = 0
    count_class_two = 0
    for image in all_images:
        head, tail = os.path.split(image)
        if "INF10^6" in image:
            count_class_two += 1
        elif "NI" in image:
            count_class_one += 1

    if count_class_two > count_class_one:  # getting the same number of entries for each class
        count_class_two = count_class_one
    else:
        count_class_one = count_class_two

    total_images = count_class_two + count_class_one
    train_proportion = int(0.7 * total_images)
    test_proportion = total_images - train_proportion

    inf_train_index = 0
    ni_train_index = 0
    print(train_proportion)
    for image in all_images:
        head, tail = os.path.split(image)
        if "INF10^6" in image and inf_train_index < train_proportion / 2:
            print("train inf ", inf_train_index)
            destination = train_inf_path + "/" + tail
            shutil.copyfile(image, destination)
            inf_train_index += 1
        elif "NI" in image and ni_train_index < train_proportion / 2:
            print("train ni ", ni_train_index)
            destination = train_ni_path + "/" + tail
            shutil.copyfile(image, destination)
            ni_train_index += 1
        elif "INF10^6" in image and inf_train_index >= train_proportion / 2:
            print("test inf ", inf_train_index)
            destination = test_inf_path + "/" + tail
            shutil.copyfile(image, destination)
            inf_train_index += 1
        elif "NI" in image and ni_train_index >= train_proportion / 2:
            print("test ni ", ni_train_index)
            destination = test_ni_path + "/" + tail
            shutil.copyfile(image, destination)
            ni_train_index += 1
