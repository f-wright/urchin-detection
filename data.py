import os
import glob
from PIL import Image
import numpy as np
import shutil

def make_yolo_folders() -> None:
    """ Generates folder structure for YOLOv5 images and labels for train, val, and test
    """
    if not os.path.exists('data'):
        for folder in ['images', 'labels']:
            for split in ['train', 'val', 'test']:
                os.makedirs(f'data/{folder}/{split}')

def get_urchin_image_folders(directory: str, labelers: list[str], image_subfolders: list[str]) -> list[str]:
    """Returns all folders with valid images in them. 

    Args:
        directory (str): base directory containing all images, relative path from repo base
        labelers (str]): list of all labelers
        image_folders (str]): list of all valid labeler subfolders which contain images directly

    Returns:
        [str]: list of paths which exist and contain images, relative paths from repo base
    """
    paths = []
    for labeler in labelers:
        for image_folder in image_subfolders:
            path = os.path.join(directory, labeler, image_folder)
            if os.path.exists(path):
                paths += [path]
    return paths

def get_urchin_label_folders(image_folders: list[str]) -> list[str]:
    """Gets label folders corresponding to image folders. Relies on particular file structure.

    Args:
        image_folders (list[str]): List of folders containing input images

    Returns:
        list[str]: List of folders containing labels and metadata about labels
    """
    label_folders = []
    for folder in image_folders:
        label_path = folder.split(os.sep)
        label_path[1] = 'labels'
        if label_path[-1] == 'images':
            del label_path[-1]
            label_path[-1] = label_path[-1] + "_images"

        label_path = os.path.join(*label_path)
        label_folders.append(label_path)
    return label_folders

def get_filenames(folder: str, is_label = False) -> set:
    """Gets all valid image filenames in a given folder

    Args:
        folder (str): folder to get filenames from

    Returns:
        set: all valid filename strings from the folder, contains the folder in the path
    """
    if is_label:
        folder = os.path.join(folder, "obj_train_data")
        extension = "*.txt"
    else:
        extension = "*.jpg"
    
    filenames = set()
    
    for path in glob.glob(os.path.join(folder, extension)):
        filenames.add(path)

    return filenames

def get_all_image_filenames(folders: list[str]) -> set:
    """Gets all valid image filenames for all folders

    Args:
        folders (str]): all folders to get images from

    Returns:
        set: all valid filename strings, relative paths from repo base directory
    """
    filenames = set()

    for folder in folders:
        folder_filenames = get_filenames(folder, is_label=False)
        filenames.update(folder_filenames)

    return filenames

def standardize_classes(label_folders: list[str]) -> None:
    """Change all .names files and label files in place to reflect the classes being ['Purple Sea Urchin\n', 'Other Sea Urchin\n']

    Args:
        label_folders (list[str]): List of all label folders to standardize
    """
    for label_folder in label_folders:
        class_names_filepath = os.path.join(label_folder, "obj.names")

        # reads in classes and strips off newline characters and trailing whitespace
        with open(class_names_filepath, "r") as class_fp:
            input_classes = class_fp.readlines()

        # classes are two different options, make sure they're the correct options and the only variation is ordering
        proper_classes = ['Purple Sea Urchin\n', 'Other Sea Urchin\n']
        flipped_classes = ['Other Sea Urchin\n', 'Purple Sea Urchin\n']
        assert input_classes in [proper_classes, flipped_classes], f"Class labels are not 'Purple Sea Urchin' and 'Other Sea Urchin' in {label_folder}"
        
        # classes in the wrong order
        if input_classes == flipped_classes:
            label_files = get_filenames(label_folder, is_label=True)

            for file in label_files:
                # read in file, then write over it, flipping class labels
                with open(file, "r+") as label_fp:
                    labels = label_fp.readlines()
                    labels = [flip_class(label) for label in labels]

                    # return to beginning of file
                    label_fp.seek(0)

                    label_fp.writelines(labels)
            
            # fix .names file
            with open(class_names_filepath, "r+") as class_fp:
                class_fp.writelines(proper_classes)
        
def flip_class(label: str) -> str:
    """Flips the class label of an input label

    Args:
        label (str): single line of a label file, class then coordinates

    Returns:
        str: single line of a label file with the class swapped between 1 and 0
    """
    if label[0] == "1":
        label = "0" + label[1:]
    else:
        label = "1" + label[1:]
    return label

def split_data(image_filenames: set, train_prop: float, val_prop: float) -> None:
    """Splits all images and labels across train, val and test folders

    Args:
        image_filenames (set): set of all image filenames as strings
        train_prop (float): proportion of total data we want to use for training
        val_prop (float): proportion of total data we want to use for validation
    """
    urchin_images = np.array(list(image_filenames))

    # shuffle data
    np.random.seed(42) # for reproducability
    np.random.shuffle(urchin_images)

    total_image_count = urchin_images.shape[0]

    train_count = int(train_prop * total_image_count)
    val_count = int(val_prop * total_image_count)

    for i, image_path in enumerate(urchin_images):
        # get path split by folders into a list
        label_path = image_path.split(os.sep)

        # get corresponding label path for image, relies on particular file structure
        label_path[-1] = label_path[-1].replace('.jpg', '.txt')
        label_path[1] = 'labels'
        if label_path[-2] == 'images':
            del label_path[-2]
            label_path[-2] = label_path[-2] + "_images"
        label_path.insert(-1, "obj_train_data")

        label_path = os.path.join(*label_path)
        
        # Split into train, val, or test
        if i < train_count:
            split = 'train'
        elif i < train_count + val_count:
            split = 'val'
        else:
            split = 'test'
        
        # make sure all of our paths exist
        assert os.path.exists(image_path), f"Image path {image_path} does not exist"
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"

        # Destination paths
        destination_filename = image_path.split(os.sep)
        destination_image_filename = destination_filename[2] + "_" + destination_filename[3] + "_" + destination_filename[-1]
        destination_label_filename = destination_image_filename.replace('.jpg', '.txt')

        target_image_folder = f'data/images/{split}/{destination_image_filename}'
        target_label_folder = f'data/labels/{split}/{destination_label_filename}'

        # Copy files
        shutil.copy(image_path, target_image_folder)
        shutil.copy(label_path, target_label_folder)

def main():
    directory = "urchin_download/images"
    labelers = ["Brittany", "Castor", "Eliza", "Francine", "James", "Katie", "Ryan"]
    image_subfolders = ["google_0", "sean_nov_3/images"]

    make_yolo_folders()

    image_folders = get_urchin_image_folders(directory, labelers, image_subfolders)
    label_folders = get_urchin_label_folders(image_folders)

    standardize_classes(label_folders)
    image_filenames = get_all_image_filenames(image_folders)
    split_data(image_filenames, 0.6, 0.2)

if __name__ == '__main__':
    main()