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

def get_urchin_image_folders(directory: str, labelers: list[str], image_folders: list[str]) -> list[str]:
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
        for image_folder in image_folders:
            path = os.path.join(directory, labeler, image_folder)
            if os.path.exists(path):
                paths += [path]
    return paths

def get_folder_filenames(folder: str) -> set:
    """Gets all valid image filenames in a given folder

    Args:
        folder (str): folder to get filenames from

    Returns:
        set: all valid filename strings from the folder, contains the folder in the path
    """
    filenames = set()
    
    for path in glob.glob(os.path.join(folder, '*.jpg')):
        # Extract the filename
        filename = path     
        filenames.add(filename)

    return filenames

def get_all_filenames(folders: list[str]) -> set:
    """Gets all valid image filenames for all folders

    Args:
        folders (str]): all folders to get images from

    Returns:
        set: all valid filename strings, relative paths from repo base directory
    """
    filenames = set()

    for folder in folders:
        folder_filenames = get_folder_filenames(folder)
        filenames.update(folder_filenames)

    return filenames


def inspect_duplicates(duplicates):
    # Show the images from the duplicated filenames
    for file in duplicates:
        for animal in ['cat', 'dog']:
            Image.open(f'download/{animal}/images/{file}').show()

def get_cats_and_dogs():
    # Dog and cat image filename sets
    dog_images = get_folder_filenames('download/dog/images')
    cat_images = get_folder_filenames('download/cat/images')

    # eliminate duplicates
    duplicates = dog_images & cat_images
    # used to check which images are duplicates and what they actually are, we know that the cats and dogs duplicates are cat images
    # inspect_duplicates(duplicates)
    dog_images -= duplicates

    dog_images = np.array(list(dog_images))
    cat_images = np.array(list(cat_images))

    # Use the same random seed for reproducability
    np.random.seed(42)
    np.random.shuffle(dog_images)
    np.random.shuffle(cat_images)

    # Cat data
    split_cats_and_dogs('cat', cat_images, train_size=400, val_size=50)

    # Dog data (reduce the number by 1 for each set due to three duplicates)
    split_cats_and_dogs('dog', dog_images, train_size=399, val_size=49)

def split_cats_and_dogs(animal, image_names, train_size, val_size):
    for i, image_name in enumerate(image_names):
        # Label filename
        label_name = image_name.replace('.jpg', '.txt')
        
        # Split into train, val, or test
        if i < train_size:
            split = 'train'
        elif i < train_size + val_size:
            split = 'val'
        else:
            split = 'test'
        
        # Source paths
        source_image_path = f'download/{animal}/images/{image_name}'
        source_label_path = f'download/{animal}/darknet/{label_name}'

        # Destination paths
        target_image_folder = f'data/images/{split}'
        target_label_folder = f'data/labels/{split}'

        # Copy files
        shutil.copy(source_image_path, target_image_folder)
        shutil.copy(source_label_path, target_label_folder)

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
    image_folders = ["google_0", "sean_nov_3/images"]

    make_yolo_folders()

    image_folders = get_urchin_image_folders(directory, labelers, image_folders)
    image_filenames = get_all_filenames(image_folders)
    split_data(image_filenames, 0.6, 0.2)

if __name__ == '__main__':
    main()