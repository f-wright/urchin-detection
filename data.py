import os
import glob
from PIL import Image
import numpy as np
import shutil

def make_yolo_folders():
    """ Sets up folder structure for YOLOv5 data.
    """
    if not os.path.exists('data'):
        for folder in ['images', 'labels']:
            for split in ['train', 'val', 'test']:
                os.makedirs(f'data/{folder}/{split}')

def get_filenames(folder):
    filenames = set()
    
    for path in glob.glob(os.path.join(folder, '*.jpg')):
        # Extract the filename
        filename = os.path.split(path)[-1]        
        filenames.add(filename)

    return filenames

def get_cats_and_dogs():
    # Dog and cat image filename sets
    dog_images = get_filenames('download/dog/images')
    cat_images = get_filenames('download/cat/images')

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
    split_dataset('cat', cat_images, train_size=400, val_size=50)

    # Dog data (reduce the number by 1 for each set due to three duplicates)
    split_dataset('dog', dog_images, train_size=399, val_size=49)


def inspect_duplicates(duplicates):
    # Show the images from the duplicated filenames
    for file in duplicates:
        for animal in ['cat', 'dog']:
            Image.open(f'download/{animal}/images/{file}').show()


def split_dataset(animal, image_names, train_size, val_size):
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
