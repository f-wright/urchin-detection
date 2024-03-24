import shutil
import os
from PIL import Image
from pathlib import Path

LABELERS = ["Brittany", "Castor", "Eliza", "Francine", "James", "Ryan"]
INPUT_DIR = "input_images"
OUTPUT_DIR = "images_to_label"
FILE_CATEGORY = "background_lsui_0"
CURRENT_DIR = os.getcwd()
VIDEO_EXTENSIONS = {".mov", ".MOV"}
IMAGE_EXTENSIONS = {".JPG", ".jpg", ".png", ".avif", ".webp", ".gif", ".jpeg"}


def distribute_videos_and_images():
    # define inputs
    input_path = os.path.join(CURRENT_DIR, INPUT_DIR)
    files_list = os.listdir(input_path)

    # define outputs
    output_dirs = [
        os.path.join(CURRENT_DIR, OUTPUT_DIR, labeler, FILE_CATEGORY)
        for labeler in LABELERS
    ]
    image_output_dirs = [os.path.join(labeler, "images") for labeler in output_dirs]
    video_output_dirs = [os.path.join(labeler, "videos") for labeler in output_dirs]
    num_groups = len(LABELERS)

    # make directories if they don't already exist
    for image_dir in image_output_dirs:
        Path(image_dir).mkdir(parents=True, exist_ok=True)
    for video_dir in video_output_dirs:
        Path(video_dir).mkdir(parents=True, exist_ok=True)

    # split images and videos
    images_list = [
        file
        for file in files_list
        if (file[-5:] in IMAGE_EXTENSIONS or file[-4:] in IMAGE_EXTENSIONS)
    ]
    videos_list = [
        file for file in files_list if (file[file.index(".") :] in VIDEO_EXTENSIONS)
    ]

    # make sure that all files are going to one of the folders
    remaining_files = [
        file
        for file in files_list
        if (file not in images_list and file not in videos_list)
        and (file != ".DS_Store")
    ]
    assert (
        len(remaining_files) == 0
    ), f"Missing extension in the list of video and image extensions, please check all are included. Files not in either list: {remaining_files}"

    # move images
    for path_i in range(len(images_list)):
        # access original image
        original_path = os.path.join(input_path, images_list[path_i])

        # Convert to jpg and change file name to index
        filename = "new_" + str(path_i) + ".jpg"
        os.rename(original_path, os.path.join(input_path, filename))
        new_path = os.path.join(input_path, filename)

        # Move image to labeler folder
        shutil.move(
            new_path, os.path.join(image_output_dirs[path_i % num_groups], filename)
        )

    # move videos
    for path_i in range(len(videos_list)):
        # access original video
        original_path = os.path.join(input_path, videos_list[path_i])

        # Convert to .MOV and change file name to index
        filename = str(path_i) + ".MOV"
        os.rename(original_path, os.path.join(input_path, filename))
        new_path = os.path.join(input_path, filename)

        # Move video to labeler folder
        shutil.move(
            new_path, os.path.join(video_output_dirs[path_i % num_groups], filename)
        )


if __name__ == "__main__":
    distribute_videos_and_images()
