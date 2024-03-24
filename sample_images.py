import shutil
import os
import random

CURRENT_DIR = os.getcwd()
OUTPUT_DIR = "input_images"
INPUT_DIR = "background_LSUI_input"


def sample_files(num_to_sample: int) -> None:
    """Sample some number of files randomly from an input directory to be labeled

    Args:
        total (int): number of files to pull from the input directory
    """
    # define inputs
    input_path = os.path.join(CURRENT_DIR, INPUT_DIR)
    files_list = os.listdir(input_path)

    assert num_to_sample <= len(
        files_list
    ), "You are trying to sample more files than are present in your input directory, try lowering your sample quantity or changing your input directory."

    # shuffle files list
    random.seed(42)
    random.shuffle(files_list)

    # define outputs
    output_dir = os.path.join(CURRENT_DIR, OUTPUT_DIR)

    # move images
    for path_i in range(num_to_sample):
        # access original image
        original_path = os.path.join(input_path, files_list[path_i])

        # Move image to distribution folder
        shutil.move(original_path, os.path.join(output_dir, files_list[path_i]))


if __name__ == "__main__":
    sample_files(100)
