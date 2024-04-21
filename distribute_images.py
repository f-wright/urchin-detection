import shutil
import os
import argparse
import cv2

from PIL import Image
from pathlib import Path

LABELERS = ["Brittany", "Castor", "Eliza", "Francine", "James", "Ryan"]
INPUT_DIR = "input_images"
OUTPUT_DIR = "images_to_label"
FILE_CATEGORY = "robot"
CURRENT_DIR = os.getcwd()
VIDEO_EXTENSIONS = {".mov", ".MOV"}
IMAGE_EXTENSIONS = {".JPG", ".jpg", ".png", ".avif", ".webp", ".gif", ".jpeg"}


def save_frames_from_video(
    video_path: str, output_dir: str, num: int, use_total: bool = False
) -> None:
    """Select frames from video and save them in a frames folder for labeling

    Args:
        video_path (str): path to video to pull frames from
        output_dir (str): path to save frames to
        num (int): if use_total, pull every num frames
                    else pull num total frames
        use_total (bool, optional): whether num is the total number of frames or the
                    number to index by. Defaults to False.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if use_total:
        # make sure there are enough frames in the video to get this total number
        assert num <= total_frames

        # convert num from a total to number of frames between frames to get
        num = total_frames // num

    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            if not os.path.exists(
                os.path.join(output_dir, video_path.split("/")[-1][:-4])
            ):
                os.mkdir(os.path.join(output_dir, video_path.split("/")[-1][:-4]))
            filename = os.path.join(
                output_dir, video_path.split("/")[-1][:-4], str(count) + ".jpg"
            )
            worked = cv2.imwrite(filename, frame)
            assert worked, f"Couldn't save to {filename}"

            count += num
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break

    os.remove(video_path)


def distribute_videos_and_images(distribute_frames: bool):
    """Distribute images and videos into labeler folders for labeling

    Args:
        distribute_frames (bool): Whether to distribute frames from videos instead of
            whole videos
    """
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
    frame_output_dirs = [os.path.join(labeler, "frames") for labeler in output_dirs]
    num_groups = len(LABELERS)

    # make directories if they don't already exist
    for image_dir in image_output_dirs:
        Path(image_dir).mkdir(parents=True, exist_ok=True)
    for video_dir in video_output_dirs:
        Path(video_dir).mkdir(parents=True, exist_ok=True)
    for frame_dir in frame_output_dirs:
        Path(frame_dir).mkdir(parents=True, exist_ok=True)

    # split image and video files
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
        if distribute_frames:
            save_frames_from_video(
                new_path, frame_output_dirs[path_i % num_groups], 20, use_total=True
            )
        else:
            shutil.move(
                new_path, os.path.join(video_output_dirs[path_i % num_groups], filename)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--distribute_frames",
        help="whether to distribute frames from a video instead of the video itself",
        action="store_true",
    )
    args = parser.parse_args()
    distribute_videos_and_images(args.distribute_frames)
