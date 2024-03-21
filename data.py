import os
import glob
import numpy as np
import shutil
import cv2
import argparse

import xml.etree.ElementTree as ET

from pathlib import Path
from PIL import Image


# change this to whatever you named your download folder with the input images and labels
# NOTE: this must have "images" and "labels" subfolders. The images subfolder may also contain videos
IMAGE_DOWNLOAD_DIR = "urchin_download"

# all people who have labeled images, the names of the corresponding download folders
# NOTE: do not change unless someone new labels images
LABELERS = ["Brittany", "Castor", "Eliza", "Francine", "James", "Katie", "Ryan"]

# all labeling rounds you want to use in training
# NOTE: must be updated when new groups of labels are done
IMAGE_SUBFOLDERS = [
    "google_0",
    "sean_nov_3/images",
    "google_negative_1/images",
    "background_lsui_0/images",
]

VIDEO_SUBFOLDERS = ["sean_nov_3/videos"]


def make_yolo_folders() -> None:
    """Generates folder structure for YOLOv5 images and labels for train, val, and test"""
    data_path = Path("data")
    if os.path.exists(data_path) and os.path.isdir(data_path):
        shutil.rmtree(data_path)

    for folder in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            os.makedirs(f"data/{folder}/{split}")


def get_urchin_image_folders(
    directory: str, labelers: list[str], image_subfolders: list[str]
) -> list[str]:
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


def get_urchin_video_folders(
    directory: str, labelers: list[str], video_subfolders: list[str]
) -> list[str]:
    """Returns all folders with valid videos in them.

    Args:
        directory (str): base directory containing all videos, relative path from repo base
        labelers (list[str]): list of all labelers
        video_subfolders (list[str]): list of all valid labeler subfolders which contain videos directly

    Returns:
        list[str]: list of paths which exist and contain videos, relative paths from repo base
    """
    paths = []
    for labeler in labelers:
        for video_folder in video_subfolders:
            path = os.path.join(directory, labeler, video_folder)
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
        label_path[1] = "labels"
        if label_path[-1] == "images":
            del label_path[-1]
            label_path[-1] = label_path[-1] + "_images"

        label_path = os.path.join(*label_path)

        label_folders.append(label_path)

    return label_folders


def polygon_to_box(label_path: str) -> None:
    """Converts polygons from a CVAT 1.1 XML file into bounding boxes, in YOLO format. Requires
    annotations.xml to be placed in the normal YOLO directory structure (google_0, google_negative_1, etc.)

    Args:
        label_path (str): Path of the folder containing the labels
    """
    annotation_path = [label_path, "annotations.xml"]
    annotation_path = os.path.join(*annotation_path)

    text_folder_path = [label_path, "obj_train_data"]
    text_folder_path = os.path.join(*text_folder_path)

    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Go through each image
    for image in root:
        if image.tag == "image":
            image_name = image.get("name")
            image_name = image_name[:-3]
            image_name = image_name + "txt"
            text_file_path = [text_folder_path, image_name]
            text_file_path = os.path.join(*text_file_path)
            h = eval(image.get("height"))
            w = eval(image.get("width"))

            label_string = ""

            # Parse each polygon
            for polygon in image:
                x_list = []
                y_list = []

                # Mask
                if "left" in polygon.attrib:
                    left = eval(polygon.get("left")) / w
                    top = eval(polygon.get("top")) / h
                    box_w = eval(polygon.get("width")) / w
                    box_h = eval(polygon.get("height")) / h
                    x_mid = left + box_w / 2.0
                    y_mid = top + box_h / 2.0

                    # Add current polygon label string to larger string
                    if polygon.get("label") == "Purple Sea Urchin":
                        label_string = (
                            label_string
                            + "0 "
                            + str(x_mid)
                            + " "
                            + str(y_mid)
                            + " "
                            + str(box_w)
                            + " "
                            + str(box_h)
                            + "\n"
                        )
                    else:
                        label_string = (
                            label_string
                            + "1 "
                            + str(x_mid)
                            + " "
                            + str(y_mid)
                            + " "
                            + str(box_w)
                            + " "
                            + str(box_h)
                            + "\n"
                        )

                # Polygon
                elif "points" in polygon.attrib:

                    point_string = polygon.get("points")
                    point_list = point_string.split(";")
                    for i in range(len(point_list)):
                        point_list[i] = point_list[i].split(",")
                    for x in point_list:
                        x_list.append(x[0])
                        y_list.append(x[1])

                    # Convert data into float, range 0-1
                    x_list = [eval(x) for x in x_list]
                    y_list = [eval(x) for x in y_list]
                    x_list = [x / w for x in x_list]
                    y_list = [x / h for x in y_list]

                    # Calculate all x, y, w, h of the bounding box
                    xMin = min(x_list)
                    xMax = max(x_list)
                    yMin = min(y_list)
                    yMax = max(y_list)

                    x_mid = (xMax + xMin) / 2.0
                    y_mid = (yMax + yMin) / 2.0
                    box_w = xMax - xMin
                    box_h = yMax - yMin

                    # Add current polygon label string to larger string
                    if polygon.get("label") == "Purple Sea Urchin":
                        label_string = (
                            label_string
                            + "0 "
                            + str(x_mid)
                            + " "
                            + str(y_mid)
                            + " "
                            + str(box_w)
                            + " "
                            + str(box_h)
                            + "\n"
                        )
                    else:
                        label_string = (
                            label_string
                            + "1 "
                            + str(x_mid)
                            + " "
                            + str(y_mid)
                            + " "
                            + str(box_w)
                            + " "
                            + str(box_h)
                            + "\n"
                        )

            # Add label string to file
            with open(text_file_path, "a+") as text_file:
                text_file.writelines(label_string)

    # Rename file when done to prevent reuse
    annotations_used_path = [label_path, "annotationsUsed.xml"]
    annotations_used_path = os.path.join(*annotations_used_path)
    os.rename(annotation_path, annotations_used_path)


def get_filenames(folder: str, is_label: bool = False, is_video: bool = False) -> set:
    """Gets all valid filenames in a given folder

    Args:
        folder (str): folder to get filenames from
        is_label (bool): whether or not the folder is for label files
        is_video (bool): whether of not the folder is for video files

    Returns:
        set: all valid filename strings from the folder, contains the folder in the path
    """
    if is_label:
        folder = os.path.join(folder, "obj_train_data")
        extension = "*.txt"
    elif is_video:
        extension = "*.MOV"
    else:
        extension = "*.jpg"

    filenames = set()

    for path in glob.glob(os.path.join(folder, extension)):
        filenames.add(path)

    return filenames


def get_all_image_filenames(folders: list[str]) -> set:
    """Gets all valid image filenames with labels for all folders

    Args:
        folders (str]): all folders to get images from

    Returns:
        set: all valid filename strings with labels, relative paths from repo base directory
    """
    filenames = set()

    for folder in folders:
        folder_filenames = get_filenames(folder, is_label=False)
        filenames.update(folder_filenames)

    # check to make sure the filenames have corresponding labels
    unlabeled_filenames = set()
    for file in filenames:
        if not os.path.exists(get_image_label_filename(file)):
            unlabeled_filenames.add(file)
    filenames = filenames - unlabeled_filenames

    return filenames


def get_all_video_filenames(folders: list[str]) -> set:
    """Gets all valid video filenames with labels for all folders

    Args:
        folders (str]): all folders to get videos from

    Returns:
        set: all valid filename strings with labels, relative paths from repo base directory
    """
    filenames = set()

    for folder in folders:
        folder_filenames = get_filenames(folder, is_video=True)
        filenames.update(folder_filenames)

    # check to make sure the filenames have corresponding labels
    unlabeled_filenames = set()
    for file in filenames:
        if not os.path.exists(get_video_label_folder(file)):
            unlabeled_filenames.add(file)
    filenames = filenames - unlabeled_filenames

    return filenames


def standardize_labels(label_folders: list[str]) -> None:
    """Standardize all labels to be bounding boxes

    Args:
        label_folders (list[str]):  List of all label folders to standardize
    """
    for label_path in label_folders:
        # Check if we exported an annotations file
        annotation_path = [label_path, "annotations.xml"]
        annotation_path = os.path.join(*annotation_path)

        # Convert polygons to bounding boxes
        if os.path.isfile(annotation_path):
            polygon_to_box(label_path)


def standardize_classes(label_folders: list[str]) -> None:
    """Change all .names files and label files in place to reflect the classes being ['Purple Sea Urchin\n', 'Other Sea Urchin\n']

    Args:
        label_folders (list[str]): List of all label folders to standardize
    """
    for label_folder in label_folders:
        class_names_filepath = os.path.join(label_folder, "obj.names")

        if os.path.exists(class_names_filepath):
            # reads in classes and strips off newline characters and trailing whitespace
            with open(class_names_filepath, "r") as class_fp:
                input_classes = class_fp.readlines()

            # classes are two different options, make sure they're the correct options and the only variation is ordering
            proper_classes = ["Purple Sea Urchin\n", "Other Sea Urchin\n"]
            flipped_classes = ["Other Sea Urchin\n", "Purple Sea Urchin\n"]
            assert input_classes in [
                proper_classes,
                flipped_classes,
            ], f"Class labels are not 'Purple Sea Urchin' and 'Other Sea Urchin' in {label_folder}"

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


def get_image_label_filename(image_filename: str) -> str:
    """Get label filename corresponding to an input image filename

    Args:
        image_filename (str): Path to an image

    Returns:
        str: Path to label file corresponding to the image
    """
    # get path split by folders into a list
    label_path = image_filename.split(os.sep)

    # make sure input is an image
    assert ".jpg" in label_path[-1], "input is not an image"

    # get corresponding label path for image, relies on particular file structure
    label_path[-1] = label_path[-1].replace(".jpg", ".txt")
    label_path[1] = "labels"
    if label_path[-2] == "images":
        del label_path[-2]
        label_path[-2] = label_path[-2] + "_images"
    label_path.insert(-1, "obj_train_data")

    label_path = os.path.join(*label_path)

    return label_path


def get_video_label_filename(video_frame_filename: str) -> str:
    """Get label filename corresponding to an input video frame

    Args:
        video_frame_filename (str): Path to a video frame

    Returns:
        str: Path to label file corresponding to the video frame
    """

    label_path = video_frame_filename.split(os.sep)

    # make sure input is an image frame
    assert ".jpg" in label_path[-1], "input is not an image"

    label_path[-1] = label_path[-1].replace(".jpg", ".txt")
    label_path[-2] = "obj_train_data"

    label_path = os.path.join(*label_path)

    return label_path


def get_video_label_folder(video_filename: str) -> str:
    """Get label folder corresponding to an input video filename

    Args:
        video_filename (str): Path to an video

    Returns:
        str: Path to label folder corresponding to the video
    """
    # get path split by folders into a list
    label_path = video_filename.split(os.sep)

    # get corresponding label path for video, relies on particular file structure
    label_path[-1] = label_path[-1].replace(".MOV", "")

    label_path[1] = "labels"
    if label_path[-2] == "videos":
        del label_path[-2]
        label_path[-2] = label_path[-2] + "_videos"

    label_path = os.path.join(*label_path)

    return label_path


def get_video_label_folders(video_filenames: list[str]) -> list[str]:
    """Get label folders corresponding to a list of input video filenames

    Args:
        video_filename (list[str]): List of paths to videos

    Returns:
        list[str]: List of paths to label folders corresponding to the input videos
    """
    label_folders = [get_video_label_folder(filename) for filename in video_filenames]
    return label_folders


def get_frame_root_filename(frame_num: int) -> str:
    """gets the root filename (no extension) given the frame number

    Args:
        frame_num (int): frame number

    Returns:
        str: root filename (no extension)
    """
    return "frame_{:0>6d}".format(frame_num)


def get_num_frames_in_video(video_path: str) -> int:
    """Gets the total number of frames in an input video

    Args:
        video_path (str): path to the video to get frames from

    Returns:
        int: total number of frames in the input video
    """
    # using https://pyimagesearch.com/2017/01/09/count-the-total-number-of-frames-in-a-video-with-opencv-and-python/
    video = cv2.VideoCapture(video_path)
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # release the video file pointer
    video.release()

    return total


def pull_frames_from_video(
    video_path: str, num: int, use_total: bool = False
) -> list[str]:
    """Select frames from video and save them in a frames folder in the label directory

    Args:
        video_path (str): path to video to pull frames from
        num (int): if use_total, pull every num frames
                    else pull num total frames
        use_total (bool, optional): whether num is the total number of frames or the
                    number to index by. Defaults to False.

    Returns:
        list[str]: list of frames saved
    """
    total_frames = get_num_frames_in_video(video_path)
    label_folder = get_video_label_folder(video_path)
    with open(os.path.join(label_folder, "train.txt")) as fp:
        total_lines = len(fp.readlines())

    assert total_frames == total_lines

    if use_total:
        # make sure there are enough frames in the video to get this total number
        assert num <= total_frames

        # convert num from a total to number of frames between frames to get
        num = total_frames // num

    cap = cv2.VideoCapture(video_path)
    count = 0

    frame_folder = os.path.join(label_folder, "frames")
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    frames_list = []

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            filename = os.path.join(
                frame_folder, get_frame_root_filename(count) + ".jpg"
            )
            frames_list.append(filename)
            worked = cv2.imwrite(filename, frame)
            assert worked

            count += num
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        else:
            cap.release()
            break

    return frames_list


def split_data(
    image_filenames: set,
    video_filenames: set,
    train_prop: float,
    val_prop: float,
    use_video: bool = True,
    use_total: bool = True,
    frame_num: int = 10,
) -> None:
    """Splits all images, videos and labels across train, val and test folders

    Args:
        image_filenames (set): set of all image filenames as strings
        video_filenames (set): set of all video filenames as strings
        train_prop (float): proportion of total data we want to use for training
        val_prop (float): proportion of total data we want to use for validation
        use_video (bool, optional): Whether or not to use video data. Defaults to True.
        video_total_frames (int, optional): Total number of frames to pull from each
            video. Will not be used if use_video=False. Defaults to 10.
    """
    urchin_images = np.array(list(image_filenames))

    # shuffle data
    np.random.seed(42)  # for reproducability
    np.random.shuffle(urchin_images)

    total_image_count = urchin_images.shape[0]

    train_image_count = int(train_prop * total_image_count)
    val_image_count = int(val_prop * total_image_count)

    # distribute images
    for i, image_path in enumerate(urchin_images):
        label_path = get_image_label_filename(image_path)

        # Split into train, val, or test
        if i < train_image_count:
            split = "train"
        elif i < train_image_count + val_image_count:
            split = "val"
        else:
            split = "test"

        # make sure all of our paths exist
        assert os.path.exists(image_path), f"Image path {image_path} does not exist"
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"

        # Destination paths
        destination_filename = image_path.split(os.sep)
        destination_image_filename = (
            destination_filename[2]
            + "_"
            + destination_filename[3]
            + "_"
            + destination_filename[-1]
        )
        destination_label_filename = destination_image_filename.replace(".jpg", ".txt")

        target_image_folder = f"data/images/{split}/{destination_image_filename}"
        target_label_folder = f"data/labels/{split}/{destination_label_filename}"

        # Copy files
        shutil.copy(image_path, target_image_folder)
        shutil.copy(label_path, target_label_folder)

    if use_video:
        urchin_videos = np.array(list(video_filenames))
        np.random.shuffle(urchin_videos)

        total_video_count = urchin_videos.shape[0]

        train_video_count = int(train_prop * total_video_count)
        val_video_count = int(val_prop * total_video_count)

        # distribute videos
        total_video_frames = 0
        for i, video_path in enumerate(urchin_videos):
            assert os.path.exists(video_path), f"Video path {video_path} does not exist"

            frames_list = pull_frames_from_video(
                video_path, frame_num, use_total=use_total
            )
            labels_list = [get_video_label_filename(frame) for frame in frames_list]

            # Split into train, val, or test
            if i < train_video_count:
                split = "train"
            elif i < train_video_count + val_video_count:
                split = "val"
            else:
                split = "test"

            # make sure all of our frame paths exist
            assert all(
                [os.path.exists(frame) for frame in frames_list]
            ), f"Frame path in {frames_list} does not exist"

            # Destination paths
            destination_filenames = [frame.split(os.sep) for frame in frames_list]

            destination_image_filenames = [
                destination_filename[2]
                + "_"
                + destination_filename[3]
                + "_"
                + destination_filename[4]
                + "_"
                + destination_filename[-1]
                for destination_filename in destination_filenames
            ]

            destination_label_filenames = [
                destination_image_filename.replace(".jpg", ".txt")
                for destination_image_filename in destination_image_filenames
            ]

            target_image_folders = [
                f"data/images/{split}/{destination_image_filename}"
                for destination_image_filename in destination_label_filenames
            ]
            target_label_folders = [
                f"data/labels/{split}/{destination_label_filename}"
                for destination_label_filename in destination_label_filenames
            ]

            # Copy files
            total_video_frames += len(frames_list)
            for i in range(len(target_image_folders)):
                shutil.copy(frames_list[i], target_image_folders[i])
                shutil.copy(labels_list[i], target_label_folders[i])
        print(f"{total_video_frames} video frames in dataset")


def main(args):
    image_dir = os.path.join(IMAGE_DOWNLOAD_DIR, "images")

    make_yolo_folders()

    image_folders = get_urchin_image_folders(image_dir, LABELERS, IMAGE_SUBFOLDERS)
    image_filenames = get_all_image_filenames(image_folders)
    image_label_folders = get_urchin_label_folders(image_folders)

    standardize_classes(image_label_folders)
    standardize_labels(image_label_folders)

    print(f"{len(image_filenames)} images in the dataset")

    if args.use_video:
        video_folders = get_urchin_video_folders(image_dir, LABELERS, VIDEO_SUBFOLDERS)
        video_filenames = get_all_video_filenames(video_folders)
        video_label_folders = get_video_label_folders(video_filenames)

        standardize_classes(video_label_folders)
    else:
        video_filenames = []

    split_data(
        image_filenames,
        video_filenames,
        train_prop=0.6,
        val_prop=0.2,
        use_video=args.use_video,
        use_total=args.use_total,
        frame_num=args.frame_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--use_video",
        help="whether to include video data in the split",
        action="store_true",
    )
    parser.add_argument(
        "-t",
        "--use_total",
        help="whether to use a total number of frames per video, or pull frames skipping some number",
        action="store_true",
    )
    parser.add_argument(
        "--frame_num",
        help="total number of frames to pull from each video, or number of frames to skip between pulls from a video",
        type=int,
    )
    args = parser.parse_args()

    main(args)
