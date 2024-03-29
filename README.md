# urchin-detection
Purple sea urchin detection for Apsis clinic project


## Setup steps
1. Clone this repository
2. Create and activate a Python environment using venv
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Download all images and labels from google drive. Put in a download folder with `images` and `labels` subfolders
4. [IF YOU WANT TO RUN YOLOv5] Clone the YOLOv5 repo and install requirements
   `git clone https://github.com/ultralytics/yolov5`
   `pip install -U -r yolov5/requirements.txt`
5. [IF YOU WANT TO RUN YOLOv8] Clone the YOLOv8 repo using `git clone https://github.com/ultralytics/ultralytics.git`


## Steps for training the model

1. First, activate your virtual environment using the command `source venv/bin/activate`, replacing `venv` with the name of your virtual environment
2. Run `data.py`. This will take your input images and labels and organize them into the proper folders for YOLO training and evaluation. If you've run `data.py` before, your previous data folder will be deleted and replaced with a the new one. Run `python3 data.py -h` to see information about what flags you can pass to run the model on different datasets. 
3. To run the YOLOv5 model: `python3 yolov5/train.py --data urchins.yaml --weights yolov5s.pt --epochs 50 --batch 4 --freeze 10`
4. To run the YOLOv8 model: `yolo detect train data='[INSERT ABSOLUTE PATH TO YOUR 'urchins.yaml' FILE HERE]' model=yolov8n.pt epochs=50 batch=4 freeze=10`

## Steps for running model inference

1. To get predictions for a particular image or video, run `yolo predict model='[PATH TO TRAINED MODEL .pt FILE]' source='[PATH TO IMAGE OR VIDEO TO RUN INFERENCE ON]`. Can use `show` parameter to see inference being done in real time.
2. To run object tracking on a particular video, run `yolo track model='[PATH TO TRAINED MODEL .pt FILE]' source='[PATH TO VIDEO TO RUN TRACKING ON]`. Can also set `conf=[SOME FLOAT], iou=[SOME FLOAT]` and the `show` parameter allows you to see the video inference being done in real time.

## References
YOLOv5 transfer learning based on tutorial: https://kikaben.com/yolov5-transfer-learning-dogs-cats/#yolov5-transfer-learning-execution
Object tracking from: https://docs.ultralytics.com/modes/track/