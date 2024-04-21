# urchin-detection
Purple sea urchin detection for Apsis clinic project

## YOLOv9

### Setup steps
1. Clone this repository
2. Create and activate a Python environment using venv
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Download all images and labels from google drive. Put in a download folder with `images` and `labels` subfolders
4. Clone the YOLOv9 repo using `git clone https://github.com/WongKinYiu/yolov9.git`

### Steps for training the YOLOv8 model

1. First, activate your virtual environment using the command `source venv/bin/activate`, replacing `venv` with the name of your virtual environment
2. Run `data.py`. This will take your input images and labels and organize them into the proper folders for YOLO training and evaluation. If you've run `data.py` before, your previous data folder will be deleted and replaced with a the new one. Run `python3 data.py -h` to see information about what flags you can pass to run the model on different datasets. 
3. To run the YOLOv9 model: `yolo detect train model=[ABSOLUTE PATH TO FILE 'yolov9c.pt] data = [ABSOLUTE PATH TO FILE 'urchins.yaml'] epochs = 50 batch = 2`. Note that batch = 3 may also work, but batch = 4 crashes the clinic computer (but may work on a more computationally efficient system).

## YOLOv8

### Setup steps
1. Clone this repository
2. Create and activate a Python environment using venv
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Download all images and labels from google drive. Put in a download folder with `images` and `labels` subfolders
4. Clone the YOLOv8 repo using `git clone https://github.com/ultralytics/ultralytics.git`
5. Install requirements for yOLOv8 `pip install ultralytics`


### Steps for training the YOLOv8 model

1. First, activate your virtual environment using the command `source venv/bin/activate`, replacing `venv` with the name of your virtual environment
2. Run `data.py`. This will take your input images and labels and organize them into the proper folders for YOLO training and evaluation. If you've run `data.py` before, your previous data folder will be deleted and replaced with a the new one. Run `python3 data.py -h` to see information about what flags you can pass to run the model on different datasets. 
3. To run the YOLOv8 model: `yolo detect train data='[INSERT ABSOLUTE PATH TO YOUR 'urchins.yaml' FILE HERE]' model=yolov8n.pt epochs=50 batch=4 freeze=10`


### Steps for running model inference on an image, video, or stream

1. To get predictions for a particular image or video, run `yolo predict model='[PATH TO TRAINED MODEL .pt FILE]' source='[PATH TO IMAGE, PATH TO VIDEO, OR STREAM ID TO RUN INFERENCE ON]`. Can use `show` parameter to see inference being done in real time.
2. To run object tracking on a particular video, run `yolo track model='[PATH TO TRAINED MODEL .pt FILE]' source='[PATH TO IMAGE, PATH TO VIDEO, OR STREAM ID TO RUN TRACKING ON]`. Can also set `conf=[SOME FLOAT], iou=[SOME FLOAT]` and the `show` parameter allows you to see the video inference being done in real time.

Example running command for robot video stream `yolo track model='/Users/fwright/Library/CloudStorage/GoogleDrive-fwright@g.hmc.edu/My Drive/clinic/urchin-detection/runs/detect/train26/weights/best.pt' source='rtsp://192.168.2.2:8554/video_udp_stream_0' show`


## YOLOv5

### Setup steps
1. Clone this repository
2. Create and activate a Python environment using venv
   `python3 -m venv venv`
   `source venv/bin/activate`
3. Download all images and labels from google drive. Put in a download folder with `images` and `labels` subfolders
4. Clone the YOLOv5 repo and install requirements
   `git clone https://github.com/ultralytics/yolov5`
   `pip install -U -r yolov5/requirements.txt`


### Steps for training the YOLOv5 model

1. First, activate your virtual environment using the command `source venv/bin/activate`, replacing `venv` with the name of your virtual environment
2. Run `data.py`. This will take your input images and labels and organize them into the proper folders for YOLO training and evaluation. If you've run `data.py` before, your previous data folder will be deleted and replaced with a the new one. Run `python3 data.py -h` to see information about what flags you can pass to run the model on different datasets. 
3. To run the YOLOv5 model: `python3 yolov5/train.py --data urchins.yaml --weights yolov5s.pt --epochs 50 --batch 4 --freeze 10`

## References
YOLOv5 transfer learning based on tutorial: https://kikaben.com/yolov5-transfer-learning-dogs-cats/#yolov5-transfer-learning-execution
Object tracking from: https://docs.ultralytics.com/modes/track/
