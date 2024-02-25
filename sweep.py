# Here is the command I have been running with
# python3 sweep.py --data urchins.yaml --weights yolov5s.pt --epochs 5 --batch 4 --freeze 10 --project yolov5/runs/sweep/degrees

from yolov5 import train
from pathlib import Path
import numpy as np
import os
import yaml

def createHyp(params, hypPath):
    """Creates a new file of hyperparameters at the given path with the desired changed parameters

    Args: 
        params (dict): dictionary of parameters to change from the default parameters
        hypPath (str): desired path of the new parameter file
    """

    # set up initial parameters
    initialFile = 'yolov5/data/hyps/hyp.scratch-low.yaml'
    with open(initialFile, 'r') as file:
        initialParams = yaml.safe_load(file)

    # replace all desired parameters
    for i in params:
        initialParams[i] = params[i]

    # write to new file
    with open(hypPath, 'w') as file:
        yaml.dump(initialParams, file)

def createSweep(param, low, high, quantity, hypDirectory):
    """Creates all hyperparameter yaml files for the desired sweep

    Args:
        param (str): parameter to be swept
        low (int): lowest value of the parameter
        high (int): highest value of the parameter
        quantity (int): number of files to create
        hypDirectory (str): directory for the files to be created in
    """

    for i in np.linspace(low, high, quantity):
        paramDict = {param : float(i)}
        hypPath = hypDirectory + '/hyp_' + param + str(i) + '.yaml' 
        createHyp(paramDict, hypPath)


def main(opt):

    hypDirectory = 'hyps'
    
    # set up all the yaml files
    createSweep('degrees', 0, 20, 21, hypDirectory)

    # run the model for each yaml file in the hyperparameter directory
    files = os.listdir(hypDirectory)
    files.sort()

    f = open(str(opt.project) + '/results.csv', 'w')
    headerString = "Class" + ',' + "Images" + ',' + "Instances" + ',' + "P" + ',' + "R" + ',' + "mAP50" + ',' + "mAP50-95" + '\n'
    f.writelines(headerString)
    f.close()

    for file in files:
        if file[:3] == 'hyp':
            opt.hyp = Path(hypDirectory + '/' + file)
            opt.name = file[:-5]
            opt.save_dir = opt.project
            train.main(opt)

if __name__ == '__main__':
    opt = train.parse_opt()
    main(opt)