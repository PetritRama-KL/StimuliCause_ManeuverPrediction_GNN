import torch
import os 
import random
import shutil
import numpy as np
import matplotlib.pyplot as plt

import cv2

from sklearn.model_selection import train_test_split

import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm

from PIL import Image, ImageDraw




datasetDir = '/raid2/Petrit/datasets/honda/HDD'

clause2ID = {
            "congestion"                            : 0,
            "sign"                                  : 1,
            "red_light"                             : 2,
            "crossing_vehicle"                      : 3,
            "parked_vehicle"                        : 4,
            "yellow_light"                          : 5,
            "crossing_pedestrian"                   : 6,
            "merging_vehicle"                       : 7,
            "on-road_bicyclist"                     : 8,
            "pedestrian_near_ego_lane"              : 9,
            "park"                                  : 10,
            "on-road_motorcyclist"                  : 11,
            "vehicle_cut-in"                        : 12,
            "road_work"                             : 13,
            "turning_vehicle"                       : 14,
            "vehicle_passing_with_lane_departure"   : 15,
            "other"                                 : 16,
            "none"                                  : 17,
}




def moveImages2Folder(filesList, dstFolder):
    for f in filesList:
        try:
            shutil.copy(f, dstFolder+'/'+f.split('/')[8]+'_'+f.split('/')[9])
        except:
            print(f)
            assert False




def moveLabels2Folder(filesList, dstFolder):
    for f in filesList:
        try:
            shutil.copy(f, dstFolder+'/'+f.split('/')[8]+'_'+f.split('/')[9][:-10]+'.txt')
        except:
            print(f)
            assert False




def plotBBox(image, annotation):
    width, height = image.size
    plottedImg = ImageDraw.Draw(image)

    annot = np.array(annotation)
    annotT = np.copy(annot)

    annotT[:,[1,3]] = annot[:,[1,3]]*width
    annotT[:,[2,4]] = annot[:,[2,4]]*height

    annotT[:,1] = annotT[:,1] - (annotT[:,3]/2)
    annotT[:,2] = annotT[:,2] - (annotT[:,4]/2)
    annotT[:,3] = annotT[:,1] + annotT[:,3]
    annotT[:,4] = annotT[:,2] + annotT[:,4]

    for ann in annotT:
        objClass, x0, y0, x1, y1 = ann

        plottedImg.rectangle(((x0,y0), (x1,y1)))
        plottedImg.text((x0, y0 - 10), list(clause2ID.keys())[list(clause2ID.values()).index(int(objClass))])

    #plt.imshow(np.array(plottedImg))
    #plt.show()

    image.save('test.jpg')
    print('test')



causeFolder = datasetDir+'/release_2020_07_15_causal_reasoning/Cause_labeled_images/'
causeFolderList  = [f.path for f in os.scandir(causeFolder) if os.path.isdir(f)]

imageList = []
labelList = []
for folder in causeFolderList:
    fileList = [f.path for f in os.scandir(folder) if os.path.isfile(f)]
    for file in fileList:
        if file[-4:]==".jpg":
            imageList.append(file)
        elif file[-9:]=="yolo5.txt":
            labelList.append(file)

imageList2 = sorted(imageList, key=lambda x: (x.split('/')[8], x.split('/')[9][:-4]))
labelList2 = sorted(labelList, key=lambda x: (x.split('/')[8], x.split('/')[9][:-10]))

trainImages, valImages, trainLabels, valLabels = train_test_split(imageList2, labelList2, test_size=0.2, random_state=1)
valImages, testImages, valLabels, testLabels = train_test_split(valImages, valLabels, test_size=0.5, random_state=1)

moveImages2Folder(trainImages,   datasetDir+'/yolov5/images/train')
moveImages2Folder(valImages,     datasetDir+'/yolov5/images/val/')
moveImages2Folder(testImages,    datasetDir+'/yolov5/images/test/')
moveLabels2Folder(trainLabels,   datasetDir+'/yolov5/labels/train/')
moveLabels2Folder(valLabels,     datasetDir+'/yolov5/labels/val/')
moveLabels2Folder(testLabels,    datasetDir+'/yolov5/labels/test/')

print('test')
