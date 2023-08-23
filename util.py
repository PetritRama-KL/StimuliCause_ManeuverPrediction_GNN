import numpy as np
import pickle
import cv2
import os
import collections

from PIL import Image
import urllib
import shutil
from math import dist
import tqdm
import pandas as pd
import seaborn as sns
import scipy.io as sio
import operator
import base64
from io import BytesIO

import torch
from torchvision import ops

import logging
import random

import dgl
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from skimage import data
from skimage import io

import sys
sys.path.insert(0, './yolov5')



targetLbl = [   
                "Intersection Passing", # 0
                "Left Turn",            # 1
                "Right Turn",           # 2
                "Left Lane Change",     # 3
                "Right Lane Change",    # 4
                "Left Lane Branch",     # 5
                "Right Lane Branch",    # 6
                "Crosswalk Passing",    # 7
                "Railroad Passing",     # 8
                "Merge",                # 9
                "U-turn",               # 10
                "Keep the Lane*"        # 11
            ]   


labelsLbl = ["", "intersection passing", "merge", "left lane change", "right lane branch", "right lane change", 
        "intersection passing", "left turn", "crosswalk passing", "park", "railroad passing", "left lane branch", "U-turn", 
        "park park", "", "Park park", "congestion", "Sign", "red light", "crossing vehicle", "Parked vehicle", "yellow light", 
        "crossing pedestrian", "merging vehicle", "on-road bicyclist", "pedestrian near ego lane", "park", "on-road motorcyclist", 
        "vehicle cut-in", "road work", "turning vehicle", "vehicle passing with lane departure", "downtown", "freeway", "tunnel",
        "red light", "on-road motorcyclist", "vehicle cut-in", "crossing vehicle", "Sign", "merging vehicle", "congestion", 
        "Parked vehicle", "yellow light", "crossing pedestrian", "road work", "on-road bicyclist", "pedestrian near ego lane", 
        "vehicle passing with lane departure", "start", "end", "roundabout", "", "highway exit", 
        "While turning left, the frontal vehicle is also turning left slowly as there is a crossing pedestrian", 
        "While U-turning, the ego car is waiting for oncoming vehicles", "atypical", "driveway", "Long Merge", "Atypical", 
        "Roundabout", "Ambiguous as it can interpret as left lane branch", "ramp", "vehicle on the hilly road", 
        "atypical T-intersection", "wheelchair", "hard example", "curve road", "curved road", "very long right turn", 
        "or right lane change", "speical situation", "hump", "crossing vehicle", "on-road bicyclist", "crossing pedestrian", 
        "merging vehicle", "Parked vehicle", "red light", "vehicle cut-in", "yellow light", "on-road motorcyclist", 
        "road work", "Sign", "pedestrian near ego lane", "stop 4 congestion", "stop 4 sign", "stop 4 light", "Avoid parked car", 
        "stop 4 pedestrian", "Stop for others", "Avoid pedestrian near ego lane", "Avoid on-road bicyclist", "Avoid TP", "empty",
        "keep the lane*", "Go*", "other*"]

areaLbl = ['Downtown', 'Freeway', 'Tunnel']

causeLbl = [
            "Congestion ",                           # 0
            "Sign ",                                 # 1
            "Red Light ",                            # 2
            "Crossing Vehicle ",                     # 3 
            "Parked Vehicle ",                       # 4
            "Yellow Light ",                         # 5
            "Crossing Pedestrian ",                  # 6
            "Merging Vehicle ",                      # 7
            "On-road Bicyclist ",                    # 8
            "Pedestrian Near Ego Lane ",             # 9 
            "Park ",                                 # 10
            "On-road Motorcyclist ",                 # 11
            "Vehicle Cut-in",                        # 12
            "Road Work ",                            # 13
            "Turning Vehicle ",                      # 14
            "Vehicle Departing Lane ",               # 15
            "Other ",                                # 16
            "None "                                  # 17
            ]
causeLbl.reverse()

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


stimuliLbl = [
                "Stop  ",     # 0
                "Avoid  ",    # 1
                "Go  ",       # 2
            ]
stimuliLbl.reverse()


datasetDir = '/raid2/Petrit/datasets/honda/HDD'




def processLabels(datasetDir):

    folderList  = [f.name for f in os.scandir(datasetDir+'camera') if os.path.isdir(f)]
    #folderName = '201709201221'

    for folder in folderList:
        
        camera = 'D:\\Datasets\\HondaDatasets\\HDD\\camera\\'+folder+'\\'
        sensor = np.load('D:\\Datasets\\HondaDatasets\\HDD\\sensor\\'+folder+'.npy')
        target = np.load('D:\\Datasets\\HondaDatasets\\HDD\\target\\'+folder+'.npy')
        goal = np.load('D:\\Datasets\\HondaDatasets\\HDD\\goal\\g'+folder+'.npy')
        cause = np.load('D:\\Datasets\\HondaDatasets\\HDD\\cause\\c'+folder+'.npy')
        stimuli = np.load('D:\\Datasets\\HondaDatasets\\HDD\\stimuli\\s'+folder+'.npy')
        attention = np.load('D:\\Datasets\\HondaDatasets\\HDD\\attention\\a'+folder+'.npy')
        attention2 = np.load('D:\\Datasets\\HondaDatasets\\HDD\\attention2\\aa'+folder+'.npy')
        area = np.load('D:\\Datasets\\HondaDatasets\\HDD\\area\\ar'+folder+'.npy')
        note = np.load('D:\\Datasets\\HondaDatasets\\HDD\\note\\n'+folder+'.npy')

        #print(np.amin(stimuli))
        #print(np.amax(stimuli))

        imagesList  = [f.path for f in os.scandir(camera) if os.path.isfile(f)]
        #imagesList  = sorted(imagesList, key=lambda x: int(x.split('/')[7][0:-4]))


        target2 = np.zeros(len(target), dtype=int)
        ## 1. extend the labeled maneuver for 15 more labels backward 
        for i in range(0, len(target)):
            if target[i]!=0:
                target2[i] = target[i]
                for j in reversed(range(i-15,i)):
                    if target[j]==0:
                        target2[j] = target[i]
                    else:
                        break

        ## 2. extend the labeled maneuver for more steps if not moving
        for i in range(0, len(target2)):
            if target2[i]!=0:
                tempLbl = target2[i]
                for j in reversed(range(0,i)):
                    if sensor[j][3]==0: 
                        for k in range(j-15,j):
                            if target2[k]==0:
                                target2[k] = tempLbl
                    else:
                        break

        ## 3. relabel the maneuver 'background' to 'keep the lane*'
        for i in range(0, len(target2)):
            if target2[i]==0:
                target2[i] = 12


        stimuli2 = np.zeros(len(stimuli), dtype=int)
        cause2 = cause.copy()
        ## 1. relabel stimuli
        for i in range(0, len(stimuli)):
            if stimuli[i]==85 or stimuli[i]==86 or stimuli[i]==87 or stimuli[i]==89 or stimuli[i]==90:
                stimuli2[i] = 0
            elif stimuli[i]==88 or stimuli[i]==91 or stimuli[i]==92 or stimuli[i]==93:
                stimuli2[i] = 1
            elif stimuli[i]==0:
                stimuli2[i] = 2
            else:
                stimuli2[i] = 3

        ## 2. Extend the stimuli labels to frames where cause!=0
        for i in range(0, len(stimuli2)):
            if cause[i]!=0 and sensor[i-2][0]==0 and stimuli2[i]==2:
                stimuli2[i] = 0

        ## 3. Extend the 'STOP' stimuli label to frames where acceleration=0
        for i in range(0, len(stimuli2)):
            tempCause = cause[i]
            if stimuli2[i]==0:
                for j in range(i+1, len(stimuli2)):
                    if  cause[j]==0 and stimuli2[j]==2 and sensor[j][0]==0:
                        stimuli2[j] = 0
                        cause2[j] = tempCause
                    else:
                        break

        ## 4. Add the 'Other' cause label to frames where cause is missing
        for i in range(0, len(stimuli2)):
            if stimuli2[i]==0 and cause2[i]==0:
                if stimuli[i]==85:
                    cause2[i]=16
                elif stimuli[i]==86:
                    cause2[i]=17
                elif stimuli[i]==87:
                    cause2[i]=18
                elif stimuli[i]==88:
                    cause2[i]=20
                elif stimuli[i]==89:
                    cause2[i]=22
                elif stimuli[i]==91:
                    cause2[i]=25
                elif stimuli[i]==92:
                    cause2[i]=24
                else:
                    cause2[i]=97

        ## 5. Reorder the cause label to frames where cause is missing
        for i in range(0, len(cause2)):
            if cause2[i]==0:
                cause2[i] = 17
            elif cause2[i]==97:
                cause2[i] = 16
            else:
                cause2[i] = cause2[i] - 16


        area = area[2:]
        note = note[2:]
        target = target[2:]
        target2 = target2[2:]
        goal = goal[2:]
        attention = attention[2:]
        attention2 = attention2[2:]
        cause = cause[2:]
        cause2 = cause2[2:]
        stimuli = stimuli[2:]
        stimuli2 = stimuli2[2:]

        startIndexs = np.where(note==49)
        endIndexs = np.where(note==50)

        if len(startIndexs[0])==0:
            startIndex = 0
        else:
            startIndex = startIndexs[0][0]
            
        if len(endIndexs[0])==0:
            endIndex = len(sensor)
        else:
            endIndex = endIndexs[0][len(endIndexs[0])-1]


        np.save('.\\hdd\\accel_pedal_info\\'+folder+'.npy', sensor[:,0])
        np.save('.\\hdd\\rtk_pos_info\\'+folder+'.npy', sensor[:,1])
        np.save('.\\hdd\\steer_info\\'+folder+'.npy', sensor[:,2])
        np.save('.\\hdd\\vel_info\\'+folder+'.npy', sensor[:,3])
        np.save('.\\hdd\\brake_pedal_info\\'+folder+'.npy', sensor[:,4])
        np.save('.\\hdd\\rtk_track_info\\'+folder+'.npy', sensor[:,5])
        np.save('.\\hdd\\turn_signal_info\\'+folder+'.npy', sensor[:,6])
        np.save('.\\hdd\\yaw_info\\'+folder+'.npy', sensor[:,6])

        np.save('.\\hdd\\area\\'+folder+'.npy', area)
        np.save('.\\hdd\\note\\'+folder+'.npy', note)
        np.save('.\\hdd\\target2\\'+folder+'.npy', target2)
        np.save('.\\hdd\\attention\\'+folder+'.npy', attention)
        np.save('.\\hdd\\attention2\\'+folder+'.npy', attention2)
        np.save('.\\hdd\\cause2\\'+folder+'.npy', cause2)
        np.save('.\\hdd\\stimuli2\\'+folder+'.npy', stimuli2)

        print(folder)




def writeLables2Images():
    for i in range(0, len(imagesList)):
        if i>startIndex and i<endIndex:
            img = cv2.imread(imagesList[i])

            cv2.putText(img, "accel: "+str(sensor[i][0]), (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "  rtk: "+str(sensor[i][1]), (10, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "steer: "+str(sensor[i][2]), (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "veloc: "+str(sensor[i][3]), (10, 80),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "brak: " +str(sensor[i][4]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "rtk_ti:"+str(sensor[i][5]), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, " turn: "+str(sensor[i][6]), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "  yaw: "+str(sensor[i][7]), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(img, "Area: "+      str(area[i])      + "-" + labelsLbl[area[i]],   (1100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Note: "+      str(note[i])      + "-" + labelsLbl[note[i]],   (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(img, "Target: "+    str(target[i])      + "-" + targetLbl[target[i]],       (10, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Target2: "+   str(target2[i])     + "-" + targetLbl[target2[i]],      (10, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Goal: "+      str(goal[i])        + "-" + labelsLbl[goal[i]],         (10, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Attention: "+ str(attention[i])   + "-" + labelsLbl[attention[i]],    (10, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Attention2: "+str(attention2[i])  + "-" + labelsLbl[attention2[i]],   (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.putText(img, "Cause: "+     str(cause[i])       + "-" + labelsLbl[cause[i]],        (1000, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Cause2: "+    str(cause2[i])      + "-" + causeLbl[cause2[i]],        (1000, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Stimuli: "+   str(stimuli[i])     + "-" + labelsLbl[stimuli[i]],      (1000, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, "Stimuli2: "+  str(stimuli2[i])    + "-" + stimuliLbl[stimuli2[i]],    (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            #cv2.imwrite('.\\images\\'+folderName+'\\'+str(i)+'.jpg', img)

            print(imagesList[i])
    
    print('done!!!!')




def objDetectionImages():
    pretrainedModels = timm.list_models('*resnet*')

    model = timm.create_model('inception_resnet_v2', pretrained=True)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    img = Image.open(datasetDir+"camera\\201702271017\\00346.jpg")
    img = img.convert('RGB')
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(tensor)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    print(probabilities.shape)

    url, filename = ("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename) 
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
    
    print('test')
    



def convertLbl2Yolo():
    lblDir = datasetDir+'/release_2020_07_15_causal_reasoning/Cause_labeled_images'

    folderList  = [f.name for f in os.scandir(lblDir) if os.path.isdir(f)]
    
    for folder in folderList:
        folderSplit = folder.split('_')

        if len(folderSplit)==6:
            cause = folderSplit[3]
        elif len(folderSplit)==7:
            cause = folderSplit[3] + '_' + folderSplit[4]
        elif len(folderSplit)==8:
            cause = folderSplit[3] + '_' + folderSplit[4] + '_' + folderSplit[5]
        elif len(folderSplit)==9:
            cause = folderSplit[3] + '_' + folderSplit[4] + '_' + folderSplit[5] + '_' + folderSplit[6]
        elif len(folderSplit)==10:
            cause = folderSplit[3] + '_' + folderSplit[4] + '_' + folderSplit[5] + '_' + folderSplit[6] + '_' + folderSplit[7]

        
        fileList = [f.name for f in os.scandir(lblDir+'/'+folder) if os.path.isfile(f)]

        for file in fileList:
            if file[-4:]==".txt" and file[-9:-4]!='yolo5':
                lines = [line.rstrip() for line in open(lblDir+'/'+folder+'/'+file)]
                for line in lines:
                    if line!='':
                        line = line.split(' ')

                        topLeftX = int(line[1]) / 1280
                        topLeftY = int(line[2]) / 720
                        bottomRightX = int(line[3]) / 1280
                        bottomRightY = int(line[4]) / 720

                        centerPointX = (topLeftX + bottomRightX) / 2
                        centerPointY = (topLeftY + bottomRightY) / 2

                        width = bottomRightX - topLeftX
                        height = bottomRightY - topLeftY

                        dirName = lblDir+'/'+folder+'/'+file[:-4]+'_yolo5.txt'
                        with open(dirName, 'a+') as f:
                            print(f"{clause2ID[cause]} {centerPointX} {centerPointY} {width} {height}", file=f)

        print(folder)




def video2frames(folder):
    folderDir = datasetDir + '/release_2019_01_20/'+folder[:4]+'_'+folder[4:6]+'_'+folder[6:8]+'_ITS1/'+folder+'/camera/center/'
    fileList  = [f.name for f in os.scandir(folderDir) if os.path.isfile(f)]
    
    os.mkdir(folderDir+'frames/')
    
    vidcap = cv2.VideoCapture(folderDir+fileList[0])
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(folderDir+'frames/frame_%d.jpg' % count, image)
        success,image = vidcap.read()
        count += 1


def video2frames2018(file):    
    os.mkdir(folderDir+'frames/')
    
    vidcap = cv2.VideoCapture(folderDir+file)
    success,image = vidcap.read()
    fileSplit = file.split('_')
    count = int(fileSplit[-2])
    while success:
        cv2.imwrite(folderDir+'frames/frame_%d.jpg' % count, image)
        success,image = vidcap.read()
        count += 1
    


def copyCleanFrames(folder):
    srcDir = datasetDir + '/release_2019_01_20/'+folder[:4]+'_'+folder[4:6]+'_'+folder[6:8]+'_ITS1/'+folder+'/camera/center/frames/'
    dstDir = datasetDir + '/release_2020_07_15_causal_reasoning/Cause_labeled_images/'

    dstFolderList  = [f.name for f in os.scandir(dstDir) if os.path.isdir(f)]
    for dstFolder in dstFolderList:
        dstFolderSplit = dstFolder.split('_')
        if dstFolderSplit[0] == folder:
            fileList  = [f.name for f in os.scandir(dstDir+dstFolder) if os.path.isfile(f)]
            for file in fileList:
                if file[-4:]=='.png':
                    endIndex = file.find('.')
                    shutil.copy(srcDir+'frame_'+file[6:endIndex]+'.jpg', dstDir+dstFolder+'/output'+file[6:endIndex]+'.jpg')
        
            print(dstFolder)
    


def copyCleanFrames2018(file):
    srcDir = folderDir+'frames/'
    dstDir = datasetDir + '/release_2020_07_15_causal_reasoning/Cause_labeled_images/'+file[0:-4]

    try:
        fileList  = [f.name for f in os.scandir(dstDir) if os.path.isfile(f)]
        for f in fileList:
            if f[-4:]=='.png':
                endIndex = f.find('.')
                shutil.copy(srcDir+'frame_'+f[6:endIndex]+'.jpg', dstDir+'/output'+f[6:endIndex]+'.jpg')
    except:
        print("ERROR: " + dstDir)

    print(dstDir)




def deleteFolderFrames(folder):
    folderDir = datasetDir + '/release_2019_01_20/'+folder[:4]+'_'+folder[4:6]+'_'+folder[6:8]+'_ITS1/'+folder+'/camera/center/frames/'
    shutil.rmtree(folderDir)


def deleteFolderFrames2018(folder):
    shutil.rmtree(folderDir+'frames/')




def drawWaypoint():
    imageDir = datasetDir + '/camera/201702271017/'
    imageList  = [f.path for f in os.scandir(imageDir) if os.path.isfile(f)]
    imageList  = sorted(imageList, key=lambda x: int(x.split('/')[8][0:-4]))

    sensorList = np.load(datasetDir+'/sensor/201702271017.npy')

    posYList = [620,570,520,480,440,410,380,360,340,330]
    posSList = [200,180,160,140,120,100,80,60,40,20]

    for image in imageList:
        img = cv2.imread(image)
        imgCount = int(image.split('/')[-1][0:-4])
        for i in range(0,10):
            steer = sensorList[imgCount+i-1][2]
            yaw   = sensorList[imgCount+i-1][7]
            vel   = sensorList[imgCount+i-1][3]
            cv2.putText(img, str(steer), (10, posSList[i]),   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, str(yaw),   (80, posSList[i]),   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(img, str(vel),   (1000, posSList[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            velScales = vel/72.95
            waypointY = int(640/(1+velScales))
            wpxYaw = 0
            wpxSteer = 0
            if yaw>0:
                steerScale = steer/331
                yawScale = yaw/29.7852
                wpxYaw = int(yawScale*600)
                wpxSteer = int(steerScale*600)
            elif yaw<0:
                steerScale = steer/531
                yawScale = yaw/37.1094
                wpxYaw = int(yawScale*680)
                wpxSteer = int(steerScale*680)
            
            cv2.drawMarker(img, (680+wpxYaw,posYList[i]), (0,0,255), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            cv2.drawMarker(img, (680+wpxSteer,posYList[i]), (0,255,0), markerType=cv2.MARKER_DIAMOND, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

        
        cv2.imwrite('./waypoints/waypoint_'+image.split('/')[-1],img)
        print(image.split('/')[-1])

    print('done')
        



def deleteEmptyImages():
    causeFolder = datasetDir+'/release_2020_07_15_causal_reasoning/Cause_labeled_images'
    folderList  = [f.path for f in os.scandir(causeFolder) if os.path.isdir(f)]

    folderList.sort()
    
    for folder in folderList:
        fileList = [f.path for f in os.scandir(folder) if os.path.isfile(f)]
        for file in fileList:
            if file[-4:]==".txt" and file[-9:-4]!='yolo5':
                if os.stat(file).st_size==0:
                    for file2 in fileList:
                        if file.split('/')[9][:-4] in file2:
                            os.remove(file2)
        
        print(folder)
    
    

def detectObjects():
    noteDir = './hdd/note/'
    cameraDir = datasetDir + '/camera/'

    logging.getLogger("yolov5").setLevel(logging.WARNING)
    model    = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/yolo_HDD/weights/best.pt', source='local')
    modelPre = torch.hub.load('./yolov5', 'yolov5s', pretrained=True, source='local')

    noteFileList = [f.path for f in os.scandir(noteDir) if os.path.isfile(f)]
    for note in tqdm.tqdm(noteFileList):
    
        os.mkdir('./hdd/graphsPre/' + str(note.split('/')[3][:-4]))
        npNote = np.load(note)

        startIndexs = np.where(npNote==49)
        if len(startIndexs[0])==0:
            startIndex = 1
        else:
            startIndex = startIndexs[0][0]
            
        endIndexs = np.where(npNote==50)
        if len(endIndexs[0])==0:
            endIndex = len(npNote)
        else:
            endIndex = endIndexs[0][len(endIndexs[0])-1]

        for i in range(startIndex+1, endIndex):
            result = model(cameraDir + '/' + str(note.split('/')[3][:-4]) + '/' + format(i, '05d') + '.jpg')
            resultPre = modelPre(cameraDir + '/' + str(note.split('/')[3][:-4]) + '/' + format(i, '05d') + '.jpg')

            dfResult = result.pandas().xyxy[0]
            dfResultPre = resultPre.pandas().xyxy[0]

            dfResultWH = result.pandas().xywh[0]
            dfResultPreWH = resultPre.pandas().xywh[0]

            img = cv2.imread(cameraDir + '/' + str(note.split('/')[3][:-4]) + '/' + format(i, '05d') + '.jpg')
            '''
            for j in dfResultPre.index:
                top_left    = (int(dfResultPre['xmin'][j]), int(dfResultPre['ymin'][j]))
                down_right  = (int(dfResultPre['xmax'][j]), int(dfResultPre['ymax'][j]))
                cv2.rectangle(img, top_left, down_right, (0,0,255), 2)
                cv2.putText(img, dfResultPre['name'][j], tuple(map(operator.add, top_left, (0,-5))), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 1)
            cv2.imwrite('./visualization/test/' + str(i) + '.jpg', img)
            '''
            
            resultGraph = []
            hddObjectFlag = False
            if len(dfResultWH)>0 and len(dfResultPreWH)==0:
                g = createGraph(dfResultWH, True)
                save_graphs('./hdd/graphsPre/'+ str(str(note.split('/')[3][:-4])) + '/' + str(i) + '.bin', [g])
            elif len(dfResultWH)==0 and len(dfResultPreWH)>0:
                for k in dfResultPreWH.index:
                    if dfResultPreWH['confidence'][k]>0.25 and resultClasses(dfResultPreWH, k):
                        resultGraph.append(dfResultPreWH.iloc[k])
                dfResultGraph = pd.DataFrame(resultGraph)
                dfResultGraph = dfResultGraph.reset_index()
                g = createGraph(dfResultGraph, True)
                save_graphs('./hdd/graphsPre/'+ str(str(note.split('/')[3][:-4])) + '/' + str(i) + '.bin', [g])
            elif len(dfResultWH)>0 and len(dfResultPreWH)>0:
                for k in dfResultWH.index:
                    for l in dfResultPreWH.index:
                        if dfResultPreWH['confidence'][l]>0.25 and resultClasses(dfResultPreWH, l):
                            resk = torch.tensor([[dfResult.iloc[k][0],dfResult.iloc[k][1],dfResult.iloc[k][2],dfResult.iloc[k][3]]],dtype=torch.float)
                            resl = torch.tensor([[dfResultPre.iloc[l][0],dfResultPre.iloc[l][1],dfResultPre.iloc[l][2],dfResultPre.iloc[l][3]]],dtype=torch.float)
                            if ops.box_iou(resk,resl)>0.5:
                                resultGraph.append(dfResultWH.iloc[k])
                                hddObjectFlag=True

                            elif len(dfResultWH)-1==k:
                                resultGraph.append(dfResultPreWH.iloc[l])
                    
                    if hddObjectFlag==False:
                        resultGraph.append(dfResultWH.iloc[k])

                dfResultGraph = pd.DataFrame(resultGraph)
                dfResultGraph = dfResultGraph.reset_index()
                g = createGraph(dfResultGraph, True)
                save_graphs('./hdd/graphsPre/'+ str(str(note.split('/')[3][:-4])) + '/' + str(i) + '.bin', [g])
            
        print(str(note.split('/')[3][:-4]))

    print('done')




def createGraph(dfResults, useCuda):
    nodeFeat    = []
    edges       = []
    edgeNorm    = []

    nodeX = []
    nodeY = []

    egoFeat = []
    egoFeat.append(scaleX(700))             # x = 0.546875
    egoFeat.append(scaleY(720))             # y = 1
    egoFeat.append(1)                       # w
    egoFeat.append(1)                       # h
    egoFeat.append(1)                       # conf
    egoFeat.extend(get_one_hot(0,26)[0])    # class
    nodeFeat.append(egoFeat)

    nodeX.append(700)
    nodeY.append(720)
    
    nrNodes = 1

    for i in dfResults.index:
        if dfResults['confidence'][i]>0.25:
            feat = []
            feat.append( scaleX(dfResults['xcenter'][i]) )  # x
            feat.append( scaleY(dfResults['ycenter'][i]) )  # y
            feat.append( scaleW(dfResults['width'][i]) )    # w
            feat.append( scaleH(dfResults['height'][i]) )   # h
            feat.append( dfResults['confidence'][i] )       # conf
            
            # class
            if dfResults['name'][i]=='person':
                feat.extend(get_one_hot(17,26)[0])
            elif dfResults['name'][i]=='bicycle':
                feat.extend(get_one_hot(18,26)[0])
            elif dfResults['name'][i]=='car':
                feat.extend(get_one_hot(19,26)[0])
            elif dfResults['name'][i]=='motorbike':
                feat.extend(get_one_hot(20,26)[0])
            elif dfResults['name'][i]=='bus':
                feat.extend(get_one_hot(21,26)[0])
            elif dfResults['name'][i]=='train':
                feat.extend(get_one_hot(22,26)[0])
            elif dfResults['name'][i]=='truck':
                feat.extend(get_one_hot(23,26)[0])
            elif dfResults['name'][i]=='traffic light':
                feat.extend(get_one_hot(24,26)[0])
            elif dfResults['name'][i]=='stop sign':
                feat.extend(get_one_hot(25,26)[0])
            else:
                feat.extend(get_one_hot(int(dfResults['class'][i]+1),26)[0])
            
            nodeFeat.append(feat)

            nodeX.append(dfResults['xcenter'][i])
            nodeY.append(dfResults['ycenter'][i])
            
            nrNodes = nrNodes + 1


    for i in range(0, nrNodes):
        for j in range(0, nrNodes):
            #if i != j:
            edges.append([i, j])
            ed = distPoints(nodeX[i], nodeY[i], nodeX[j], nodeY[j])
            edgeNorm.append(ed)
    

    g = dgl.graph(edges, num_nodes=nrNodes)

    edgeNorm    = torch.tensor(edgeNorm,    dtype=torch.float32).unsqueeze(1)
    nodeFeat    = torch.tensor(nodeFeat,    dtype=torch.float32)

    if useCuda:
        device      = torch.device('cuda:3')
        edgeNorm    = edgeNorm.to(device)
        nodeFeat    = nodeFeat.to(device)
        g           = g.to(device)

    g.edata.update( {'norm': edgeNorm} )
    g.ndata['feat'] = nodeFeat

    return g





def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res


def scaleX(xValue, xMin=0, xMax=1402):
    return (xValue - xMin) / (xMax - xMin)


def scaleY(yValue, yMin=0, yMax=720):
    return (yValue - yMin) / (yMax - yMin)


def scaleW(wValue, wMin=0, wMax=1280):
    return (wValue - wMin) / (wMax - wMin)


def scaleH(hValue, hMin=0, hMax=721):
    return (hValue - hMin) / (hMax - hMin)


def distPoints(x1, y1, x2, y2):
    a = (x1, y1)
    b = (x2, y2)
    return 1-(dist(a,b)/1468.6048)


def scaleAccel(aValue, aMin=0, aMax=100):
    return (aValue - aMin) / (aMax - aMin)

def descaleAccel(aValue, aMin=0, aMax=100):
    return aValue * (aMax-aMin) + aMin


def scaleRtk(rValue, rMin=-481, rMax=474):
    return (rValue - rMin) / (rMax - rMin)

def descaleRtk(rValue, rMin=-481, rMax=474):
    return rValue * (rMax-rMin) + rMin


def scaleBrake(bValue, bMin=0, bMax=7332):
    return (bValue - bMin) / (bMax - bMin)
    
def descaleBrake(bValue, bMin=0, bMax=7332):
    return bValue * (bMax-bMin) + bMin


def scaleVel(vValue, vMin=0, vMax=138):
    return (vValue - vMin) / (vMax - vMin)
    
def descaleVel(vValue, vMin=0, vMax=138):
    return vValue * (vMax-vMin) + vMin


def scaleYaw(yValue, yMin=-60, yMax=45):
    return (yValue - yMin) / (yMax - yMin)

def descaleYaw(yValue, yMin=-60, yMax=45):
    return yValue * (yMax-yMin) + yMin


def scaleSteer(sValue, sMin=-747, sMax=761):
    return (sValue - sMin) / (sMax - sMin)
    
def descaleSteer(sValue, sMin=-747, sMax=761):
    return sValue * (sMax-sMin) + sMin


def findMaxMin():
    inputDir = '/raid2/Petrit/datasets/honda/HDD/sensor/'
    files = [f.path for f in os.scandir(inputDir) if os.path.isfile(f)]

    minmaxVal = -9999
    for f in files:
        if f[-4:]=='.npy':
            npf = np.load(f)
            for i in range(len(npf[:,7])):
                if npf[:,7][i]>minmaxVal:
                    minmaxVal = npf[:,7][i]
    
    print(minmaxVal)




def plotConfusionMatrix(confusionMatrix, filename, head):
    plt.figure(figsize=(15,10))

    classNamesStimuli = [
                        "Stop",
                        "Avoid",
                        "Go"
                        ]

    classNamesCause = [
                        "Congestion",
                        "Sign",
                        "Red Light",
                        "Crossing Vehicle",
                        "Parked Vehicle",
                        "Yellow Light",
                        "Crossing Pedestrian",
                        "Merging Vehicle",
                        "On-road Bicyclist",
                        "Pedestrian Near Ego Lane",
                        "Park",
                        "On-road Motorcyclist",
                        "Vehicle Cut-in",
                        "Road Work",
                        "Turning Vehicle",
                        "Vehicle Passing With Lane Departure",
                        "Other",
                        "None"
                        ]
    
    if head=='stimuli':
        dfCM = pd.DataFrame(confusionMatrix, index=classNamesStimuli, columns=classNamesStimuli).astype(int)
    elif head=='cause':
        dfCM = pd.DataFrame(confusionMatrix, index=classNamesCause, columns=classNamesCause).astype(int)

    heatmap = sns.heatmap(dfCM, annot=True, fmt="d")

    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(filename)




def deleteOddFrames(folder):
    imageList = [f.path for f in os.scandir(folder) if os.path.isfile(f)]
    imageList = sorted(imageList, key=lambda x: int(x.split('/')[3].split('_')[1][0:-4]))

    for img in imageList:
        endIndex = img.split('/')[-1].find('.')
        indx = int(img.split('/')[-1][6:endIndex])

        if indx % 2 != 0:
            os.remove(img)




def inference(model, folderName, useCuda):

    logging.getLogger("yolov5").setLevel(logging.WARNING)
    modelYolo    = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/yolo_HDD/weights/best.pt', source='local')
    modelYoloPre = torch.hub.load('./yolov5', 'yolov5s', pretrained=True, source='local')

    folder  = folderName.split('/')[2].split('_')[0]
    start   = folderName.split('/')[2].split('_')[1]
    end     = folderName.split('/')[2].split('_')[2]

    imageList = [f.path for f in os.scandir(folderName) if os.path.isfile(f)]
    imageList = sorted(imageList, key=lambda x: int(x.split('/')[3].split('_')[1][0:-4]))

    accel_pedal_info    = np.load('./hdd/accel_pedal_info/'+folder+'.npy')
    area                = np.load('./hdd/area/'+folder+'.npy')
    attention           = np.load('./hdd/attention/'+folder+'.npy')
    brake_pedal_info    = np.load('./hdd/brake_pedal_info/'+folder+'.npy')
    cause               = np.load('./hdd/cause2/'+folder+'.npy')
    rtk_pos_info        = np.load('./hdd/rtk_pos_info/'+folder+'.npy')
    rtk_track_info      = np.load('./hdd/rtk_track_info/'+folder+'.npy')
    steer_info          = np.load('./hdd/steer_info/'+folder+'.npy')
    stimuli             = np.load('./hdd/stimuli2/'+folder+'.npy')
    target              = np.load('./hdd/target2/'+folder+'.npy')
    turn_signal_info    = np.load('./hdd/turn_signal_info/'+folder+'.npy')
    vel_info            = np.load('./hdd/vel_info/'+folder+'.npy')
    yaw_info            = np.load('./hdd/yaw_info/'+folder+'.npy')
    
    with torch.no_grad():
        for img in imageList[800:]:

            endIndex = img.split('/')[-1].find('.')-1
            indx = int(img.split('/')[-1][6:endIndex])
            
            sensor = []
            sensor.append(scaleAccel(accel_pedal_info[indx]))
            sensor.append(scaleRtk(rtk_pos_info[indx]))
            sensor.append(scaleSteer(steer_info[indx]))
            sensor.append(scaleVel(vel_info[indx]))
            sensor.append(scaleBrake(brake_pedal_info[indx]))
            sensor.append(rtk_track_info[indx])
            sensor.append(turn_signal_info[indx])
            sensor.append(scaleYaw(yaw_info[indx]))
            
            result = modelYolo(img)
            resultPre = modelYoloPre(img)

            dfResultXY = result.pandas().xyxy[0]
            dfResultPreXY = resultPre.pandas().xyxy[0]

            dfResultWH = result.pandas().xywh[0]
            dfResultPreWH = resultPre.pandas().xywh[0]

            resultGraphXY = []
            resultGraphWH = []
            hddObjectFlag = False
            preObjectFlag = False
            if len(dfResultWH)>0 and len(dfResultPreWH)==0:
                dfResultGraphXY = dfResultXY
                dfResultGraphWH = dfResultWH
                g = createGraph(dfResultGraphWH, True)
            elif len(dfResultWH)==0 and len(dfResultPreWH)>0:
                for k in dfResultPreWH.index:
                    if dfResultPreWH['confidence'][k]>0.25 and resultClasses(dfResultPreWH, k):
                        resultGraphXY.append(dfResultPreXY.iloc[k])
                        resultGraphWH.append(dfResultPreWH.iloc[k])
                dfResultGraphXY = pd.DataFrame(resultGraphXY)
                dfResultGraphXY = dfResultGraphXY.reset_index()
                dfResultGraphWH = pd.DataFrame(resultGraphWH)
                dfResultGraphWH = dfResultGraphWH.reset_index()
                g = createGraph(dfResultGraphWH, True)
            elif len(dfResultWH)>0 and len(dfResultPreWH)>0:
                for p in dfResultWH.index:
                    resultGraphXY.append(dfResultXY.iloc[p])
                    resultGraphWH.append(dfResultWH.iloc[p])
                for k in dfResultPreWH.index:
                    for l in dfResultWH.index:
                        if dfResultPreWH['confidence'][k]>0.25 and resultClasses(dfResultPreWH, k):
                            preObjectFlag = True
                            resk = torch.tensor([[dfResultPreXY.iloc[k][0],dfResultPreXY.iloc[k][1],dfResultPreXY.iloc[k][2],dfResultPreXY.iloc[k][3]]],dtype=torch.float)
                            resl = torch.tensor([[dfResultXY.iloc[l][0],dfResultXY.iloc[l][1],dfResultXY.iloc[l][2],dfResultXY.iloc[l][3]]],dtype=torch.float)
                            if ops.box_iou(resk,resl)>0.4:
                                hddObjectFlag = True
                    
                    if preObjectFlag==True and hddObjectFlag==False:
                        resultGraphXY.append(dfResultPreXY.iloc[k])
                        resultGraphWH.append(dfResultPreWH.iloc[k])
                    
                    preObjectFlag = False
                    hddObjectFlag = False

                #resultGraphXY.reverse()
                #resultGraphWH.reverse()

                dfResultGraphXY = pd.DataFrame(resultGraphXY)
                dfResultGraphXY = dfResultGraphXY.reset_index()
                dfResultGraphWH = pd.DataFrame(resultGraphWH)
                dfResultGraphWH = dfResultGraphWH.reset_index()
                g = createGraph(dfResultGraphWH, True)
            
            else:
                nodeFeat = []
                egoFeat = []
                egoFeat.append(scaleX(700))             # x = 0.546875
                egoFeat.append(scaleY(720))             # y = 1
                egoFeat.append(1)                       # w
                egoFeat.append(1)                       # h
                egoFeat.append(1)                       # conf
                egoFeat.extend(get_one_hot(0,25)[0])    # class
                nodeFeat.append(egoFeat)
                g = dgl.graph([[0,0]])
                
                edgeNorm    = torch.tensor([1],      dtype=torch.float32).unsqueeze(1)
                nodeFeat    = torch.tensor(nodeFeat, dtype=torch.float32)

                g.edata.update( {'norm': edgeNorm} )
                g.ndata['feat'] = nodeFeat

            
            target1h = get_one_hot(target[indx]-1,12)

            if area[indx]==0:
                area1h = get_one_hot(0,3)
            else:
                area1h = get_one_hot(area[indx]-32,3)
            
            tSensor  = torch.tensor(sensor).unsqueeze(0)
            tTarget  = torch.tensor(target1h)
            tArea    = torch.tensor(area1h)
            
            if useCuda:
                device      = torch.device('cuda:3')
                tSensor     = tSensor.to(device)
                tTarget     = tTarget.to(device)
                tArea       = tArea.to(device)
                        
            predStimuli,predCause = model(g, tSensor, tTarget, tArea)
            predStimuli = torch.softmax(predStimuli, -1)
            predCause = torch.softmax(predCause, -1)

            predStimuli = predStimuli.cpu().detach().numpy()[0] * 100
            predCause = predCause.cpu().detach().numpy()[0] * 100

            #predStimuli = np.array( [ 0.0, 0.0, 100.0] )
            #predCause   = np.array( [ [0.0] * 16 ] )
            #predCause   = np.append(predCause, 100.0)

            visualizeScene(img, dfResultGraphXY)
            visualizePlot(img, dfResultGraphWH, sensor, target[indx], area[indx], predStimuli, predCause)




def visualizeScene(imgPath, dfResult):
    img = cv2.imread(imgPath)

    for i in dfResult.index:
        if dfResult['name'][i]=='congestion':    # Congestion
            color = (0, 255, 0)
        elif dfResult['name'][i]=='sign':  # Sign
            color = (255, 0, 0)
        elif dfResult['name'][i]=='red_light':  # Red Light
            color = (0, 0, 255)
        elif dfResult['name'][i]=='crossing_vehicle':  # Crossing Vehicle
            color = (0, 128, 255)
        elif dfResult['name'][i]=='yellow_light':  # Yellow Light
            color = (0, 255, 255)
        else:
            color = (255, 255, 255)

        top_left    = (int(dfResult['xmin'][i]), int(dfResult['ymin'][i]))
        down_right  = (int(dfResult['xmax'][i]), int(dfResult['ymax'][i]))
        
        cv2.rectangle(img, top_left, down_right, color, 2)
        cv2.putText(img, dfResult['name'][i], tuple(map(operator.add, top_left, (0,-5))), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)
    
    cv2.imwrite('./visualization/'+imgPath.split('/')[2]+'/plots/'+imgPath.split('/')[3][6:], img)




def visualizePlot(imgPath, dfResult, sensor, target, area, predStimuli, predCause):
    
    nodeX       = []
    nodeY       = []
    nodeColor   = []
    nodeText    = []
    textPos     = []
    
    edgeX       = []
    edgeY       = []
    edgeNorm    = []

    nodeX.append(640)
    nodeY.append(0)
    nodeColor.append('black')
    nodeText.append('EgoNode')
    textPos.append('bottom center')

    for i in dfResult.index:
        nodeX.append(dfResult['xcenter'][i])
        nodeY.append(720-dfResult['ycenter'][i])

        edgeX.append(640)
        edgeY.append(0)
        edgeX.append(dfResult['xcenter'][i])
        edgeY.append(720-dfResult['ycenter'][i])

        norm = 1 - (abs(dist((dfResult['xcenter'][i], 720-dfResult['ycenter'][i]), (640, 0)))/963)
        norm = max(0, norm)
        edgeNorm.append(norm*2)

        if dfResult['name'][i]=='congestion':
            nodeColor.append('green')
        elif dfResult['name'][i]=='sign':
            nodeColor.append('blue')
        elif dfResult['name'][i]=='red_light':
            nodeColor.append('red')
        elif dfResult['name'][i]=='crossing_vehicle':
            nodeColor.append('orange')
        elif dfResult['name'][i]=='yellow_light':
            nodeColor.append('yellow')
        else:
            nodeColor.append('gray')

        nodeText.append( dfResult['name'][i] + f"[{dfResult['confidence'][i]:.2f}]")
        textPos.append('top center')
    
    if area!=0:
        area = area - 32

    # Make Subplot
    fig = make_subplots(
        rows=4, cols=3,
        shared_xaxes=False,
        column_widths=[0.2, 0.6, 0.2],
        row_heights=[0.2, 0.4, 0.3, 0.1],
        horizontal_spacing=0.025,
        vertical_spacing=0.025,
        subplot_titles=['','','<b>Stimuli Prediction</b>','','','','<b>Cause Prediction</b>'],
        specs=[[{"type":"table", 't':0.036}, 
                {"type":"image", 'rowspan':2, 'r':0.1}, 
                {"type":"bar", 'rowspan':2, 't':0.044, 'b':0.4}],
                [{"type":"table", 't':-0.016, 'b':0},
                None,
                None],
                [{"type":"table", 't':0, 'b':-0.04},
                {"type":"scatter", 'rowspan':2, 'r':0.1}, 
                {"type":"bar", 'rowspan':2, 't':-0.34}],
                [{"type":"table", 't':0.028, 'b': 0}, 
                None,
                None]]
        )
    fig.update_annotations(font_size=44)
    #fig.for_each_trace(lambda t: t.update(header_fill_color = 'rgba(0,0,0,0)'))
    #fig.layout['template']['data']['table'][0]['header']['fill']['color']='rgba(0,0,0,0)'


    # add the model and scene info
    fig.add_trace(
        go.Table(
            header=dict(
                fill_color='rgba(0,0,0,0)',
                height=0
            ),
            cells=dict(
                values=[ ['<b>Scenario:</b>', '<b>Frame:</b>', '<b>YOLOv5:</b>', '<b>Model:</b>'], 
                        [imgPath.split('/')[2].split('_')[0], 
                        int(imgPath.split('/')[3].split('_')[1][:-4])/10,
                        'HDD-PTY', 'EGAT'] ],
                fill_color=[ ['#C8D4E4', 'aliceblue']*2 ],
                align = ['right', 'left'],
                font=dict(size=40),
                height=50
            )
        ),
        row=1, col=1
    )


    # add the maneuver of the ego-vehicle
    targetCells = ['Keep the Lane', 'Intersection Passing', 'Left Turn',
                    'Right Turn', 'Left Lane Change', 'Right Lane Change', 
                    'Left Lane Branch', 'Right Lane Branch', 'Crosswalk Passing',
                    'Railroad Passing', 'Merge', 'U-turn']
    
    targetColor = ['aliceblue'] * 12
    
    if target == 12:
        targetCells[0] = '<b>Keep the Lane</b>'
    elif target == 1:
        targetCells[1] = '<b>Intersection Passing</b>'
    elif target == 2:
        targetCells[2] = '<b>Left Turn</b>'
    elif target == 3:
        targetCells[3] = '<b>Right Turn</b>'
    elif target == 4:
        targetCells[4] = '<b>Left Lane Change</b>'
    elif target == 5:
        targetCells[5] = '<b>Right Lane Change</b>'
    elif target == 6:
        targetCells[6] = '<b>Left Lane Branch</b>'
    elif target == 7:
        targetCells[7] = '<b>Right Lane Branch</b>'
    elif target == 8:
        targetCells[8] = '<b>Crosswalk Passing</b>'
    elif target == 9:
        targetCells[9] = '<b>Railroad Passing</b>'
    elif target == 10:
        targetCells[10] = '<b>Merge</b>'
    elif target == 11:
        targetCells[11] = '<b>U-turn</b>'
    
    if target == 12:
        targetColor[0] = 'PaleGreen'
    else:
        targetColor[target] = 'PaleGreen'

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Maneuver of ego-vehicle'],
                align = ['center'],
                line_color='darkslategray',
                font=dict(size=54),
                height=66
            ),
            cells=dict(
                values=[ targetCells ],
                line_color='darkslategray',
                fill_color=[ targetColor ],
                align = ['center'],
                font=dict(size=40),
                height=50
            )
        ),
        row=2, col=1
    )


    # add the traffic scene
    img = io.imread('./visualization/'+imgPath.split('/')[2]+'/plots/'+imgPath.split('/')[3][6:])
    fig.add_trace(go.Image(z=img), 1, 2)


    # add the predicted stimuli
    barsStimuli = go.Bar(
        y=stimuliLbl, 
        x=np.flip(predStimuli),
        showlegend=False,
        marker=dict(color=['darkgreen', 'darkblue', 'darkred']),
        orientation='h'
    )
    fig.add_trace(barsStimuli, 1, 3)


    # add sensor info in the table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Sensor', 'Value'],
                align = ['center'],
                line_color='darkslategray',
                font=dict(size=54),
                height=66
            ),
            cells=dict(
                values=[
                    [ '<b>Acceleration Pedal:</b>', '<b>RTK Position:</b> ','<b>Steering Info:</b>', 
                    '<b>Velocity Info:</b>', '<b>Brake Pedal:</b>', '<b>Left Indicator:</b>', 
                    '<b>Right Indicator:</b>', '<b>Yaw:</b>'],
                    [ f"{descaleAccel(sensor[0]):.4f}", f"{descaleRtk(sensor[1]):.4f}", 
                    f"{descaleSteer(sensor[2]):.4f}", f"{descaleVel(sensor[3]):.4f}", 
                    f"{descaleBrake(sensor[4]):.4f}", sensor[5], sensor[6], 
                    f"{descaleYaw(sensor[7]):.4f}"]
                ],
                align = ["right", "left"],
                fill_color='aliceblue',
                line_color='darkslategray',
                font=dict(size=44),
                height=57
                ),
                columnwidth=[0.65, 0.35]
        ),
        row=3, col=1
    )


    # add area info in the table
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Area'],
                align = ['center'],
                line_color='darkslategray',
                font=dict(size=54),
                height=66
            ),
            cells=dict(
                values=[
                    [areaLbl[area]]
                ],
                align = ["center"],
                fill_color='aliceblue',
                line_color='darkslategray',
                font=dict(size=44),
                height=57
                ),
                columnwidth=[0.7, 0.3]
        ),
        row=4, col=1
    )


    # add the graph plotting
    for i in range(0, len(edgeNorm)):
        fig.add_trace(go.Scatter(x=edgeX[2*i:2*i+2], y=edgeY[2*i:2*i+3], 
                                    mode='lines',
                                    showlegend=False,
                                    line=dict(color='#444', width=edgeNorm[i]*10)
    ), 3, 2)

    node_trace = go.Scatter(
        x=nodeX, y=nodeY,
        mode='markers+text',
        text=nodeText,
        textposition=textPos,
        showlegend=False,
        textfont=dict(
            size=44,
            color=nodeColor
        ),
        marker=dict(
            showscale=False,
            colorscale='YlGnBu',
            reversescale=True,
            color=nodeColor,
            size=44,
            #colorbar=dict(
            #    thickness=15,
            #    title='Node Connections',
            #    xanchor='left',
            #    titleside='right'
            #),
            line_width=2)
        )
    fig.add_trace(node_trace, 3, 2)


    # add the predicted cause
    barsCause = go.Bar(
        y=causeLbl, 
        x=np.flip(predCause),
        showlegend=False,
        marker=dict(
            color='darkblue'),
        orientation='h'
    )
    fig.add_trace(barsCause, 3, 3)


    fig.update_xaxes(row=1, col=2, showgrid=False, showticklabels=False)
    fig.update_yaxes(row=1, col=2, showgrid=False, showticklabels=False)

    fig.update_xaxes(range=[0, 1280], row=3, col=2, showgrid=False, showticklabels=False, 
                        title_text='Ego Node')
    fig.update_yaxes(range=[0, 720],  row=3, col=2, showgrid=False, showticklabels=False)

    fig.update_xaxes(range=[0, 100], row=1, col=3)
    fig.update_xaxes(range=[0, 100], row=3, col=3)

    fig.update_layout(
        autosize=False,
        title_font_size=44,
        font_size=40,
        width=3840,
        height=2160)

    fig.write_image('./visualization/'+imgPath.split('/')[2]+'/scenes/'+imgPath.split('/')[3][0:-4]+'.pdf')
    #fig.write_image('./plots/plot_' + imgPath.split('/')[3].split('_')[1][:-4] + '.svg')




def makeVideo(folderName, width=3840, height=2160):
    imageList = []
    imageList = [f.path for f in os.scandir(folderName+'/scenes/') if os.path.isfile(f)]
    imageList = sorted(imageList, key=lambda x: int(x.split('/')[4][0:-4]))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(folderName + '/' + folderName.split('/')[2]+'_videoPre.mp4', fourcc, 10, (width,height))
    #video = cv2.VideoWriter('./plots/'+path.split('/')[5]+'/camera/'+'_'+name+'.mp4', fourcc, 5, (width,height))

    for image in imageList:
        video.write(cv2.imread(image))
        
    cv2.destroyAllWindows()
    video.release()




def bbox_iou(box1, box2, eps=1e-9):    
    box2 = box2.T

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3] 
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3] 

    # Intersection area
    inter = (min(b1_x2, b2_x2) - max(b1_x1, b2_x1)) * (min(b1_y2, b2_y2) - max(b1_y1, b2_y1)) 
    
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps 
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps 
    union = w1 * h1 + w2 * h2 - inter + eps

    return inter / union




def resultClasses(dfResultPre, k):
    if dfResultPre['class'][k]==0:
        return True
    elif dfResultPre['class'][k]==1:
        return True
    elif dfResultPre['class'][k]==2:
        return True
    elif dfResultPre['class'][k]==3:
        return True
    elif dfResultPre['class'][k]==5:
        return True
    elif dfResultPre['class'][k]==6:
        return True
    elif dfResultPre['class'][k]==7:
        return True
    elif dfResultPre['class'][k]==9:
        return True
    elif dfResultPre['class'][k]==11:
        return True
    else:
        return False



def addSensor2Datalist(accel_pedal_info, rtk_pos_info, steer_info, vel_info, brake_pedal_info, 
                        rtk_track_info, turn_signal_info, yaw_info):
    tempSensor = []
    tempSensor.append(scaleAccel(accel_pedal_info))
    tempSensor.append(scaleRtk(rtk_pos_info))
    tempSensor.append(scaleSteer(steer_info))
    tempSensor.append(scaleVel(vel_info))
    tempSensor.append(scaleBrake(brake_pedal_info))
    tempSensor.append(rtk_track_info)
    tempSensor.append(turn_signal_info)
    tempSensor.append(scaleYaw(yaw_info))
    return tempSensor


# DOS

def dos_g_n(g):
    g1 = dgl.graph(g.edges(), num_nodes=g.number_of_nodes())
    g1.ndata['feat'] = g.ndata['feat'].detach().clone()
    g1.edata['norm'] = g.edata['norm'].detach().clone()

    featureList = [0,1,2,3,4]
    ruleList = ['up', 'down']
    valueList = [1.01, 1.02, 1.03]
    for i in range(1, g1.ndata['feat'].shape[0]):
        randomFeature = random.choice(featureList)
        randomRule = random.choice(ruleList)
        randomValue = random.choice(valueList)
        if randomRule=='up':
            g1.ndata['feat'][i][randomFeature] = g1.ndata['feat'][i][randomFeature] * randomValue
        else:
            g1.ndata['feat'][i][randomFeature] = g1.ndata['feat'][i][randomFeature] / randomValue

    return g1


def dos_g_e(g):
    g1 = dgl.graph(g.edges(), num_nodes=g.number_of_nodes())
    g1.ndata['feat'] = g.ndata['feat'].detach().clone()
    g1.edata['norm'] = g.edata['norm'].detach().clone()

    ruleList = ['up', 'down']
    valueList = [1.01, 1.02, 1.03]

    for i in range(0, int(g1.edata['norm'].shape[0]*0.1)):
        randomEdge = random.randint(0, g1.edata['norm'].shape[0]-1)
        randomRule = random.choice(ruleList)
        randomValue = random.choice(valueList)
        if randomRule=='up':
            g1.edata['norm'][randomEdge] = g1.edata['norm'][randomEdge] * randomValue
        else:
            g1.edata['norm'][randomEdge] = g1.edata['norm'][randomEdge] / randomValue
    
    return g1


def dos_sensor(sensor, i):
    sensorNew = sensor.copy()

    ruleList = ['up', 'down']
    valueList = [1.01, 1.02, 1.03]

    randomRule = random.choice(ruleList)
    randomValue = random.choice(valueList)
    if randomRule=='up':
        sensorNew[i] = sensor[i] * randomValue
    else:
        sensorNew[i] = sensor[i] / randomValue
    
    return sensorNew