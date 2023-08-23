import os
from random import sample, shuffle
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm
import random

import dgl
from dgl.data.utils import load_graphs
from dgl.data import MiniGCDataset

import util


class MiniGCNDataset(object):
    def __init__(self):
        super(MiniGCNDataset, self).__init__()
        self.graphs  = []
        self.sensor  = []
        self.target  = []
        self.area    = []
        self.stimuli = []
        self.cause   = []

    def __len__(self):
        return len(self.graphs)

    def __add__(self, graph, sensor, target, area, stimuli, cause):
        self.graphs.append(graph)
        self.sensor.append(sensor)
        self.target.append(target)
        self.area.append(area)
        self.stimuli.append(stimuli)
        self.cause.append(cause)

    def __getitem__(self, idx):
        return self.graphs[idx], self.sensor[idx], self.target[idx], self.area[idx], self.stimuli[idx], self.cause[idx]




def CreateDataset(nrStimuliClasses, nrCauseClasses, datasetDir, ratio, timesteps=1, useCuda=False):

    countTrainStimuli   = [0.0] * nrStimuliClasses
    countTestStimuli    = [0.0] * nrStimuliClasses
    
    countTrainCause = [0.0] * nrCauseClasses
    countTestCause  = [0.0] * nrCauseClasses

    graphList   = []
    sensorList  = []
    areaList    = []
    attenList   = []
    causeList   = []
    stimuliList = []
    targetList  = []
    targetList2 = []

    graphTrainList   = []
    sensorTrainList  = []
    areaTrainList    = []
    attenTrainList   = []
    causeTrainList   = []
    stimuliTrainList = []
    targetTrainList  = []
    targetTrainList2 = []
    
    graphTestList   = []
    sensorTestList  = []
    areaTestList    = []
    attenTestList   = []
    causeTestList   = []
    stimuliTestList = []
    targetTestList  = []
    targetTestList2 = []

    noteDir         = './hdd/note/'
    noteFileList    = [f.path for f in os.scandir(noteDir) if os.path.isfile(f)]
    for note in tqdm.tqdm(noteFileList):

        npNote = np.load(note)

        startIndexs = np.where(npNote==49)
        if len(startIndexs[0])==0:
            startIndex = 0
        else:
            startIndex = startIndexs[0][0]
            
        endIndexs = np.where(npNote==50)
        if len(endIndexs[0])==0:
            endIndex = len(npNote)-1
        else:
            endIndex = endIndexs[0][len(endIndexs[0])-1]


        folderName = note.split('/')[3][:-4]
        cameraDir           = datasetDir + '/camera/' + folderName
        accel_pedal_info    = readInputFeature('accel_pedal_info', folderName)
        area                = readInputFeature('area', folderName)
        attention           = readInputFeature('attention', folderName)
        brake_pedal_info    = readInputFeature('brake_pedal_info', folderName)
        cause               = readInputFeature('cause2', folderName)
        rtk_pos_info        = readInputFeature('rtk_pos_info', folderName)
        rtk_track_info      = readInputFeature('rtk_track_info', folderName)
        steer_info          = readInputFeature('steer_info', folderName)
        stimuli             = readInputFeature('stimuli2', folderName)
        target              = readInputFeature('target2', folderName)
        turn_signal_info    = readInputFeature('turn_signal_info', folderName)
        vel_info            = readInputFeature('vel_info', folderName)
        yaw_info            = readInputFeature('yaw_info', folderName)

        for i in range(startIndex, endIndex):
            if os.path.isfile('./hdd/graphsPre/'+folderName+'/'+str(i)+'.bin'):
                
                # DUS
                '''
                intersectionFlag = False
                keeplaneFlag = False
                
                if target[i]-1==0 and cause[i]==17:
                    if random.randint(0,100) < 50:
                        intersectionFlag = True
                elif target[i]-1==11 and cause[i]==17:
                    if random.randint(0,100) < 50:
                        keeplaneFlag = True
                else:
                    intersectionFlag = True
                    keeplaneFlag = True

                if intersectionFlag or keeplaneFlag:
                '''
                g = load_graphs('./hdd/graphsPre/'+folderName+'/'+str(i)+'.bin')
                
                graphList.append(g[0][0])
                tempSensor = []
                tempSensor.append(util.scaleAccel(accel_pedal_info[i]))
                tempSensor.append(util.scaleRtk(rtk_pos_info[i]))
                tempSensor.append(util.scaleSteer(steer_info[i]))
                tempSensor.append(util.scaleVel(vel_info[i]))
                tempSensor.append(util.scaleBrake(brake_pedal_info[i]))
                tempSensor.append(rtk_track_info[i])
                tempSensor.append(turn_signal_info[i])
                tempSensor.append(util.scaleYaw(yaw_info[i]))
                sensorList.append(tempSensor)
                targetList.append(util.get_one_hot(target[i]-1,12)[0].tolist())
                targetList2.append(target[i]-1)
                if area[i]==0:
                    areaList.append(util.get_one_hot(0,3)[0].tolist())
                else:
                    areaList.append(util.get_one_hot(area[i]-32,3)[0].tolist())
                attenList.append(attention[i])
                stimuliList.append(stimuli[i])
                causeList.append(cause[i])


    datasetList = list(zip(graphList, sensorList, targetList, areaList, stimuliList, causeList))
    random.shuffle(datasetList)
    graphList2, sensorList2, targetList2, areaList2, stimuliList2, causeList2 = zip(*datasetList)

    
    splitValue = int( len(graphList2) * 0.34 )

    trainGraph      = graphList2[:splitValue]
    trainSensor     = sensorList2[:splitValue]
    trainTarget     = targetList2[:splitValue]
    trainArea       = areaList2[:splitValue]
    trainStimuli    = stimuliList2[:splitValue]
    trainCause      = causeList2[:splitValue]

    testGraph       = graphList2[splitValue:]
    testSensor      = sensorList2[splitValue:]
    testTarget      = targetList2[splitValue:]
    testArea        = areaList2[splitValue:]
    testStimuli     = stimuliList2[splitValue:]
    testCause       = causeList2[splitValue:]

    trainGraph      = list(trainGraph)
    trainSensor     = list(trainSensor)
    trainTarget     = list(trainTarget)
    trainArea       = list(trainArea)
    trainStimuli    = list(trainStimuli)
    trainCause      = list(trainCause)

    testGraph       = list(testGraph)
    testSensor      = list(testSensor)
    testTarget      = list(testTarget)
    testArea        = list(testArea)
    testStimuli     = list(testStimuli)
    testCause       = list(testCause)

    
    for d in range(0, len(testGraph)-1):
        if testTarget[d][11]==1 or testCause[d]==17 or testCause[d]==16:
            if random.randint(0,100) < 44:
                trainGraph.append(testGraph[d])
                trainSensor.append(testSensor[d])
                trainTarget.append(testTarget[d])
                trainArea.append(testArea[d])
                trainStimuli.append(testStimuli[d])
                trainCause.append(testCause[d])
            else:
                graphTestList.append(testGraph[d])
                sensorTestList.append(testSensor[d])
                targetTestList.append(testTarget[d])
                areaTestList.append(testArea[d])
                stimuliTestList.append(testStimuli[d])
                causeTestList.append(testCause[d])
        elif testTarget[d][0]==1 or testCause[d]==1 or testCause[d]==3:
            if random.randint(0,100) < 24:
                trainGraph.append(testGraph[d])
                trainSensor.append(testSensor[d])
                trainTarget.append(testTarget[d])
                trainArea.append(testArea[d])
                trainStimuli.append(testStimuli[d])
                trainCause.append(testCause[d])
            else:
                graphTestList.append(testGraph[d])
                sensorTestList.append(testSensor[d])
                targetTestList.append(testTarget[d])
                areaTestList.append(testArea[d])
                stimuliTestList.append(testStimuli[d])
                causeTestList.append(testCause[d])
        else:
            graphTestList.append(testGraph[d])
            sensorTestList.append(testSensor[d])
            targetTestList.append(testTarget[d])
            areaTestList.append(testArea[d])
            stimuliTestList.append(testStimuli[d])
            causeTestList.append(testCause[d])
    

    #####################################################
    # DOS

    for j in range(0, len(trainGraph)):

        graphTrainList.append(trainGraph[j])
        sensorTrainList.append(trainSensor[j])
        targetTrainList.append(trainTarget[j])
        targetTrainList2.append(trainTarget[j].index(1))
        if trainArea[j]==0:
            areaTrainList.append([1,0,0])
        else:
            areaTrainList.append(trainArea[j])
        stimuliTrainList.append(trainStimuli[j])
        causeTrainList.append(trainCause[j])

        # Maneuver + Graph Nodes
        if trainCause[j]!=17:
            if random.randint(0,100) < 44:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Maneuver + Graph Nodes
        if trainCause[j]!=17:
            if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Graph Edges
        if (trainTarget[j][11])!=1:
            if trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_e(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Graph Nodes - 2
        if (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Graph Edges - 2
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_e(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Graph Nodes - 3
        if (trainTarget[j][11])!=1:
            if trainCause[j]!=16 and trainCause[j]!=17:
                if random.randint(0,100) < 64:
                    glt = util.dos_g_n(trainGraph[j])
                    graphTrainList.append(glt)
                    sensorTrainList.append(trainSensor[j])
                    targetTrainList.append(trainTarget[j])
                    targetTrainList2.append(trainTarget[j].index(1))
                    if trainArea[j]==0:
                        areaTrainList.append([1,0,0])
                    else:
                        areaTrainList.append(trainArea[j])
                    stimuliTrainList.append(trainStimuli[j])
                    causeTrainList.append(trainCause[j])

        # Cause + Graph Edges - 3
        if (trainTarget[j][11])!=1 and trainStimuli[j]==1:
            if trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_e(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Graph Nodes - 4
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                if random.randint(0,100) < 64:
                    glt = util.dos_g_n(trainGraph[j])
                    graphTrainList.append(glt)
                    sensorTrainList.append(trainSensor[j])
                    targetTrainList.append(trainTarget[j])
                    targetTrainList2.append(trainTarget[j].index(1))
                    if trainArea[j]==0:
                        areaTrainList.append([1,0,0])
                    else:
                        areaTrainList.append(trainArea[j])
                    stimuliTrainList.append(trainStimuli[j])
                    causeTrainList.append(trainCause[j])

        # Cause + Graph Nodes - 4
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Target + Cause + Graph Nodes - 5
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Target + Cause + Graph Edges - 5
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_e(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Target + Cause + Graph Nodes - 5
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_n(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Target + Cause + Graph Edges - 5
        if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                glt = util.dos_g_e(trainGraph[j])
                graphTrainList.append(glt)
                sensorTrainList.append(trainSensor[j])
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Acceleration
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][0]!=0:
            if trainCause[j]!=0 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],0)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + RTK
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][1]!=0:
            if trainCause[j]!=0 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],1)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Steering
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][2]!=0:
            if trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],2)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Velocity
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][3]!=0:
            if trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],3)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Brake
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][4]!=0:
            if trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],4)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Yaw
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][5]!=0:
            if trainCause[j]!=0 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],5)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])
        
        # Cause + Acceleration
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][0]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],0)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + RTK
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][1]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],1)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Steering
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][2]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],2)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Velocity
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][3]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],3)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Brake
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][4]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],4)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])

        # Cause + Yaw
        if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1 and trainSensor[j][5]!=0:
            if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                graphTrainList.append(trainGraph[j])
                tempTrainSensor1 = util.dos_sensor(trainSensor[j],5)
                sensorTrainList.append(tempTrainSensor1)
                targetTrainList.append(trainTarget[j])
                targetTrainList2.append(trainTarget[j].index(1))
                if trainArea[j]==0:
                    areaTrainList.append([1,0,0])
                else:
                    areaTrainList.append(trainArea[j])
                stimuliTrainList.append(trainStimuli[j])
                causeTrainList.append(trainCause[j])
        
        for r in range(20):
            rr = random.randint(0,len(trainSensor[j])-1)
            if trainSensor[j][rr]!=0:
                if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
                    if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                        graphTrainList.append(trainGraph[j])
                        tempTrainSensor1 = util.dos_sensor(trainSensor[j],rr)
                        sensorTrainList.append(tempTrainSensor1)
                        targetTrainList.append(trainTarget[j])
                        targetTrainList2.append(trainTarget[j].index(1))
                        if trainArea[j]==0:
                            areaTrainList.append([1,0,0])
                        else:
                            areaTrainList.append(trainArea[j])
                        stimuliTrainList.append(trainStimuli[j])
                        causeTrainList.append(trainCause[j])
        
        for r in range(44):
            if (trainTarget[j][0])!=1 and (trainTarget[j][11])!=1:
                if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=3 and trainCause[j]!=16 and trainCause[j]!=17:
                    glt = util.dos_g_n(trainGraph[j])
                    graphTrainList.append(glt)
                    sensorTrainList.append(trainSensor[j])
                    targetTrainList.append(trainTarget[j])
                    targetTrainList2.append(trainTarget[j].index(1))
                    if trainArea[j]==0:
                        areaTrainList.append([1,0,0])
                    else:
                        areaTrainList.append(trainArea[j])
                    stimuliTrainList.append(trainStimuli[j])
                    causeTrainList.append(trainCause[j])
        
        for r in range(44):
            if (trainTarget[j][0])!=1 and (trainTarget[j][1])!=1 and (trainTarget[j][2])!=1 and (trainTarget[j][11])!=1:
                if trainCause[j]!=0 and trainCause[j]!=1 and trainCause[j]!=2 and trainCause[j]!=16 and trainCause[j]!=17:
                    glt = util.dos_g_e(trainGraph[j])
                    graphTrainList.append(glt)
                    sensorTrainList.append(trainSensor[j])
                    targetTrainList.append(trainTarget[j])
                    targetTrainList2.append(trainTarget[j].index(1))
                    if trainArea[j]==0:
                        areaTrainList.append([1,0,0])
                    else:
                        areaTrainList.append(trainArea[j])
                    stimuliTrainList.append(trainStimuli[j])
                    causeTrainList.append(trainCause[j])

        #####################################################


    datasetTrainList = list(zip(graphTrainList, sensorTrainList, targetTrainList, areaTrainList, stimuliTrainList, causeTrainList))
    random.shuffle(datasetTrainList)
    graphTrainList2, sensorTrainList2, targetTrainList2, areaTrainList2, stimuliTrainList2, causeTrainList2 = zip(*datasetTrainList)


    print("INFO [processing.py]  Number of train datapoints  = ", len(graphTrainList))
    print("INFO [processing.py]  Number of test datapoints   = ", len(graphTestList))

    traintrainTarget = []
    testtestTarget   = []
    for k in range(len(targetTrainList2)):
        traintrainTarget.append(targetTrainList2[k].index(1))
    for l in range(len(targetTestList)):
        testtestTarget.append(targetTestList[l].index(1))

    
    print('----')
    print('target train')
    for i in range(0,20):
        print( str(i) + ' - ' + str(traintrainTarget.count(i)) )

    print('----')
    print('stimuli train')
    for i in range(0,20):
        print( str(i) + ' - ' + str(stimuliTrainList2.count(i)) )

    print('----')
    print('cause train')
    for i in range(0,20):
        print( str(i) + ' - ' + str(causeTrainList2.count(i)) )
    

    print('----')
    print('target test')
    for i in range(0,20):
        print( str(i) + ' - ' + str(testtestTarget.count(i)) )

    print('----')
    print('stimuli test')
    for i in range(0,20):
        print( str(i) + ' - ' + str(stimuliTestList.count(i)) )

    print('----')
    print('cause test')
    for i in range(0,20):
        print( str(i) + ' - ' + str(causeTestList.count(i)) )

    
    trainset = MiniGCNDataset()
    testset  = MiniGCNDataset()
    
    for i in range(0,len(graphTrainList2)):
        trainset.__add__(graphTrainList2[i], sensorTrainList2[i], targetTrainList2[i], areaTrainList2[i], stimuliTrainList2[i], causeTrainList2[i])
        countTrainStimuli[stimuliTrainList2[i]] += 1
        countTrainCause[causeTrainList2[i]] += 1
    
    for i in range(0,len(graphTestList)):
        testset.__add__(graphTestList[i], sensorTestList[i], targetTestList[i], areaTestList[i], stimuliTestList[i], causeTestList[i])
        countTestStimuli[stimuliTestList[i]] += 1
        countTestCause[causeTestList[i]] += 1


    print("INFO [processing.py]  Dataset Created!")
    print("INFO [processing.py]  Number of Training Graphs = ", len(trainset))
    print("INFO [processing.py]  Number of Testing  Graphs = ", len(testset))

    return trainset, countTrainStimuli, countTrainCause, testset, countTestStimuli, countTestCause


def CreateDatasetSeq(nrStimuliClasses, nrCauseClasses, datasetDir, ratio, timesteps=10, useCuda=False):

    countTrainStimuli   = [0.0] * nrStimuliClasses
    countTestStimuli    = [0.0] * nrStimuliClasses
    
    countTrainCause = [0.0] * nrCauseClasses
    countTestCause  = [0.0] * nrCauseClasses

    graphListRNN   = []
    gEdgeListRNN   = []
    gNodeFeatRNN   = []
    gEdgeNormRNN   = []
    sensorListRNN  = []
    areaListRNN    = []
    attenListRNN   = []
    causeListRNN   = []
    stimuliListRNN = []
    targetListRNN  = []

    noteDir         = './hdd/note/'
    noteFileList    = [f.path for f in os.scandir(noteDir) if os.path.isfile(f)]
    random.shuffle(noteFileList)
    for note in tqdm.tqdm(noteFileList[0:1]):

        npNote = np.load(note)

        startIndexs = np.where(npNote==49)
        if len(startIndexs[0])==0:
            startIndex = 0
        else:
            startIndex = startIndexs[0][0]
            
        endIndexs = np.where(npNote==50)
        if len(endIndexs[0])==0:
            endIndex = len(npNote)
        else:
            endIndex = endIndexs[0][len(endIndexs[0])-1]


        folderName = note.split('/')[3][:-4]
        cameraDir           = datasetDir + '/camera/' + folderName
        accel_pedal_info    = readInputFeature('accel_pedal_info', folderName)
        area                = readInputFeature('area', folderName)
        attention           = readInputFeature('attention', folderName)
        brake_pedal_info    = readInputFeature('brake_pedal_info', folderName)
        cause               = readInputFeature('cause2', folderName)
        rtk_pos_info        = readInputFeature('rtk_pos_info', folderName)
        rtk_track_info      = readInputFeature('rtk_track_info', folderName)
        steer_info          = readInputFeature('steer_info', folderName)
        stimuli             = readInputFeature('stimuli2', folderName)
        target              = readInputFeature('target2', folderName)
        turn_signal_info    = readInputFeature('turn_signal_info', folderName)
        vel_info            = readInputFeature('vel_info', folderName)
        yaw_info            = readInputFeature('yaw_info', folderName)

        graphList   = []
        sensorList  = []
        areaList    = []
        attenList   = []
        causeList   = []
        stimuliList = []
        targetList  = []

        downsample = 0

        for i in range(startIndex, endIndex, 4):
            for j in range(i, i+timesteps):
                if i+timesteps<endIndex:                    
                    if os.path.isfile('./hdd/graphsPre/'+folderName+'/'+str(j)+'.bin'):

                        g = load_graphs('./hdd/graphsPre/'+folderName+'/'+str(j)+'.bin')
                        graphList.append(g[0][0])

                        tempSensor = []
                        tempSensor.append(util.scaleAccel(accel_pedal_info[j]))
                        tempSensor.append(util.scaleRtk(rtk_pos_info[j]))
                        tempSensor.append(util.scaleSteer(steer_info[j]))
                        tempSensor.append(util.scaleVel(vel_info[j]))
                        tempSensor.append(util.scaleBrake(brake_pedal_info[j]))
                        tempSensor.append(rtk_track_info[j])
                        tempSensor.append(turn_signal_info[j])
                        tempSensor.append(util.scaleYaw(yaw_info[j]))
                        sensorList.append(tempSensor)
                        targetList.append(util.get_one_hot(target[j]-1,12)[0].tolist())
                        if area[j]==0:
                            areaList.append(util.get_one_hot(0,3)[0].tolist())
                        else:
                            areaList.append(util.get_one_hot(area[j]-32,3)[0].tolist())
                        attenList.append(attention[j])
                        stimuliList.append(stimuli[j])
                        causeList.append(cause[j])

                    else:
                        graphList   = []
                        sensorList  = []
                        areaList    = []
                        attenList   = []
                        causeList   = []
                        stimuliList = []
                        targetList  = []
                        break
            
            downsample = 0
            if len(graphList)!=0:
                for i in range(0,timesteps):
                    if targetList[i][11]==1 and stimuliList[i]==2 and causeList[i]==17:
                        downsample += 1

                if downsample!=timesteps:                        
                    graphListRNN.append(graphList)
                    sensorListRNN.append(sensorList)
                    areaListRNN.append(areaList)
                    attenListRNN.append(attenList)
                    causeListRNN.append(causeList)
                    stimuliListRNN.append(stimuliList)
                    targetListRNN.append(targetList)
                #else:
                #    if random.randint(0,100) < 50:
                #        graphListRNN.append(graphList)
                #        sensorListRNN.append(sensorList)
                #        areaListRNN.append(areaList)
                #        attenListRNN.append(attenList)
                #        causeListRNN.append(causeList)
                #        stimuliListRNN.append(stimuliList)
                #        targetListRNN.append(targetList)
                    
                graphList   = []
                sensorList  = []
                areaList    = []
                attenList   = []
                causeList   = []
                stimuliList = []
                targetList  = []
        

    print("INFO [processing.py]  Number of datapoints  = ", len(graphListRNN)*timesteps)

    #for i in range(0,20):
    #    print( str(i) + '-' + str(targetList.count(i)) )


    datasetList = list(zip(graphListRNN,sensorListRNN,targetListRNN,areaListRNN,stimuliListRNN,causeListRNN))
    random.shuffle(datasetList)
    graphListRNN2,sensorListRNN2,targetListRNN2,areaListRNN2,stimuliListRNN2,causeListRNN2 = zip(*datasetList)

    splitValue = int( len(graphListRNN2) * ratio )

    trainGraph      = graphListRNN2[:splitValue]
    trainSensor     = sensorListRNN2[:splitValue]
    trainTarget     = targetListRNN2[:splitValue]
    trainArea       = areaListRNN2[:splitValue]
    trainStimuli    = stimuliListRNN2[:splitValue]
    trainCause      = causeListRNN2[:splitValue]
    
    testGraph       = graphListRNN2[splitValue:]
    testSensor      = sensorListRNN2[splitValue:]
    testTarget      = targetListRNN2[splitValue:]
    testArea        = areaListRNN2[splitValue:]
    testStimuli     = stimuliListRNN2[splitValue:]
    testCause       = causeListRNN2[splitValue:]
    
    trainset = MiniGCNDataset()
    testset  = MiniGCNDataset()
    
    for i in range(0,len(trainGraph)):
        trainset.__add__(trainGraph[i], trainSensor[i], trainTarget[i], trainArea[i], trainStimuli[i], trainCause[i])
        for j in range(0,timesteps):
            countTrainStimuli[trainStimuli[i][j]] += 1
            countTrainCause[trainCause[i][j]] += 1
    
    for i in range(0,len(testGraph)):
        testset.__add__(testGraph[i], testSensor[i], testTarget[i], testArea[i], testStimuli[i], testCause[i])
        for j in range(0,timesteps):
            countTestStimuli[testStimuli[i][j]] += 1
            countTestCause[testCause[i][j]] += 1


    print("INFO [processing.py]  Dataset Created!")
    print("INFO [processing.py]  Number of Training Graphs = ", len(trainset)*timesteps)
    print("INFO [processing.py]  Number of Testing  Graphs = ", len(testset)*timesteps)

    return trainset, countTrainStimuli, countTrainCause, testset, countTestStimuli, countTestCause




def CreateBatch(trainset, testset):
    trainsetLoader  = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=Collate)
    testsetLoader   = DataLoader(testset,  batch_size=32, shuffle=True, collate_fn=Collate)
    return [trainsetLoader, testsetLoader]


def CreateBatchSeq(trainset, testset):
    trainsetLoader  = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True, collate_fn=CollateSeq)
    testsetLoader   = DataLoader(testset,  batch_size=64, shuffle=True, drop_last=True, collate_fn=CollateSeq)
    return [trainsetLoader, testsetLoader]




def Collate(samples):
    graphs, sensor, target, area, stimuli, cause = map(list, zip(*samples))
    bGraphs  = dgl.batch(graphs)
    bSensor  = torch.tensor(sensor, dtype=torch.float32)
    bTarget  = torch.tensor(target, dtype=torch.float32)
    bArea    = torch.tensor(area, dtype=torch.float32)
    bStimuli = torch.tensor(stimuli, dtype=torch.float32)
    bCause   = torch.tensor(cause, dtype=torch.float32)
    return [bGraphs, bSensor, bTarget, bArea, bStimuli, bCause]
    

def CollateSeq(samples):
    graphs, sensor, target, area, stimuli, cause = map(list, zip(*samples))

    bSensor  = torch.tensor(sensor,     dtype=torch.float32)
    bTarget  = torch.tensor(target,     dtype=torch.float32)
    bArea    = torch.tensor(area,       dtype=torch.float32)
    bStimuli = torch.tensor(stimuli,    dtype=torch.float32)
    bCause   = torch.tensor(cause,      dtype=torch.float32)
    
    bGraphs = []
    for i in range(0, len(graphs)):
        bGraphs.append(dgl.batch(graphs[i]))
    
    return [graphs, bSensor, bTarget, bArea, bStimuli, bCause]




def readInputFeature(inputFeat, folder):
    inputFeat   = './hdd/'+inputFeat+'/'+folder+'.npy'
    return np.load(inputFeat)