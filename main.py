import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import sklearn.metrics as skm
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from pathlib import Path

import dgl

from processing import *
from model import *
import util

datasetDir = '/raid2/Petrit/datasets/honda/HDD'


stimuliLabelsNr = [0,1,2]
stimuliLabels = [
                "Stop",     # 0
                "Avoid",    # 1
                "Go",       # 2
                ]

causeLabelsNr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
causeLabels = [   
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
    "Road Work",
    "Turning Vehicle",
    "Vehicle Passing with Lane Departure",
    "Other",
    "None"
]


goalLabelsNr = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
goalLabels = [
    "Right Turn",
    "Intersection Passing",
    "Merge",
    "Left Lane Change",
    "Right Lane Branch",
    "Right Lane Change",
    "Intersection Passing",
    "Left Turn",
    "Crosswalk Passing",
    "Park",
    "Railroad Passing",
    "Left Lane Branch",
    "U-Turn",
    "Park Park",
    "",
    "Park Park"
]


areaLabelsNr = [0,1,2]
areaLabels = ["Downtown", "Freeway", "Tunnel"]


sensorLabels = [
    "accel_pedal_info",
    "rtk_pos_info",
    "steer_info",
    "vel_info",
    "brake_pedal_info",
    "rtk_track_info",
    "turn_signal_info",
    "yaw_info"
]

targetLabelsNr = [0,1,2,3,4,5,6,7,8,9,10,11]
targetLabels = [
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




useCuda = True
useMultiCuda = False

nrCauseClasses = 18     # (0)  Congestion
                        # (1)  Sign
                        # (2)  Red Light
                        # (3)  Crossing Vehicle
                        # (4)  Parked Vehicle
                        # (5)  Yellow Light
                        # (6)  Crossing Pedestrian
                        # (7)  Merging Vehicle
                        # (8)  On-road Bicyclist
                        # (9)  Pedestrian Near Ego Lane
                        # (10) Park
                        # (11) On-road Motorcyclist
                        # (12) Vehicle Cut-in
                        # (13) Road Work
                        # (14) Turning Vehicle
                        # (15) Vehicle Passing with Lane Departure
                        # (16) Other
                        # (17) None

nrStimuliClasses = 3    # (0) Stop
                        # (1) Avoid
                        # (2) Go

#nodeFeatDim = 21
nodeFeatDim = 31        # (0)  x coordinate
                        # (1)  y coordinate
                        # (2)  width
                        # (3)  height
                        # (4)  confidence
                        # one-hot vecor of class
                        # (5)  Ego-Car
                        # (6)  Congestion
                        # (7)  Sign
                        # (8)  Red Light
                        # (9)  Crossing Vehicle
                        # (10) Parked Vehicle
                        # (11) Yellow Light
                        # (12) Crossing Pedestrian
                        # (13) Merging Vehicle
                        # (14) On-road Bicyclist
                        # (15) Pedestrian Near Ego Lane
                        # (16) Park
                        # (17) On-road Motorcyclist
                        # (18) Vehicle Cut-in
                        # (19) Road Work
                        # (20) Turning Vehicle
                        # (21) Vehicle Passing with Lane Departure
                        #  - 
                        # (22) person
                        # (23) bicycle
                        # (24) car
                        # (25) motorbike
                        # (26) bus
                        # (27) train
                        # (28) truck
                        # (29) traffic light
                        # (30) stop sign

sensorDim = 8           # (0) accel_pedal_info
                        # (1) rtk_pos_info
                        # (2) steer_info
                        # (3) vel_info
                        # (4) brake_pedal_info
                        # (5) rtk_track_info
                        # (6) turn_signal_info
                        # (7) yaw_info

targetDim = 12+3


nrRels = 1

hiddenDim       = 32
hiddenDimGNN    = 32
hiddenDimMLP    = 32
hiddenDimRNN    = 32
hiddenDimFC     = 32

nrRNNlayers = 1
timesteps   = 1

norm        = None      # None | batch | layer

dropoutGNN  = 0.1
dropoutMLP  = 0
dropoutFC   = 0

nrEpochs    = 100
lr          = 0.001
min_lr      = 0.00001
l2norm      = 1e-4

splitRatio  = 0.7

bestAccTrain = 0
bestAccTest  = 0

# Accuracy
trainAccs   = []
testAccs    = []

# Loss
trainLosses = []
testLosses  = []

# F1 Score Stimuli
testF1macroStimuli     = []
testF1microStimuli     = []
testF1weightStimuli    = []
testF1samplesStimuli   = []

# F1 Score Cause
testF1macroCause     = []
testF1microCause     = []
testF1weightCause    = []
testF1samplesCause   = []

# ROC Score Stimuli
testROCAUCmacroStimuli     = []
testROCAUCmicroStimuli     = []
testROCAUCweightStimuli    = []
testROCAUCsamplesStimuli   = []

# ROC Score Cause
testROCAUCmacroCause     = []
testROCAUCmicroCause     = []
testROCAUCweightCause    = []
testROCAUCsamplesCause   = []


#util.detectObjects()
#print('util.detectObjects')
#exit()


model = GNN_MLP_model(nodeFeatDim, sensorDim, targetDim, hiddenDimGNN, hiddenDimMLP, hiddenDimFC, nrStimuliClasses, nrCauseClasses, dropoutGNN, dropoutMLP, dropoutFC, norm, useCuda)


#index = torch.tensor(list(range(32)))

if useMultiCuda:
    model = nn.DataParallel(model).cuda()
    index = index.cuda()
elif useCuda:
    device = torch.device("cuda:3")
    model = model.to(device)



#print('201704151140_4050_4550_CrossingPedestrian')

#model.load_state_dict(torch.load("./ledom/117/bestModel.pt"))
#util.inference(model, './inference/201704151140_4050_4550_CrossingPedestrian', useCuda)
#util.makeVideo('./visualization/201703061541_1240_1880_PedestrianNearEgoLane')

#print('done!!!!')
#exit()


print('#####################################################################################')
ledom = './ledom/136/'
Path(ledom).mkdir(parents=True, exist_ok=True)
print(ledom)


for param in model.parameters():
    print(param.shape)

print("INFO [main.py]  Total epochs = ", nrEpochs)
print("INFO [main.py]  Model Created!")


trainset, countTrainStimuli, countTrainCause, testset, countTestStimuli, countTestCause = CreateDataset(nrStimuliClasses, nrCauseClasses, datasetDir, splitRatio, timesteps, useCuda)
[trainsetLoader, testsetLoader] = CreateBatch(trainset, testset)

WhStimuli   = [1 - (x / sum(countTrainStimuli)) for x in countTrainStimuli]
WhCause     = [1 - (x / sum(countTrainCause)) for x in countTrainCause]

np.set_printoptions(precision=4)
countTrainStimuliPerc   = [ (x/len(trainset))*100.0/timesteps for x in countTrainStimuli]
countTestStimuliPerc    = [ (x/len(testset))*100.0/timesteps  for x in countTestStimuli]
countTrainCausePerc     = [ (x/len(trainset))*100.0/timesteps for x in countTrainCause]
countTestCausePerc      = [ (x/len(testset))*100.0/timesteps  for x in countTestCause]

print("INFO [main.py]  Stimuli - Train Class Count  = ", countTrainStimuli)
print("INFO [main.py]  Stimuli - Train Class Count  = ", countTrainStimuliPerc)
print("INFO [main.py]  Stimuli - Test Class Count   = ", countTestStimuli)
print("INFO [main.py]  Stimuli - Test Class Count   = ", countTestStimuliPerc)

print("INFO [main.py]  Cause - Train Class Count  = ", countTrainCause)
print("INFO [main.py]  Cause - Train Class Count  = ", countTrainCausePerc)
print("INFO [main.py]  Cause - Test Class Count   = ", countTestCause)
print("INFO [main.py]  Cause - Test Class Count   = ", countTestCausePerc)

WhStimuli = torch.from_numpy(np.array(WhStimuli))
WhStimuli = WhStimuli.float()

WhCause = torch.from_numpy(np.array(WhCause))
WhCause = WhCause.float()

print("INFO [main.py]  Wh Stimuli = ", WhStimuli)
print("INFO [main.py]  Wh Cause   = ", WhCause)

if useMultiCuda:
    WhStimuli   = WhStimuli.cuda()
    WhCause     = WhCause.cuda()
elif useCuda:
    WhStimuli   = WhStimuli.to(device)
    WhCause     = WhCause.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2norm)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=min_lr, patience=8, verbose=True)


lossFuncStimuli = nn.CrossEntropyLoss(weight=WhStimuli)
lossFuncCause   = nn.CrossEntropyLoss(weight=WhCause)


for epoch in range(nrEpochs):
    model.train()

    trainLoss = 0
    testLoss  = 0

    correctTrain = 0
    correctTest  = 0

    correctTrainStimuli = 0
    totalTrainStimuli   = 0
    correctTestStimuli  = 0
    totalTestStimuli    = 0

    correctTrainCause   = 0
    totalTrainCause     = 0
    correctTestCause    = 0
    totalTestCause      = 0

    if useMultiCuda:
        testPredStimuliList = torch.tensor([]).cuda()
        testLblStimuliList  = torch.tensor([]).cuda()
        testPredCauseList   = torch.tensor([]).cuda()
        testLblCauseList    = torch.tensor([]).cuda()
    elif useCuda:
        testPredStimuliList = torch.tensor([]).to(device)
        testLblStimuliList  = torch.tensor([]).to(device)
        testPredCauseList   = torch.tensor([]).to(device)
        testLblCauseList    = torch.tensor([]).to(device)
    else:
        testPredStimuliList = torch.tensor([])
        testLblStimuliList  = torch.tensor([])
        testPredCauseList   = torch.tensor([])
        testLblCauseList    = torch.tensor([])

    start = time.time()

    for _, (graph,sensor,target,area,lblStimuli,lblCause) in enumerate(trainsetLoader):
        
        if useMultiCuda:
            bSensor = sensor.cuda()
            bTarget = target.cuda()
            bArea   = area.cuda()
        elif useCuda:
            bSensor = sensor.to(device)
            bTarget = target.to(device)
            bArea   = area.to(device)

        predStimuli,predCause = model(graph,bSensor,bTarget,bArea)

        lblStimuli  = lblStimuli.contiguous().view(-1)
        lblCause    = lblCause.contiguous().view(-1)

        lblStimuli  = lblStimuli.type(torch.LongTensor)
        lblCause    = lblCause.type(torch.LongTensor)
        
        if useMultiCuda:
            lblStimuli  = lblStimuli.cuda()
            lblCause    = lblCause.cuda()
        elif useCuda:
            lblStimuli  = lblStimuli.to(device)
            lblCause    = lblCause.to(device)

        lossStimuli = lossFuncStimuli(predStimuli, lblStimuli)
        lossCause   = lossFuncCause(predCause, lblCause)
        loss        = lossStimuli + lossCause

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        trainLoss += loss.detach().item()

        predStimuliMax  = predStimuli.argmax(1)
        predCauseMax    = predCause.argmax(1)

        correctTrainStimuli = (predStimuliMax==lblStimuli).sum().item()
        correctTrainCause   = (predCauseMax==lblCause).sum().item()
        correctTrain        += correctTrainStimuli + correctTrainCause

        totalTrainStimuli   += lblStimuli.size(0)
        totalTrainCause     += lblCause.size(0)
        totalTrain          = totalTrainStimuli + totalTrainCause


    trainLoss = trainLoss / len(trainsetLoader)
    trainAcc = correctTrain * 100. / totalTrain


    model.eval()
    
    if useMultiCuda:
        testLossSchdlr = (torch.tensor(0.0)).cuda()
    elif useCuda:
        testLossSchdlr = (torch.tensor(0.0)).to(device)
    else:
        testLossSchdlr = (torch.tensor(0.0)).to('cpu')

    for _, (graph,sensor,target,area,lblStimuli,lblCause) in enumerate(testsetLoader):
        with torch.no_grad():
            
            if useMultiCuda:
                bSensor = sensor.cuda()
                bTarget = target.cuda()
                bArea   = area.cuda()
            elif useCuda:
                bSensor = sensor.to(device)
                bTarget = target.to(device)
                bArea   = area.to(device)

            predStimuli,predCause = model(graph,bSensor,bTarget,bArea)

            lblStimuli  = lblStimuli.contiguous().view(-1)
            lblCause    = lblCause.contiguous().view(-1)

            lblStimuli  = lblStimuli.type(torch.LongTensor)
            lblCause    = lblCause.type(torch.LongTensor)
            
            if useMultiCuda:
                lblStimuli  = lblStimuli.cuda()
                lblCause    = lblCause.cuda()
            elif useCuda:
                lblStimuli  = lblStimuli.to(device)
                lblCause    = lblCause.to(device)

            lossStimuli = lossFuncStimuli(predStimuli, lblStimuli)
            lossCause   = lossFuncCause(predCause, lblCause)
            loss        = lossStimuli + lossCause
            
            testLoss += loss.detach().item()
            testLossSchdlr += loss.detach().item()

            predStimuliMax  = predStimuli.argmax(1)
            predCauseMax    = predCause.argmax(1)

            correctTestStimuli = (predStimuliMax==lblStimuli).sum().item()
            correctTestCause   = (predCauseMax==lblCause).sum().item()
            correctTest        += correctTestStimuli + correctTestCause

            totalTestStimuli   += lblStimuli.size(0)
            totalTestCause     += lblCause.size(0)
            totalTest          = totalTestStimuli + totalTestCause

            testPredStimuliList = torch.cat((testPredStimuliList, predStimuliMax),  dim=0)
            testLblStimuliList  = torch.cat((testLblStimuliList, lblStimuli),       dim=0)
            testPredCauseList   = torch.cat((testPredCauseList, predCauseMax),      dim=0)
            testLblCauseList    = torch.cat((testLblCauseList, lblCause),           dim=0)


    scheduler.step(testLossSchdlr)
    testLoss = testLoss / len(testsetLoader)
    testAcc = correctTest * 100. / totalTest

    end = time.time()

    print('------------------------------------------------------------------')
    print("Epoch {:04d}                    |  Epoch Time   =".format(epoch+1), time.strftime("%H:%M:%S", time.gmtime(end-start)))
    print("Train Accuracy   = {:.4f}    |  Train Loss   = {:.4f} ".format(trainAcc, trainLoss))
    print("Test Accuracy    = {:.4f}    |  Test Loss    = {:.4f} ".format(testAcc, testLoss))

    if trainAcc > bestAccTrain:
        bestAccTrain = trainAcc
        torch.save(model.state_dict(), ledom+'bestModel.pt')

    if testAcc > bestAccTest:
        bestAccTest = testAcc

    np.set_printoptions(precision=6)

    # list of Accuracies and Losses for vizualization
    trainAccs.append(trainAcc)
    trainLosses.append(trainLoss)
    testAccs.append(testAcc)
    testLosses.append(testLoss)

    # to CPU
    testPredStimuliList2 = testPredStimuliList.cpu()
    testLblStimuliList2  = testLblStimuliList.cpu()
    testPredCauseList2   = testPredCauseList.cpu()
    testLblCauseList2    = testLblCauseList.cpu()
    
    # F1 Score
    testF1macroStimuli.append(skm.f1_score(testLblStimuliList2, testPredStimuliList2, average='macro')*100)
    testF1microStimuli.append(skm.f1_score(testLblStimuliList2, testPredStimuliList2, average='micro')*100)
    testF1weightStimuli.append(skm.f1_score(testLblStimuliList2, testPredStimuliList2, average='weighted')*100)
        
    testF1macroCause.append(skm.f1_score(testLblCauseList2, testPredCauseList2, average='macro')*100)
    testF1microCause.append(skm.f1_score(testLblCauseList2, testPredCauseList2, average='micro')*100)
    testF1weightCause.append(skm.f1_score(testLblCauseList2, testPredCauseList2, average='weighted')*100)
    
    # Convert two outputs to one-hot-vectors
    testPredStimuliList1hot = label_binarize(testPredStimuliList2, classes=stimuliLabelsNr)
    testLblStimuliList1hot = label_binarize(testLblStimuliList2, classes=stimuliLabelsNr)
    
    testPredCauseList1hot = label_binarize(testPredCauseList2, classes=causeLabelsNr)
    testLblCauseList1hot = label_binarize(testLblCauseList2, classes=causeLabelsNr)

    # ROC Score
    testROCAUCmicroStimuli.append(skm.roc_auc_score(testLblStimuliList1hot, testPredStimuliList1hot, multi_class='ovo', average='micro')*100)
    testROCAUCmacroStimuli.append(skm.roc_auc_score(testLblStimuliList1hot, testPredStimuliList1hot, multi_class='ovo', average='macro')*100)
    testROCAUCweightStimuli.append(skm.roc_auc_score(testLblStimuliList1hot, testPredStimuliList1hot, multi_class='ovo', average='weighted')*100)
    
    testROCAUCmicroCause.append(skm.roc_auc_score(testLblCauseList1hot, testPredCauseList1hot, multi_class='ovo', average='micro')*100)
    #testROCAUCmacroCause.append(skm.roc_auc_score(testLblCauseList1hot, testPredCauseList1hot, multi_class='ovo', average='macro')*100)
    #testROCAUCweightCause.append(skm.roc_auc_score(testLblCauseList1hot, testPredCauseList1hot, multi_class='ovo', average='weighted')*100)
    
    # plotting the results
    plt.plot(trainAccs)
    plt.plot(testAccs)
    #plt.ylim(bottom=50, top=90)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.title('Model Accuracy')
    plt.savefig('./ledom/acc.png')
    plt.clf()

    plt.plot(trainLosses)
    plt.plot(testLosses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.title('Model Loss')
    plt.savefig('./ledom/loss.png')
    plt.clf()


print('------------------------------------------------------------------')

print('#####################################################################################')
print("INFO [HL_DM_AD_GCN-RNN]  Best Train Accuracy    = {:.4f}".format(bestAccTrain))
print("INFO [HL_DM_AD_GCN-RNN]  Best Test Accuracy     = {:.4f}".format(bestAccTest))
print('#####################################################################################')


#print(skm.classification_report(testTargList2, testPredList2, labels=classesLabelsNr, target_names=classesLabels, zero_division=0))

#print("INFO [HL_DM_AD_GCN-RNN]  precision_recall_fscore_support (macro)    = ", skm.precision_recall_fscore_support(testTargList2, testPredList2, average='macro'))
#print("INFO [HL_DM_AD_GCN-RNN]  precision_recall_fscore_support (micro)    = ", skm.precision_recall_fscore_support(testTargList2, testPredList2, average='micro'))
#print("INFO [HL_DM_AD_GCN-RNN]  precision_recall_fscore_support (weighted) = ", skm.precision_recall_fscore_support(testTargList2, testPredList2, average='weighted'))
#
#print('#####################################################################################')

print(ledom)
print('#####################################################################################')

plt.figure()
plt.plot(trainAccs)
plt.plot(testAccs)
#plt.ylim(bottom=50, top=90)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.title('Model Accuracy')
plt.savefig(ledom+'acc.png')
plt.clf()

plt.figure()
plt.plot(trainLosses)
plt.plot(testLosses)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.title('Model Loss')
plt.savefig(ledom+'loss.png')
plt.clf()

plt.figure()
plt.plot(testF1macroStimuli)
plt.plot(testF1microStimuli)
plt.plot(testF1weightStimuli)
#plt.plot(testF1samples)
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['f1_macro_stimuli', 'f1_micro_stimuli', 'f1_weighted_stimuli'])
plt.title('Model F1 Score Stimuli')
plt.savefig(ledom+'f1score_stimuli.png')
plt.clf()

plt.figure()
plt.plot(testF1macroCause)
plt.plot(testF1microCause)
plt.plot(testF1weightCause)
#plt.plot(testF1samples)
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['f1_macro_cause', 'f1_micro_cause', 'f1_weighted_cause'])
plt.title('Model F1 Score Cause')
plt.savefig(ledom+'f1score_cause.png')
plt.clf()

plt.figure()
plt.plot(testROCAUCmicroStimuli)
plt.plot(testROCAUCmacroStimuli)
plt.plot(testROCAUCweightStimuli)
#plt.plot(testROCAUCsamples)
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['ROC_micro_stimuli', 'ROC_macro_stimuli', 'ROC_weighted_stimuli'])
plt.title('Model ROC Score Stimuli')
plt.savefig(ledom+'rocScore_stimuli.png')
plt.clf()

plt.figure()
plt.plot(testROCAUCmicroCause)
plt.plot(testROCAUCmacroCause)
plt.plot(testROCAUCweightCause)
#plt.plot(testROCAUCsamples)
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['ROC_micro_cause', 'ROC_macro_cause', 'ROC_weighted_cause'])
plt.title('Model ROC Score Cause')
plt.savefig(ledom+'rocScore_cause.png')
plt.clf()


with open(ledom+'trainAccs', "wb") as fp:
    pickle.dump(trainAccs, fp)
with open(ledom+'testAccs', "wb") as fp:
    pickle.dump(testAccs, fp)


with open(ledom+'trainLosses', "wb") as fp:
    pickle.dump(trainLosses, fp)
with open(ledom+'testLosses', "wb") as fp:
    pickle.dump(testLosses, fp)


with open(ledom+'testF1macroStimuli', "wb") as fp:
    pickle.dump(testF1macroStimuli, fp)
with open(ledom+'testF1macroCause', "wb") as fp:
    pickle.dump(testF1macroCause, fp)

with open(ledom+'testF1microStimuli', "wb") as fp:
    pickle.dump(testF1microStimuli, fp)
with open(ledom+'testF1microCause', "wb") as fp:
    pickle.dump(testF1microCause, fp)

with open(ledom+'testF1weightStimuli', "wb") as fp:
    pickle.dump(testF1weightStimuli, fp)
with open(ledom+'testF1weightCause', "wb") as fp:
    pickle.dump(testF1weightCause, fp)


with open(ledom+'testROCAUCmacroStimuli', "wb") as fp:
    pickle.dump(testROCAUCmacroStimuli, fp)
with open(ledom+'testROCAUCmacroCause', "wb") as fp:
    pickle.dump(testROCAUCmacroCause, fp)

with open(ledom+'testROCAUCmicroStimuli', "wb") as fp:
    pickle.dump(testROCAUCmicroStimuli, fp)
with open(ledom+'testROCAUCmicroCause', "wb") as fp:
    pickle.dump(testROCAUCmicroCause, fp)

with open(ledom+'testROCAUCweightStimuli', "wb") as fp:
    pickle.dump(testROCAUCweightStimuli, fp)
with open(ledom+'testROCAUCweightCause', "wb") as fp:
    pickle.dump(testROCAUCweightCause, fp)


testConfusionMatrixStimuli  = skm.confusion_matrix(testLblStimuliList2, testPredStimuliList2)
testConfusionMatrixCause    = skm.confusion_matrix(testLblCauseList2,   testPredCauseList2)

util.plotConfusionMatrix(testConfusionMatrixStimuli, ledom+'testStimuliCM.png', 'stimuli')
util.plotConfusionMatrix(testConfusionMatrixCause,   ledom+'testCauseCM.png',   'cause')