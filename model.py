import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from dgl import DGLGraph
from functools import partial

from layers import gnn_layer, gat_layer




####################################################################################




class GNN_MLP_model(nn.Module):
    def __init__(self, inputDimGNN, inputDimSensor, inputDimTarget, hiddenDimGNN, hiddenDimMLP, hiddenDimFC,
                    outputDimStimuli, outputDimClause, dropoutGNN, dropoutMLP, dropoutFC, norm, useCuda=False):
        super(GNN_MLP_model, self).__init__()

        self.inputDimGNN        = inputDimGNN
        self.inputDimSensor     = inputDimSensor
        self.inputDimTarget     = inputDimTarget
        self.hiddenDimGNN       = hiddenDimGNN
        self.hiddenDimMLP       = hiddenDimMLP
        self.hiddenDimFC        = hiddenDimFC
        self.outputDimStimuli   = outputDimStimuli
        self.outputDimClause    = outputDimClause
        self.activation         = nn.ReLU()
        self.useCuda            = useCuda


        # normalization
        if norm == 'batch':
            self.normGNN = nn.BatchNorm1d(self.hiddenDimGNN, affine=True)
            self.normMLP = nn.BatchNorm1d(self.hiddenDimMLP, affine=True)
            self.normFC  = nn.BatchNorm1d(self.hiddenDimFC,  affine=True)
        elif norm == 'layer':
            self.normGNN = nn.LayerNorm(self.hiddenDimGNN, elementwise_affine=True)
            self.normMLP = nn.LayerNorm(self.hiddenDimMLP, elementwise_affine=True)
            self.normFC  = nn.LayerNorm(self.hiddenDimFC, elementwise_affine=True)
        elif norm == 'instance':
            self.normGNN = nn.InstanceNorm1d(self.hiddenDimGNN, affine=False)
            self.normMLP = nn.InstanceNorm1d(self.hiddenDimMLP, affine=False)
            self.normFC  = nn.InstanceNorm1d(self.hiddenDimFC, affine=False)
        else:
            self.normGNN = None
            self.normMLP = None
            self.normFC  = None

        # dropout
        if dropoutGNN:
            self.dropoutGNN = nn.Dropout(dropoutGNN)
        else:
            self.dropoutGNN = nn.Dropout(0)

        if dropoutMLP:
            self.dropoutMLP = nn.Dropout(dropoutMLP)
        else:
            self.dropoutMLP = nn.Dropout(0)

        if dropoutFC:
            self.dropoutFC = nn.Dropout(dropoutFC)
        else:
            self.dropoutFC = nn.Dropout(0)

        self._gnn1 = gnn_layer(self.inputDimGNN,   self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        self._gnn2 = gnn_layer(self.hiddenDimGNN,  self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        self._gnn3 = gnn_layer(self.hiddenDimGNN,  self.hiddenDimFC,     self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)

        self._gat1 = gat_layer(self.inputDimGNN,   self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        self._gat2 = gat_layer(self.hiddenDimGNN,  self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        self._gat3 = gat_layer(self.hiddenDimGNN,  self.hiddenDimFC,     self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)

        self.gcn1 = dglnn.GraphConv(self.inputDimGNN,   self.hiddenDimGNN,  norm='both',    weight=True,   bias=True)
        self.gcn2 = dglnn.GraphConv(self.hiddenDimGNN,  self.hiddenDimGNN,  norm='both',    weight=True,   bias=True)
        self.gcn3 = dglnn.GraphConv(self.hiddenDimGNN,  self.hiddenDimFC,   norm='both',    weight=True,   bias=True)

        self.gat1 = dglnn.GATConv(self.inputDimGNN,     self.hiddenDimGNN, num_heads=3,   bias=True)
        self.gat2 = dglnn.GATConv(self.hiddenDimGNN,    self.hiddenDimGNN, num_heads=3,   bias=True)
        self.gat3 = dglnn.GATConv(self.hiddenDimGNN*3,  self.hiddenDimFC,  num_heads=1,   bias=True)
        
        numHeads = 3
        self.egat1 = dglnn.EGATConv(self.inputDimGNN,           1,        self.hiddenDimGNN,  1,  num_heads=numHeads, bias=True)
        self.egat2 = dglnn.EGATConv(self.hiddenDimGNN*numHeads, numHeads, self.hiddenDimGNN,  1,  num_heads=numHeads, bias=True)
        self.egat3 = dglnn.EGATConv(self.hiddenDimGNN*numHeads, numHeads, self.hiddenDimFC,   1,  num_heads=1,        bias=True)

        self.mlpSensor1 = nn.Linear(self.inputDimSensor, self.hiddenDimMLP)
        self.mlpSensor2 = nn.Linear(self.hiddenDimMLP,   self.hiddenDimMLP)
        self.mlpSensor3 = nn.Linear(self.hiddenDimMLP,   self.hiddenDimFC)

        self.mlpTarget1 = nn.Linear(self.inputDimTarget,    self.hiddenDimMLP)
        self.mlpTarget2 = nn.Linear(self.hiddenDimMLP,      self.hiddenDimFC)

        self.linearFC1 = nn.Linear(3*self.hiddenDimFC,  self.hiddenDimFC)
        self.linearFC2 = nn.Linear(self.hiddenDimFC,    self.hiddenDimFC)

        self.linearStimuli  = nn.Linear(self.hiddenDimFC,   self.outputDimStimuli)
        self.linearCause    = nn.Linear(self.hiddenDimFC,   self.outputDimClause)

        self.reset_parameters()


    ## Reinitialize learnable parameters
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlpSensor1.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpSensor2.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpSensor3.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpTarget1.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpTarget2.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearFC1.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearFC2.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearCause.weight,    gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearStimuli.weight,  gain=nn.init.calculate_gain('relu'))


    def forward(self, g, sensor, target, area):
        if self.useCuda:
            device = torch.device('cuda:3')
            g = g.to(device)
            sensor = sensor.to(device)
            target = target.to(device)
            area   = area.to(device)

        # GNN
        feat = g.ndata['feat']
        edgeNorm = g.edata['norm']

        #g2, hG = self._gnn1(g, feat)
        #g2, hG = self._gnn2(g, hG)
        #g2, hG = self._gnn3(g, hG)
        
        #hG = self._gat1(g, feat)
        #hG = self._gat2(g, hG)
        #hG = self._gat3(g, hG)

        #norm = EdgeWeightNorm(norm='both')
        #edgeWeight = norm(g, edgeNorm)
        #hG = self.gcn1(g, feat, edge_weight=edgeNorm)
        #hG = self.dropoutGNN(hG)
        #hG = self.activation(hG)
        #hG = self.gcn2(g,  hG)
        #hG = self.gcn3(g, hG, edge_weight=edgeNorm)
        #hG = self.dropoutGNN(hG)
        #hG = self.activation(hG)

        #hG = self.gat1(g,  feat)
        #hG = hG.view(-1, hG.size(1) * hG.size(2))
        #hG = self.gat3(g,  hG)
        #hG = hG.view(-1, hG.size(1) * hG.size(2))

        hG, hEdges = self.egat1(g, feat, edgeNorm)
        hG = hG.view(-1, hG.size(1) * hG.size(2))
        hEdges = hEdges.view(-1, hEdges.size(1) * hEdges.size(2))
        hG = self.dropoutGNN(hG)
        hG = self.activation(hG)
        #hNodes, hEdges = self.egat2(g, hNodes, hEdges)
        #hNodes = hNodes.view(-1, hNodes.size(1) * hNodes.size(2))
        #hEdges = hEdges.view(-1, hEdges.size(1) * hEdges.size(2))
        hG, hEdges = self.egat3(g, hG, hEdges)
        hG = hG.view(-1, hG.size(1) * hG.size(2))
        hEdges = hEdges.view(-1, hEdges.size(1) * hEdges.size(2))
        hG = self.dropoutGNN(hG)
        hG = self.activation(hG)

        egoIndexes = (feat[:,5]==1).nonzero(as_tuple=False).flatten()
        hGraph = hG[egoIndexes]

        # MLP Sensor
        hSensor = self.mlpSensor1(sensor.float())
        if self.normMLP:
            hSensor = self.normMLP(hSensor)
        hSensor = self.dropoutMLP(hSensor)
        hSensor = self.activation(hSensor)

        #hSensor = self.mlpSensor2(hSensor)
        #if self.normMLP:
        #    hSensor = self.normMLP(hSensor)
        #hSensor = self.dropoutMLP(hSensor)
        #hSensor = self.activation(hSensor)

        hSensor = self.mlpSensor3(hSensor)
        if self.normMLP:
            hSensor = self.normMLP(hSensor)
        hSensor = self.dropoutMLP(hSensor)
        hSensor = self.activation(hSensor)

        # MLP Target
        hTA = torch.cat((target.squeeze(1).float(),area.squeeze(1).float()),1)
        hTarget = self.mlpTarget1(hTA)
        if self.normMLP:
            hTarget = self.normMLP(hTarget)
        hTarget = self.dropoutMLP(hTarget)
        hTarget = self.activation(hTarget)
        
        hTarget = self.mlpTarget2(hTarget)
        if self.normMLP:
            hTarget = self.normMLP(hTarget)
        hTarget = self.dropoutMLP(hTarget)
        hTarget = self.activation(hTarget)

        # FC concat representations
        h = torch.cat((hGraph, hSensor, hTarget),1)
        h = self.linearFC1(h)
        if self.normFC:
            h = self.normFC(h)
        h = self.dropoutFC(h)
        h = self.activation(h)

        #h = self.linearFC2(h)
        #if self.normFC:
        #    h = self.normFC(h)
        #h = self.dropoutFC(h)
        #h = self.activation(h)

        # two heads
        hStimuli = self.linearStimuli(h)
        hCause = self.linearCause(h)

        return hStimuli,hCause




####################################################################################




class MLP_MLP_model(nn.Module):
    def __init__(self, inputDimGNN, inputDimSensor, inputDimTarget, hiddenDimGNN, hiddenDimMLP, hiddenDimFC,
                    outputDimStimuli, outputDimClause, dropoutGNN, dropoutMLP, dropoutFC, norm, useCuda=False):
        super(MLP_MLP_model, self).__init__()

        self.inputDimGNN        = inputDimGNN
        self.inputDimSensor     = inputDimSensor
        self.inputDimTarget     = inputDimTarget
        self.hiddenDimGNN       = hiddenDimGNN
        self.hiddenDimMLP       = hiddenDimMLP
        self.hiddenDimFC        = hiddenDimFC
        self.outputDimStimuli   = outputDimStimuli
        self.outputDimClause    = outputDimClause
        self.activation         = nn.ReLU()
        self.useCuda            = useCuda


        # normalization
        if norm == 'batch':
            self.normGNN = nn.BatchNorm1d(self.hiddenDimGNN, affine=True)
            self.normMLP = nn.BatchNorm1d(self.hiddenDimMLP, affine=True)
            self.normFC  = nn.BatchNorm1d(self.hiddenDimFC,  affine=True)
        elif norm == 'layer':
            self.normGNN = nn.LayerNorm(self.hiddenDimGNN, elementwise_affine=True)
            self.normMLP = nn.LayerNorm(self.hiddenDimMLP, elementwise_affine=True)
            self.normFC  = nn.LayerNorm(self.hiddenDimFC, elementwise_affine=True)
        elif norm == 'instance':
            self.normGNN = nn.InstanceNorm1d(self.hiddenDimGNN, affine=False)
            self.normMLP = nn.InstanceNorm1d(self.hiddenDimMLP, affine=False)
            self.normFC  = nn.InstanceNorm1d(self.hiddenDimFC, affine=False)
        else:
            self.normGNN = None
            self.normMLP = None
            self.normFC  = None

        # dropout
        if dropoutGNN:
            self.dropoutGNN = nn.Dropout(dropoutGNN)
        else:
            self.dropoutGNN = nn.Dropout(0)

        if dropoutMLP:
            self.dropoutMLP = nn.Dropout(dropoutMLP)
        else:
            self.dropoutMLP = nn.Dropout(0)

        if dropoutFC:
            self.dropoutFC = nn.Dropout(dropoutFC)
        else:
            self.dropoutFC = nn.Dropout(0)

        self.gnn1 = nn.Linear(self.inputDimGNN+1,   self.hiddenDimGNN)
        self.gnn2 = nn.Linear(self.hiddenDimGNN,    self.hiddenDimGNN)
        self.gnn3 = nn.Linear(self.hiddenDimGNN,    self.hiddenDimFC)

        self.mlpSensor1 = nn.Linear(self.inputDimSensor, self.hiddenDimMLP)
        self.mlpSensor2 = nn.Linear(self.hiddenDimMLP,   self.hiddenDimMLP)
        self.mlpSensor3 = nn.Linear(self.hiddenDimMLP,   self.hiddenDimFC)

        self.mlpTarget1 = nn.Linear(self.inputDimTarget, self.hiddenDimMLP)
        self.mlpTarget2 = nn.Linear(self.hiddenDimMLP,   self.hiddenDimFC)

        self.linearFC1 = nn.Linear(3*self.hiddenDimFC,  self.hiddenDimFC)
        self.linearFC2 = nn.Linear(self.hiddenDimFC,    self.hiddenDimFC)

        self.linearStimuli = nn.Linear(self.hiddenDimFC,  self.outputDimStimuli)
        self.linearCause = nn.Linear(self.hiddenDimFC,    self.outputDimClause)

        self.reset_parameters()


    ## Reinitialize learnable parameters
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gnn1.weight,           gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.gnn2.weight,           gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.gnn3.weight,           gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpSensor1.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpSensor2.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpSensor3.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpTarget1.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.mlpTarget2.weight,     gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearFC1.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearFC2.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearCause.weight,    gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearStimuli.weight,  gain=nn.init.calculate_gain('relu'))


    def forward(self, g, sensor, target, area):
        if self.useCuda:
            device = torch.device('cuda:3')
            g = g.to(device)
            sensor = sensor.to(device)
            target = target.to(device)
            area   = area.to(device)

        # GNN
        g1 = dgl.unbatch(g)
        nodeFeats = []
        edgeNorms = []
        for graph in g1:
            nodeFeats.append(torch.sum(graph.ndata['feat'],0).tolist())
            edgeNorms.append(torch.sum(graph.edata['norm'][0:graph.number_of_nodes()],0).tolist())

        nodeFeats = torch.FloatTensor(nodeFeats)

        edgeNorms1 = [item for sublist in edgeNorms for item in sublist]
        edgeTensor = torch.FloatTensor(edgeNorms1)

        if self.useCuda:
            nodeFeats = nodeFeats.to(device)
            edgeTensor = edgeTensor.to(device)

        feat = torch.cat((nodeFeats,edgeTensor.unsqueeze(1)),-1)

        # MLP Graph
        hG = self.gnn1(feat)
        if self.normGNN:
            hSensor = self.normGNN(hG)
        hG = self.dropoutGNN(hG)
        hG = self.activation(hG)
        
        #hG = self.gnn2(hG)
        #if self.normGNN:
        #    hSensor = self.normGNN(hG)
        #hG = self.dropoutGNN(hG)
        #hG = self.activation(hG)

        hG = self.gnn3(hG)
        if self.normGNN:
            hSensor = self.normGNN(hG)
        hG = self.dropoutGNN(hG)
        hG = self.activation(hG)

        # MLP Sensor
        hSensor = self.mlpSensor1(sensor.float())
        if self.normMLP:
            hSensor = self.normMLP(hSensor)
        hSensor = self.dropoutMLP(hSensor)
        hSensor = self.activation(hSensor)

        #hSensor = self.mlpSensor2(hSensor)
        #if self.normMLP:
        #    hSensor = self.normMLP(hSensor)
        #hSensor = self.dropoutMLP(hSensor)
        #hSensor = self.activation(hSensor)

        hSensor = self.mlpSensor3(hSensor)
        if self.normMLP:
            hSensor = self.normMLP(hSensor)
        hSensor = self.dropoutMLP(hSensor)
        hSensor = self.activation(hSensor)

        # MLP Target
        hTA = torch.cat((target.squeeze(1).float(),area.squeeze(1).float()),1)
        hTarget = self.mlpTarget1(hTA)
        if self.normMLP:
            hTarget = self.normMLP(hTarget)
        hTarget = self.dropoutMLP(hTarget)
        hTarget = self.activation(hTarget)
        
        hTarget = self.mlpTarget2(hTarget)
        if self.normMLP:
            hTarget = self.normMLP(hTarget)
        hTarget = self.dropoutMLP(hTarget)
        hTarget = self.activation(hTarget)

        # FC concat representations
        h = torch.cat((hG, hSensor, hTarget),1)
        h = self.linearFC1(h)
        if self.normFC:
            h = self.normFC(h)
        h = self.dropoutFC(h)
        h = self.activation(h)

        #h = self.linearFC2(h)
        #if self.normFC:
        #    h = self.normFC(h)
        #h = self.dropoutFC(h)
        #h = self.activation(h)

        # two heads
        hStimuli = self.linearStimuli(h)
        hCause = self.linearCause(h)

        return hStimuli,hCause




####################################################################################




class GNN_MLP_RNN_model(nn.Module):
    def __init__(self, inputDimGNN, inputDimSensor, inputDimTarget, hiddenDimGNN, hiddenDimMLP, hiddenDimFC, hiddenDimRNN, nrRNNlayers,
                    outputDimStimuli, outputDimClause, dropoutGNN, dropoutMLP, dropoutFC, dropoutRNN, norm, timesteps, useCuda=False):
        super(GNN_MLP_RNN_model, self).__init__()

        self.inputDimGNN        = inputDimGNN
        self.inputDimSensor     = inputDimSensor
        self.inputDimTarget     = inputDimTarget
        self.hiddenDimGNN       = hiddenDimGNN
        self.hiddenDimMLP       = hiddenDimMLP
        self.hiddenDimFC        = hiddenDimFC
        self.hiddenDimRNN       = hiddenDimRNN
        self.outputDimStimuli   = outputDimStimuli
        self.outputDimClause    = outputDimClause
        self.nrRNNlayers        = nrRNNlayers
        self.activation         = nn.ReLU()
        self.timesteps          = timesteps
        self.useCuda            = useCuda
        self.useMultiCuda       = False
        self.device             = torch.device('cuda:3')

        # normalization
        if norm == 'batch':
            self.normGNN = nn.BatchNorm1d(self.hiddenDimGNN, affine=True)
            self.normMLP = nn.BatchNorm1d(self.hiddenDimMLP, affine=True)
            self.normFC  = nn.BatchNorm1d(self.hiddenDimFC,  affine=True)
        elif norm == 'layer':
            self.normGNN = nn.LayerNorm(self.hiddenDimGNN, elementwise_affine=True)
            self.normMLP = nn.LayerNorm(self.hiddenDimMLP, elementwise_affine=True)
            self.normFC  = nn.LayerNorm(self.hiddenDimFC, elementwise_affine=True)
        elif norm == 'instance':
            self.normGNN = nn.InstanceNorm1d(self.hiddenDimGNN, affine=False)
            self.normMLP = nn.InstanceNorm1d(self.hiddenDimMLP, affine=False)
            self.normFC  = nn.InstanceNorm1d(self.hiddenDimFC, affine=False)
        else:
            self.normGNN = None
            self.normMLP = None
            self.normFC  = None

        # dropout
        if dropoutGNN:
            self.dropoutGNN = nn.Dropout(dropoutGNN)
        else:
            self.dropoutGNN = nn.Dropout(0)

        if dropoutMLP:
            self.dropoutMLP = nn.Dropout(dropoutMLP)
        else:
            self.dropoutMLP = nn.Dropout(0)

        if dropoutFC:
            self.dropoutFC = nn.Dropout(dropoutFC)
        else:
            self.dropoutFC = nn.Dropout(0)

        if dropoutRNN:
            self.dropoutRNN = nn.Dropout(dropoutRNN)
        else:
            self.dropoutRNN = nn.Dropout(0)

        # GNN module
        gnn1 = gnn_layer(self.inputDimGNN,   self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        gnn2 = gnn_layer(self.hiddenDimGNN,  self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        gnn3 = gnn_layer(self.hiddenDimGNN,  self.hiddenDimRNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        
        # GAT module
        gat1 = gat_layer(self.inputDimGNN,   self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        gat2 = gat_layer(self.hiddenDimGNN,  self.hiddenDimGNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        gat3 = gat_layer(self.hiddenDimGNN,  self.hiddenDimRNN,    self.dropoutGNN,    self.normGNN,  self.activation,  self.useCuda)
        
        if self.useMultiCuda:
            self.gnn1c = gnn1.cuda()
            self.gnn2c = gnn2.cuda()
            self.gnn3c = gnn3.cuda()
            self.gat1c = gat1.cuda()
            self.gat2c = gat2.cuda()
            self.gat3c = gat3.cuda()
        elif self.useCuda:
            self.gnn1c = gnn1.to(self.device)
            self.gnn2c = gnn2.to(self.device)
            self.gnn3c = gnn3.to(self.device)
            self.gat1c = gat1.to(self.device)
            self.gat2c = gat2.to(self.device)
            self.gat3c = gat3.to(self.device)
        else:
            self.gnn1c = gnn1
            self.gnn2c = gnn2
            self.gnn3c = gnn3
            self.gat1c = gat1
            self.gat2c = gat2
            self.gat3c = gat3
        

        #self.rnnG   = nn.RNN(self.hiddenDimRNN, self.hiddenDimFC, self.nrRNNlayers, nonlinearity='tanh', batch_first=True, dropout=dropoutRNN)
        #self.lstmG  = nn.LSTM(self.hiddenDimRNN, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)
        self.gruG   = nn.GRU(self.hiddenDimRNN, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)

        #self.rnnS   = nn.RNN(self.inputDimSensor, self.hiddenDimFC, self.nrRNNlayers, nonlinearity='tanh', batch_first=True, dropout=dropoutRNN)
        #self.lstmS  = nn.LSTM(self.inputDimSensor, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)
        self.gruS   = nn.GRU(self.inputDimSensor, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)

        #self.rnnT   = nn.RNN(self.inputDimTarget, self.hiddenDimFC, self.nrRNNlayers, nonlinearity='tanh', batch_first=True, dropout=dropoutRNN)
        #self.lstmT  = nn.LSTM(self.inputDimTarget, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)
        self.gruT   = nn.GRU(self.inputDimTarget, self.hiddenDimFC, self.nrRNNlayers, batch_first=True, dropout=dropoutRNN)

        self.linearFC1 = nn.Linear(3*self.hiddenDimFC,  self.hiddenDimFC)
        self.linearFC2 = nn.Linear(self.hiddenDimFC,    self.hiddenDimFC)

        self.linearStimuli = nn.Linear(self.hiddenDimFC,  self.outputDimStimuli)
        self.linearCause = nn.Linear(self.hiddenDimFC,  self.outputDimClause)

        self.reset_parameters()


    ## Reinitialize learnable parameters
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linearFC1.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearFC2.weight,      gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearStimuli.weight,  gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linearCause.weight,    gain=nn.init.calculate_gain('relu'))


    def forward(self, bG, bSensor, bTarget, bArea):

        #bG = bG[int(index[0]) : int(index[-1]) + 1]

        bgList = []
        for i in range(0, len(bG)):
            hgList = []
            for t in range(0,self.timesteps):
                if self.useMultiCuda:
                    bG[i][t] = bG[i][t].to(bSensor.device)
                elif self.useCuda:
                    bG[i][t] = bG[i][t].to(self.device)
                feat = bG[i][t].ndata['feat']

                g, hG = self.gnn1c(bG[i][t], feat)
                #g2, hG = self.gnn2(g, hG)
                g2, hG = self.gnn3c(bG[i][t], hG)

                #g, hG = self.gat1c(bG[i][t], feat)
                #g2, hG = self.gnn2(g, hG)
                #g2, hG = self.gat3c(bG[i][t], hG)

                hgList.append(hG[0].unsqueeze(0))
            hgTensor = torch.cat(hgList)
            bgList.append(hgTensor.unsqueeze(0))
        graphSeq = torch.cat(bgList)
        
        
        if self.useMultiCuda:
            h0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC).cuda()
            #c0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC).cuda()
        elif self.useCuda:
            h0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC).to(self.device)
            #c0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC).to(self.device)
        else:
            h0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC)
            #c0 = torch.zeros(self.nrRNNlayers, len(bG), self.hiddenDimFC)
        

        #self.lstmG.flatten_parameters()
        #graphOut, _ = self.lstmG(graphSeq, (h0, c0))
        self.gruG.flatten_parameters()
        graphOut, _ = self.gruG(graphSeq, h0)

        #self.lstmS.flatten_parameters()
        #sensorOut, _ = self.lstmS(bSensor, (h0, c0))
        self.gruS.flatten_parameters()
        sensorOut, _ = self.gruS(bSensor, h0)

        #self.lstmT.flatten_parameters()
        #targetOut, _ = self.lstmT(bTarget, (h0, c0))
        self.gruT.flatten_parameters()
        targetOut, _ = self.gruT(bTarget, h0)


        output = torch.cat((graphOut,sensorOut,targetOut),2)
        output = output.contiguous().view(-1, 3*self.hiddenDimFC)

        hOut = self.linearFC1(output)
        if self.normFC:
            hOut = self.normFC(hOut)
        hOut = self.dropoutFC(hOut)
        hOut = self.activation(hOut)

        #h = self.linearFC2(h)
        #if self.normFC:
        #    h = self.normFC(h)
        #h = self.dropoutFC(h)
        #h = self.activation(h)
        
        # two heads
        hStimuli = self.linearStimuli(hOut)
        hCause = self.linearCause(hOut)

        return hStimuli, hCause