import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
import dgl.function as fn

import numpy as np



class gnn_layer(nn.Module):
    def __init__(self, inputDim, outputDim, dropout=0.0, norm=None, activation=None, useCuda=False, bias=True):
        super(gnn_layer, self).__init__()
        self.inputDim   = inputDim
        self.outputDim  = outputDim
        self.dropout    = dropout
        self.activation = activation
        self.norm       = norm
        self.useCuda    = useCuda

        self.linear = nn.Linear(self.inputDim, self.outputDim, bias=True)
        self.reset_parameters()

        # weight
        if self.useCuda:
            device = torch.device('cuda:3')

        # bias
        if bias:
            bias = nn.Parameter(torch.Tensor(self.outputDim))
            if self.useCuda:
                self.bias = bias.to(device)
        else:
            self.bias = None

    
    ## Reinitialize learnable parameters.
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))


    ## forward method
    def forward(self, g, h):

        def message_func(edges):
            msg = edges.src['h']
            msg = msg * edges.data['norm']
            return {'msg': msg}
        
        
        with g.local_scope():
            g.ndata['h'] = h
            #g.update_all(fn.copy_u('h', 'msg'), fn.sum('msg', 'h'))
            g.update_all(message_func, fn.sum('msg', 'h'))
            h = g.ndata['h']

            h = self.linear(h)

            if self.bias is not None:
                h = h + self.bias
            #if self.norm:
            #    h = self.norm(h)
            if self.dropout:
                h = self.dropout(h)
            if self.activation:
                h = self.activation(h)

            return g, h




class gat_layer(nn.Module):
    def __init__(self, inputDim, outputDim, dropout=0.0, norm=None, useCuda=False, bias=False):
        super(gat_layer, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.fcAttn = nn.Linear(2*outputDim, 1)
        self.reset_parameters()
        

    ## Reinitialize learnable parameters.
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fcAttn.weight, gain=nn.init.calculate_gain('relu'))


    def edge_attention(self, edges):
        z2 = torch.cat( [edges.src['z'],edges.dst['z']], dim=1 )
        a = self.fcAttn(z2)
        return {'e': F.leaky_relu(a)}

    
    def message_func(self, edges):
        return { 'z':edges.src['z'], 'e':edges.data['e'] }

    
    def reduce_func(self, nodes):
        aa = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(aa*nodes.mailbox['z'], dim=1)
        return {'h': h}

    
    def forward(self, g, h):
        z = self.fc(h)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')
