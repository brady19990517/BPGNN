# @title [RUN] Import python modules

from distutils.log import error
from multiprocessing.spawn import prepare
import os
import random
import time
from pprint import pprint
import seaborn as sns

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
from scipy.stats import ortho_group
from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import (dense_to_sparse, remove_self_loops,
                                   to_dense_adj)
from torch_scatter import scatter
from tqdm import tqdm
import json

from BP import *
from factor import *
from factor_graph import *
from dataset import *



class EncoderLayer(MessagePassing):
    def __init__(self, h_dim=32, msg_dim=2, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(msg_dim+h_dim, h_dim, bias)
    
    def forward(self, msg, h_msg):
        input = torch.cat((msg, h_msg), dim=1).float()
        return self.lin(input)

class LoopEncoderLayer(MessagePassing):
    def __init__(self, h_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(h_dim, h_dim, bias)
    
    def forward(self, h_msg):
        return self.lin(h_msg)

class DummyEncoderLayer(MessagePassing):
    def __init__(self, h_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        # self.lin = nn.Linear(h_dim, h_dim, bias)
    
    def forward(self, h_msg):
        return h_msg

class MPNN_Layer(MessagePassing):
    def __init__(self, h_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.M = nn.Sequential(nn.Linear(2*h_dim+h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.N_node = nn.Sequential(nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU())
        self.h_msg_temp = None

    def forward(self, h_node, edge_index, h_msg, encoded_msg):
        h_node = self.propagate(edge_index, h_node=h_node, h_msg=h_msg, encoded_msg=encoded_msg)
        return h_node, self.h_msg_temp

    def message(self, h_node_i, h_node_j, encoded_msg, edge_index):
        aggr_msg = self.N_node(h_node_j) + self.M(torch.cat([h_node_i, h_node_j, encoded_msg], dim=-1))
        # aggr_msg = self.M(torch.cat([h_node_i, h_node_j, encoded_msg], dim=-1))
        self.h_msg_temp = aggr_msg
        return aggr_msg

    def update(self, aggr_out, h_node):
        h_node = self.U(torch.cat((h_node, aggr_out), dim=1))
        return h_node

class MPNN_SenderAggr_Layer(MessagePassing):
    def __init__(self, h_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.M = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.N = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.N_node = nn.Sequential(nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU())
        self.h_msg_temp = None

    def forward(self, h_node, edge_index, h_msg, encoded_msg):
        # Message aggregation at all destination node
        # aggr_msgs = scatter(self.M(torch.cat([encoded_msg, h_msg], dim=-1)), edge_index[1].unsqueeze(-1), dim=self.node_dim, reduce='sum') # NxHIDDED
        aggr_msgs = scatter(encoded_msg, edge_index[1].unsqueeze(-1), dim=self.node_dim, reduce='sum') # NxHIDDED
        
        # aggr_msgs_i = aggr_msgs[edge_index[0]]
        # aggr_msg = self.N(torch.cat([aggr_msgs_i, encoded_msg], dim=-1))
        # self.h_msg_temp = aggr_msg

        h_node = self.propagate(edge_index, h_node=h_node, h_msg=h_msg, encoded_msg=encoded_msg, aggr_msgs=aggr_msgs)

        return h_node, self.h_msg_temp

    def message(self, h_node_i, h_node_j, encoded_msg, edge_index, aggr_msgs_j):
        # print("First edge source node:", edge_index[0][0])
        # print("aggr_msgs_j", aggr_msgs_j[0])
        # print("aggr_msgs[edge_index[0]]", aggr_msgs[edge_index[0]][0])

        aggr_msg = self.N_node(h_node_j) + self.N(torch.cat([aggr_msgs_j, encoded_msg], dim=-1))
        # aggr_msg = self.N(torch.cat([aggr_msgs_j, encoded_msg], dim=-1))
        self.h_msg_temp = aggr_msg
        return aggr_msg

    def update(self, aggr_out, h_node):
        h_node = self.U(torch.cat((h_node, aggr_out), dim=1))
        return h_node

class MPNN_Structure2Vec_Layer(MessagePassing):
    def __init__(self, h_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.M = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.N = nn.Sequential(nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.N_node = nn.Sequential(nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(h_dim, h_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU())
        self.h_msg_temp = None

    def forward(self, h_node, edge_index, h_msg, encoded_msg):
        # Message aggregation at all destination node
        # aggr_msgs = scatter(self.M(torch.cat([encoded_msg, h_msg], dim=-1)), edge_index[1].unsqueeze(-1), dim=self.node_dim, reduce='sum') # NxHIDDED
        # h_encoded = self.M(torch.cat([encoded_msg, h_msg], dim=-1))
        aggr_msgs = scatter(encoded_msg, edge_index[1].unsqueeze(-1), dim=self.node_dim, reduce='sum') # NxHIDDED
        
        # aggr_msgs_i = aggr_msgs[edge_index[0]]
        # aggr_msg = self.N(torch.cat([aggr_msgs_i, encoded_msg], dim=-1))
        # self.h_msg_temp = aggr_msg

        h_node = self.propagate(edge_index, h_node=h_node, h_msg=h_msg, encoded_msg=encoded_msg, aggr_msgs=aggr_msgs)

        return h_node, self.h_msg_temp

    def message(self, h_node_i, h_node_j, encoded_msg, edge_index, aggr_msgs_j,aggr_msgs):

        aggr_msg = self.N_node(h_node_j) + self.N(aggr_msgs_j - encoded_msg)
        self.h_msg_temp = aggr_msg
        return aggr_msg

    def update(self, aggr_out, h_node):
        h_node = self.U(torch.cat((h_node, aggr_out), dim=1))
        return h_node
    

class MessageDecoderLayer(MessagePassing):
    # Implement decoder node?
    def __init__(self, hidden_dim=32,msg_dim=2, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(hidden_dim, msg_dim, bias)
    
    def forward(self, h_msg):
        # Normalise message for prediction
        return torch.softmax(self.lin(h_msg), dim=-1)
        
class BeliefsDecoderLayer(MessagePassing):
    # Implement decoder node?
    def __init__(self, hidden_dim=32,beliefs_dim=2, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(hidden_dim, beliefs_dim, bias)
    
    def forward(self, h_node, x_feat):
        # Normalise message for prediction
        mask = x_feat[:, 0] == 1
        variable_h_node = h_node[mask]
        # for i in range(len(x_feat)):
        #     if x_feat[i][0] == 1:
        #         variable_h_node.append(h_node[i])
        # variable_h_node = torch.stack(variable_h_node)
        return torch.softmax(self.lin(variable_h_node), dim=-1)
    
class MPNN(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor = MPNN_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        #decoded beliefs
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs
        # return h_msg, y_msg

class MPNN_SenderAggr(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor_2 = MPNN_SenderAggr_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_Structure2Vec(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor_3 = MPNN_Structure2Vec_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_3(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_2Layer(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor = MPNN_Layer()
        self.processor_2 = MPNN_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs


class MPNN_Loop(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = LoopEncoderLayer()
        self.processor = MPNN_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg):
        # print(data.x.shape)
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        #decoded beliefs
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs
        # return h_msg, y_msg

class MPNN_SenderAggr_Loop(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = LoopEncoderLayer()
        self.processor_2 = MPNN_SenderAggr_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_Structure2Vec_Loop(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = LoopEncoderLayer()
        self.processor_3 = MPNN_Structure2Vec_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_3(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_2Layer_Loop(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = LoopEncoderLayer()
        self.processor = MPNN_Layer()
        self.processor_2 = MPNN_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_Loop_Transfer(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2, model=None):
        super().__init__()
        self.model = model
        self.encoder = LoopEncoderLayer()

    def forward(self, data, h_msg):
        h_node = self.model.lin_in(data.x)
        encoded_msg = self.encoder(h_msg)
        h_node, h_msg = self.model.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.model.decoder(h_msg)
        #decoded beliefs
        y_beliefs = self.model.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs


class MPNN_SenderAggr_Loop_Transfer(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2, model=None):
        super().__init__()
        self.model = model
        self.encoder = LoopEncoderLayer()

    def forward(self, data, h_msg):
        h_node = self.model.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.model.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.model.decoder(h_msg)
        y_beliefs = self.model.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_Structure2Vec_Loop_Transfer(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2, model=None):
        super().__init__()
        self.model = model
        self.encoder = LoopEncoderLayer()

    def forward(self, data, h_msg):
        h_node = self.model.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.model.processor_3(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.model.decoder(h_msg)
        y_beliefs = self.model.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_2Layer_Loop_Transfer(Module):
    def __init__(self, x_dim=3, h_dim=32, msg_dim=2, model=None):
        super().__init__()
        self.model = model
        self.encoder = LoopEncoderLayer()

    def forward(self, data, h_msg):
        h_node = self.model.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(h_msg)
        # hidden_node -> h
        h_node, h_msg = self.model.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        h_node, h_msg = self.model.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.model.decoder(h_msg)
        y_beliefs = self.model.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs


def train(model, train_loader, optimizer, device, epoch):
    torch.autograd.set_detect_anomaly(True)
    model.train()

    loss_all = 0
    msg_loss_all = 0
    belief_loss_all = 0

    for data in train_loader:
        # Transpose batch
        data = prepare_batch(data)
        data = data.to(device)
        data.x, data.edge_attr = data.x.float(), data.edge_attr.float()
        optimizer.zero_grad()
        h_dim=32
        # All 0 for first h_msg
        h_msg = torch.zeros(data.edge_attr[0].size(dim=0), h_dim)

        y_msg_pred = []
        y_beliefs_pred = []
        y_msg = data.edge_attr[0]
        #Start from step_index 1, there's no prediction for step_index 0
        for step_idx in range(1,data.edge_attr.size(dim=0)):
            h_msg, y_msg, y_beliefs = model(data, h_msg=h_msg, y_msg=y_msg)
            y_msg_pred.append(y_msg)
            y_beliefs_pred.append(y_beliefs)
            y_msg = data.edge_attr[step_idx]

        y_msg_pred = torch.stack(y_msg_pred)
        y_beliefs_pred = torch.stack(y_beliefs_pred)
    
        loss_message = F.l1_loss(y_msg_pred, data.edge_attr[1:])
        loss_belief = F.l1_loss(y_beliefs_pred, data.y[1:])
        loss = loss_message + loss_belief
        # loss = loss_message
        loss.backward(retain_graph=False)

        loss_all += loss.item() * data.num_graphs # number of graphs per batch
        msg_loss_all += loss_message.item() * data.num_graphs
        belief_loss_all += loss_belief.item() * data.num_graphs

        optimizer.step()
    return loss_all / len(train_loader.dataset), msg_loss_all / len(train_loader.dataset), belief_loss_all / len(train_loader.dataset)
    # number of total graphs

def eval(model, loader, device, epoch):
    model.eval()
    error = 0
    msg_loss_all = 0
    belief_loss_all = 0
    error_iter = None
    error_iter_msg = None
    error_iter_belief = None
    for data in loader:
        data = prepare_batch(data)
        data = data.to(device)
        data.x, data.edge_attr = data.x.float(), data.edge_attr.float()
        h_dim=32
        h_msg = torch.zeros(data.edge_attr[0].size(dim=0), h_dim)

        y_msg = data.edge_attr[0]
        y_msg_pred = []
        y_beliefs_pred = []
        with torch.no_grad():
            #TODO: change the step index to 20
            for step_idx in range(1, 20):
                h_msg, y_msg, y_beliefs = model(data, h_msg=h_msg, y_msg=y_msg)
                # h_msg, y_msg = model(data, h_msg=h_msg, y_msg=y_msg)
                y_msg_pred.append(y_msg)
                y_beliefs_pred.append(y_beliefs)
    
            y_msg_pred = torch.stack(y_msg_pred)
            y_beliefs_pred = torch.stack(y_beliefs_pred)
         
            loss_msg = F.l1_loss(y_msg_pred, data.edge_attr[1:])
            loss_belief = F.l1_loss(y_beliefs_pred, data.y[1:])

            # print("y_msg_pred: ",y_msg_pred.shape)
            # print("data.edge_attr[1:]: ", data.edge_attr[1:].shape)
            # print("f1 shape: ", F.l1_loss(y_msg_pred, data.edge_attr[1:], reduction='none').shape)
            # print("f1 shape: ", torch.mean(F.l1_loss(y_msg_pred, data.edge_attr[1:], reduction='none'),(1,2)).shape)
            loss_msg_iter = torch.mean(F.l1_loss(y_msg_pred, data.edge_attr[1:], reduction='none'),(1,2))
            loss_belief_iter = torch.mean(F.l1_loss(y_beliefs_pred, data.y[1:], reduction='none'),(1,2))
            error_iter = loss_msg_iter + loss_belief_iter
            error_iter_msg = loss_msg_iter
            error_iter_belief = loss_belief_iter

            error += loss_msg.item() * data.num_graphs
            msg_loss_all += loss_msg.item() * data.num_graphs

            error += loss_belief.item() * data.num_graphs
            belief_loss_all += loss_belief.item() * data.num_graphs

    return error / len(loader.dataset), msg_loss_all / len(loader.dataset), belief_loss_all / len(loader.dataset), error_iter, error_iter_belief, error_iter_msg

def train_loop(model, train_loader, optimizer, device, epoch):
    torch.autograd.set_detect_anomaly(True)
    model.train()

    loss_all = 0

    for data in train_loader:
        # Transpose batch
        # data = prepare_batch(data)

        data.x = data.x.float()
        optimizer.zero_grad()
        h_dim=32
        # All 0 for first h_msg
        h_msg = torch.zeros(data.edge_index.size(dim=1), h_dim)

        # y_msg_pred = []
        y_beliefs_pred = []
        # y_msg = data.edge_attr[0]
        #Start from step_index 1, there's no prediction for step_index 0
        #TODO: change the step index to 20
        for step_idx in range(1,20):
            h_msg, y_msg, y_beliefs = model(data, h_msg=h_msg)
            # y_msg_pred.append(y_msg)
            y_beliefs_pred.append(y_beliefs)
            # y_msg = data.edge_attr[step_idx]

        # y_msg_pred = torch.stack(y_msg_pred)
        y_beliefs_pred = torch.stack(y_beliefs_pred)
    
        # loss_message = F.l1_loss(y_msg_pred, data.edge_attr[1:])
        loss = F.l1_loss(y_beliefs_pred[-1], data.y)
        # loss = loss_message + loss_belief
        loss.backward(retain_graph=False)

        loss_all += loss.item() * data.num_graphs # number of graphs per batch
        # msg_loss_all += loss_message.item() * data.num_graphs


        optimizer.step()
    return loss_all / len(train_loader.dataset)

def eval_loop(model, loader, device, epoch):
    model.eval()
    error = 0
    error_iter = None
    for data in loader:
        # data = prepare_batch(data)
        # data = data.to(device)
        data.x = data.x.float()
        h_dim=32
        h_msg = torch.zeros(data.edge_index.size(dim=1), h_dim)

        y_beliefs_pred = []
        with torch.no_grad():
            for step_idx in range(1,20):
                h_msg, y_msg, y_beliefs = model(data, h_msg=h_msg)
                # h_msg, y_msg = model(data, h_msg=h_msg, y_msg=y_msg)
                # y_msg_pred.append(y_msg)
                y_beliefs_pred.append(y_beliefs)
    
            # y_msg_pred = torch.stack(y_msg_pred)
            y_beliefs_pred = torch.stack(y_beliefs_pred)

            # print(y_beliefs_pred.shape)
            # print(data.beliefs[1:].transpose(1, 0).shape)
            # batch.edge_attr = batch.edge_attr.transpose(1, 0)
            # loss_msg = F.l1_loss(y_msg_pred, data.edge_attr[1:])
            loss_belief = F.l1_loss(y_beliefs_pred[-1], data.y)
            
            # print(F.l1_loss(y_beliefs_pred[-1], data.y, reduction='none').shape)
            error_iter = torch.mean(F.l1_loss(y_beliefs_pred, data.beliefs.transpose(1, 0)[1:], reduction='none'),(1,2))
            # print(F.l1_loss(y_beliefs_pred[-1], data.y, reduction='none').shape)

            # error += loss_msg.item() * data.num_graphs
            # msg_loss_all += loss_msg.item() * data.num_graphs

            error += loss_belief.item() * data.num_graphs
            # belief_loss_all += loss_belief.item() * data.num_graphs

    return error / len(loader.dataset), error_iter


def run_experiment(model, model_name, train_loader, val_loader, test_loader, n_epochs=100, patience=50):

    print(
        f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    # Adam optimizer with LR 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # LR scheduler which decays LR when validation metric doesn't improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5, min_lr=0.00001)

    print("\nStart training:")
    best_val_error = None
    best_test_error = None
    best_test_msg_error = None
    best_test_belief_error = None
    perf_per_epoch = []  # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    p_counter = 0
    error_iter = None
    for epoch in range(1, n_epochs+1):
        # print('epoch: ', epoch)
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        train_loss, train_msg_loss, train_belief_loss = train(model, train_loader, optimizer, device, epoch)

        # Evaluate model on validation set
        val_error, val_msg_error, val_belief_error, error_iter, error_iter_belief, error_iter_msg = eval(model, val_loader, device, epoch)

        
        # Evaluate model on test set if validation metric improves
        test_error, test_msg_error, test_belief_error, error_iter, error_iter_belief, error_iter_msg = eval(model, test_loader, device, epoch)

        if best_val_error is None or val_error <= best_val_error:
            best_val_error = val_error
            p_counter = 0
            torch.save(model, 'model/'+model_name+'_15nodes.pth')

        if best_test_error is None or test_error <= best_test_error:
            best_test_error = test_error
        
        if best_test_msg_error is None or test_msg_error <= best_test_msg_error:
            best_test_msg_error = test_msg_error

        if best_test_belief_error is None or test_belief_error <= best_test_belief_error:
            best_test_belief_error = test_belief_error

        if val_error > best_val_error:
            p_counter += 1

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {train_loss:.7f}, '
                  f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}, Test Msg MAE: {test_msg_error:.7f}, Test Belief MAE: {test_belief_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((train_loss, train_msg_loss, train_belief_loss, test_error, test_msg_error, test_belief_error, 
                                    val_error, val_msg_error, val_belief_error, epoch, model_name))

        if p_counter >= patience:
            print("Validation error not improving...Terminate training")
            break

    t = time.time() - t
    train_time = t/60
    print(
        f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_error:.7f}, Best test MAE: {best_test_error:.7f}, Best test msg MAE {best_test_msg_error:.7f}, Best test belief MAE {best_test_belief_error:.7f}")

    return best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,error_iter

def run_experiment_loop(model, model_name, train_loader, val_loader, test_loader, n_epochs=100, patience=50):

    print(
        f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\nModel architecture:")
    print(model)
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    model = model.to(device)

    # Adam optimizer with LR 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # LR scheduler which decays LR when validation metric doesn't improve
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5, min_lr=0.00001)

    print("\nStart training:")
    best_val_error = None
    best_test_error = None

    perf_per_epoch = []  # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    p_counter = 0
    for epoch in range(1, n_epochs+1):
        # print('epoch: ', epoch)
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        train_loss = train_loop(model, train_loader, optimizer, device, epoch)

        # Evaluate model on validation set
        val_error, error_iter = eval_loop(model, val_loader, device, epoch)

        
        # Evaluate model on test set if validation metric improves
        test_error, error_iter = eval_loop(model, test_loader, device, epoch)

        if best_val_error is None or val_error <= best_val_error:
            best_val_error = val_error
            p_counter = 0
            torch.save(model, 'model/'+model_name+'_15nodes.pth')

        if best_test_error is None or test_error <= best_test_error:
            best_test_error = test_error

        if val_error > best_val_error:
            p_counter += 1

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {train_loss:.7f}, '
                  f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}, Test Msg MAE: {test_msg_error:.7f}, Test Belief MAE: {test_belief_error:.7f}')

        scheduler.step(val_error)

        train_msg_loss = train_belief_loss = val_msg_error = val_belief_error = test_msg_error = test_belief_error = 0

        perf_per_epoch.append((train_loss, train_msg_loss, train_belief_loss, test_error, test_msg_error, test_belief_error, 
                                    val_error, val_msg_error, val_belief_error, epoch, model_name))

        if p_counter >= patience:
            print("Validation error not improving...Terminate training")
            break

    t = time.time() - t
    train_time = t/60
    best_test_msg_error = best_test_belief_error = 0
    print(
        f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_error:.7f}, Best test MAE: {best_test_error:.7f}, Best test msg MAE {best_test_msg_error:.7f}, Best test belief MAE {best_test_belief_error:.7f}")

    return best_val_error, best_test_error, train_time, perf_per_epoch

def prepare_batch(batch):
    batch.edge_attr = batch.edge_attr.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    return batch

def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def plot(DF_RESULTS):
    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Train MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    p.get_figure().savefig('trainMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Train Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Message Error')
    plt.show()
    p.get_figure().savefig('trainMsgMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Train Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Belief Error')
    plt.show()
    p.get_figure().savefig('trainBeliefMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    p.get_figure().savefig('testMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Test Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Message Error')
    plt.show()
    p.get_figure().savefig('testMsgMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Test Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Belief Error')
    plt.show()
    p.get_figure().savefig('testBeliefMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    p.get_figure().savefig('valMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Val Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Message Error')
    plt.show()
    p.get_figure().savefig('valMsgMAE_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Val Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    p.set(xlabel='Epoch', ylabel='Belief Error')
    plt.show()
    p.get_figure().savefig('valBeliefMAE_15nodes.png')

def plot_loop(DF_RESULTS):
    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Train MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    # plt.show()
    p.get_figure().savefig('trainMAE_loop_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    # plt.show()
    p.get_figure().savefig('testMAE_loop_15nodes.png')

    plt.figure()
    sns.set()
    p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    # plt.show()
    p.get_figure().savefig('valMAE_loop_15nodes.png')

def save_load_model(model,path):
    torch.save(model,path)
    model = torch.load(path)
    return model

def log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log.txt'):
    RESULTS[model_name] = (best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Train Msg MAE", "Train Belief MAE", "Test MAE", "Test Msg MAE", 
                                "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)
    with open(path, 'w') as f:
        f.write('\n------------------------------------\n')
        dfAsString = DF_RESULTS.to_string(header=True, index=False)
        f.write(dfAsString+'\n')
        f.write(json.dumps(RESULTS))
        f.write('\n------------------------------------\n')
    return DF_RESULTS


if __name__ == "__main__":
    RESULTS = {}
    DF_RESULTS = pd.DataFrame(columns=["Train MAE", "Train Msg MAE", "Train Belief MAE", "Test MAE", "Test Msg MAE", 
                                "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"])
    # ------------- Prepare data --------------------------------
    dataset = BPDataset(root='data/',num_data=1000, node_len=15, loop=False)
    train_dataset = dataset[0:800]
    val_dataset = dataset[800:900]
    test_dataset = dataset[900:1000]
    # train_dataset = [dataset[0]]
    # val_dataset = [dataset[0]]
    # test_dataset = [dataset[0]]
    print("Single graph data shape: ", train_dataset[0])
    print(f"Total number of training samples (graphs): {len(train_dataset)}.")
    print(
        f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # ------------------------------ MPNN --------------------------------
    # model = MPNN(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,error_iter = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_15nodes.pth')
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_15nodes.txt')
    # # plot(DF_RESULTS)

    # #------------------------------ MPNN_SenderAggr --------------------------------
    # model = MPNN_SenderAggr(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,error_iter = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_SenderAggr_15nodes.pth')
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_15nodes.txt')
    # # plot(DF_RESULTS)

    # # #------------------------------ MPNN_Structure2Vec --------------------------------
    # model = MPNN_Structure2Vec(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,error_iter = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_Structure2Vec_15nodes.pth')
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_15nodes.txt')
    # # plot(DF_RESULTS)

    # # #------------------------------ MPNN_2Layer --------------------------------
    # model = MPNN_2Layer(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,error_iter = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_2Layer_15nodes.pth')
    # the_model = torch.load('model/MPNN_2Layer_15nodes.pth')
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_15nodes.txt')
    # plot(DF_RESULTS)
    #-------------------------------
    
    # #----------------------- Test on larger graphs----------------------
    # MPNN_model = torch.load('model/MPNN_15nodes.pth')
    # MPNN_SenderAggr_model = torch.load('model/MPNN_SenderAggr_15nodes.pth')
    # MPNN_Structure2Vec_model = torch.load('model/MPNN_Structure2Vec_15nodes.pth')
    # MPNN_2Layer_model = torch.load('model/MPNN_2Layer_15nodes.pth')

    # dataset_15 = test_dataset
    # dataset_20 = BPDataset(root='data/',num_data=100, node_len=20, loop=False)
    # dataset_25 = BPDataset(root='data/',num_data=100, node_len=25, loop=False)
    # dataset_30 = BPDataset(root='data/',num_data=100, node_len=30, loop=False)

    # general_data = [dataset_15,dataset_20,dataset_25,dataset_30]
    # # general_data = [dataset_30]
    # models = [MPNN_model,MPNN_SenderAggr_model,MPNN_Structure2Vec_model,MPNN_2Layer_model]
    
    # data = []
    # data_msg = []
    # data_belief = []
    # bar_data_all = []
    # bar_data_msg = []
    # bar_data_belief = []
    # for model in models:
    #     print(type(model).__name__)
    #     for d in general_data:
    #         # print(d[0])
    #         test_loader = DataLoader(d, batch_size=100, shuffle=True)
    #         test_error, test_msg_error, test_belief_error,error_iter,error_iter_belief, error_iter_msg = eval(model, test_loader, 'cpu', 0)
    #         # print(error_iter)
    #         # print(error_iter_belief)
    #         # print(error_iter_msg)
    #         for i,e in enumerate(error_iter):
    #             data.append([e.item(), i, type(model).__name__])
    #             data_msg.append([error_iter_msg[i].item(), i, type(model).__name__])
    #             data_belief.append([error_iter_belief[i].item(), i, type(model).__name__])

    #         # For generalisation
    #         bar_data_all.append([test_error,type(model).__name__,d[0].x.size(dim=0)])
    #         bar_data_msg.append([test_msg_error,type(model).__name__,d[0].x.size(dim=0)])
    #         bar_data_belief.append([test_belief_error,type(model).__name__,d[0].x.size(dim=0)])
    
            
    # df = pd.DataFrame(data, columns = ['y','x','Model'])
    # df['x'] = df['x'].astype(int)
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # inspect = "MPNN_2Layer"
    # print(df[df.Model=='MPNN_2Layer'])
    # print((df['y'][(df.Model == inspect) & (df.x==18)].values[0] - df['y'][(df.Model == inspect) & (df.x==0)].values[0]) / df['y'][(df.Model == inspect) & (df.x==0)].values[0])
    # sns.set_theme()
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # p = sns.lineplot(x='x', y='y', hue="Model", data=df, ci=None)
    # p.set(xlabel='BP Iteration', ylabel='MAE (Message + Belief)')
    # p.set(ylim=(0, 0.6))
    # # p.get_figure().savefig('paper_iterVSmae.png')
    # # plt.show()

    # df = pd.DataFrame(data_msg, columns = ['y','x','Model'])
    # df['x'] = df['x'].astype(int)
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # print(df[df.Model=='MPNN_2Layer'])
    # print((df['y'][(df.Model == inspect) & (df.x==18)].values[0] - df['y'][(df.Model == inspect) & (df.x==0)].values[0]) / df['y'][(df.Model == inspect) & (df.x==0)].values[0])
    # sns.set_theme()
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # p = sns.lineplot(x='x', y='y', hue="Model", data=df, ci=None)
    # p.set(xlabel='BP Iteration', ylabel='MAE (Message Only)')
    # p.set(ylim=(0, 0.6))
    # # p.get_figure().savefig('paper_iterVSmae_msg.png')
    # # plt.show()

    # df = pd.DataFrame(data_belief, columns = ['y','x','Model'])
    # df['x'] = df['x'].astype(int)
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # print((df['y'][(df.Model == inspect) & (df.x==18)].values[0] - df['y'][(df.Model == inspect) & (df.x==0)].values[0]) / df['y'][(df.Model == inspect) & (df.x==0)].values[0])
    # sns.set_theme()
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # p = sns.lineplot(x='x', y='y', hue="Model", data=df, ci=None)
    # p.set(xlabel='BP Iteration', ylabel='MAE (Belief Only)')
    # p.set(ylim=(0, 0.6))
    # p.get_figure().savefig('paper_iterVSmae_belief.png')
    # plt.show()

    # -------- Generalisation
    # df = pd.DataFrame(bar_data_all, columns = ['y','Model','Graph Size'])
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # print(df['y'])
    # sns.set_theme()
    # p = sns.barplot(x='Graph Size', y="y", hue='Model', data=df)
    # p.set(xlabel='Number of Nodes', ylabel='MAE (Message + Belief)')
    # p.set(ylim=(0, 0.4))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #       ncol=3, fancybox=True, shadow=True, prop={'size': 7})
    # # for container in p.containers:
    # #     p.bar_label(container)
    # # p.get_figure().savefig('paper_bar.png')
    # # plt.show()


    # df = pd.DataFrame(bar_data_msg, columns = ['y','Model','Graph Size'])
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # print(df['y'])
    # sns.set_theme()
    # p = sns.barplot(x='Graph Size', y="y", hue='Model', data=df)
    # p.set(xlabel='Number of Nodes', ylabel='MAE ( Message Only)')
    # p.set(ylim=(0, 0.4))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #       ncol=3, fancybox=True, shadow=True, prop={'size': 7})
    # # p.get_figure().savefig('paper_bar_msg.png')
    # # plt.show()


    # df = pd.DataFrame(bar_data_belief, columns = ['y','Model','Graph Size'])
    # df.loc[df.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # print(df['y'])
    # sns.set_theme()
    # p = sns.barplot(x='Graph Size', y="y", hue='Model', data=df)
    # p.set(xlabel='Number of Nodes', ylabel='MAE (Belief Only)')
    # p.set(ylim=(0, 0.4))
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #       ncol=3, fancybox=True, shadow=True, prop={'size': 7})
    # # p.get_figure().savefig('paper_bar_belief.png')
    # # plt.show()


    

    #----------------------- Transfer Learning Loopy Graph ----------------------

    dataset = BPDataset(root='data/',num_data=1000, node_len=15, loop=True)
    train_dataset = dataset[0:800]
    val_dataset = dataset[800:900]
    test_dataset = dataset[900:1000]
    print(len(dataset))
    # train_dataset = [dataset[0]]
    # val_dataset = [dataset[0]]
    # test_dataset = [dataset[0]]

    print("Single graph data shape: ", train_dataset[0])
    print(f"Total number of training samples (graphs): {len(train_dataset)}.")
    print(
        f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



    # MPNN_model = torch.load('model/MPNN_15nodes.pth')
    # MPNN_SenderAggr_model = torch.load('model/MPNN_SenderAggr_15nodes.pth')
    # MPNN_Structure2Vec_model = torch.load('model/MPNN_Structure2Vec_15nodes.pth')
    # MPNN_2Layer_model = torch.load('model/MPNN_2Layer_15nodes.pth')

    
    # model = MPNN_Loop_Transfer(model=MPNN_model)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_Loop_Transfer.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_SenderAggr_Loop_Transfer(model=MPNN_SenderAggr_model)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_SenderAggr_Loop_Transfer.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_Structure2Vec_Loop_Transfer(model=MPNN_Structure2Vec_model)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_Structure2Vec_Loop_Transfer.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_2Layer_Loop_Transfer(model=MPNN_2Layer_model)
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_2Layer_Loop_Transfer.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_Loop()
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_Loop.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)


    # model = MPNN_SenderAggr_Loop()
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_SenderAggr_Loop.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_Structure2Vec_Loop()
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_Structure2Vec_Loop.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # plot_loop(DF_RESULTS)

    # model = MPNN_2Layer_Loop()
    # model_name = type(model).__name__
    # best_val_error, best_test_error, train_time, perf_per_epoch = run_experiment_loop(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=2000
    # )

    # # save_load_model(model,'model/MPNN_2Layer_Loop.pth')
    # best_test_msg_error = best_test_belief_error = 0
    # DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch,path='log_transfer_15nodes.txt')
    # # print(DF_RESULTS)
    # plot_loop(DF_RESULTS)

    
    # ----------------------- Test on larger graphs----------------------
    MPNN_Loop_Transfer_model = torch.load('model/MPNN_Loop_Transfer_15nodes.pth')
    MPNN_SenderAggr_Loop_Transfer_model = torch.load('model/MPNN_SenderAggr_Loop_Transfer_15nodes.pth')
    MPNN_Structure2Vec_Loop_Transfer_model = torch.load('model/MPNN_Structure2Vec_Loop_Transfer_15nodes.pth')
    MPNN_2Layer_Loop_Transfer_model = torch.load('model/MPNN_2Layer_Loop_Transfer_15nodes.pth')

    MPNN_Loop_model = torch.load('model/MPNN_Loop_15nodes.pth')
    MPNN_SenderAggr_Loop_model = torch.load('model/MPNN_SenderAggr_Loop_15nodes.pth')
    MPNN_Structure2Vec_Loop_model = torch.load('model/MPNN_Structure2Vec_Loop_15nodes.pth')
    MPNN_2Layer_Loop_model = torch.load('model/MPNN_2Layer_Loop_15nodes.pth')

    dataset_15 = test_dataset
    dataset_20 = BPDataset(root='data/',num_data=100, node_len=20, loop=True)
    dataset_25 = BPDataset(root='data/',num_data=100, node_len=25, loop=True)
    dataset_30 = BPDataset(root='data/',num_data=100, node_len=30, loop=True)

    general_data = [dataset_15,dataset_20,dataset_25,dataset_30]
    # general_data = [dataset_30]
    models = [MPNN_Loop_Transfer_model,MPNN_SenderAggr_Loop_Transfer_model,MPNN_Structure2Vec_Loop_Transfer_model,MPNN_2Layer_Loop_Transfer_model,
                MPNN_Loop_model,MPNN_SenderAggr_Loop_model,MPNN_Structure2Vec_Loop_model,MPNN_2Layer_Loop_model]
    
    data =[]
    bar_data = []
    
    for model in models:
        print(type(model).__name__)
        for d in general_data:
            # print(d[0])
            test_loader = DataLoader(d, batch_size=100)
            test_error, error_iter = eval_loop(model, test_loader, 'cpu', 0)


            # For error iteration
            for i,e in enumerate(error_iter):
                data.append([e.item(), i+1, type(model).__name__])

            # For generalisation
            bar_data.append([test_error,type(model).__name__,d[0].x.size(dim=0)])

            
    # -------- For error iteration
    # error = [] # size 20
    # for i in range(1,20):
    #     loss = 0 #Current Iteration across all graph
    #     for d in dataset_30:
    #         belief = d.beliefs.transpose(1, 0)
    #         # print(belief.shape)
    #         loss += F.l1_loss(belief[i], d.y).item()
    #     error.append(loss/100)

    # for i in range(19):
    #     data.append([error[i], i+1, 'Loopy BP'])



    # df = pd.DataFrame(data, columns = ['y','x','Model'])
    # df.loc[df.Model == 'MPNN_Loop_Transfer', 'Model'] = "MPNN_Transfer"
    # df.loc[df.Model == 'MPNN_SenderAggr_Loop_Transfer', 'Model'] = "MPNN_SenderAggr_Transfer"
    # df.loc[df.Model == 'MPNN_Structure2Vec_Loop_Transfer', 'Model'] = "MPNN_Struct2Vec_Transfer"
    # df.loc[df.Model == 'MPNN_2Layer_Loop_Transfer', 'Model'] = "MPNN_2Layer_Transfer"

    # df.loc[df.Model == 'MPNN_Loop', 'Model'] = "MPNN_Loopy"
    # df.loc[df.Model == 'MPNN_SenderAggr_Loop', 'Model'] = "MPNN_SenderAggr_Loopy"
    # df.loc[df.Model == 'MPNN_Structure2Vec_Loop', 'Model'] = "MPNN_Struct2Vec_Loopy"
    # df.loc[df.Model == 'MPNN_2Layer_Loop', 'Model'] = "MPNN_2Layer_Loopy"

    # inspect = "MPNN_2Layer_Transfer"
    # print(df[(df.Model == inspect)])
    # # print(df[(df.Model == inspect)])
    # # print(df['y'][(df.Model == inspect) & (df.x==0)].values)
    # # for inspect in ["MPNN_Transfer","MPNN_SenderAggr_Transfer","MPNN_Struct2Vec_Transfer","MPNN_2Layer_Transfer","MPNN_Loopy","MPNN_SenderAggr_Loopy","MPNN_Struct2Vec_Loopy","MPNN_2Layer_Loopy","Loopy BP"]:
    # #     print((df['y'][(df.Model == inspect) & (df.x==19)].values[0] - df['y'][(df.Model == inspect) & (df.x==1)].values[0]) / df['y'][(df.Model == inspect) & (df.x==1)].values[0])
    # sns.set_theme()
    # ax = plt.figure().gca()
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # p = sns.lineplot(x='x', y='y', hue="Model", data=df,ci=None)
    # p.set(xlabel='BP Iteration', ylabel='MAE (Belief Only)')
    # p.set(ylim=(0, 0.6))
    # plt.show()
    # p.get_figure().savefig('paper_iterVSmae_loop.png')

    # -------- Generalisation
    error = [] # size 20

    for data in general_data:
        loss = 0
        for d in data:
            for i in range(1,20):
                belief = d.beliefs.transpose(1, 0)
                # print(belief.shape)
                loss += F.l1_loss(belief[i], d.y).item()
        error.append(loss/2000)

  
    bar_data.append([error[0],'Loopy BP',15])
    bar_data.append([error[1],'Loopy BP',20])
    bar_data.append([error[2],'Loopy BP',25])
    bar_data.append([error[3],'Loopy BP',30])




    df = pd.DataFrame(bar_data, columns = ['y','Model','Graph Size'])

    df.loc[df.Model == 'MPNN_Loop_Transfer', 'Model'] = "MPNN_Transfer"
    df.loc[df.Model == 'MPNN_SenderAggr_Loop_Transfer', 'Model'] = "MPNN_SenderAggr_Transfer"
    df.loc[df.Model == 'MPNN_Structure2Vec_Loop_Transfer', 'Model'] = "MPNN_Struct2Vec_Transfer"
    df.loc[df.Model == 'MPNN_2Layer_Loop_Transfer', 'Model'] = "MPNN_2Layer_Transfer"

    df.loc[df.Model == 'MPNN_Loop', 'Model'] = "MPNN_Loopy"
    df.loc[df.Model == 'MPNN_SenderAggr_Loop', 'Model'] = "MPNN_SenderAggr_Loopy"
    df.loc[df.Model == 'MPNN_Structure2Vec_Loop', 'Model'] = "MPNN_Struct2Vec_Loopy"
    df.loc[df.Model == 'MPNN_2Layer_Loop', 'Model'] = "MPNN_2Layer_Loopy"
    print(df)
    sns.set_theme()

    p = sns.barplot(x='Graph Size', y="y", hue='Model', data=df)
    p.set(xlabel='Number of Nodes', ylabel='MAE (Belief Only)')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=True, prop={'size': 7})
    p.get_figure().savefig('paper_bar_loop.png')
    # p.legend(loc='center left', bbox_to_anchor=(1, 0.5),)    # p.set(ylim=(0, 0.6))
    plt.show()


    # ------------------------------ Plot ----------------------------
    # data = pd.read_csv('temp.txt', sep="\s+", header=None)
    # data.columns = ["Train MAE", "Train Msg MAE" , "Train Belief MAE", "Test MAE", "Test Msg MAE", "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"]
    # data.loc[data.Model == 'MPNN_Structure2Vec', 'Model'] = "MPNN_Struct2Vec"
    # DF_RESULTS = data
    # print(data.loc[data['Model'] == 'MPNN_SenderAggr'])
    # plt.figure()
    # sns.set()
    # p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    # p.set(ylim=(0, 0.8))
    # p.set(xlim=(0, 200))
    # p.set(xlabel='Epoch', ylabel='MAE (Message + Belief)')
    # plt.show()
    # p.get_figure().savefig('paper_testMAE.png')

    # plt.figure()
    # sns.set()
    # p = sns.lineplot(x="Epoch", y="Test Msg MAE", hue="Model", data=DF_RESULTS)
    # p.set(ylim=(0, 0.8))
    # p.set(xlim=(0, 200))
    # p.set(xlabel='Epoch', ylabel='MAE ( Message Only)')
    # plt.show()
    # p.get_figure().savefig('paper_testMAE_Msg.png')

    # plt.figure()
    # sns.set()
    # p = sns.lineplot(x="Epoch", y="Test Belief MAE", hue="Model", data=DF_RESULTS)
    # p.set(ylim=(0, 0.8))
    # p.set(xlim=(0, 200))
    # p.set(xlabel='Epoch', ylabel='MAE (Belief Only)')
    # plt.show()
    # p.get_figure().savefig('paper_testMAE_Belief.png')

    # # #------------------------------- Plot Loop--------------------------
    # data = pd.read_csv('temp2.txt', sep="\s+", header=None)
    # data.columns = ["Train MAE", "Train Msg MAE" , "Train Belief MAE", "Test MAE", "Test Msg MAE", "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"]
    
    # data.loc[data.Model == 'MPNN_Loop_Transfer', 'Model'] = "MPNN_Transfer"
    # data.loc[data.Model == 'MPNN_SenderAggr_Loop_Transfer', 'Model'] = "MPNN_SenderAggr_Transfer"
    # data.loc[data.Model == 'MPNN_Structure2Vec_Loop_Transfer', 'Model'] = "MPNN_Struct2Vec_Transfer"
    # data.loc[data.Model == 'MPNN_2Layer_Loop_Transfer', 'Model'] = "MPNN_2Layer_Transfer"

    # data.loc[data.Model == 'MPNN_Loop', 'Model'] = "MPNN_Loopy"
    # data.loc[data.Model == 'MPNN_SenderAggr_Loop', 'Model'] = "MPNN_SenderAggr_Loopy"
    # data.loc[data.Model == 'MPNN_Structure2Vec_Loop', 'Model'] = "MPNN_Struct2Vec_Loopy"
    # data.loc[data.Model == 'MPNN_2Layer_Loop', 'Model'] = "MPNN_2Layer_Loopy"

    # DF_RESULTS = data
    # plt.figure()
    # sns.set()
    # p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    # p.set(ylim=(0, 0.5))
    # p.set(xlim=(0, 500))
    # p.set(xlabel='Epoch', ylabel='MAE (Belief Only)')
    # plt.show()
    # p.get_figure().savefig('paper_testMAE_loop.png')