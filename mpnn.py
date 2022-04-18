# @title [RUN] Import python modules

from multiprocessing.spawn import prepare
import os
import random
import time
from pprint import pprint
import seaborn as sns

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
        # aggr_msg = self.N_node(h_node_j) + self.M(torch.cat([h_node_i, h_node_j, encoded_msg], dim=-1))
        aggr_msg = self.M(torch.cat([h_node_i, h_node_j, encoded_msg], dim=-1))
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

        # aggr_msg = self.N_node(h_node_j) + self.N(torch.cat([aggr_msgs_j, encoded_msg], dim=-1))
        aggr_msg = self.N(torch.cat([aggr_msgs_j, encoded_msg], dim=-1))
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
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
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
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
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
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
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

class MPNN_2Structure2VecLayer(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor = MPNN_Structure2Vec_Layer()
        self.processor_3 = MPNN_Structure2Vec_Layer()
        self.decoder = MessageDecoderLayer()
        self.belief_decoder = BeliefsDecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        h_node, h_msg = self.processor_3(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        y_beliefs = self.belief_decoder(h_node,data.x)
        return h_msg, y_msg, y_beliefs

class MPNN_2Layer(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
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
            for step_idx in range(1, data.edge_attr.size(dim=0)):
                h_msg, y_msg, y_beliefs = model(data, h_msg=h_msg, y_msg=y_msg)
                # h_msg, y_msg = model(data, h_msg=h_msg, y_msg=y_msg)
                y_msg_pred.append(y_msg)
                y_beliefs_pred.append(y_beliefs)
    
            y_msg_pred = torch.stack(y_msg_pred)
            y_beliefs_pred = torch.stack(y_beliefs_pred)
         
            loss_msg = F.l1_loss(y_msg_pred, data.edge_attr[1:])
            loss_belief = F.l1_loss(y_beliefs_pred, data.y[1:])

            error += loss_msg.item() * data.num_graphs
            msg_loss_all += loss_msg.item() * data.num_graphs

            error += loss_belief.item() * data.num_graphs
            belief_loss_all += loss_belief.item() * data.num_graphs

    return error / len(loader.dataset), msg_loss_all / len(loader.dataset), belief_loss_all / len(loader.dataset)


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
    for epoch in range(1, n_epochs+1):
        # print('epoch: ', epoch)
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        train_loss, train_msg_loss, train_belief_loss = train(model, train_loader, optimizer, device, epoch)

        # Evaluate model on validation set
        val_error, val_msg_error, val_belief_error = eval(model, val_loader, device, epoch)

        
        # Evaluate model on test set if validation metric improves
        test_error, test_msg_error, test_belief_error = eval(model, test_loader, device, epoch)

        if best_val_error is None or val_error <= best_val_error:
            best_val_error = val_error
            p_counter = 0

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

    return best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch

def prepare_batch(batch):
    batch.edge_attr = batch.edge_attr.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    return batch

def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def plot(DF_RESULTS):
    sns.set()
    p = sns.lineplot(x="Epoch", y="Train MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('trainMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Train Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('trainMsgMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Train Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('trainBeliefMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Test MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('testMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Test Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('testMsgMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Test Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('testBeliefMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Val MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('valMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Val Msg MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('valMsgMAE.png')

    sns.set()
    p = sns.lineplot(x="Epoch", y="Val Belief MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()
    # p.get_figure().savefig('valBeliefMAE.png')

def save_load_model(model,path):
    torch.save(model,path)
    model = torch.load(path)
    return model

def log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch):
    RESULTS[model_name] = (best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Train Msg MAE", "Train Belief MAE", "Test MAE", "Test Msg MAE", 
                                "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)
    with open('log.txt', 'a') as f:
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
    dataset = BPDataset(root='data/',num_data=1000, loop=False)
    train_dataset = dataset[0:800]
    val_dataset = dataset[800:900]
    test_dataset = dataset[900:1000]
    print("Single graph data shape: ", train_dataset[0])
    print(f"Total number of training samples (graphs): {len(train_dataset)}.")
    print(
        f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    #------------------------------ MPNN --------------------------------
    model = MPNN(x_dim=3)
    model_name = type(model).__name__
    best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=2000
    )

    save_load_model(model,'model/MPNN.pth')
    DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch)
    # plot(DF_RESULTS)

    #------------------------------ MPNN_SenderAggr --------------------------------
    model = MPNN_SenderAggr(x_dim=3)
    model_name = type(model).__name__
    best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=2000
    )

    save_load_model(model,'model/MPNN_SenderAggr.pth')
    DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch)
    # plot(DF_RESULTS)

    # #------------------------------ MPNN_Structure2Vec --------------------------------
    model = MPNN_Structure2Vec(x_dim=3)
    model_name = type(model).__name__
    best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=2000,
        patience=100
    )

    save_load_model(model,'model/MPNN_Structure2Vec.pth')
    DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch)
    plot(DF_RESULTS)

    # #------------------------------ MPNN_2Layer --------------------------------
    model = MPNN_2Layer(x_dim=3)
    model_name = type(model).__name__
    best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=2000
    )

    save_load_model(model,'model/MPNN_2Layer.pth')
    DF_RESULTS = log(RESULTS, DF_RESULTS, best_val_error, best_test_error, best_test_msg_error, best_test_belief_error, train_time, perf_per_epoch)
    plot(DF_RESULTS)


    # data = pd.read_csv('temp.txt', sep=" ", header=None)
    # data.columns = ["Train MAE", "Train Msg MAE" , "Train Belief MAE", "Test MAE", "Test Msg MAE", "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"]
    # df = data[data.Model != 'MPNN_Structure2Vec']

    # print(df)


    # data2 = pd.read_csv('temp2.txt', sep=" ", header=None)
    # data2.columns = ["Train MAE", "Train Msg MAE" , "Train Belief MAE", "Test MAE", "Test Msg MAE", "Test Belief MAE", "Val MAE", "Val Msg MAE", "Val Belief MAE", "Epoch", "Model"]

    # print(data2)
    # result = pd.concat([df, data2], ignore_index=True, sort=False)

    # print(result)
    # plot(result)