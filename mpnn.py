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

from BP import *
from factor import *
from factor_graph import *
from dataset import *

RESULTS = {}
DF_RESULTS = pd.DataFrame(columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])

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
        self.U = nn.Sequential(nn.Linear(2*h_dim, h_dim, bias=bias),
                               nn.LeakyReLU())
        self.h_msg_temp = None

    def forward(self, h_node, edge_index, h_msg, encoded_msg):
        h_node = self.propagate(edge_index, h_node=h_node, h_msg=h_msg, encoded_msg=encoded_msg)
        return h_node, self.h_msg_temp

    def message(self, h_node_i, h_node_j, encoded_msg, edge_index):
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
        # print("First edge source node:", edge_index[1][0])
        # print("aggr_msgs_j", aggr_msgs_j)
        # print("encoded_msg", encoded_msg)
        print("diff", (aggr_msgs_j-encoded_msg).shape)
        # print("aggr_msgs[edge_index[0]]", aggr_msgs[5])
        # print("edge_indx: ")
        # print(edge_index)

        aggr_msg = self.N(aggr_msgs_j - encoded_msg)
        self.h_msg_temp = aggr_msg
        return aggr_msg

    def update(self, aggr_out, h_node):
        h_node = self.U(torch.cat((h_node, aggr_out), dim=1))
        return h_node
    

class DecoderLayer(MessagePassing):
    # Implement decoder node?
    def __init__(self, hidden_dim=32,msg_dim=2, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)
        self.lin = nn.Linear(hidden_dim, msg_dim, bias)
    
    def forward(self, h_msg):
        # Normalise message for prediction
        return torch.softmax(self.lin(h_msg), dim=-1)

class MPNN(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor = MPNN_Layer()
        self.decoder = DecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        return h_msg, y_msg

class MPNN_SenderAggr(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor_2 = MPNN_SenderAggr_Layer()
        self.decoder = DecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        return h_msg, y_msg

class MPNN_Structure2Vec(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor_3 = MPNN_Structure2Vec_Layer()
        self.decoder = DecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor_3(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        return h_msg, y_msg

class MPNN_2Layer(Module):
    def __init__(self, x_dim=2, h_dim=32, msg_dim=2):
        super().__init__()
        self.lin_in = Linear(x_dim, h_dim)
        self.h_dim=h_dim
        self.encoder = EncoderLayer()
        self.processor = MPNN_Layer()
        self.processor_2 = MPNN_Layer()
        self.decoder = DecoderLayer()

    def forward(self, data, h_msg, y_msg):
        h_node = self.lin_in(data.x)
        # encoded_message -> z
        encoded_msg = self.encoder(y_msg,h_msg)
        # hidden_node -> h
        h_node, h_msg = self.processor(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        h_node, h_msg = self.processor_2(h_node,data.edge_index,h_msg=h_msg,encoded_msg=encoded_msg)
        # decoded_message -> y
        y_msg = self.decoder(h_msg)
        return h_msg, y_msg


def train(model, train_loader, optimizer, device, epoch):
    torch.autograd.set_detect_anomaly(True)
    model.train()
    loss_all = 0
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
        y_msg = data.edge_attr[0]
        #Start from step_index 1, there's no prediction for step_index 0
        for step_idx in range(1,data.edge_attr.size(dim=0)):
            h_msg, y_msg = model(data, h_msg=h_msg, y_msg=y_msg)
            y_msg_pred.append(y_msg)
            y_msg = data.edge_attr[step_idx]

        y_msg_pred = torch.stack(y_msg_pred)
        # print("y_msg_pred: ", y_msg_pred[-1,-1])
        # print("data.edge_attr: ", data.edge_attr[-1,-1])
        # if epoch % 10 == 0:
        #     print("[Last Iteration] y_msg_pred: ", (y_msg_pred[-1]*10**3).round() / (10**3)  )
        #     print("[Last Iteration] data.edge_attr: ", data.edge_attr[-1])
        loss = F.l1_loss(y_msg_pred, data.edge_attr[1:])

        loss.backward(retain_graph=False)
        # print("train number of graphs per batch: ", data.num_graphs)
        # print("train loss (one batch): ", loss.item())
        loss_all += loss.item() * data.num_graphs # number of graphs per batch
        optimizer.step()

    # print('total graph: ', len(train_loader.dataset))
    return loss_all / len(train_loader.dataset) # number of total graphs



def eval(model, loader, device, epoch):
    model.eval()
    error = 0
    for data in loader:
        data = prepare_batch(data)
        data = data.to(device)
        data.x, data.edge_attr = data.x.float(), data.edge_attr.float()
        h_dim=32
        h_msg = torch.zeros(data.edge_attr[0].size(dim=0), h_dim)

        y_msg = data.edge_attr[0]
        y_msg_pred = []
        with torch.no_grad():
            for step_idx in range(1, data.edge_attr.size(dim=0)):
                h_msg, y_msg = model(data, h_msg=h_msg, y_msg=y_msg)
                y_msg_pred.append(y_msg)
            # Mean Absolute Error
            y_msg_pred = torch.stack(y_msg_pred)
            # print("y_msg_pred: ", y_msg_pred[-1])
            # print("data.edge_attr: ", data.edge_attr[-1])
            
            # TODO: (Done): train MSE val MAE
            # TODO (Done): Make sure to normalise data message
            # error += F.mse_loss(y_msg_pred, data.edge_attr[1:])
            # print("eval number of graphs per batch: ", data.num_graphs)
            # print("eval loss (one batch): ", F.l1_loss(y_msg_pred, data.edge_attr[1:]))
            # print("eval loss (one batch): ", )
            # error += (y_msg_pred - data.edge_attr[1:]).abs().sum().item()
            error += F.l1_loss(y_msg_pred, data.edge_attr[1:]).item() * data.num_graphs
            if epoch == 2000:
                print(F.kl_div(y_msg_pred, data.edge_attr[1:]))
            # print(data.edge_index[:,0])
            # breakpoint()
    # print('total graph: ', len(loader.dataset))
    return error / len(loader.dataset)


def run_experiment(model, model_name, train_loader, val_loader, test_loader, n_epochs=100):

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
    perf_per_epoch = []  # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # print('epoch: ', epoch)
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device, epoch)

        # Evaluate model on validation set
        val_error = eval(model, val_loader, device, epoch)

        if best_val_error is None or val_error <= best_val_error:
            # Evaluate model on test set if validation metric improves
            test_error = eval(model, test_loader, device, epoch)
            best_val_error = val_error

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                  f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((loss, test_error, val_error, epoch, model_name))

    t = time.time() - t
    train_time = t/60
    print(
        f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_error:.7f}, corresponding test MAE: {test_error:.7f}.")

    return best_val_error, test_error, train_time, perf_per_epoch

def prepare_batch(batch):
    batch.edge_attr = batch.edge_attr.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    return batch

# def prepare_datatset():
#     mrf = string2factor_graph('f1(a,b)f2(b,c,d)f3(c)')
#     f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
#     f2 = factor(['b', 'd', 'c'], np.array([[[7,2],[1,5]],[[8,3],[6,4]]]))
#     f3 = factor(['c'],           np.array([5, 1]))
#     mrf.change_factor_distribution('f1', f1)
#     mrf.change_factor_distribution('f2', f2)
#     mrf.change_factor_distribution('f3', f3)
#     lbp = loopy_belief_propagation(mrf)
#     lbp.belief('a', 10)
#     msg, belief = lbp.get_msg_belief()
#     data = create_data(mrf,msg, belief,2)
#     print(data.edge_index)
#     dataset = [data]
#     # dataset = [data]*100
#     # train_dataset = dataset[:30]
#     # val_dataset = dataset[30:50]
#     # test_dataset = dataset[50:100]
#     train_dataset = dataset
#     val_dataset = dataset
#     test_dataset = dataset
#     return train_dataset, val_dataset, test_dataset

def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

if __name__ == "__main__":
    # Prepare data
    dataset = BPDataset(root='data/',num_data=3000)
    # train_dataset, val_dataset, test_dataset = split(dataset, 3)

    train_dataset =  val_dataset = test_dataset = [dataset[0]]

    print("Single graph data shape: ", train_dataset[0])
    print(f"Total number of training samples (graphs): {len(train_dataset)}.")
    print(
        f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # Create dataloaders with batch size = 16
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    # Training
    # model = MPNN(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=200
    # )
    
    # RESULTS[model_name] = (best_val_error, test_error, train_time)
    # df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])
    # DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)


    # model = MPNN_SenderAggr(x_dim=3)
    # model_name = type(model).__name__
    # best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=200
    # )
    
    # RESULTS[model_name] = (best_val_error, test_error, train_time)
    # df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])
    # DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

    model = MPNN_Structure2Vec(x_dim=3)
    model_name = type(model).__name__
    best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=200
    )
    
    RESULTS[model_name] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

    model = MPNN_2Layer(x_dim=3)
    model_name = type(model).__name__
    best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
        model, 
        model_name, 
        train_loader,
        val_loader, 
        test_loader,
        n_epochs=200
    )
    
    RESULTS[model_name] = (best_val_error, test_error, train_time)
    df_temp = pd.DataFrame(perf_per_epoch, columns=["Train MAE", "Test MAE", "Val MAE", "Epoch", "Model"])
    DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)

    sns.set()
    p = sns.lineplot(x="Epoch", y="Train MAE", hue="Model", data=DF_RESULTS)
    p.set(ylim=(0, 1))
    plt.show()


    # Save and load model
    # path = 'model/model.pth'
    # path = 'model/model_3000.pth'
    # torch.save(model,path)
    # model = torch.load(path)

   

    #TODO: Loss graph
    #TODO: loopy graph


    
    
