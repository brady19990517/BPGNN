# @title [RUN] Import python modules

from multiprocessing.spawn import prepare
import os
import random
import time
from pprint import pprint

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


class MPNNLayer(MessagePassing):
    def __init__(self, hidden_dim=32, aggr='add',bias=True):
        # Set the aggregation function
        super().__init__(aggr=aggr)

        self.M = nn.Sequential(nn.Linear(2*hidden_dim+hidden_dim, hidden_dim, bias=bias),
                               nn.LeakyReLU(),
                               nn.Linear(hidden_dim, hidden_dim, bias=bias),
                               nn.LeakyReLU()
                               )
        self.U = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim, bias=bias),
                               nn.LeakyReLU())
        self.latent_msg = None

    def forward(self, x, edge_index, hidden, z):
        # print("x_feat: ", x.shape)
        # print("hiddent feat before: ", hidden)
        hidden = self.propagate(edge_index, x=x, hidden=hidden, z=z)
        # print("hidden feat after: ", hidden)
        print(self.latent_msg)
        return hidden

    def message(self, x_i, x_j, z):
        # [1,3][1,3][1,32]
        print("x_i: ",x_i.shape)
        print("x_j: ",x_j.shape)
        print("z: ",z.shape)
        msg = self.M(torch.cat([x_i, x_j, z], dim=-1))
        self.latent_msg = msg
        return msg

    def update(self, aggr_out, x):
        hidden = self.U(torch.cat((x, aggr_out), dim=1))
        return hidden

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class MPNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1):
        """Message Passing Neural Network model for graph property prediction

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()

        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)

        # Stack of MPNN layers
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MPNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(data.x)  # (n, d_n) -> (n, d)

        for conv in self.convs:
            # (n, d) -> (n, d)
            h = h + conv(h, data.edge_index, data.edge_attr)
            # Note that we add a residual connection after each MPNN layer

        h_graph = self.pool(h, data.batch)  # (n, d) -> (batch_size, d)

        out = self.lin_pred(h_graph)  # (batch_size, d) -> (batch_size, 1)

        return out.view(-1)

# @title [RUN] Helper functions for managing experiments, training, and evaluating models.


def train(model, train_loader, optimizer, device):
    model.train()
    loss_all = 0

    for data in train_loader:
        # Transpose batch
        data = prepare_batch(data)
        data = data.to(device)
        optimizer.zero_grad()




        y_pred = model(data)
        loss = F.mse_loss(y_pred, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def eval(model, loader, device):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            y_pred = model(data)
            # Mean Absolute Error using std (computed when preparing data)
            error += (y_pred * std - data.y * std).abs().sum().item()
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
        # Call LR scheduler at start of each epoch
        lr = scheduler.optimizer.param_groups[0]['lr']

        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device)

        # Evaluate model on validation set
        val_error = eval(model, val_loader, device)

        if best_val_error is None or val_error <= best_val_error:
            # Evaluate model on test set if validation metric improves
            test_error = eval(model, test_loader, device)
            best_val_error = val_error

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {lr:5f}, Loss: {loss:.7f}, '
                  f'Val MAE: {val_error:.7f}, Test MAE: {test_error:.7f}')

        scheduler.step(val_error)
        perf_per_epoch.append((test_error, val_error, epoch, model_name))

    t = time.time() - t
    train_time = t/60
    print(
        f"\nDone! Training took {train_time:.2f} mins. Best validation MAE: {best_val_error:.7f}, corresponding test MAE: {test_error:.7f}.")

    return best_val_error, test_error, train_time, perf_per_epoch

def prepare_batch(batch):
    print("before batch edge_attr shape: ", batch.edge_attr.shape)
    print("before batch y shape: ", batch.y.shape)
    # batch = batch.to(_DEVICE)
    # if len(batch.x.shape) == 2:
    #     batch.x = batch.x.unsqueeze(-1)
    batch.edge_attr = batch.edge_attr.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    print("after batch edge_attr shape: ", batch.edge_attr.shape)
    print("after batch y shape: ", batch.y.shape)
    return batch

if __name__ == "__main__":
    mrf = string2factor_graph('f1(a,b)f2(b,c,d)f3(c)')
    f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
    f2 = factor(['b', 'd', 'c'], np.array([[[7,2],[1,5]],[[8,3],[6,4]]]))
    f3 = factor(['c'],           np.array([5, 1]))
    mrf.change_factor_distribution('f1', f1)
    mrf.change_factor_distribution('f2', f2)
    mrf.change_factor_distribution('f3', f3)
    lbp = loopy_belief_propagation(mrf)
    lbp.belief('a', 10)
    msg, belief = lbp.get_msg_belief()
    data = create_data(mrf,msg, belief,2)

    dataset = [data]*100
    print("Single datapoint shape: ", data)

    print(f"Total number of samples: {len(dataset)}.")


    train_dataset = dataset[:30]
    val_dataset = dataset[30:50]
    test_dataset = dataset[50:100]

    print(
        f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # Create dataloaders with batch size = 32
    train_loader = DataLoader(train_dataset, batch_size=15, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Transpose Batch
    batch = next(iter(train_loader))

    print("Before batch: ",batch)
    batch = prepare_batch(batch)
    print("After batch: ",batch)
    print(batch.x[0].shape)
    print(batch.y[0].shape)

    hidden_dim = 32
    hidden = torch.zeros(batch.edge_attr[0].size(dim=0), hidden_dim)
    # for i in range(1):
    i=0
    msg = batch.edge_attr[i]
    # print(msg.shape)
    # print(hidden.shape)
    input = torch.cat((msg, hidden), dim=1).float()
    print(input.shape)
    m = nn.Linear(2+hidden_dim, hidden_dim, bias=True)
    z = m(input)
    print(z.shape)
    mpnn = MPNNLayer()
    # project node features
    t = nn.Linear(2, hidden_dim, bias=True)
    print(batch.x.shape)
    latent_x = t(batch.x.float())
    print(latent_x.shape)
    hidden = mpnn(latent_x, batch.edge_index, hidden=hidden, z=z)
    print("latent_node shape: ", hidden.shape)
    print("latent_msg shape: ",mpnn.latent_msg.shape)


    # Y = ….
    # Ye = ge…(ze, hei, hej)
    #decodeer
    decoder_edge = nn.Linear(hidden_dim, 2, bias=True)
    output_msg = decoder_edge(mpnn.latent_msg)
    print(output_msg.shape)






    # for data in train_loader:
    #     print("Before batch: ",data)
    #     data = prepare_batch(data)
    #     print("After batch: ",data)
    #     print(data.x[0].shape)
    #     print(data.y[0].shape)

    # model = MPNNModel(num_layers=4, emb_dim=64, in_dim=11, edge_dim=4, out_dim=1)
    # model_name = type(model).__name__

    # best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    #     model, 
    #     model_name, 
    #     train_loader,
    #     val_loader, 
    #     test_loader,
    #     n_epochs=100
    # )

    
    
    
    
    
    
    
    
    
    
    # model = MPNNModel(num_layers=4, emb_dim=64,
    #                   in_dim=11, edge_dim=4, out_dim=1)
    # model_name = type(model).__name__
    # best_val_error, test_error, train_time, perf_per_epoch = run_experiment(
    #     model,
    #     model_name,
    #     train_loader,
    #     val_loader,
    #     test_loader,
    #     n_epochs=100
    # )

    # RESULTS = {}
    # DF_RESULTS = pd.DataFrame(
    #     columns=["Test MAE", "Val MAE", "Epoch", "Model"])
    # RESULTS[model_name] = (best_val_error, test_error, train_time)
    # df_temp = pd.DataFrame(perf_per_epoch, columns=[
    #                        "Test MAE", "Val MAE", "Epoch", "Model"])
    # DF_RESULTS = DF_RESULTS.append(df_temp, ignore_index=True)
