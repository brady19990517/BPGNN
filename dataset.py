from BP import *
from factor_graph import *
from factor import *
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Dataset, Data
import numpy as np 
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import networkx as nx
import os
from tqdm import tqdm
from pprint import pprint
from torch_geometric.utils.convert import from_networkx

# print(f"Torch version: {torch.__version__}")
# print(f"Torch geometric version: {torch_geometric.__version__}")

#Factor Types
# Sum
# train size


def create_data(pgm, msgs, beliefs, num_classes=2):
    #Create Features
    feat = []
    for v in pgm.get_graph().vs:
        feat.append(1) if v["is_factor"] else feat.append(0)
    
    #Create edge index
    edge_index = []
    for e in pgm.get_graph().es:
        edge_index.append([e.tuple[0],e.tuple[1]])
        edge_index.append([e.tuple[1],e.tuple[0]])
    edge_index = np.array(edge_index)

    #Create edge attr
    iters = len(msgs)
    edge_attr = []
    for e in edge_index:
        temp = []
        for iter in range(iters):
            temp.append(np.array(msgs[iter][(pgm.get_graph().vs[int(e[0])]['name'],pgm.get_graph().vs[int(e[1])]['name'])]))
        edge_attr.append(np.array(temp))
    edge_attr = np.array(edge_attr)
    
    #Create label
    y = []
    for v in pgm.get_graph().vs:
        temp = []
        if not v["is_factor"]:
            for iter in range(iters):
                temp.append(np.array(beliefs[iter][v['name']]))
            y.append(np.array(temp))
    y = np.array(y)

    #Create Data Object
    data = Data(F.one_hot(torch.tensor(feat),num_classes),
                edge_index=torch.tensor(np.transpose(edge_index)),
                edge_attr=torch.tensor(edge_attr),
                y=torch.tensor(y)) 
    return data


# mrf = string2factor_graph('f1(a,b)f2(b,c,d)f3(c)')
# f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
# f2 = factor(['b', 'd', 'c'], np.array([[[7,2],[1,5]],[[8,3],[6,4]]]))
# f3 = factor(['c'],           np.array([5, 1]))
# mrf.change_factor_distribution('f1', f1)
# mrf.change_factor_distribution('f2', f2)
# mrf.change_factor_distribution('f3', f3)
# lbp = loopy_belief_propagation(mrf)
# lbp.belief('a', 10)
# msg, belief = lbp.get_msg_belief()
# data_1 = create_data(mrf,msg, belief,2)



# mrf = string2factor_graph('f1(a,b)')
# f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
# mrf.change_factor_distribution('f1', f1)
# lbp = loopy_belief_propagation(mrf)
# lbp.belief('a', 10)
# msg, belief = lbp.get_msg_belief()
# data_2 = create_data(mrf,msg, belief,2)


# print(data_1)
# # print(data_1.x)
# # print(data_1.edge_index)
# # print(data_1.edge_attr)
# # print(data_1.y)
# print(data_2)

# data_list = [data_1, data_2]
# batch = Batch.from_data_list(data_list)
# print(batch)

# # Create DataLoader
# loader = DataLoader(data_list, batch_size=2, shuffle=False)
# it = iter(loader)
# batch_1 = next(it)
# batch_2 = next(it)

# assert (batch_1.x == data_1.x).all() and (batch_2.x == data_2.x).all()


# graph = gen_graph(10,20)
# # print(graph)


if __name__ == "__main__":
    # graph, variables, factors = gen_graph()
    # print(graph)
    # # data = from_networkx(graph,group_node_attrs,group_edge_attrs)
    # # print(data)
    # node_attrs = []
    # for i in graph.nodes():
    #     if i in variables:
    #         node_attrs.append(0)
    #     else:
    #         node_attrs.append(random.randint(1, 2))
    # group_node_attrs = F.one_hot(torch.tensor(node_attrs),3)
    # print(group_node_attrs)



    mrf = gen_graph()
    for v in  mrf.get_graph().vs:
        if v["is_factor"]:
            print(v['name'])
            print(mrf.get_graph().vs[mrf.get_graph().neighbors(v['name'])]['name'])
    # f1 = factor(['a', 'b'],      np.array([[2,3],[6,4]]))
    # f2 = factor(['b', 'd', 'c'], np.array([[[7,2],[1,5]],[[8,3],[6,4]]]))
    # f3 = factor(['c'],           np.array([5, 1]))
    # mrf.change_factor_distribution('f1', f1)
    # mrf.change_factor_distribution('f2', f2)
    # mrf.change_factor_distribution('f3', f3)
    # lbp = loopy_belief_propagation(mrf)
    # lbp.belief('a', 10)
    # msg, belief = lbp.get_msg_belief()
    # data_1 = create_data(mrf,msg, belief,2)

