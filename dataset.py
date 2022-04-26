import dis
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
from torch_geometric.data import InMemoryDataset, download_url

# print(f"Torch version: {torch.__version__}")
# print(f"Torch geometric version: {torch_geometric.__version__}")

#Factor Types
# Sum
# train size
class BPDataset(Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 num_data=100,
                 loop=False,
                 node_len=10):
        self.num_data = num_data
        self.loop = loop
        self.node_len=node_len
        super(BPDataset, self).__init__(root, transform, pre_transform)


    @property
    def processed_file_names(self):
        if self.loop:
            if self.node_len == 15:
                return [f'data_{i}_loop_15.pt' for i in range(self.num_data)]
            elif self.node_len == 20:
                return [f'data_{i}_loop_20.pt' for i in range(self.num_data)]
            elif self.node_len == 25:
                return [f'data_{i}_loop_25.pt' for i in range(self.num_data)]
            elif self.node_len == 30:
                return [f'data_{i}_loop_30.pt' for i in range(self.num_data)]
        else:
            if self.node_len == 15:
                return [f'data_{i}_15.pt' for i in range(self.num_data)]
            elif self.node_len == 20:
                return [f'data_{i}_20.pt' for i in range(self.num_data)]
            elif self.node_len == 25:
                return [f'data_{i}_25.pt' for i in range(self.num_data)]
            elif self.node_len == 30:
                return [f'data_{i}_30.pt' for i in range(self.num_data)]


    def process(self):
        data_list = self.create_dataset(num_data=self.num_data,loop=self.loop,min_node_len=self.node_len,max_node_len=self.node_len)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        for i,data in enumerate(data_list):
            if self.loop:
                if self.node_len == 15:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_loop_15.pt'))
                elif self.node_len == 20:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_loop_20.pt'))
                elif self.node_len == 25:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_loop_25.pt'))
                elif self.node_len == 30:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_loop_30.pt'))
            else:
                if self.node_len == 15:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_15.pt'))
                elif self.node_len == 20:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_20.pt'))
                elif self.node_len == 25:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_25.pt'))
                elif self.node_len == 30:
                    torch.save(data, os.path.join(self.processed_dir, f'data_{i}_30.pt'))
                
        
    def get(self, i):
        if self.loop:
            if self.node_len == 15:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_loop_15.pt'))
            elif self.node_len == 20:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_loop_20.pt'))
            elif self.node_len == 25:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_loop_25.pt'))
            elif self.node_len == 30:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_loop_30.pt'))
        else:
            if self.node_len == 15:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_15.pt'))
            elif self.node_len == 20:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_20.pt'))
            elif self.node_len == 25:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_25.pt'))
            elif self.node_len == 30:
                data = torch.load(os.path.join(self.processed_dir, f'data_{i}_30.pt'))
        return data

    def len(self):
        return self.num_data

    def create_dataset(self,num_data=100, min_node_len=10, max_node_len=20, iteration=20, factor_class=3,loop=False):
        dataset = []
        for _ in tqdm(range(num_data)):
            mrf = None
            try:
                mrf = gen_graph(random.randint(min_node_len,max_node_len),loop=loop)
            except Exception as e:
                print(e)
                print("Exception occur skip current loop")
                continue
            node_type = []
            factor_list = []
            variable_list = []
            for v in  mrf.get_graph().vs:
                if v["is_factor"]:
                    nei = mrf.get_graph().vs[mrf.get_graph().neighbors(v['name'])]['name']
                    nei_size = len(nei)
                    factor_type = random.randint(0,1)
                    distribution = self.create_distribution(nei_size,factor_type)
                    f = factor(nei, distribution)
                    mrf.change_factor_distribution(v['name'], f)
                    node_type.append(factor_type+1)
                    factor_list.append(f)
                else:
                    node_type.append(0)
                    variable_list.append(v['name'])
            lbp = loopy_belief_propagation(mrf)
            exact_beliefs = []
            if loop is False:
                bp = belief_propagation(mrf)
                # print((lbp.belief('0', iteration).get_distribution()==bp.belief('0').get_distribution()).all())
                assert((lbp.belief('0', iteration).get_distribution()==bp.belief('0').get_distribution()).all())
            else:
                # lbp.belief('0', iteration)
                skip_loop = False
                for i, v in enumerate(variable_list):
                    # print(factor_list)
                    # print(variable_list[0:i]+variable_list[i+1:])
                    exact = factor_marginalization(joint_distribution(factor_list), variable_list[0:i]+variable_list[i+1:]).get_distribution()
                    if np.sum(exact) == 0:
                        exact = np.array([0.5,0.5])
                    else:
                        exact = exact / np.sum(exact)
                    #-----------
                    # if v == '0':
                    #     print('plot')
                    #     tol = []

                        
                    #     for j in range(15):
                    #         lbp = loopy_belief_propagation(mrf)
                    #         tol.append(np.linalg.norm(lbp.belief(v, j).get_distribution() - exact))
                    #     print(tol)
                    #     plt.figure(figsize=(8,6))
                    #     plt.semilogy(tol, label=r"$\|b_k - b^*\|_2$", color='navy')
                    #     plt.xlabel(r"Number of iteration, $k$", fontsize=15)
                    #     plt.ylabel(r"Convergence rate", fontsize=15)
                    #     plt.xticks(fontsize = 15)
                    #     plt.yticks(fontsize = 15)
                    #     plt.legend(loc="best", fontsize=15)
                    #     plt.plot(tol)
                    #     plt.show()
                    
                    # time.sleep(100)
                    #----------------

                    if np.isnan(exact).all():
                        print(exact)
                        skip_loop = True
                        break
                    # print('normal exact: ', exact)
                    lbp.belief('0', iteration)
                    exact_beliefs.append(exact)
                if skip_loop:
                    skip_loop = False
                    print("ignoring current data")
                    continue
            # print(exact_beliefs)
            msg, belief = lbp.get_msg_belief()
            data = self.create_data(mrf,msg,belief,np.array(node_type),factor_class,exact_beliefs,loop=loop)
            dataset.append(data)
        return dataset

    def create_distribution(self, nei_size,factor_type):
        assert(nei_size >= 1)

        base = np.array([0,1])
        result = base
        for i in range(nei_size-1):
            temp = result + 1
            # result = np.concatenate((result,temp),axis=-1)
            result = np.array([result,temp])

        if factor_type == 1: # negative factor type
            result = nei_size - result
        assert(result.shape == tuple(2 for _ in range(nei_size)))
        return result


    def create_data(self, pgm, msgs, beliefs, node_type,factor_class,exact_beliefs,loop=False):
        
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
        if loop is False:
            data = Data(F.one_hot(torch.tensor(node_type),factor_class),
                        edge_index=torch.tensor(np.transpose(edge_index)),
                        edge_attr=torch.tensor(edge_attr),
                        y=torch.tensor(y)) 
        else:
            # print(exact_beliefs)
            data = Data(F.one_hot(torch.tensor(node_type),factor_class),
                        edge_index=torch.tensor(np.transpose(edge_index)),
                        edge_attr=torch.tensor(edge_attr),
                        y=torch.from_numpy(np.stack(exact_beliefs)),
                        beliefs=torch.tensor(y))
        return data


if __name__ == "__main__":
    dataset = BPDataset(root='data/',num_data=1500,loop=True)
    print(len(dataset))
    print(dataset[1])
    print(dataset[999].y)
    # dataset = BPDataset(root='data/',num_data=3000,loop=False)
    # print(len(dataset))
    # print(dataset[0])
    