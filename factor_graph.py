import igraph as ig
import pyvis.network as net
from factor import *
import networkx as nx
import random
import time

class factor_graph:
    def __init__(self):
        self._graph = ig.Graph()
    
    # ----------------------- Factor node functions ---------
    def add_factor_node(self, f_name, factor_):
        if (self.get_node_status(f_name) != False) or (f_name in factor_.get_variables()):
            raise Exception('Invalid factor name')
        if type(factor_) is not factor:
            raise Exception('Invalid factor_')
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == 'factor':
                raise Exception('Invalid factor')
        
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 1)
        # Create variables
        for v_name in factor_.get_variables():
            if self.get_node_status(v_name) == False:
                self.__create_variable_node(v_name)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Add node and corresponding edges
        self.__create_factor_node(f_name, factor_)

    def change_factor_distribution(self, f_name, factor_):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        if set(factor_.get_variables()) != set(self._graph.vs[self._graph.neighbors(f_name)]['name']):
            raise Exception('invalid factor distribution')
        
        # Check ranks
        self.__check_variable_ranks(f_name, factor_, 0)
        # Set ranks
        self.__set_variable_ranks(f_name, factor_)
        # Set data
        self._graph.vs.find(name=f_name)['factor_'] = factor_

    def remove_factor(self, f_name, remove_zero_degree=False):
        if self.get_node_status(f_name) != 'factor':
            raise Exception('Invalid variable name')
        
        neighbors = self._graph.neighbors(f_name, mode="out")
        self._graph.delete_vertices(f_name)
        
        if remove_zero_degree:
            for v_name in neighbors:
                if self._graph.vs.find(v_name).degree() == 0:
                    self.remove_variable(v_name)

    def __create_factor_node(self, f_name, factor_):
        # Create node
        self._graph.add_vertex(f_name)
        self._graph.vs.find(name=f_name)['is_factor'] = True
        self._graph.vs.find(name=f_name)['factor_']   = factor_
        
        # Create corresponding edges
        start = self._graph.vs.find(name=f_name).index
        edge_list = [tuple([start, self._graph.vs.find(name=i).index]) for i in factor_.get_variables()]
        self._graph.add_edges(edge_list)
    
    # ----------------------- Rank functions -------
    def __check_variable_ranks(self, f_name, factor_, allowded_v_degree):
        for counter, v_name in enumerate(factor_.get_variables()):
            if (self.get_node_status(v_name) == 'variable') and (not factor_.is_none()):
                if     (self._graph.vs.find(name=v_name)['rank'] != factor_.get_shape()[counter]) \
                and (self._graph.vs.find(name=v_name)['rank'] != None) \
                and (self._graph.vs.find(v_name).degree() > allowded_v_degree):
                    raise Exception('Invalid shape of factor_')

    def __set_variable_ranks(self, f_name, factor_):
        for counter, v_name in enumerate(factor_.get_variables()):
            if factor_.is_none():
                self._graph.vs.find(name=v_name)['rank'] = None
            else:
                self._graph.vs.find(name=v_name)['rank'] = factor_.get_shape()[counter]
        
    # ----------------------- Variable node functions -------
    def add_variable_node(self, v_name):
        if self.get_node_status(v_name) != False:
            raise Exception('Node already exists')
        self.__create_variable_node(v_name)

    def remove_variable(self, v_name):
        if self.get_node_status(v_name) != 'variable':
            raise Exception('Invalid variable name')
        if self._graph.vs.find(v_name).degree() != 0:
            raise Exception('Can not delete variables with degree >0')
        self._graph.delete_vertices(self._graph.vs.find(v_name).index)  

    def __create_variable_node(self, v_name, rank=None):
        self._graph.add_vertex(v_name)
        self._graph.vs.find(name=v_name)['is_factor'] = False
        self._graph.vs.find(name=v_name)['rank'] = rank

    # ----------------------- Info --------------------------
    def get_node_status(self, name):
        if len(self._graph.vs) == 0:
            return False
        elif len(self._graph.vs.select(name_eq=name)) == 0:
            return False
        else:
            if self._graph.vs.find(name=name)['is_factor'] == True:
                return 'factor'
            else:
                return 'variable'
    
    # ----------------------- Graph structure ---------------
    def get_graph(self):
        return self._graph

    def is_connected(self):
        return self._graph.is_connected()

    def is_loop(self):
        return any(self._graph.is_loop())


##################################

def string2factor_graph(str_):
    res_factor_graph = factor_graph()
    
    str_ = [i.split('(') for i in str_.split(')') if i != '']
    for i in range(len(str_)):
        str_[i][1] = str_[i][1].split(',')
        
    for i in str_:
        res_factor_graph.add_factor_node(i[0], factor(i[1]))
    
    return res_factor_graph

def plot_factor_graph(x):
    graph = net.Network(notebook=True, width="100%")
    graph.toggle_physics(False)
    
    # Vertices
    label = x.get_graph().vs['name']
    color = ['#2E2E2E' if i is True else '#F2F2F2' for i in x.get_graph().vs['is_factor']]
    graph.add_nodes(range(len(x.get_graph().vs)), label=label, color=color)
    
    # Edges
    graph.add_edges(x.get_graph().get_edgelist())

    return graph.show("graph.html")

def gen_graph(node_len=10, loop=False):
    sequence = [random.randint(0,node_len-2+1) for i in range(node_len-2)]
    tree = nx.from_prufer_sequence(sequence)
    traverse_edge = list(nx.edge_bfs(tree))
    depth = nx.single_source_shortest_path_length(tree,0)
    d = {}
    for key, value in depth.items():
        d.setdefault(value, []).append(key)
    var, fac= [], []
    for key in d.keys():
        if key % 2 == 0:
            var+=d[key]
        else:
            fac+=d[key]

    # print("Before: ", nx.find_cycle(tree, orientation="original"))
    if loop:
        f = random.choice(fac)
        f_depth = depth[f]
        connect_f_depth = f_depth + 3 if f_depth + 3 in d else f_depth - 3
        assert(connect_f_depth in d)
        v = random.choice(d[connect_f_depth])
        tree.add_edge(f,v)

    # print("After: ", nx.find_cycle(tree, orientation="original"))
    res_factor_graph = factor_graph()
    for f in fac:
        res_factor_graph.add_factor_node(str(f), factor([str(n) for n in list(tree.neighbors(f))]))
    return res_factor_graph
