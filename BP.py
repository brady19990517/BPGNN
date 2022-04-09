import numpy as np
from factor import *
from factor_graph import *



class belief_propagation():
    def __init__(self, pgm):
        if type(pgm) is not factor_graph:
            raise Exception('PGM is not a factor graph')
        if not (pgm.is_connected() and not pgm.is_loop()):
            raise Exception('PGM is not a tree')
        
        self.__msg = {}
        self.__pgm = pgm
    
    def belief(self, v_name):
        incoming_messages = []
        for f_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(v_name)]['name']:
            incoming_messages.append(self.get_factor2variable_msg(f_name_neighbor, v_name))
        return self.__normalize_msg(joint_distribution(incoming_messages))
    
    # ----------------------- Variable to factor ------------
    def get_variable2factor_msg(self, v_name, f_name):
        key = (v_name, f_name)
        if key not in self.__msg:
            self.__msg[key] = self.__compute_variable2factor_msg(v_name, f_name)
        return self.__msg[key]
    
    def __compute_variable2factor_msg(self, v_name, f_name):
        incoming_messages = []
        for f_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(v_name)]['name']:
            if f_name_neighbor != f_name:
                incoming_messages.append(self.get_factor2variable_msg(f_name_neighbor, v_name))
        
        if not incoming_messages:
            # if the variable does not have its own distribution
            return factor([v_name], np.array([1.]*self.__pgm.get_graph().vs.find(name=v_name)['rank']))
        else:
            # Since all messages have the same dimension (1, order of v_name) the expression after
            # ```return``` is equivalent to ```factor(v_name, np.prod(incoming_messages))```
            return self.__normalize_msg(joint_distribution(incoming_messages))
    
    # ----------------------- Factor to variable ------------
    def get_factor2variable_msg(self, f_name, v_name):
        key = (f_name, v_name)
        if key not in self.__msg:
            self.__msg[key] = self.__compute_factor2variable_msg(f_name, v_name)
        return self.__msg[key]
    
    def __compute_factor2variable_msg(self, f_name, v_name):
        incoming_messages = [self.__pgm.get_graph().vs.find(f_name)['factor_']]
        marginalization_variables = []
        for v_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(f_name)]['name']:
            if v_name_neighbor != v_name:
                incoming_messages.append(self.get_variable2factor_msg(v_name_neighbor, f_name))
                marginalization_variables.append(v_name_neighbor)
        return self.__normalize_msg(factor_marginalization(
            joint_distribution(incoming_messages),
            marginalization_variables
        ))
    
    # ----------------------- Other -------------------------
    def __normalize_msg(self, message):
        return factor(message.get_variables(), message.get_distribution()/np.sum(message.get_distribution()))

class loopy_belief_propagation():
    def __init__(self, pgm):
        if type(pgm) is not factor_graph:
            raise Exception('PGM is not a factor graph')
        if not pgm.is_connected():
            raise Exception('PGM is not connected')
        if len(pgm.get_graph().es) - 1 == len(pgm.get_graph().vs):
            raise Exception('PGM is a tree')
        
        self.__t       = 0
        self.__msg     = {}
        self.__msg_new = {}
        self.__pgm     = pgm
        self.ret_belief  = []
        self.ret_msg = []
        
        # Initialization of messages
        for edge in self.__pgm.get_graph().es:
            start_index, end_index = edge.tuple[0], edge.tuple[1]
            start_name, end_name = self.__pgm.get_graph().vs[start_index]['name'], self.__pgm.get_graph().vs[end_index]['name']
            
            if self.__pgm.get_graph().vs[start_index]['is_factor']:
                self.__msg[(start_name, end_name)] = factor([end_name],   np.array([1.]*self.__pgm.get_graph().vs[end_index]['rank']))
            else:
                self.__msg[(start_name, end_name)] = factor([start_name], np.array([1.]*self.__pgm.get_graph().vs[start_index]['rank']))
            self.__msg[(end_name, start_name)] = self.__msg[(start_name, end_name)]
            
            self.__msg_new[(start_name, end_name)] = 0
            self.__msg_new[(end_name, start_name)] = 0
    
    def belief(self, v_name, num_iter):
        if self.__t > num_iter:
            raise Exception('Invalid number of iterations. Current number: ' + str(self.__t))
        elif self.__t < num_iter:
            self.__loop(num_iter)
        
        incoming_messages = []
        for f_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(v_name)]['name']:
            
            incoming_messages.append(self.get_factor2variable_msg(f_name_neighbor, v_name))
            # print("Message from: ",f_name_neighbor,"to",v_name,self.get_factor2variable_msg(f_name_neighbor, v_name).get_distribution())
        return self.__normalize_msg(joint_distribution(incoming_messages))
    
    # ----------------------- Variable to factor ------------
    def get_variable2factor_msg(self, v_name, f_name):
        return self.__msg[(v_name, f_name)]
    
    def __compute_variable2factor_msg(self, v_name, f_name):
        incoming_messages = []
        for f_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(v_name)]['name']:
            if f_name_neighbor != f_name:
                incoming_messages.append(self.get_factor2variable_msg(f_name_neighbor, v_name))
        
        if not incoming_messages:
            return factor([v_name], np.array([1]*self.__pgm.get_graph().vs.find(name=v_name)['rank']))
        else:
            return self.__normalize_msg(joint_distribution(incoming_messages))
    
    # ----------------------- Factor to variable ------------
    def get_factor2variable_msg(self, f_name, v_name):
        return self.__msg[(f_name, v_name)]
    
    def __compute_factor2variable_msg(self, f_name, v_name):
        incoming_messages = [self.__pgm.get_graph().vs.find(f_name)['factor_']]
        marginalization_variables = []
        for v_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(f_name)]['name']:
            if v_name_neighbor != v_name:
                incoming_messages.append(self.get_variable2factor_msg(v_name_neighbor, f_name))
                marginalization_variables.append(v_name_neighbor)
        return self.__normalize_msg(factor_marginalization(
            joint_distribution(incoming_messages),
            marginalization_variables
        ))
    
    # ----------------------- Other -------------------------
    def __loop(self, num_iter):
        # Message updating
        while self.__t < num_iter:

            #Collect current message
            # print(self.__t)
            temp = {}
            for k,v in self.__msg.items():
                temp[k] = v.get_distribution()
                # print(k,v.get_distribution())
            self.ret_msg.append(temp)

            temp = {}
            #Calculate current belief
            for v in self.__pgm.get_graph().vs:
                if not v["is_factor"]:
                    incoming_messages = []
                    for f_name_neighbor in self.__pgm.get_graph().vs[self.__pgm.get_graph().neighbors(v['name'])]['name']:
                        incoming_messages.append(self.get_factor2variable_msg(f_name_neighbor, v['name']))
                        # print("Message from: ",f_name_neighbor,"to",v_name,self.get_factor2variable_msg(f_name_neighbor, v_name).get_distribution())
                    temp[v['name']] = self.__normalize_msg(joint_distribution(incoming_messages)).get_distribution()
                    # print(v['name'],self.__normalize_msg(joint_distribution(incoming_messages)).get_distribution())
            self.ret_belief.append(temp)
            #Update message
            for edge in self.__pgm.get_graph().es:
                start_index, end_index = edge.tuple[0], edge.tuple[1]
                start_name, end_name   = self.__pgm.get_graph().vs[start_index]['name'], self.__pgm.get_graph().vs[end_index]['name']

                if self.__pgm.get_graph().vs[start_index]['is_factor']:
                    self.__msg_new[(start_name, end_name)] = self.__compute_factor2variable_msg(start_name, end_name)
                    self.__msg_new[(end_name, start_name)] = self.__compute_variable2factor_msg(end_name, start_name)
                else:
                    self.__msg_new[(start_name, end_name)] = self.__compute_variable2factor_msg(start_name, end_name)
                    self.__msg_new[(end_name, start_name)] = self.__compute_factor2variable_msg(end_name, start_name)
            self.__msg.update(self.__msg_new)
            self.__t += 1
            
            
            
    
    def __normalize_msg(self, message):
        return factor(message.get_variables(), message.get_distribution()/np.sum(message.get_distribution()))

    def get_msg_belief(self):
        return self.ret_msg, self.ret_belief
