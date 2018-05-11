import networkx as nx
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


inputgraf = []
inputlength = []
graphnumber=[]
graphLeaveCount=[]
gr=[]
begin=time.time()
numberOfGraphs=1
for j in range(numberOfGraphs):
    graph= nx.read_gml('graph'+str(j)+'.gml',label='id')
    n_nodes = int(graph.number_of_nodes())
    print(list(graph.edges).__len__())
    nodes=list(graph.nodes())
    gr.extend(list(nx.topological_sort(graph)))#topologic order

    maximumLine=0
    for i in range(n_nodes):
        inputlength.append(graph.in_degree(i))
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))
    leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphLeaveCount.append(leafs.__len__())
    graphnumber.extend([n_nodes])
    #print(time.time() - begin)

print(time.time() - begin)

class ListModule(object):
    #Should work with all kind of module
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

values = [0] * (800 + 1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        idx=0
        self.neurons = ListModule(self, 'Neurons_')

        for x in inputlength:
            #print(ind[x])
            if(x==0):
                self.neurons.append(nn.Linear(1,1))
            else:
                self.neurons.append(nn.Linear(x,1))
        #print("nuda")



    def forward(self, x):
        a=0

        tst=time.time()
        h=step
        for h in range(0,skip):
            #print(neuron)
            i=step+gr[step+h]
            value_indicator=gr[step+h]
            neuron = self.neurons[i]
            if inputlength[i] == 0:
                values[value_indicator]=F.sigmoid(Variable(torch.ones([1]))*(x[a])+Variable(torch.ones([1])))
                a=a+1
            elif inputlength[i]>1:

                l = torch.cat([values[j] for j in inputgraf[i]],0)
                #print(l)
                # print(inp)
                values[value_indicator] = F.sigmoid(neuron(l))
            else:
                l = values[inputgraf[i][0]]
                #print(l)
                # print(inp)
                values[value_indicator] = F.sigmoid(Variable(torch.ones([1])) * l + Variable(torch.ones([1])))
        return values[skip-1]

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.FloatTensor([5]))


target=Variable(torch.FloatTensor([0.6]))
start = time.time()
for j in range(186):
    step=0
    for i in range(numberOfGraphs):

        count=graphLeaveCount[i]
        skip=graphnumber[i]
        input = Variable(torch.ones(count))
        net.train()
        optimizer.zero_grad()
        out=net(input)
        loss = criterion(out , target)
        #loss.backward()
        #optimizer.step()
        step = step + graphnumber[i]

print(time.time() - start)



