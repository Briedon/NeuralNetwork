import networkx as nx
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


numberOfGraphs=49
i=0
for number_of_graph in range(1,numberOfGraphs):
    inputgraf = []
    inputlength = []
    outputlength = []
    outputsize = 0;
    outputsizes = []
    graphnumber = []
    graphLeaveCount = []
    gr = []
    begin = time.time()
    j=0
    outputsize=0
    graph= nx.read_gml('./generated graphs/genr'+str(number_of_graph*50)+'nor.gml',label='id')
    n_nodes = int(graph.number_of_nodes())

    nodes=list(graph.nodes())
    gr.extend(list(nx.topological_sort(graph)))#topologic order

    maximumLine=0
    for i in range(n_nodes):
        inputlength.append(graph.in_degree(i))
        outputlength.append(graph.out_degree(i))
        if(graph.out_degree(i)==0):
            outputsize=outputsize+1
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))
    leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphLeaveCount.append(leafs.__len__())
    graphnumber.extend([n_nodes])
    outputsizes.append([outputsize])


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

    values = [0] * (nodes.__sizeof__())
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
            return_value=[]
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
                if outputlength[i]==0:
                    return_value.extend([values[value_indicator]])

            return torch.cat(return_value)

    net = Net()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad()
    criterion = nn.MSELoss()



    start = time.time()

    step=0
    i=0
    target = Variable(torch.rand(outputsizes[i]))
    start = time.time()
    for j in range(100):
        count=graphLeaveCount[i]
        skip=graphnumber[i]
        input = Variable(torch.ones(count))
        net.train()
        optimizer.zero_grad()
        out=net(input)
        loss = criterion(out , target)
        loss.backward()
        optimizer.step()
    print(time.time() - start)




