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

for i in range(186):
    graph= nx.read_gml('graph'+str(i)+'.gml',label='id')

    n_nodes=int(graph.number_of_nodes())
    nodes=list(graph.nodes)
    gr.append(list(nx.topological_sort(graph)))#topologic order

    maximumLine=0
    for i in range(n_nodes):
        inputlength.append(graph.in_degree[i])
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))
#



    leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphLeaveCount.append(leafs.__len__())
    graphnumber.extend([n_nodes])
    print(time.time() - begin)

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
        values = [0] * (800 + 1)
        tst=time.time()
        h=step
        for h in range(0,skip):
            #print(neuron)
            i=step+ortology[h]
            neuron = self.neurons[i]
            #print(ind[i])
            if inputlength[i] == 0:
                values[ortology[h]]=F.sigmoid(neuron(x[a]))
                a=a+1
            elif inputlength[i]>1:

                l = torch.cat([values[j] for j in inputgraf[i]],0)
                #print(l)
                # print(inp)
                values[ortology[h]] = F.sigmoid(neuron(l))
            else:

                l = values[inputgraf[i][0]]
                #print(l)
                # print(inp)
                values[ortology[h] ] = F.sigmoid(neuron(l))
        print(time.time()-tst)
        return values[skip-1]

net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.FloatTensor([5])).cuda()
start = time.time()

target=Variable(torch.FloatTensor([0.6]))
for j in range(100):
    step=0
    for i in range(186):
        count=graphLeaveCount[i]
        ortology=gr[i]
        skip=graphnumber[i]
        input = Variable(torch.ones(count))
        net.train()
        optimizer.zero_grad()
        out=net(input)
        loss = criterion(out , target)
        loss.backward()
        optimizer.step()
        step = step + graphnumber[i]
        print(time.time() - start)





