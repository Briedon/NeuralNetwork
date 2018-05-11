import networkx as nx
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


graph= nx.reverse(nx.read_gml('graph19.gml',label='id'),True)

n_nodes=int(graph.number_of_nodes())
nodes=list(graph.nodes)
gr=list(nx.topological_sort(graph))#topologic order
ind=[graph.in_degree(i) for i in nodes]
inputgraf=[]
inputlength=[]
maximumLine=0
for i in range(n_nodes):
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0,0])

        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))
            if maximumLine < graph.in_degree(nodes[i]):
                maximumLine=graph.in_degree(nodes[i])



leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
count= leafs.__len__()

graph=0

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
        self.igraf=torch.LongTensor(n_nodes,maximumLine+1).cuda()
        for i in range(n_nodes):
            for j in range(len(inputgraf[i])):

                #pridat tensor velikosti

                self.igraf[i][j]=inputgraf[i][j]
        for x in gr:
            #print(ind[x])
            if(ind[x]==0):
                self.neurons.append(nn.Linear(1,1).cuda())
            else:
                self.neurons.append(nn.Linear(ind[x],1).cuda())
        #print("nuda")



    def forward(self, x):
        a=0
        values = [0] * (n_nodes + 1)
        tst=time.time()
        h=0
        for neuron in self.neurons:
            #print(neuron)
            i=gr[h]
            #print(ind[i])
            if ind[i] == 0:
                #print(a)
                #print(inp)
                values[i]=F.relu(neuron(x[a]))
                a=a+1
            elif ind[i]>1:
                l = torch.cat([values[self.igraf[i][j]] for j in range(ind[i])],0)

                # print(inp)
                values[i] = F.sigmoid(neuron(l))
            else:
                l = values[self.igraf[i][0]]
                # print(inp)
                values[i] = F.sigmoid(neuron(l))
            h=h+1
        print(time.time()-tst)
        return values[n_nodes-1]

net = Net()
net.cuda()
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.FloatTensor([5])).cuda()
start = time.time()
print(torch.cuda.is_available())
for i in range(186):
    net.train()
    optimizer.zero_grad()
    out=net(Variable(torch.randn(count)).cuda())


print(time.time()-start)


