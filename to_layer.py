import networkx as nx
import random
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

begin=time.time()
model = nx.read_gml('graph80.gml',label='id')
print(time.time()-begin)


leafs=[]

number_of_nodes=int(model.number_of_nodes())
length_of_inputs=[0]*number_of_nodes
list_of_predecessors=[-1]*number_of_nodes
layer=[0]*number_of_nodes
nodes=list(model.nodes())
for i in range(number_of_nodes):
    length_of_inputs[i]=model.in_degree(nodes[i])
    list_of_predecessors[i]=(list(model.predecessors(nodes[i])))

topological_order=nx.topological_sort(model)
layer_number=0
layer_list=[]
for i in topological_order:
    contains=[layer[j] for j in list_of_predecessors[i]]
    if(contains.__len__()==0):
        layer[i]=0
        print(i)
        continue
    the_biggest_predecessor=max(contains)
    layer[i]=the_biggest_predecessor+1


size0=[i for i in range(number_of_nodes) if layer[i]==0].__len__()
size1=[i for i in range(number_of_nodes) if layer[i]==1].__len__()
size2=[i for i in range(number_of_nodes) if layer[i]==2].__len__()
size3=[i for i in range(number_of_nodes) if layer[i]==3].__len__()

print(min(layer))
print(max(layer))



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.neuron = nn.Linear(size0,size1).cuda()
        self.neuron1 =nn.Linear(size1,size2).cuda()
        self.neuron2 = nn.Linear(size2, size3).cuda()


    def forward(self, x):

        tst=time.time()
        value=F.sigmoid(self.neuron(x))
        value = F.sigmoid(self.neuron1(value))
        value = F.sigmoid(self.neuron2(value))
        return value



net = Net()
net.cuda();
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.ones(size3)).cuda()


start=time.time()
for j in range(2000):
    input = Variable(torch.ones(size0)).cuda()
    out=net(input)
    optimizer.zero_grad()
    net.train()
    loss = criterion(out , target)
    loss.backward()
    optimizer.step()


print(time.time() - start)
