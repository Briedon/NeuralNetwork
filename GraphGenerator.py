import networkx as nx
import random
import numpy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

mindensity = 0.1
maxdensity=0.3
number_of_nodes = 600
juming_neurons = 0
number_of_layers = 5
number_of_leaves = 156
number_of_root =1


model = nx.DiGraph()

layers={}
edge_layer={}
size_layers={}
edges={} #input edges of each neuron
layers[0]=[i for i in range(0,number_of_leaves-1)]
for i in range(0,number_of_leaves-1):
    edge_layer[i]=0
    edges[i]=[]
size_layers[0]=number_of_leaves
next = number_of_leaves-1
for i in range(1,number_of_layers-1):
    number_nodes_layer=random.randint(1,numpy.maximum(int((2*((number_of_nodes-next)-(number_of_layers-i+1))/(number_of_layers-i))),1))
    size_layers[i]=number_nodes_layer

    layers[i]=[j for j in range(next,next+number_nodes_layer)]
    for j in range(next,next+number_nodes_layer):
        edge_layer[j]=i
    next=next+number_nodes_layer



layers[number_of_layers-1]=[i for i in range(next,number_of_nodes-1)]
for i in range(next,number_of_nodes-1):
    edge_layer[i]=number_of_layers-1
layers[number_of_layers]=[number_of_nodes-1]
edge_layer[number_of_nodes-1]=number_of_layers
size_layers[number_of_layers-1]=number_of_nodes-2-next

edge=0
for i in range(number_of_leaves-1,number_of_nodes):
    choices=numpy.array(numpy.random.choice(size_layers[edge_layer[i]-1]-1,numpy.maximum(int(size_layers[edge_layer[i]-1]*random.uniform(mindensity,maxdensity)),1),replace=False)).tolist()
    #print(edge_layer[i]-1)
    layer=layers[edge_layer[i]-1]
    edge=edge+len(choices)
    edges[i]=[layer[j] for j in choices]
edges[number_of_nodes]=layers[number_of_layers]


for i in range(0,juming_neurons):
    neuron=random.choice(range(number_of_leaves,number_of_nodes))
    smaller_layer=random.choice(range(1,edge_layer[neuron]+1))
    smaller_neuron=random.choice(layers[smaller_layer])
    edges[neuron].extend([smaller_neuron])

#count layers


for i in range(number_of_nodes):
    model.add_node(i)
    for node_edge in edges[i]:
        model.add_edge(i,node_edge)

print(edge)
nx.write_gml(model,"./generated0.gml")

#the testing of the classic approach of the pytorch

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

        for i in range(0,number_of_nodes+1):
            x=len(edges[i])
            if(x==0):
                self.neurons.append(nn.Linear(1,1))
            else:
                self.neurons.append(nn.Linear(x,1))



    def forward(self, x):
        a=0
        values = [0] * (number_of_nodes+1)
        tst=time.time()
        for i in range(0,number_of_nodes+1):
            neuron = self.neurons[i]
            length=len(edges[i])
            if length == 0:
                values[i]=F.sigmoid(neuron(x[a]))
                a=a+1
            elif length >1:
                l = torch.cat([values[j] for j in edges[i]])
                values[i] = F.sigmoid(neuron(l))
            else:
                l = values[edges[i][0]]
                values[i] = F.sigmoid(neuron(l))
        return values[number_of_nodes]

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.FloatTensor([0.6]))
start=time.time()
for j in range(1):
    input = Variable(torch.ones(number_of_leaves))
    out=net(input)
    optimizer.zero_grad()
    net.train()
    loss = criterion(out , target)
    loss.backward()
    optimizer.step()
print(time.time() - start)


class LinNet(nn.Module):

    def __init__(self):
        super(LinNet, self).__init__()
        idx=0
        self.neuron = nn.Linear(number_of_leaves,size_layers[1])
        self.neuron1 =nn.Linear(size_layers[1],size_layers[2])
        self.neuron2 = nn.Linear(size_layers[2], size_layers[3])
        self.neuron3 = nn.Linear(size_layers[3], size_layers[4])
        self.neuron4 = nn.Linear(size_layers[4], 1)




    def forward(self, x):

        tst=time.time()
        value=F.sigmoid(self.neuron(x))
        value = F.sigmoid(self.neuron1(value))
        value = F.sigmoid(self.neuron2(value))
        value = F.sigmoid(self.neuron3(value))
        value = self.neuron4(value)
        return value



net = LinNet()
optimizer = optim.SGD(net.parameters(), lr=0.1)
optimizer.zero_grad()
criterion = nn.MSELoss()
target=Variable(torch.FloatTensor([0.6]))
start=time.time()
for j in range(186):
    input = Variable(torch.ones(number_of_leaves))
    out=net(input)
    optimizer.zero_grad()
    net.train()
    loss = criterion(out , target)
    loss.backward()
    optimizer.step()

print(time.time() - start)