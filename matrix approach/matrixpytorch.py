import networkx as nx
import random
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

numberOfGraphs=50


for number in range(1,numberOfGraphs):
    model = nx.read_gml('./generated graphs/genr' + str(number * 50) + 'dense.gml', label='id')

    number_of_nodes = int(model.number_of_nodes())
    length_of_inputs = [0] * number_of_nodes
    list_of_predecessors = [-1] * number_of_nodes
    layer = [0] * number_of_nodes
    nodes = list(model.nodes())
    for i in range(number_of_nodes):
        length_of_inputs[i] = model.in_degree(nodes[i])
        list_of_predecessors[i] = (list(model.predecessors(nodes[i])))

    topological = list(nx.topological_sort(model))

    layer_number = 0
    layer_list = []
    for i in topological:
        contains = [layer[j] for j in list_of_predecessors[i]]
        if (contains.__len__() == 0):
            layer[i] = 0
            continue
        the_biggest_predecessor = max(contains)
        layer[i] = the_biggest_predecessor + 1

    input_layer = {}
    layer_nodes = {}
    matrices = {}
    maximum = max(layer)
    sizes = [0] * (maximum + 1)
    for i in range(maximum + 1):
        input_layer[i] = []
        layer_nodes[i] = [j for j in topological if layer[j] == i]
        sizes[i] = layer_nodes[i].__len__()

    c = 0
    for edge in model.edges():
        a = layer[edge[0]] - 1
        b = layer[edge[1]] - 1
        if (a < b):
            a = b
            c = edge[0]
        else:
            c = edge[1]
        not_found = True
        for i in input_layer[a]:
            if (i == c):
                not_found = False
                break
        if not_found:
            input_layer[a].extend([c])

    for i in range(maximum):
        matrices[i] = np.zeros((sizes[i + 1], input_layer[i].__len__()))

    for i in range(maximum):
        nodes = layer_nodes[i + 1]
        next = -1
        for node in nodes:
            next = next + 1
            for edge in model.in_edges(node):
                k = 0
                for j in range(input_layer[i].__len__()):
                    if (edge[0] != input_layer[i][j]):
                        k = k + 1
                    else:
                        break
                matrices[i][next][k] = 1


    class MyLinear(torch.nn.Module):
        def __init__(self, weight,order,bias):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(MyLinear, self).__init__()
            self.weight=nn.Parameter(weight)
            self.bias=nn.Parameter(bias)
            self.order=order

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            out = self.weight.mul(self.order)
            out=out.float()
            out = torch.mv(out,x)

            out=out+self.bias
            return out

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
            self.layers=ListModule(self,'layers')
            self.input_layer=nn.Linear(sizes[0],sizes[0])
            for i in range(maximum):
                weight=torch.from_numpy(matrices[i])
                order=Variable(torch.from_numpy(matrices[i],))
                bias=torch.rand(sizes[i+1])
                layer=MyLinear(weight,order,bias)
                self.layers.append(layer)

        def forward(self, x):
            outputs=[0]*number_of_nodes
            out=F.relu(self.input_layer(x))
            for j in range(out.__len__()):
                outputs[layer_nodes[0][j]]=out[j]
            for i in range(0,maximum):
                inp=[0]*input_layer[i].__len__()
                for j in range(inp.__len__()):
                    inp[j]=outputs[input_layer[i][j]]
                inp=torch.cat((inp),0)
                out=F.relu(self.layers[i](inp))
                for j in range(out.__len__()):
                    outputs[layer_nodes[i+1][j]] = out[j]
            return out


    net = Net()
    net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    start = time.time()

    step = 0
    i = 0
    target = Variable(torch.rand(sizes[maximum]))
    start = time.time()
    for j in range(100):
        input = Variable(torch.ones(sizes[0]))
        net.train()
        optimizer.zero_grad()
        out = net(input)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    print(time.time() - start)
