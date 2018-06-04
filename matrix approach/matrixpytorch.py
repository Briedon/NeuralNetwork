import networkx as nx
import random
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

numberOfGraphs=2


for number in range(0,numberOfGraphs):
    model = nx.read_gml('./graphs/graph' + str(number) + '.gml', label='id')

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

    #tu zacina pytorch
    #linearnu model znazornujuci
    class MyLinear(torch.nn.Module):
        #weight je vaha, order je struktura grafu zapisana s jednotkami a bias je offset
        def __init__(self, weight,order,bias):
            """
            v konstruktore inicializujeme premenne, ktore neskor pouzijeme
            pouzivame self lebo budeme je pouyivat mimo funkcie ale stale v triede
            """
            #zdedime valstnosti predka
            super(MyLinear, self).__init__()
            #nastavime vahu
            self.weight=nn.Parameter(weight)
            #bias
            self.bias=nn.Parameter(bias)
            #ulozime ako vyzera vaha
            self.order=order

        def forward(self, x):
            """
            ve forward pouzijeme inicializovane moduly z __init__
            """
            #prenasobenia vah maticou znaciaacou strukturu grafu
            out = self.weight.mul(self.order)
            #hodenie do float hodnoty
            out=out.float()
            #vetkorve prenasobenie matice
            out = torch.mv(out,x)

            out=out+self.bias
            return out

    # list modulov, ktore su potrebne pre pytorch
    # pytorchov autograd neumi delat s polem modulov alebo premmennych
    class ListModule(object):
        #mal by fungovat s hocujakym modulom
        def __init__(self, module, prefix, *args):
            self.module = module
            self.prefix = prefix
            self.num_module = 0
            for new_module in args:
                self.append(new_module)
        #pridaj dalsi modul do listu rovnako ako v pythone
        def append(self, new_module):
            if not isinstance(new_module, nn.Module):
                raise ValueError('Not a Module')
            else:
                self.module.add_module(self.prefix + str(self.num_module), new_module)
                self.num_module += 1
        #funkcia na dlzku
        def __len__(self):
            return self.num_module
        #dostan modul z listu
        def __getitem__(self, i):
            if i < 0 or i >= self.num_module:
                raise IndexError('Out of bound')
            return getattr(self.module, self.prefix + str(i))

    #trida aktualneho modulu
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            #definujem si vrstvy ako list modulov pomocou ListModule
            self.layers=ListModule(self,'layers')
            #vstupna vrstva iba idnetitn=a matica
            self.input_layer=nn.Linear(sizes[0],sizes[0])
            for i in range(maximum):
                #pre kazdu jednu vrstvu si vytvorim
                weight=torch.from_numpy(matrices[i])
                order=Variable(torch.from_numpy(matrices[i],), requires_grad=False)
                bias=torch.rand(sizes[i+1])
                #vytovri instanciu novej vrstvy
                layer=MyLinear(weight,order,bias)
                self.layers.append(layer)

        def forward(self, x):
            #kod priebehu neuralnej siete
            #vystupi
            outputs=[0]*number_of_nodes
            #vystup nultej vrstvy
            out=F.relu(self.input_layer(x))
            #zapis nultej vrstvy do
            for j in range(out.__len__()):
                outputs[layer_nodes[0][j]]=out[j]
            for i in range(0,maximum):
                #vytvorenie vstupu do novej matice
                inp=[0]*input_layer[i].__len__()
                for j in range(inp.__len__()):
                    #priadnie vysledkov predchadzajucich vsrtiev do vstupu novej
                    inp[j]=outputs[input_layer[i][j]]
                inp=torch.cat((inp),0)
                #zaznamenanie vystupu
                out=F.relu(self.layers[i](inp))
                for j in range(out.__len__()):
                    outputs[layer_nodes[i+1][j]] = out[j]
            #posledny vystup
            return out


    net = Net()
    #deklariacia modelu
    #ak chcem pouzivat gpu potrebujeme
    #net.cuda()
    optimizer = optim.SGD(net.parameters(), lr=0.00001)
    optimizer.zero_grad()
    criterion = nn.MSELoss()

    #random vytvoreny output
    target = Variable(torch.rand(sizes[maximum]))

    for j in range(100):
        input = Variable(torch.ones(sizes[0]))
        net.train()
        #vynulovanie gradient zostupu
        optimizer.zero_grad()
        out = net(input)
        #vypocitanie chyby
        loss = criterion(out, target)
        loss.backward()
        #optimalizacia
        optimizer.step()
