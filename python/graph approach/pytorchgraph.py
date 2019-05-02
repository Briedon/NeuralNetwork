import networkx as nx
import torch
import random
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


numberOfGraphs=2
i=0
for number_of_graph in range(0,numberOfGraphs):
    #definovanie a inicializacia premennych pozdeji pouzitych
    inputgraf = []
    inputlength = []
    outputlength = []
    outputsize = 0;
    outputsizes = []
    graphnumber = []
    graphInput = []
    gr = []
    begin = time.time()
    j=0
    outputsize=0
    #nacitanie grafu
    graph= nx.read_gml('C:/Users/mbriedon/Documents/GitHub/NeuralNetwork/python/graph approach/generated input/genr'+str((number_of_graph+1)*50)+'nor.gml',label='id')
    n_nodes = int(graph.number_of_nodes())

    nodes=list(graph.nodes())
    gr.extend(list(nx.topological_sort(graph)))#topologic order

    maximumLine=0
    for i in range(n_nodes):
        #vygenerovanie vstupov a vystupov pre kazdy neuron
        inputlength.append(graph.in_degree(i))
        outputlength.append(graph.out_degree(i))
        if(graph.out_degree(i)==0):
            outputsize=outputsize+1
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))
    graphInput.append([x for x in graph.nodes() if graph.in_degree(x) == 0].__len__())
    graphnumber.extend([n_nodes])
    outputsizes.append([outputsize])
    #je to podobne ako pre tensorflow

    class ListModule(object):
        #pouzivam ho ako nahradu pole
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
            #inicializovanie listu neuronov
            self.neurons = ListModule(self, 'Neurons_')

            for x in inputlength:
                #priradenie neuronov podla velkosti vstupu
                if(x==0):
                    self.neurons.append(nn.Linear(1,1))
                else:
                    self.neurons.append(nn.Linear(x,1))



        def forward(self, x):
            a=0

            return_value=[]
            for h in range(0,number_of_nodes):
                #print(neuron)
                i=gr[h]
                value_indicator=gr[h]
                neuron = self.neurons[i]
                #pridavanie hodnot podla velkosti vstupu
                if inputlength[i] == 0:
                    #neurona na vstupu
                    values[value_indicator]=F.sigmoid(Variable(torch.ones([1]))*(x[a])+Variable(torch.ones([1])))
                    a=a+1
                elif inputlength[i]>1:
                    #neuron s viacerymi vstupmi
                    l = torch.cat([values[j] for j in inputgraf[i]],0)
                    values[value_indicator] = F.sigmoid(neuron(l))
                else:
                    # pre neurony len s jednym vstupom
                    l = values[inputgraf[i][0]]
                    values[value_indicator] = F.sigmoid(Variable(torch.ones([1])) * l + Variable(torch.ones([1])))
                #ak nemaju vystup tak ich pridam do vysledkoveho vektoru
                if outputlength[i]==0:
                    return_value.extend([values[value_indicator]])
            #vratenie vysledkoveho vektoru
            return torch.cat(return_value)
    #definovanie neuronovej siete
    net = Net()
    # deklaracia optimizatora
    optimizer = optim.SGD(net.parameters(), lr=0.1)
    #vynulovanie optimizatora
    optimizer.zero_grad()
    #nastavenie kriteria ktore je priemerna hodnota druhej mocniny
    criterion = nn.MSELoss()
    outputsizes.append(outputsize)

    #vygenerovnaie nahodneho vystupu na ucenie
    target = Variable(torch.rand(outputsizes.__len__()))
    start = time.time()
    for j in range(100):
        #
        count=graphInput[number_of_graph]
        number_of_nodes=graphnumber[number_of_graph]
        #vytvorenie vstupu
        input = Variable(torch.ones(count))
        #vynulovanie gradinetov
        optimizer.zero_grad()
        #dostanie vysledku zo vstupu
        out=net(input)
        #chyba pri uceni
        loss = criterion(out , target)
        #rozprestriet chybu
        loss.backward()
        #ucenie sa
        optimizer.step()
        net.train()