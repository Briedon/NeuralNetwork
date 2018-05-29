import networkx as nx
import random
import numpy
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

for n in range(1,50):
    mindensity = 0.80
    maxdensity=0.95
    number_of_nodes = n*50
    juming_neurons = 300
    number_of_layers = 5



    model = nx.DiGraph()

    layers={}
    for i in range(number_of_layers):
        layers[i]=[]
    edge_layer={}
    size_layers=[0]*number_of_layers
    edges={} #input edges of each neuron
    node_layers=numpy.random.randint(number_of_layers,size=number_of_nodes)
    node_layers=numpy.sort(node_layers)
    edge_layer=node_layers
    for i in range(number_of_nodes):
        size_layers[node_layers[i]]=size_layers[node_layers[i]]+1
        layers[node_layers[i]].extend([i])

    edge=0
    used = [0] * number_of_nodes
    inputsize=size_layers[0];
    outputsize=size_layers[number_of_layers-1];
    for i in range(1,number_of_layers):

        for neuron in layers[i]:
            choices=numpy.array(numpy.random.choice(size_layers[i-1]-1,numpy.maximum(int((size_layers[i-1])*random.uniform(mindensity,maxdensity)),1),replace=False)).tolist()
            layer=layers[i-1]
            edge=edge+len(choices)
            for j in choices:
                used[layer[j]]=1
            edges[neuron]=[layer[j] for j in choices]
        for j in layers[i-1]:
            if(used[j]==0):
                add=random.choice(range(size_layers[i]))
                edges[layers[i][add]].extend([j])


    for i in range(0,juming_neurons):
        layer=random.choice(range(2,number_of_layers-1))
        from_layer=random.choice(range(0,layer-1))
        first_neuron=random.choice(layers[layer])
        second_layer=random.choice(layers[from_layer])
        edges[first_neuron].extend([second_layer])

    #count layers


    for i in range(number_of_nodes):
        model.add_node(i)
        if node_layers[i]==0:
            continue
        for node_edge in edges[i]:
            model.add_edge(i,node_edge)


    f=open("./generated input/genr"+str(n*50)+"nor.txt","w")

    for i in range(200):
        inp=numpy.random.randint(1,4,size=inputsize)
        out = numpy.random.randint(1,2, size=outputsize)
        f.write(numpy.array2string(inp,max_line_width=(2*inputsize+1)))
        f.write(" ")
        f.write(numpy.array2string(out,max_line_width=(2*outputsize+1)))
        f.write("\n")
    f.close()
    nx.write_gml(model, "./generated graphs/genr" + str(n * 50) + "nor.gml")


