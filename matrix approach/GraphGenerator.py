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
    #definovanie zakladnych parametrov novych grafov
    mindensity = 0.80
    maxdensity=0.95
    number_of_nodes = n*50
    juming_neurons = 300
    number_of_layers = 5


    #definovanie modelu pre networkx
    model = nx.DiGraph()
    #definovanie premennych pouyitych
    layers={}
    for i in range(number_of_layers):
        layers[i]=[]
    edge_layer={}
    size_layers=[0]*number_of_layers
    edges={} #vstupne hrany pre vsetky vrcholy
    #priradenie prislusnych vrcholov do vrstiev
    node_layers=numpy.random.randint(number_of_layers,size=number_of_nodes)
    #nasledne je zoradim aby som zoskupil vsechny vrcholy vo svojich vrstvach
    node_layers=numpy.sort(node_layers)
    edge_layer=node_layers
    # zoberem kazdy vrchol a pridadim ho do vrstvy a pripocitam do velikosti vrstvy
    for i in range(number_of_nodes):
        size_layers[node_layers[i]]=size_layers[node_layers[i]]+1
        layers[node_layers[i]].extend([i])
    #pripravim si premenne na generovanie hran
    edge=0
    #sluzi na zachytenie nepouyitych vrcholov
    used = [0] * number_of_nodes
    for i in range(1,number_of_layers):
        for neuron in layers[i]:
            #vyberem vrcholi z predchadzajucej vrstvy
            choices=numpy.array(numpy.random.choice(size_layers[i-1]-1,numpy.maximum(int((size_layers[i-1])*random.uniform(mindensity,maxdensity)),1),replace=False)).tolist()
            #nactu si nazvy vsech predchadyajucich vrcholov z vrstiev
            layer=layers[i-1]
            #pripocitam si pocet hran k celkovemu
            edge=edge+len(choices)
            # ulozim si ktore vrcholi som predtim pouzil
            for j in choices:
                used[layer[j]]=1
            #vytvorim si pole kde ulozim vsechny vrholy vstupujuce do tohto vrcholu
            edges[neuron]=[layer[j] for j in choices]
        for j in layers[i-1]:
            #kontorla ci som vyuzil vsechny vrholi, ak nie pridam nahodnemu vrcholu ako vstupnu hranu
            if(used[j]==0):
                add=random.choice(range(size_layers[i]))
                edges[layers[i][add]].extend([j])

    #pridanie "skakjucich neuronov"
    for i in range(0,juming_neurons):
        #zoberem nahodnu vrstvu od druhej po poslednu
        layer=random.choice(range(2,number_of_layers))
        from_layer=random.choice(range(0,layer-1))
        #prijimaci vrchol
        first_neuron=random.choice(layers[layer])
        #odchadzajuci vrchol
        second_layer=random.choice(layers[from_layer])
        #pridanie do hran
        edges[first_neuron].extend([second_layer])

    #count layers

    #pridanie vygenerovanych vrcholov do modelu
    for i in range(number_of_nodes):
        model.add_node(i)
        if node_layers[i]==0:
            continue
        for node_edge in edges[i]:
            model.add_edge(i,node_edge)


    f=open("./generated input/genr"+str(n*50)+"dens.txt","w")
    #vygenerovanie inputu
    for i in range(200):
        inp=numpy.random.randint(1,4,size=inputsize)
        out = numpy.random.randint(1,2, size=outputsize)
        #zmena inputsize aby bol napasny na jednom riakdu
        f.write(numpy.array2string(inp,max_line_width=(2*inputsize+1)))
        f.write(" ")
        f.write(numpy.array2string(out,max_line_width=(2*outputsize+1)))
        f.write("\n")
    f.close()
    nx.write_gml(model, "./generated graphs/genr" + str(n * 50) + "dens.gml")


