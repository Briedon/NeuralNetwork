import networkx as nx
import random
import numpy as np
import time


numberOfGraphs=50


for number in range(1,numberOfGraphs):
    model = nx.read_gml('./generated graphs/genr' + str(number * 50) + 'dens.gml', label='id')

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