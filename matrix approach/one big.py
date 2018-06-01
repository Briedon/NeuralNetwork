import networkx as nx
import random
import numpy as np
import time


numberOfGraphs=50



graph={}
nodes=0
big_nodes={}
index=0
for number in range(0,numberOfGraphs):
    graph[number] = nx.read_gml('./graphs/graph' + str(number ) + '.gml')

    for node in list(graph[number].nodes()):
        separated = node.split("\n")[0]
        nodes=nodes+1
        if not separated in big_nodes:
            big_nodes[separated]=index
            index=index+1

model = nx.DiGraph()

for name in big_nodes:
    model.add_node(big_nodes[name])

for number in range(0,numberOfGraphs):
    nodes=graph[number].nodes()
    edges =list(graph[number].edges())
    print(list(graph[number].edges()).__len__())
    for edge in edges:
        separated0 = edge[0].split("\n")[0]
        separated1 = edge[1].split("\n")[0]
        model.add_edge(big_nodes[separated0],big_nodes[separated1])
graph.clear()
print("number of nodes after grouping")
print(list(model.nodes()).__len__())
model=nx.convert_node_labels_to_integers(model)
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
    matrices[i]= np.zeros((sizes[i+1],input_layer[i].__len__()))


for i in range(maximum):
    nodes=layer_nodes[i+1]
    next=-1
    for node in nodes:
        next=next+1
        for edge in model.in_edges(node):
            k=0
            for j in range(input_layer[i].__len__()):
                if(edge[0]!=input_layer[i][j]):
                    k=k+1
                else:
                    break
            matrices[i][next][k]=1
print("matrix")
print(sizes)
