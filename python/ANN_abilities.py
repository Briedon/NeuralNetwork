import networkx as nx
import random
import numpy as np
import time

begin=time.time()
model = nx.read_gml('./graphs/graph150.gml',label='id')
print(time.time()-begin)


leafs=[]
number_of_nodes=int(model.number_of_nodes())
number_of_edges=int(model.number_of_edges())
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
        continue
    the_biggest_predecessor=max(contains)
    layer[i]=the_biggest_predecessor+1

print("number of nodes")
print(number_of_nodes)
print("Number_of_layers")
print(max(layer))
print(layer)
print("size of layers")
layer_size=[0]*(max(layer)+1)
for i in range(nodes.__len__()):
    layer_size[layer[i]]=layer_size[layer[i]]+1

print(layer_size)
all_edges=0
for i in range(0,max(layer)-1):
    lower=[1 for j in range(number_of_nodes) if layer[j]==i].__len__()
    upper = [1 for j in range(number_of_nodes) if layer[j] == (i+1)].__len__()
    all_edges=all_edges+lower*upper

print("Density of the graph")
print(number_of_edges/all_edges)

