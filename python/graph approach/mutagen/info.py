import networkx as nx
import os

edges=0
nodes=0
avgedges=float(0)
avgnodes=0
for a in range(187,188):
    graph=nx.DiGraph(nx.drawing.nx_agraph.read_dot(str(a) + 'graph.dot'))
    nx.write_gml(graph,'graph'+str(a)+'.gml')
    avgnodes=max(avgnodes,graph.number_of_nodes())
    print(a)
    print(graph.number_of_nodes())

print(avgnodes)
