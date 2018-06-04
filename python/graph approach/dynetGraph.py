import networkx as nx
import dynet as dy
import random
import numpy as np
import time


b=20

inputgraf = []
inputlength = []
graphnumber=[]
graphLeaveCount=[]
gr=[]
begin=time.time()
numberOfGraphs=20
neuronWeights=[]
a=0
for j in range(0,numberOfGraphs):
    graph= nx.read_gml('./generated graphs/genr'+str(j*49)+'dens.gml',label='id')
    number_of_nodes =list(graph.nodes()).__len__()
    n_nodes=int(graph.number_of_nodes())
    nodes=list(graph.nodes())
    gr.extend(list(nx.topological_sort(graph)))#topologic order
    gr.extend(list(nx.topological_sort(graph)))#topologic order
    maximumLine=0
    for i in range(n_nodes):
        inputlength.append(graph.in_degree(i))
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))


    leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphLeaveCount.append(leafs.__len__())
    graphnumber.extend([n_nodes])
    #print(time.time() - begin)


    a=a+1


    m = dy.Model()
    input=dy.vecInput(graphLeaveCount[0])
    trainer = dy.SimpleSGDTrainer(m)
    values=[0]*number_of_nodes
    weight=[0]*number_of_nodes
    bias=[0]*number_of_nodes
    next=0
    a=0

    for i in range(n_nodes):
        next_neuron=gr[i]
        if inputlength[next_neuron] == 0:
            weight[next_neuron]= m.add_parameters(1)
            bias[next_neuron]= m.add_parameters(1)
            values[next_neuron] = dy.rectify( (weight[next_neuron]*input[a])+bias[next_neuron])
            a = a + 1
        elif inputlength[next_neuron] > 1:
            weight[next_neuron]= m.add_parameters((inputlength[next_neuron]))
            bias[next_neuron]= m.add_parameters(1)
            l = dy.concatenate([values[j] for j in inputgraf[next_neuron]])

            values[next_neuron] = dy.rectify(dy.dot_product(weight[next_neuron],l) + bias[next_neuron])

        else:
            l = values[inputgraf[next_neuron][0]]
            # print(l)
            weight[next_neuron] = m.add_parameters(1)
            bias[next_neuron] = m.add_parameters(1)
            values[next_neuron] = dy.rectify(weight[next_neuron] * l + bias[next_neuron])
    y_pred=values[gr[n_nodes-1]]
    y=dy.scalarInput(0)
    loss = dy.squared_distance(y_pred, y)
    begin=time.time()
    for iter in range(100):
        mloss = 0.0
        input.set([random.uniform(0,2)])
        y.set(15)
        mloss += loss.scalar_value()
        loss.backward()
        trainer.update()
        #print("loss: %0.9f" % mloss)

    print(time.time() -begin)