import networkx as nx
import dynet as dy
import random
import numpy as np
import time



numberOfGraphs=80
for j in range(0,numberOfGraphs):
    inputgraf = []
    inputlength = []
    graphnumber = []
    #vystpu z neuronu
    outputlength = []
    outputsize=0
    neuronWeights = []
    #nacitanie grafu
    graph= nx.read_gml('./graphs/graph'+str(j*50)+'.gml',label='id')
    #pocet vrcholov
    number_of_nodes =list(graph.nodes()).__len__()
    nodes=list(graph.nodes())
    #vygenerovanie topologickeho zoradenia vrcholov
    gr=list(nx.topological_sort(graph))
    maximumLine=0
    for i in range(number_of_nodes):
        #prechadzania grafom a pridavanie vrcholov do predpripavenych premennych
        inputlength.append(graph.in_degree(i))
        #pridanie vystupu grafov
        outputlength.append(graph.out_degree(i))
        if(graph.out_degree(i)==0):
            outputsize=outputsize+1
        #pridanie vstupov grafu a predbezny fromat vrcholu
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([0,0])
        else:
            inputgraf.append(list(graph.predecessors(nodes[i])))


    graphInput=[x for x in graph.nodes() if graph.in_degree(x) == 0].__len__()
    graphnumber.extend([number_of_nodes])
    #print(time.time() - begin)


    #pouzitie dynet modelu
    m = dy.Model()
    #vygenerovanie input vektoru pre dynet model
    input=dy.vecInput(graphInput)
    #vytvorenie automatickeho pocitadla gradientu
    trainer = dy.SimpleSGDTrainer(m)
    #definovanie premennych
    values=[0]*number_of_nodes
    weight=[0]*number_of_nodes
    bias=[0]*number_of_nodes
    #pocitadlo vstupov
    a=0
    out=[]
    for i in range(number_of_nodes):
        next_neuron=gr[i]
        #dlzka vstupu
        if inputlength[next_neuron] == 0:
            #vygenerovanie vstupnych neuronov
            weight[next_neuron]= m.add_parameters(1)
            bias[next_neuron]= m.add_parameters(1)
            values[next_neuron] = dy.rectify( (weight[next_neuron]*input[a])+bias[next_neuron])
            a = a + 1

        elif inputlength[next_neuron] > 1:
            # vygeneorvnaie neuronov s viacerymi vstupmi
            weight[next_neuron]= m.add_parameters((inputlength[next_neuron]))
            bias[next_neuron]= m.add_parameters(1)
            l = dy.concatenate([values[j] for j in inputgraf[next_neuron]])

            values[next_neuron] = dy.rectify(dy.dot_product(weight[next_neuron],l) + bias[next_neuron])

        else:
            #vytvorenie vrcholu pre jeden vstup
            l = values[inputgraf[next_neuron][0]]
            weight[next_neuron] = m.add_parameters(1)
            bias[next_neuron] = m.add_parameters(1)
            values[next_neuron] = dy.rectify(weight[next_neuron] * l + bias[next_neuron])
        if outputlength[next_neuron]==0:
            out.extend([values[next_neuron]])

    y_pred=dy.concatenate(out)
    y=dy.vecInput(outputsize)
    #chyba v grafe
    loss = dy.squared_distance(y_pred, y)
    for iter in range(100):
        #vygenerovanie vstupov
        input.set(np.random.uniform(0,2,graphInput))
        y.set(np.random.uniform(0,2,outputsize))
        #pustenie chyb backpropagaciou
        loss.backward()
        #uplatnenie ucenia
        trainer.update()

