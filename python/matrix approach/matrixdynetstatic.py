import networkx as nx
import random
import numpy as np
import scipy

#import dynet_config
#dynet_config.set_gpu()
import dynet as dy

for number in range(1,50):
    model = nx.read_gml('./generated graphs/genr'+str(number*50)+'dens.gml',label='id')

    number_of_nodes=int(model.number_of_nodes())
    length_of_inputs=[0]*number_of_nodes
    list_of_predecessors=[-1]*number_of_nodes
    layer=[0]*number_of_nodes
    nodes=list(model.nodes())
    for i in range(number_of_nodes):
        length_of_inputs[i]=model.in_degree(nodes[i])
        list_of_predecessors[i]=(list(model.predecessors(nodes[i])))

    topological=list(nx.topological_sort(model))


    for i in topological:
        contains=[layer[j] for j in list_of_predecessors[i]]
        if(contains.__len__()==0):
            layer[i]=0
            continue
        the_biggest_predecessor=max(contains)
        layer[i]=the_biggest_predecessor+1


    input_layer={}
    layer_nodes={}
    matrices={}
    maximum=max(layer)
    sizes=[0]*(maximum+1)
    for i in range(maximum+1):
        input_layer[i]=[]
        layer_nodes[i] = [j for j in topological if layer[j]==i]
        sizes[i]=layer_nodes[i].__len__()


    c=0
    for edge in model.edges():
        a=layer[edge[0]]-1
        b=layer[edge[1]]-1
        if(a<b):
            a=b
            c=edge[0]
        else:
            c=edge[1]
        not_found=True
        for i in input_layer[a]:
            if(i==c):
                not_found=False
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
    #tu zacina dynet implementace stickeho modelu
    m = dy.Model()
    #zapisanie input vektoru
    input = dy.vecInput(sizes[0])
    trainer = dy.SimpleSGDTrainer(m)
    values = [0] * number_of_nodes
    weight = [0] * maximum
    bias = [0] * maximum
    a = 0

    for i in range(maximum):
        if(i==0):
            a=matrices[i]
            weight[i]=m.add_parameters(a.shape,init=a)
            bias=m.add_parameters((layer_nodes[i+1].__len__()))
            con=dy.const_parameter(m.add_parameters(a.shape,init=a))
            weight[i]=dy.cmult(weight[i],con)
            result=weight[i]*input
            result =dy.logistic(result + bias)
            for j in range(layer_nodes[i+1].__len__()):
                values[layer_nodes[i+1][j]]=result[j]
        else:
            inp=[]
            for node in input_layer[i]:
                inp.extend([values[node]])
            a=matrices[i]
            weight[i] = m.add_parameters(a.shape, init=a)
            inp=dy.concatenate(inp)
            bias = m.add_parameters(layer_nodes[i+1].__len__())
            con = dy.const_parameter(m.add_parameters(a.shape, init=a))
            weight[i] = dy.cmult(weight[i], con)
            result = weight[i]*inp
            result=dy.logistic(result+bias)
            for j in range(layer_nodes[i+1].__len__()):
                values[layer_nodes[i+1][j]] = result[j]
    y_pred = result
    y = dy.vecInput(sizes[maximum])
    loss = dy.squared_distance(y_pred, y)

    for iter in range(100):
        mloss = 0.0
        input.set(np.random.uniform(0,2,sizes[0]))
        y.set(np.random.uniform(0,2,sizes[maximum]))
        mloss += loss.scalar_value()
        loss.backward()
        trainer.update()
    dy.renew_cg()

