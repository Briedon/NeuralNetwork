import networkx as nx
import torch
import random
import numpy as np
import scipy
from scipy.sparse import coo_matrix
#import dynet_config
#dynet_config.set_gpu()
import dynet as dy

for number in range(1,50):
    model = nx.read_gml('./generated graphs/genr'+str(number*50)+'dense.gml',label='id')

    number_of_nodes=int(model.number_of_nodes())
    length_of_inputs=[0]*number_of_nodes
    list_of_predecessors=[-1]*number_of_nodes
    layer=[0]*number_of_nodes
    nodes=list(model.nodes())
    for i in range(number_of_nodes):
        length_of_inputs[i]=model.in_degree(nodes[i])
        list_of_predecessors[i]=(list(model.predecessors(nodes[i])))

    topological=list(nx.topological_sort(model))

    layer_number=0
    layer_list=[]
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


    def create_graph(init_weight, init_con, init_bias, weight, bias, con, input, target):
        x = dy.vecInput(input.size)
        x.set(input)
        y=dy.vecInput(target.size)
        y.set(target)
        result = init_weight * x
        hgt = dy.cmult(init_weight, init_con)
        init_weight.set_value(hgt.value())
        result = dy.logistic(result + init_bias)
        for j in range(layer_nodes[0].__len__()):
            values[layer_nodes[0][j]] = result[j]
        for i in range(maximum):
            inp = []
            for node in input_layer[i]:
                inp.extend([values[node]])
            inp = dy.concatenate(inp)
            weight[i].set_value(dy.cmult(weight[i], con[i]).value())
            result = weight[i] * inp
            result = dy.logistic(result + bias[i])
            for j in range(layer_nodes[i + 1].__len__()):
                values[layer_nodes[i + 1][j]] = result[j]
        loss = dy.squared_distance(y, result)
        return loss


    m = dy.Model()
    input = dy.vecInput(sizes[0])

    values = [0] * number_of_nodes
    init_weight=m.add_parameters((sizes[0],sizes[0]),init='identity')
    init_con=m.add_parameters((sizes[0],sizes[0]),init='identity')
    init_bias=m.add_parameters((sizes[0]))
    weight = [0] * maximum
    con = [0] * maximum
    bias = [0] * maximum
    a = 0

    for i in range(maximum):
            a=matrices[i]
            weight[i]=m.add_parameters(a.shape,init=a)
            bias[i]=m.add_parameters((layer_nodes[i+1].__len__()))
            con[i]=m.add_parameters(a.shape,init=a)

    trainer = dy.SimpleSGDTrainer(m)




    for iter in range(100):
        dy.renew_cg()
        mloss = 0.0
        loss=create_graph(init_weight, init_con,init_bias,weight,bias,con,np.random.uniform(0,2,sizes[0]),np.random.uniform(0,2,sizes[maximum]))

        mloss += loss.scalar_value()
        loss.backward()
        trainer.update()