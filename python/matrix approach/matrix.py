import networkx as nx
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import time
import tensorflow as tf

# pocet grafov ktore chceme vyskusat
numberOfGraphs = 1

models = {}

inputs = {}
outputs = {}
for number in range(0, numberOfGraphs):
    # nacitanie grafu pomocou matice
    start = time.time()
    model = nx.drawing.nx_pydot.read_dot(
        'C:/Users/mbriedon/Documents/GitHub/NeuralNetwork/python/graph approach/mutagen/' + str(number) + 'graph.dot')
    nodes = list(model.nodes())
    mapping = {}
    for i in range(nodes.__len__()):
        mapping[nodes[i]] = i
    model = nx.convert_node_labels_to_integers(model)
    print(time.time() - start)
    # pocet vrcholov v danom grafe
    number_of_nodes = int(model.number_of_nodes())
    # inicializacia vstupov
    # dlzka vstupov pre dane vrcholy
    length_of_inputs = [0] * number_of_nodes
    # pocet predchodcov pre dany vrchol
    list_of_predecessors = [-1] * number_of_nodes
    # pole prislusnosti vrcholov k vrstvam
    layer = [0] * number_of_nodes
    # vrcholi
    nodes = list(model.nodes())
    # priradnie vstupov a predchodcov do poli
    for i in range(number_of_nodes):
        length_of_inputs[i] = model.in_degree(nodes[i])
        list_of_predecessors[i] = (list(model.predecessors(nodes[i])))
    # vytvorenie topologick=eho usporiadania
    topological = list(nx.topological_sort(model))
    # cislo vrstvy
    for i in topological:
        # toplogicky prechadzam graf a pridavam vrcholi do vrstiev
        # v ktorych vrstvach je najvacsi spolocni precchodca
        contains = [layer[j] for j in list_of_predecessors[i]]
        if (contains.__len__() == 0):
            layer[i] = 0
            continue
        the_biggest_predecessor = max(contains)
        layer[i] = the_biggest_predecessor + 1
    # vstupna vrstva
    input_layer = {}
    # vrchole vrstvy
    layer_nodes = {}
    # matice
    matrices = {}
    # najvyssia hodnota vrstvy, maxim=alna vrstva
    maximum = max(layer)
    # init velikosti vrstiev
    sizes = [0] * (maximum + 1)
    for i in range(maximum + 1):
        input_layer[i] = []
        # topologicky prejdeme vrcholi a priradime ktore vrccholy do ktoreho patria
        layer_nodes[i] = [j for j in topological if layer[j] == i]
        sizes[i] = layer_nodes[i].__len__()

    # prejdem vrcholy a pridam ich do vstupov pre jednotlive hrany
    c = 0
    for edge in model.edges():
        a = layer[edge[0]] - 1
        b = layer[edge[1]] - 1
        # kontorla vrstiev a ak je to prehodene tak ich vymenime
        if (a < b):
            a = b
            c = edge[0]
        else:
            c = edge[1]
        not_found = True
        # zistim ci je uz vo vstupnom vektore danej vrstvy
        for i in input_layer[a]:
            if (i == c):
                not_found = False
                break
        # ak este nebol pridany do vektora tak sa prida
        if not_found:
            input_layer[a].extend([c])
    # inicializovanie matic nulami v numpy
    for i in range(maximum):
        matrices[i] = np.zeros((sizes[i + 1], input_layer[i].__len__()))
    # aktualne vytovrenie matic
    for i in range(maximum):
        # nacitanie vrcholov vo vrstve
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
    # layer_nodes su posunute o jednu vrstvu

    dns=[]
    inp = tf.keras.layers.Dense(sizes[1], input_dim=sizes[0], activation='relu')
    dns.append(inp)
    for i in range(sizes.__len__() - 1):
        inp = tf.keras.layers.Dense(sizes[i + 1], activation='relu')
        dns.append(inp)
    model = tf.keras.models.Sequential(dns)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # for i in range(maximum-1):
    #     var=inp[i].get_weights()
    #     inp[i].set_weights(matrices[i])
    X = np.random.randint(0, 2, [1, sizes[0]])
    Y = np.random.randint(0, 2, [1, sizes[maximum]])

    # st = time.time()
    # model.fit(X, Y, epochs=100, batch_size=100)
    # print(time.time() - st)


    start = time.time()
    for i in range(1000):
        model.fit(X, Y, epochs=1, batch_size=1)
    print(time.time() - start)



    models[number] = model
    inputs[number] = X
    outputs[number] = Y

# for epochs in range(0,200):
#     start = time.time()
#     for number in range(0,numberOfGraphs):
#         models[number].fit(inputs[number], outputs[number], epochs=1, batch_size=1)
#     print(time.time() - start)
