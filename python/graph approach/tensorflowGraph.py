import networkx as nx
import random
import numpy as np
import keras
import time
import tensorflow as tf




numberOfGraphs=2
for j in range(1,numberOfGraphs):
    j=10
    #inicializace
    #vstup jednotlivych neuronov
    inputgraf = []
    #vstupna dlzka neuronov
    inputlength = []
    #cislo grafu
    graphnumber = []
    #vstupna vrstva grafu
    graphInput = []
    #vahy neuronov
    neuronWeights = []
    #premenne pre poslednu vrstvu grafu
    outputlength = []
    outputsize=0
    # nactenie grafu
    graph= nx.read_gml('C:/Users/mbriedon/Documents/GitHub/NeuralNetwork/python/graph approach/generated input/genr'+str(j*50)+'nor.gml',label='id')
    #cislovanie vrcholov
    n_nodes=int(graph.number_of_nodes())
    #vrcholy
    nodes=list(graph.nodes)
    # topologicke usporidanie grafu
    gr=list(nx.topological_sort(graph))#topologic order
    #pre kazdy vrchol vyplnime incializovane premenne
    for i in range(n_nodes):
        #pridame dlzku vstupu pre graf
        inputlength.append(graph.in_degree[i])
        #pridanie vystupu grafov
        outputlength.append(graph.out_degree(i))
        if(graph.out_degree(i)==0):
            outputsize=outputsize+1
        #predvyplnime typ vrcholu pre graf
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([1,1])
        else:
            inputgraf.append(list(graph.predecessors(nodes[i])))

    #vytvorenie vstupu z vrcholov
    input=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphInput = input.__len__()



    #vytvorenie vstupu a vystupu a definovanie ich velikosti
    input=tf.placeholder(tf.float32,shape=[graphInput])
    output=tf.placeholder(tf.float32,shape=[outputsize])
    values=[0]*(n_nodes+1)
    a=0
    #vytvorenie pocitacieho grafu v tensorflow
    config = tf.ConfigProto()
    config.intra_op_parallelism_threads = 1
    config.inter_op_parallelism_threads = 1
    sess=tf.Session(config=config)
    out=[]
    start_train=time.time()
    for i in range(n_nodes):
        #dalsi neuron na pridanie do vypocenteho grafu
        next_neuron=gr[i]
        #rozdelenie podla toho ci je to vstup, ma len jedne vstup alebo vela vstupov
        if inputlength[next_neuron] == 0:
            #neuron pouzity ako vstup
            values[next_neuron] =tf.nn.relu( tf.Variable(tf.ones([1]))*tf.slice(input,[a],[1])+tf.Variable(tf.ones([1])))
            a = a + 1
        elif inputlength[next_neuron] > 1:
            #neuron s viacerymi vstupmi
            #najprv vytvorime vektor vstupov
            l = tf.stack([values[j] for j in inputgraf[next_neuron]],axis=1)
            #pak ich dame do jednej vrstvy use matmul
            inp=tf.ones([inputlength[next_neuron],1])
            vars=tf.Variable(inp)
            sum=tf.matmul(l,vars)+tf.Variable(tf.ones([1]))
            sum=tf.reduce_sum(sum,0)
            values[next_neuron] = tf.nn.relu(sum)
            #values[next_neuron] =tf.reduce_sum(tf.keras.layers.Input(input=l,units=1,activation=tf.nn.relu),0)
        else:
            #neurony s jednym vstupom postupujeme rovnako ako pre vstup
            l = values[inputgraf[next_neuron][0]]
            values[next_neuron] =tf.nn.relu(tf.Variable(tf.ones([1]))*l+tf.Variable(tf.ones([1])))
        if outputlength[next_neuron]==0:
            out.extend([values[next_neuron]])
    out=tf.stack(out)
    #chyba predikcie vyt
    prediction_error=tf.reduce_sum(output-out)
    #vytvorenie trenovacieho nastroja pre graf
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)
    #inicializacia grfau
    sess.run(tf.global_variables_initializer())
    #trenovaci vstup
    training_input=np.random.uniform(0,2,graphInput)
    #trenovaci vystup
    training_output = np.random.uniform(0,2,outputsize)
    print("finish creating network")
    print(time.time() - start_train)
    start=time.time()
    print("start training")
    for step in range(10):
        #priebeh grafu a jehi ucenie
        sess.run(fetches=[train_op], feed_dict={input: training_input,output: training_output})
    print(time.time() - start)
    print(j)

