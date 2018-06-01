import networkx as nx
import random
import numpy as np
import time
import tensorflow as tf




numberOfGraphs=50
for j in range(1,numberOfGraphs):
    inputgraf = []
    inputlength = []
    graphnumber = []
    graphLeaveCount = []
    gr = []
    neuronWeights = []
    graph= nx.read_gml('./generated graphs/genr'+str(j*50)+'nor.gml',label='id')
    n_nodes=int(graph.number_of_nodes())
    nodes=list(graph.nodes)
    gr.extend(list(nx.topological_sort(graph)))#topologic order
    maximumLine=0
    for i in range(n_nodes):
        inputlength.append(graph.in_degree[i])
        if graph.in_degree(nodes[i])==0:
            inputgraf.append([1,1])
        else:

            inputgraf.append(list(graph.predecessors(nodes[i])))


    leafs=[x for x in graph.nodes() if graph.in_degree(x) == 0]
    graphLeaveCount.append(leafs.__len__())
    graphnumber.extend([n_nodes])
    #print(time.time() - begin)



    input=tf.placeholder(tf.float32,shape=[graphLeaveCount[0]])
    output=tf.placeholder(tf.float32,shape=[None,1])
    values=[0]*(n_nodes+1)
    a=0

    sess=tf.Session()
    for i in range(n_nodes):
        next_neuron=gr[i]
        if inputlength[next_neuron] == 0:
            values[next_neuron] =tf.nn.relu( tf.Variable(tf.ones([1]))*tf.slice(input,[a],[1])+tf.Variable(tf.ones([1])))
            a = a + 1
        elif inputlength[next_neuron] > 1:

            l = tf.stack([values[j] for j in inputgraf[next_neuron]])
            values[next_neuron] =tf.reduce_sum(tf.layers.dense(inputs=l,units=1,activation=tf.nn.relu),0)
        else:
            l = values[inputgraf[next_neuron][0]]
            # print(l)

            values[next_neuron] =tf.nn.relu(tf.Variable(tf.ones([1]))*l+tf.Variable(tf.ones([1])))

    prediction_error=tf.reduce_sum(output-values[n_nodes-1])
    train_op = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(prediction_error)
    sess.run(tf.global_variables_initializer())
    training_input=np.random.uniform(0,2,graphLeaveCount[0])
    training_output = [[1]]

    begin = time.time()
    for step in range(100):

        sess.run(fetches=[train_op], feed_dict={input: training_input,output: training_output})

    print(time.time() - begin)
#print(sess.run(fetches=[values[n_nodes-1]], feed_dict={input: training_input,output: training_output}))